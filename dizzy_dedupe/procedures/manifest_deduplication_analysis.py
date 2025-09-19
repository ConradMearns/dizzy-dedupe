#!/usr/bin/env python3
"""
Enhanced DuckDB-based manifest deduplication analysis script.

This script reads all CSV files in the manifests directory and processes them to:
1. Add filename mapping column
2. Calculate total size by path
3. Reorganize data by hash for deduplication
4. Count unique paths and filenames
5. Calculate total size of deduplicated files by hash
"""

import duckdb
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import List, Tuple, Dict


def find_manifest_csv_files(manifests_dir: str = "manifests") -> List[str]:
    """
    Find all CSV files in the manifests directory.
    
    Args:
        manifests_dir: Directory containing manifest CSV files
        
    Returns:
        List of CSV file paths
    """
    manifests_path = Path(manifests_dir)
    if not manifests_path.exists():
        raise FileNotFoundError(f"Manifests directory not found: {manifests_dir}")
    
    csv_files = list(manifests_path.glob("*.csv"))
    return [str(f) for f in sorted(csv_files)]


def analyze_manifests_with_deduplication(csv_files: List[str]) -> Dict:
    """
    Analyze manifest CSV files using DuckDB with enhanced deduplication analysis.
    
    Args:
        csv_files: List of CSV file paths to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    console = Console()
    
    # Create DuckDB connection
    conn = duckdb.connect()
    
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Step 1: Create temporary table with filename mapping
        progress.add_task("Creating temporary table with filename mapping...", total=None)
        conn.execute("""
            CREATE TEMPORARY TABLE manifest_files (
                path VARCHAR,
                hash VARCHAR,
                size_bytes BIGINT,
                source_filename VARCHAR,
                filename_only VARCHAR
            )
        """)
        
        # Step 2: Load all CSV files with filename mapping
        for csv_file in csv_files:
            task = progress.add_task(f"Loading {Path(csv_file).name}...", total=None)
            
            # Extract just the filename from the CSV path for mapping
            source_filename = Path(csv_file).name
            
            conn.execute(f"""
                INSERT INTO manifest_files 
                SELECT 
                    path, 
                    hash, 
                    size_bytes, 
                    '{source_filename}' as source_filename,
                    regexp_extract(path, '[^/\\\\]+$') as filename_only
                FROM read_csv_auto('{csv_file}')
            """)
            
            progress.remove_task(task)
        
        # Step 3: Calculate total size by path
        progress.add_task("Calculating total size by path...", total=None)
        
        path_sizes = conn.execute("""
            SELECT 
                path,
                SUM(size_bytes) as total_size_bytes,
                COUNT(*) as occurrence_count
            FROM manifest_files
            GROUP BY path
            ORDER BY total_size_bytes DESC
            LIMIT 10
        """).fetchall()
        
        results['top_paths_by_size'] = path_sizes
        
        # Step 4: Reorganize data by hash for deduplication
        progress.add_task("Reorganizing data by hash for deduplication...", total=None)
        
        # Create deduplication table
        conn.execute("""
            CREATE TEMPORARY TABLE deduplicated_files AS
            SELECT
                hash,
                MIN(size_bytes) as size_bytes,  -- Take first size (should be same for same hash)
                COUNT(DISTINCT path) as unique_paths,
                COUNT(DISTINCT filename_only) as unique_filenames,
                COUNT(*) as total_occurrences,
                array_agg(DISTINCT source_filename) as source_manifests,
                array_agg(DISTINCT path ORDER BY path) as sample_paths
            FROM manifest_files
            WHERE hash IS NOT NULL AND hash != ''
            GROUP BY hash
        """)
        
        # Step 5: Count unique paths and filenames analysis
        progress.add_task("Analyzing unique paths and filenames...", total=None)
        
        # Get overall statistics
        overall_stats = conn.execute("""
            SELECT 
                COUNT(*) as total_file_entries,
                COUNT(DISTINCT hash) as unique_hashes,
                COUNT(DISTINCT path) as unique_paths,
                COUNT(DISTINCT filename_only) as unique_filenames,
                SUM(size_bytes) as total_size_all_files
            FROM manifest_files
            WHERE hash IS NOT NULL AND hash != ''
        """).fetchone()
        
        results['overall_stats'] = {
            'total_file_entries': overall_stats[0],
            'unique_hashes': overall_stats[1],
            'unique_paths': overall_stats[2],
            'unique_filenames': overall_stats[3],
            'total_size_all_files': overall_stats[4]
        }
        
        # Step 6: Calculate total size of deduplicated files by hash
        progress.add_task("Calculating deduplicated size by hash...", total=None)
        
        dedup_stats = conn.execute("""
            SELECT 
                COUNT(*) as unique_file_count,
                SUM(size_bytes) as deduplicated_total_size,
                SUM(size_bytes * (total_occurrences - 1)) as space_that_could_be_saved,
                AVG(unique_paths) as avg_paths_per_hash,
                AVG(unique_filenames) as avg_filenames_per_hash
            FROM deduplicated_files
        """).fetchone()
        
        results['deduplication_stats'] = {
            'unique_file_count': dedup_stats[0],
            'deduplicated_total_size': dedup_stats[1],
            'space_that_could_be_saved': dedup_stats[2],
            'avg_paths_per_hash': dedup_stats[3],
            'avg_filenames_per_hash': dedup_stats[4]
        }
        
        # Get top duplicates by space wasted
        top_duplicates = conn.execute("""
            SELECT 
                hash,
                size_bytes,
                total_occurrences,
                unique_paths,
                unique_filenames,
                size_bytes * (total_occurrences - 1) as space_wasted,
                sample_paths
            FROM deduplicated_files
            WHERE total_occurrences > 1
            ORDER BY space_wasted DESC
            LIMIT 15
        """).fetchall()
        
        results['top_duplicates'] = top_duplicates
        
        # Get files that appear across multiple manifests
        cross_manifest_files = conn.execute("""
            SELECT
                hash,
                size_bytes,
                len(source_manifests) as manifest_count,
                source_manifests,
                unique_paths,
                sample_paths[1] as example_path
            FROM deduplicated_files
            WHERE len(source_manifests) > 1
            ORDER BY manifest_count DESC, size_bytes DESC
            LIMIT 10
        """).fetchall()
        
        results['cross_manifest_files'] = cross_manifest_files
        
        # Get file type distribution
        progress.add_task("Analyzing file type distribution...", total=None)
        
        file_type_distribution = conn.execute("""
            WITH file_type_raw AS (
                SELECT
                    CASE
                        WHEN filename_only LIKE '%.%' THEN
                            UPPER(regexp_extract(filename_only, '\\.([^.]+)$', 1))
                        ELSE
                            'NO_EXTENSION'
                    END as file_extension,
                    hash,
                    size_bytes
                FROM manifest_files
                WHERE hash IS NOT NULL AND hash != ''
            ),
            file_type_stats AS (
                SELECT
                    file_extension,
                    COUNT(*) as file_count,
                    COUNT(DISTINCT hash) as unique_files,
                    SUM(size_bytes) as total_size,
                    AVG(size_bytes) as avg_size,
                    MIN(size_bytes) as min_size,
                    MAX(size_bytes) as max_size
                FROM file_type_raw
                GROUP BY file_extension
            ),
            file_type_dedup AS (
                SELECT
                    file_extension,
                    SUM(min_size_per_hash) as deduplicated_total_size
                FROM (
                    SELECT
                        file_extension,
                        hash,
                        MIN(size_bytes) as min_size_per_hash
                    FROM file_type_raw
                    GROUP BY file_extension, hash
                ) grouped_by_hash
                GROUP BY file_extension
            )
            SELECT
                fts.file_extension,
                fts.file_count,
                fts.unique_files,
                fts.total_size,
                ftd.deduplicated_total_size,
                fts.avg_size,
                fts.min_size,
                fts.max_size,
                (fts.total_size - ftd.deduplicated_total_size) as space_saved_by_type
            FROM file_type_stats fts
            JOIN file_type_dedup ftd ON fts.file_extension = ftd.file_extension
            ORDER BY fts.total_size DESC
            LIMIT 20
        """).fetchall()
        
        results['file_type_distribution'] = file_type_distribution
    
    conn.close()
    return results


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable format.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    if bytes_value == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(bytes_value)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}"


def display_enhanced_results(results: Dict) -> None:
    """
    Display the enhanced deduplication analysis results using Rich formatting.
    
    Args:
        results: Dictionary containing all analysis results
    """
    console = Console()
    
    overall = results['overall_stats']
    dedup = results['deduplication_stats']
    
    # Calculate key metrics
    duplicate_entries = overall['total_file_entries'] - overall['unique_hashes']
    space_saved = dedup['space_that_could_be_saved']
    savings_percentage = (space_saved / overall['total_size_all_files'] * 100) if overall['total_size_all_files'] > 0 else 0
    
    # Main results panel
    console.print(Panel.fit(
        f"[bold blue]Enhanced Manifest Deduplication Analysis[/bold blue]\n\n"
        f"[green]Total File Entries:[/green] {overall['total_file_entries']:,}\n"
        f"[green]Unique Hashes:[/green] {overall['unique_hashes']:,}\n"
        f"[green]Unique Paths:[/green] {overall['unique_paths']:,}\n"
        f"[green]Unique Filenames:[/green] {overall['unique_filenames']:,}\n"
        f"[red]Duplicate Entries:[/red] {duplicate_entries:,}\n\n"
        f"[green]Total Size (All Files):[/green] {format_bytes(overall['total_size_all_files'])}\n"
        f"[green]Size After Deduplication:[/green] {format_bytes(dedup['deduplicated_total_size'])}\n"
        f"[yellow]Space That Could Be Saved:[/yellow] {format_bytes(space_saved)}\n"
        f"[yellow]Savings Percentage:[/yellow] {savings_percentage:.1f}%\n\n"
        f"[cyan]Avg Paths per Hash:[/cyan] {dedup['avg_paths_per_hash']:.2f}\n"
        f"[cyan]Avg Filenames per Hash:[/cyan] {dedup['avg_filenames_per_hash']:.2f}",
        border_style="blue",
        title="Enhanced Summary"
    ))
    
    # Top paths by size
    if results['top_paths_by_size']:
        console.print("\n")
        table = Table(title="Top 10 Paths by Total Size", show_header=True, header_style="bold magenta")
        table.add_column("Path", style="dim", min_width=30)
        table.add_column("Total Size", style="green", justify="right")
        table.add_column("Occurrences", style="yellow", justify="right")
        
        for path, size, count in results['top_paths_by_size']:
            # Truncate long paths for display
            display_path = path if len(path) <= 50 else "..." + path[-47:]
            table.add_row(
                display_path,
                format_bytes(size),
                str(count)
            )
        
        console.print(table)
    
    # Top duplicates by wasted space
    if results['top_duplicates']:
        console.print("\n")
        table = Table(title="Top 15 Duplicates by Wasted Space", show_header=True, header_style="bold red")
        table.add_column("Hash (first 16)", style="dim", min_width=18)
        table.add_column("File Size", style="green", justify="right")
        table.add_column("Occurrences", style="red", justify="right")
        table.add_column("Unique Paths", style="yellow", justify="right")
        table.add_column("Unique Names", style="cyan", justify="right")
        table.add_column("Space Wasted", style="red", justify="right")
        
        for hash_val, size, occurrences, paths, names, wasted, sample_paths in results['top_duplicates']:
            table.add_row(
                hash_val[:16] + "...",
                format_bytes(size),
                str(occurrences),
                str(paths),
                str(names),
                format_bytes(wasted)
            )
        
        console.print(table)
    
    # Files appearing across multiple manifests
    if results['cross_manifest_files']:
        console.print("\n")
        table = Table(title="Files Appearing Across Multiple Manifests", show_header=True, header_style="bold cyan")
        table.add_column("Hash (first 16)", style="dim", min_width=18)
        table.add_column("Size", style="green", justify="right")
        table.add_column("Manifest Count", style="cyan", justify="right")
        table.add_column("Unique Paths", style="yellow", justify="right")
        table.add_column("Example Path", style="dim", min_width=30)
        
        for hash_val, size, manifest_count, manifests, paths, example_path in results['cross_manifest_files']:
            # Truncate long paths for display
            display_path = example_path if len(example_path) <= 40 else "..." + example_path[-37:]
            table.add_row(
                hash_val[:16] + "...",
                format_bytes(size),
                str(manifest_count),
                str(paths),
                display_path
            )
        
        console.print(table)
    
    # File type distribution table
    if results['file_type_distribution']:
        console.print("\n")
        table = Table(title="Top 20 File Types by Total Size (with Deduplication)", show_header=True, header_style="bold green")
        table.add_column("Extension", style="cyan", min_width=12)
        table.add_column("File Count", style="yellow", justify="right")
        table.add_column("Unique Files", style="green", justify="right")
        table.add_column("Total Size", style="blue", justify="right")
        table.add_column("Deduplicated Size", style="bright_green", justify="right")
        table.add_column("Space Saved", style="red", justify="right")
        table.add_column("Savings %", style="magenta", justify="right")
        
        for ext, count, unique, total_size, dedup_size, avg_size, min_size, max_size, space_saved in results['file_type_distribution']:
            # Handle potential None values
            if avg_size is None:
                avg_size = 0
            if min_size is None:
                min_size = 0
            if max_size is None:
                max_size = 0
            if dedup_size is None:
                dedup_size = 0
            if space_saved is None:
                space_saved = 0
                
            # Calculate savings percentage
            savings_pct = (space_saved / total_size * 100) if total_size > 0 else 0
            
            table.add_row(
                ext if ext else "NO_EXT",
                f"{count:,}",
                f"{unique:,}",
                format_bytes(total_size),
                format_bytes(dedup_size),
                format_bytes(space_saved),
                f"{savings_pct:.1f}%"
            )
        
        console.print(table)


def run(manifests_dir: str = "manifests") -> None:
    """
    Main function to run the enhanced manifest deduplication analysis.
    
    Args:
        manifests_dir: Directory containing manifest CSV files
    """
    console = Console()
    
    # Display header
    console.print(Panel.fit(
        "[bold blue]Dizzy Dedupe - Enhanced Manifest Analysis[/bold blue]\n"
        "Analyzing manifest CSV files for comprehensive deduplication insights using DuckDB",
        border_style="blue"
    ))
    
    try:
        # Find CSV files
        csv_files = find_manifest_csv_files(manifests_dir)
        
        if not csv_files:
            console.print(f"[red]No CSV files found in {manifests_dir} directory![/red]")
            return
        
        console.print(f"\n[green]Found {len(csv_files)} manifest CSV files:[/green]")
        for csv_file in csv_files:
            console.print(f"  • {Path(csv_file).name}")
        
        # Analyze files
        console.print("\n[yellow]Performing enhanced analysis...[/yellow]")
        results = analyze_manifests_with_deduplication(csv_files)
        
        # Display results
        console.print("\n")
        display_enhanced_results(results)
        
    except Exception as e:
        console.print(f"\n[red]Error during analysis: {e}[/red]")
        raise


if __name__ == "__main__":
    run()