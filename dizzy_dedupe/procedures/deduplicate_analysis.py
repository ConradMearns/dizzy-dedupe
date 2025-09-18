#!/usr/bin/env python3
"""
DuckDB-based CSV deduplication analysis script.

This script reads all CSV files in the current directory and processes them
to calculate the total size after deduplicating files based on their hash values.
"""

import duckdb
import glob
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import List, Tuple


def find_csv_files() -> List[str]:
    """
    Find all CSV files in the current directory.
    
    Returns:
        List of CSV file paths
    """
    csv_files = glob.glob("*.csv")
    return sorted(csv_files)


def analyze_csv_files(csv_files: List[str]) -> Tuple[int, int, int, int]:
    """
    Analyze CSV files using DuckDB to calculate deduplication statistics.
    
    Args:
        csv_files: List of CSV file paths to analyze
        
    Returns:
        Tuple of (total_files, unique_files, total_size, deduplicated_size)
    """
    console = Console()
    
    # Create DuckDB connection
    conn = duckdb.connect()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Create a temporary table to hold all data
        progress.add_task("Creating temporary table...", total=None)
        conn.execute("""
            CREATE TEMPORARY TABLE all_files (
                path VARCHAR,
                hash VARCHAR,
                size_bytes BIGINT,
                source_file VARCHAR
            )
        """)
        
        # Load all CSV files into the temporary table
        for csv_file in csv_files:
            task = progress.add_task(f"Loading {csv_file}...", total=None)
            
            # Use DuckDB's CSV reading capabilities
            conn.execute(f"""
                INSERT INTO all_files 
                SELECT path, hash, size_bytes, '{csv_file}' as source_file
                FROM read_csv_auto('{csv_file}')
            """)
            
            progress.remove_task(task)
        
        # Calculate statistics
        progress.add_task("Calculating statistics...", total=None)
        
        # Total files and size
        total_result = conn.execute("""
            SELECT COUNT(*) as total_files, SUM(size_bytes) as total_size
            FROM all_files
        """).fetchone()
        
        total_files = total_result[0]
        total_size = total_result[1]
        
        # Unique files and deduplicated size (group by hash, take first occurrence)
        unique_result = conn.execute("""
            SELECT COUNT(*) as unique_files, SUM(size_bytes) as deduplicated_size
            FROM (
                SELECT hash, MIN(size_bytes) as size_bytes
                FROM all_files
                GROUP BY hash
            )
        """).fetchone()
        
        unique_files = unique_result[0]
        deduplicated_size = unique_result[1]
    
    conn.close()
    return total_files, unique_files, total_size, deduplicated_size


def get_duplicate_analysis(csv_files: List[str]) -> List[Tuple[str, int, int]]:
    """
    Get detailed analysis of duplicate files.
    
    Args:
        csv_files: List of CSV file paths to analyze
        
    Returns:
        List of tuples (hash, count, total_size) for files with duplicates
    """
    conn = duckdb.connect()
    
    # Create temporary table and load data
    conn.execute("""
        CREATE TEMPORARY TABLE all_files (
            path VARCHAR,
            hash VARCHAR,
            size_bytes BIGINT,
            source_file VARCHAR
        )
    """)
    
    for csv_file in csv_files:
        conn.execute(f"""
            INSERT INTO all_files 
            SELECT path, hash, size_bytes, '{csv_file}' as source_file
            FROM read_csv_auto('{csv_file}')
        """)
    
    # Find duplicates (files with same hash appearing more than once)
    duplicates = conn.execute("""
        SELECT 
            hash,
            COUNT(*) as duplicate_count,
            SUM(size_bytes) as total_wasted_space
        FROM all_files
        GROUP BY hash
        HAVING COUNT(*) > 1
        ORDER BY total_wasted_space DESC
        LIMIT 10
    """).fetchall()
    
    conn.close()
    return duplicates


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


def display_results(total_files: int, unique_files: int, total_size: int, deduplicated_size: int, duplicates: List[Tuple[str, int, int]]) -> None:
    """
    Display the deduplication analysis results using Rich formatting.
    
    Args:
        total_files: Total number of files
        unique_files: Number of unique files (after deduplication)
        total_size: Total size of all files
        deduplicated_size: Size after deduplication
        duplicates: List of duplicate file information
    """
    console = Console()
    
    # Calculate savings
    space_saved = total_size - deduplicated_size
    duplicate_files = total_files - unique_files
    savings_percentage = (space_saved / total_size * 100) if total_size > 0 else 0
    
    # Main results panel
    console.print(Panel.fit(
        f"[bold blue]Deduplication Analysis Results[/bold blue]\n\n"
        f"[green]Total Files:[/green] {total_files:,}\n"
        f"[green]Unique Files:[/green] {unique_files:,}\n"
        f"[red]Duplicate Files:[/red] {duplicate_files:,}\n\n"
        f"[green]Total Size:[/green] {format_bytes(total_size)}\n"
        f"[green]Size After Deduplication:[/green] {format_bytes(deduplicated_size)}\n"
        f"[yellow]Space That Could Be Saved:[/yellow] {format_bytes(space_saved)}\n"
        f"[yellow]Savings Percentage:[/yellow] {savings_percentage:.1f}%",
        border_style="blue",
        title="Summary"
    ))
    
    # Top duplicates table
    if duplicates:
        console.print("\n")
        table = Table(title="Top 10 Files by Wasted Space", show_header=True, header_style="bold magenta")
        table.add_column("Hash (first 16 chars)", style="dim", min_width=18)
        table.add_column("Duplicate Count", style="red", justify="right")
        table.add_column("Total Wasted Space", style="yellow", justify="right")
        
        for hash_val, count, wasted_space in duplicates:
            table.add_row(
                hash_val[:16] + "...",
                str(count),
                format_bytes(wasted_space)
            )
        
        console.print(table)


def main():
    """Main function to run the deduplication analysis."""
    console = Console()
    
    # Display header
    console.print(Panel.fit(
        "[bold blue]Dizzy Dedupe - CSV Analysis[/bold blue]\n"
        "Analyzing CSV files for deduplication opportunities using DuckDB",
        border_style="blue"
    ))
    
    try:
        # Find CSV files
        csv_files = find_csv_files()
        
        if not csv_files:
            console.print("[red]No CSV files found in the current directory![/red]")
            return
        
        console.print(f"\n[green]Found {len(csv_files)} CSV files:[/green]")
        for csv_file in csv_files:
            console.print(f"  • {csv_file}")
        
        # Analyze files
        console.print("\n[yellow]Analyzing files...[/yellow]")
        total_files, unique_files, total_size, deduplicated_size = analyze_csv_files(csv_files)
        
        # Get duplicate analysis
        console.print("[yellow]Analyzing duplicates...[/yellow]")
        duplicates = get_duplicate_analysis(csv_files)
        
        # Display results
        console.print("\n")
        display_results(total_files, unique_files, total_size, deduplicated_size, duplicates)
        
    except Exception as e:
        console.print(f"\n[red]Error during analysis: {e}[/red]")
        raise


if __name__ == "__main__":
    main()