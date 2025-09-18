"""Map device manifest - creates a CSV manifest of all files on a block device."""

import os
import hashlib
from pathlib import Path
from typing import Union


def run(blk_device: Union[str, Path], output_csv: Union[str, Path] = "device_manifest.csv") -> None:
    """
    Iterate over every file on a block device and create a CSV manifest.
    
    Args:
        blk_device: Path to the block device or mount point to scan
        output_csv: Path to the output CSV file (default: "device_manifest.csv")
    """
    blk_device = Path(blk_device)
    output_csv = Path(output_csv)
    
    # Ensure the block device/mount point exists
    if not blk_device.exists():
        raise FileNotFoundError(f"Block device or mount point not found: {blk_device}")
    
    # Create CSV header if file doesn't exist
    if not output_csv.exists():
        with open(output_csv, 'w', encoding='utf-8') as f:
            f.write("path,hash,size_bytes\n")
    
    # Walk through all files on the device
    for root, dirs, files in os.walk(blk_device):
        for file_name in files:
            file_path = Path(root) / file_name
            
            try:
                # Skip if not a regular file (symlinks, devices, etc.)
                if not file_path.is_file():
                    continue
                
                # Get file size
                file_size = file_path.stat().st_size
                
                # Calculate SHA256 hash
                file_hash = _calculate_file_hash(file_path)
                
                # Escape any commas or quotes in the path for CSV
                escaped_path = str(file_path).replace('"', '""')
                if ',' in escaped_path or '"' in str(file_path):
                    escaped_path = f'"{escaped_path}"'
                
                # Write line to CSV manually
                csv_line = f"{escaped_path},{file_hash},{file_size}\n"
                
                with open(output_csv, 'a', encoding='utf-8') as f:
                    f.write(csv_line)
                
                print(f"Processed: {file_path}")
                
            except (PermissionError, OSError) as e:
                print(f"Error processing {file_path}: {e}")
                continue


def _calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hexadecimal SHA256 hash string
    """
    hash_sha256 = hashlib.sha256()
    
    try:
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
    except (PermissionError, OSError):
        # Return empty hash for files we can't read
        return ""
    
    return hash_sha256.hexdigest()