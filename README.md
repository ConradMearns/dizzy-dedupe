# Dizzy Dedupe

A file deduplication utility that creates manifests of block devices for analysis and deduplication.

## Features

- Interactive device selection using Rich UI
- **Automatic device mounting** - Mount unmounted devices temporarily for scanning
- Complete file system scanning with SHA256 hashing
- CSV manifest generation with file paths, hashes, and sizes
- Support for all block devices and mount points
- Automatic cleanup of temporarily mounted devices
- Raw device scanning option for advanced users

## Installation

1. Install dependencies using Poetry:
```bash
poetry install
```

2. Activate the virtual environment:
```bash
poetry shell
```

## Usage

### Interactive Mode

Run the main application for an interactive experience:

```bash
python -m dizzy_dedupe.main
```

This will:
1. Display available block devices in a Rich table (showing mount status)
2. Allow you to select a device interactively
3. **For unmounted devices**: Offer to mount automatically or scan raw device
4. **Automatically suggest UUID-based filename** for the CSV output
5. Prompt for output CSV file path (with UUID default)
6. Create a complete manifest of all files on the device
7. Automatically unmount temporarily mounted devices when done

### Programmatic Usage

You can also use the modules directly in your Python code:

```python
from dizzy_dedupe.procedures.device_selector import run_device_selector
from dizzy_dedupe.procedures.map_device_manifest import run

# Select device interactively
device_path = run_device_selector()

# Create manifest
if device_path:
    run(device_path, "my_manifest.csv")
```

### Direct Manifest Creation

If you know the device path, you can create a manifest directly:

```python
from dizzy_dedupe.procedures.map_device_manifest import run

# Create manifest for a specific device/mount point
run("/dev/sda1", "output.csv")
# or
run("/mnt/my_drive", "output.csv")
```

## Project Structure

```
dizzy_dedupe/
├── __init__.py
├── main.py                    # Main application entry point
└── procedures/
    ├── __init__.py
    ├── device_selector.py     # Rich-based device selection UI
    └── map_device_manifest.py # File system scanning and CSV generation
```

## Output Format

The generated CSV file contains the following columns:
- `path`: Full file path
- `hash`: SHA256 hash of the file content
- `size_bytes`: File size in bytes

The CSV filename is automatically generated using the device's UUID for easy identification and organization.

Example output filename: `a1b2c3d4-e5f6-7890-abcd-ef1234567890.csv`

Example CSV content:
```csv
path,hash,size_bytes
"/home/user/document.txt","a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3","123"
"/home/user/image.jpg","e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","45678"
```

## Testing

Test the device selector:
```bash
python test_device_selector.py
```

Test the manifest creation:
```bash
python test_manifest.py
```

## Requirements

- Python 3.11+
- Rich library for UI
- Standard library modules: os, hashlib, pathlib, subprocess, json

## Device Mounting

When you select an unmounted device, the application will offer three options:

1. **Mount** (default): Automatically mount the device to a temporary directory
2. **Scan Raw**: Scan the raw device directly (requires elevated permissions)
3. **Cancel**: Go back to device selection

### Mounting Features:
- Uses `sudo mount` to temporarily mount devices
- Creates temporary mount points in `/tmp/dizzy_mount_*`
- Automatically detects filesystem type or allows manual specification
- Cleans up mount points when the application exits
- Offers to keep devices mounted after scanning for further use

## Notes

- The application requires appropriate permissions to read files on the target device
- **Mounting requires sudo privileges** for the mount/umount commands
- Large devices may take considerable time to scan
- Files that cannot be read (due to permissions) will be skipped with a warning
- The CSV output uses proper escaping for paths containing commas or quotes
- Temporarily mounted devices are automatically unmounted on exit or error