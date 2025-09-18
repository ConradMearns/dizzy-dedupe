#!/usr/bin/env python3
"""Main entry point for dizzy-dedupe application."""

import sys
import atexit
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .procedures.device_selector import run_device_selector, unmount_device, select_device
from .procedures.map_device_manifest import run as create_manifest

# Global variable to track mounted devices for cleanup
_mounted_devices = []

def cleanup_mounted_devices():
    """Clean up any mounted devices on exit."""
    global _mounted_devices
    console = Console()
    
    for mount_point in _mounted_devices:
        console.print(f"[yellow]Cleaning up mounted device: {mount_point}[/yellow]")
        unmount_device(mount_point)
    
    _mounted_devices.clear()

# Register cleanup function
atexit.register(cleanup_mounted_devices)


def main():
    """Main application entry point."""
    global _mounted_devices
    console = Console()
    
    # Display welcome message
    console.print(Panel.fit(
        "[bold blue]Dizzy Dedupe[/bold blue]\n"
        "File deduplication utility - Device manifest creator",
        border_style="blue"
    ))
    
    try:
        # Step 1: Select device
        console.print("\n[bold cyan]Step 1: Select Device[/bold cyan]")
        selected_device = select_device()
        
        if not selected_device:
            console.print("[yellow]No device selected. Exiting.[/yellow]")
            return
        
        # Check if we mounted the device
        if hasattr(selected_device, '_was_mounted_by_us') and selected_device._was_mounted_by_us:
            _mounted_devices.append(selected_device.mountpoint)
            console.print(f"[green]Device mounted at:[/green] {selected_device.mountpoint}")
        
        # Get the device path to scan
        device_path = selected_device.mountpoint or f"/dev/{selected_device.name.split('/')[-1]}"
        
        # Step 2: Get output file path
        console.print(f"\n[bold cyan]Step 2: Configure Output[/bold cyan]")
        
        # Use UUID as default filename if available
        if selected_device.uuid:
            default_output = f"{selected_device.uuid}.csv"
            console.print(f"[green]Using device UUID as filename:[/green] {selected_device.uuid}")
        else:
            # Fallback to device name if no UUID
            device_name = selected_device.name.split('/')[-1]
            default_output = f"{device_name}_manifest.csv"
            console.print(f"[yellow]No UUID found, using device name:[/yellow] {device_name}")
        
        output_file = Prompt.ask(
            "Output CSV file path",
            default=default_output
        )
        
        # Step 3: Create manifest
        console.print(f"\n[bold cyan]Step 3: Creating Manifest[/bold cyan]")
        console.print(f"[green]Scanning device:[/green] {device_path}")
        console.print(f"[green]Output file:[/green] {output_file}")
        
        confirm = Prompt.ask("Start manifest creation?", choices=["y", "n"], default="y")
        if confirm.lower() != 'y':
            console.print("[yellow]Operation cancelled.[/yellow]")
            return
        
        # Run the manifest creation
        console.print("\n[yellow]Creating manifest... This may take a while for large devices.[/yellow]")
        create_manifest(device_path, output_file)
        
        console.print(f"\n[bold green]✓ Manifest created successfully![/bold green]")
        console.print(f"[green]Output file:[/green] {Path(output_file).absolute()}")
        
        # Offer to keep device mounted or unmount
        if _mounted_devices:
            keep_mounted = Prompt.ask(
                "Keep device mounted for further use?",
                choices=["y", "n"],
                default="n"
            )
            if keep_mounted.lower() == 'n':
                cleanup_mounted_devices()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        cleanup_mounted_devices()
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        cleanup_mounted_devices()
        sys.exit(1)


if __name__ == "__main__":
    main()