"""Device selector using Rich interface to display and select block devices."""

import subprocess
import json
import tempfile
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text


class DeviceInfo:
    """Represents information about a block device."""
    
    def __init__(self, name: str, size: str, label: str, fstype: str, uuid: str, serial: str = "", mountpoint: str = ""):
        self.name = name
        self.size = size
        self.label = label or ""
        self.fstype = fstype or ""
        self.uuid = uuid or ""
        self.serial = serial or ""
        self.mountpoint = mountpoint or ""


def get_block_devices() -> List[DeviceInfo]:
    """
    Get block device information using lsblk command.
    
    Returns:
        List of DeviceInfo objects containing device information
    """
    try:
        # Run lsblk with JSON output for easier parsing
        result = subprocess.run(
            ["lsblk", "-J", "-o", "NAME,SIZE,LABEL,FSTYPE,UUID,MOUNTPOINT"],
            capture_output=True,
            text=True,
            check=True
        )
        
        data = json.loads(result.stdout)
        devices = []
        
        def process_device(device_data, parent_name=""):
            """Recursively process device data including children."""
            name = device_data.get("name", "")
            if parent_name:
                name = f"{parent_name}/{name}"
            
            device = DeviceInfo(
                name=name,
                size=device_data.get("size", ""),
                label=device_data.get("label", ""),
                fstype=device_data.get("fstype", ""),
                uuid=device_data.get("uuid", ""),
                mountpoint=device_data.get("mountpoint", "")
            )
            devices.append(device)
            
            # Process children (partitions)
            children = device_data.get("children", [])
            for child in children:
                process_device(child, name)
        
        for device in data.get("blockdevices", []):
            process_device(device)
        
        return devices
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run lsblk command: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse lsblk output: {e}")


def detect_filesystem_type(device_path: str) -> Optional[str]:
    """
    Detect filesystem type using blkid command.
    
    Args:
        device_path: Path to the device (e.g., /dev/sdd)
    
    Returns:
        Detected filesystem type or None if detection fails
    """
    try:
        result = subprocess.run(
            ["sudo", "blkid", "-o", "value", "-s", "TYPE", device_path],
            capture_output=True,
            text=True,
            check=True
        )
        fstype = result.stdout.strip()
        return fstype if fstype else None
    except subprocess.CalledProcessError:
        return None


def mount_device(device_path: str, fstype: str = None) -> Tuple[bool, str]:
    """
    Mount a device to a temporary directory with enhanced Windows drive support.
    
    Args:
        device_path: Path to the device (e.g., /dev/sdd1)
        fstype: Filesystem type (optional, will auto-detect if not provided)
    
    Returns:
        Tuple of (success: bool, mount_point_or_error: str)
    """
    console = Console()
    
    try:
        # Create a temporary mount point
        temp_dir = tempfile.mkdtemp(prefix="dizzy_mount_")
        
        # If no filesystem type provided, try to detect it
        if not fstype:
            console.print(f"[yellow]Detecting filesystem type for {device_path}...[/yellow]")
            fstype = detect_filesystem_type(device_path)
            if fstype:
                console.print(f"[cyan]Detected filesystem: {fstype}[/cyan]")
        
        # List of filesystem types to try for Windows drives
        fstypes_to_try = []
        
        if fstype:
            fstypes_to_try.append(fstype)
        
        # Add common Windows filesystem types if not already specified
        common_windows_types = ["ntfs", "vfat", "exfat"]
        for fs in common_windows_types:
            if fs not in fstypes_to_try:
                fstypes_to_try.append(fs)
        
        # Also try auto-detection
        if None not in fstypes_to_try:
            fstypes_to_try.append(None)
        
        last_error = ""
        
        for attempt_fstype in fstypes_to_try:
            try:
                # Prepare mount command
                mount_cmd = ["sudo", "mount"]
                
                if attempt_fstype:
                    mount_cmd.extend(["-t", attempt_fstype])
                    console.print(f"[yellow]Attempting to mount {device_path} as {attempt_fstype} to {temp_dir}...[/yellow]")
                else:
                    console.print(f"[yellow]Attempting auto-detection mount of {device_path} to {temp_dir}...[/yellow]")
                
                # Add read-only option for safety
                mount_cmd.extend(["-o", "ro"])
                mount_cmd.extend([device_path, temp_dir])
                
                # Execute mount command
                result = subprocess.run(
                    mount_cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                console.print(f"[green]✓ Device mounted successfully at {temp_dir}[/green]")
                if attempt_fstype:
                    console.print(f"[green]✓ Filesystem type: {attempt_fstype}[/green]")
                return True, temp_dir
                
            except subprocess.CalledProcessError as e:
                last_error = e.stderr.strip() if e.stderr else str(e)
                if attempt_fstype:
                    console.print(f"[dim]Failed to mount as {attempt_fstype}: {last_error}[/dim]")
                continue
        
        # If all attempts failed
        error_msg = f"Failed to mount device with any filesystem type. Last error: {last_error}"
        console.print(f"[red]✗ {error_msg}[/red]")
        
        # Clean up temp directory if mount failed
        try:
            os.rmdir(temp_dir)
        except:
            pass
            
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error during mount: {str(e)}"
        console.print(f"[red]✗ {error_msg}[/red]")
        return False, error_msg


def unmount_device(mount_point: str) -> bool:
    """
    Unmount a device and clean up the temporary mount point.
    
    Args:
        mount_point: Path to the mount point
    
    Returns:
        True if successful, False otherwise
    """
    console = Console()
    
    try:
        # Unmount the device
        result = subprocess.run(
            ["sudo", "umount", mount_point],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Remove the temporary directory
        os.rmdir(mount_point)
        
        console.print(f"[green]✓ Device unmounted and cleanup completed[/green]")
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to unmount: {e.stderr.strip() if e.stderr else str(e)}"
        console.print(f"[red]✗ {error_msg}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ Unexpected error during unmount: {str(e)}[/red]")
        return False


def display_devices_table(devices: List[DeviceInfo]) -> None:
    """
    Display devices in a Rich table format.
    
    Args:
        devices: List of DeviceInfo objects to display
    """
    console = Console()
    
    table = Table(title="Available Block Devices", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Name", style="cyan", min_width=12)
    table.add_column("Size", style="green", min_width=8)
    table.add_column("Label", style="yellow", min_width=10)
    table.add_column("FS Type", style="blue", min_width=8)
    table.add_column("UUID", style="dim", min_width=20)
    table.add_column("Mount Point", style="bright_green", min_width=12)
    
    for i, device in enumerate(devices, 1):
        # Highlight mounted devices
        name_style = "bright_cyan" if device.mountpoint else "cyan"
        
        table.add_row(
            str(i),
            Text(device.name, style=name_style),
            device.size,
            device.label or "-",
            device.fstype or "-",
            device.uuid[:8] + "..." if len(device.uuid) > 8 else device.uuid or "-",
            device.mountpoint or "-"
        )
    
    console.print(table)


def select_device() -> Optional[DeviceInfo]:
    """
    Interactive device selection using Rich interface.
    
    Returns:
        Selected DeviceInfo object or None if cancelled
    """
    console = Console()
    
    try:
        # Get available devices
        devices = get_block_devices()
        
        if not devices:
            console.print("[red]No block devices found![/red]")
            return None
        
        # Display header
        console.print(Panel.fit(
            "[bold blue]Device Selector[/bold blue]\n"
            "Select a block device to scan for file manifest creation.",
            border_style="blue"
        ))
        
        # Display devices table
        display_devices_table(devices)
        
        # Get user selection
        console.print("\n[yellow]Enter device number (or 'q' to quit):[/yellow]")
        
        while True:
            choice = Prompt.ask("Selection", default="q")
            
            if choice.lower() == 'q':
                console.print("[yellow]Selection cancelled.[/yellow]")
                return None
            
            try:
                device_num = int(choice)
                if 1 <= device_num <= len(devices):
                    selected_device = devices[device_num - 1]
                    
                    # Show confirmation and handle mounting
                    console.print(f"\n[green]Selected device:[/green] {selected_device.name}")
                    
                    if selected_device.mountpoint:
                        console.print(f"[green]Mount point:[/green] {selected_device.mountpoint}")
                        confirm = Prompt.ask("Proceed with this device?", choices=["y", "n"], default="y")
                        if confirm.lower() == 'y':
                            return selected_device
                        else:
                            console.print("[yellow]Please select a different device:[/yellow]")
                            continue
                    else:
                        console.print("[yellow]Warning: Device is not mounted[/yellow]")
                        
                        # Offer mounting options
                        mount_choice = Prompt.ask(
                            "What would you like to do?",
                            choices=["mount", "scan_raw", "cancel"],
                            default="mount"
                        )
                        
                        if mount_choice == "mount":
                            device_path = f"/dev/{selected_device.name.split('/')[-1]}"
                            success, result = mount_device(device_path, selected_device.fstype)
                            
                            if success:
                                # Update the device info with the new mount point
                                selected_device.mountpoint = result
                                selected_device._was_mounted_by_us = True  # Mark that we mounted this
                                console.print(f"[green]Device is now mounted at:[/green] {result}")
                                return selected_device
                            else:
                                console.print(f"[red]Mount failed: {result}[/red]")
                                console.print("[yellow]Please select a different device or try again:[/yellow]")
                                continue
                        elif mount_choice == "scan_raw":
                            console.print("[yellow]Proceeding with raw device scan (may require elevated permissions)[/yellow]")
                            return selected_device
                        else:  # cancel
                            console.print("[yellow]Please select a different device:[/yellow]")
                            continue
                else:
                    console.print(f"[red]Invalid selection. Please enter a number between 1 and {len(devices)}[/red]")
            except ValueError:
                console.print("[red]Invalid input. Please enter a number or 'q' to quit.[/red]")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None


def run_device_selector() -> Optional[str]:
    """
    Run the device selector and return the selected device path.
    
    Returns:
        Device path (name or mountpoint) or None if cancelled
    """
    selected_device = select_device()
    
    if selected_device:
        # Return mountpoint if available, otherwise return device name
        if selected_device.mountpoint:
            return selected_device.mountpoint
        else:
            # Handle nested device names (e.g., "sdd/sdd1" -> "sdd1")
            device_name = selected_device.name.split('/')[-1]
            return f"/dev/{device_name}"
    
    return None


def run_device_selector_with_mount_info() -> Optional[tuple]:
    """
    Run the device selector and return both device path and mount info.
    
    Returns:
        Tuple of (device_path, mount_point) or None if cancelled
        mount_point is None if device was already mounted or not mounted
    """
    selected_device = select_device()
    
    if selected_device:
        if selected_device.mountpoint:
            # Device was already mounted
            return selected_device.mountpoint, None
        else:
            # Device was mounted by us or scanned raw
            device_name = selected_device.name.split('/')[-1]
            device_path = f"/dev/{device_name}"
            
            # Check if mountpoint was set during selection (meaning we mounted it)
            if hasattr(selected_device, '_temp_mounted') and selected_device.mountpoint:
                return selected_device.mountpoint, selected_device.mountpoint
            else:
                return device_path, None
    
    return None


if __name__ == "__main__":
    # Demo the device selector
    console = Console()
    device_path = run_device_selector()
    
    if device_path:
        console.print(f"\n[bold green]Selected device path:[/bold green] {device_path}")
    else:
        console.print("\n[yellow]No device selected.[/yellow]")