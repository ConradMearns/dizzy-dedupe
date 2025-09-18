#!/usr/bin/env python3
"""
NSFW Image Detection Script

This script scans images in a directory recursively, detects NSFW content using 
Hugging Face Vision Transformer model, and outputs results to a CSV file with 
image hash and path for detected cases.

Requirements:
    pip install transformers torch pillow

Usage:
    python nudity_detector.py <directory_path> [output_csv]
"""

import os
import sys
import csv
import hashlib
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image
from transformers import pipeline, AutoModelForImageClassification, ViTImageProcessor


class NSFWChecker:
    """Handles NSFW detection for images using Hugging Face Vision Transformer."""
    
    def __init__(self):
        """Initialize the NSFW detector."""
        print("Loading Hugging Face Vision Transformer model...")
        try:
            # Use the pipeline approach for simplicity
            self.classifier = pipeline(
                "image-classification", 
                model="Falconsai/nsfw_image_detection"
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def check_image_nsfw(self, image_path: str, confidence_threshold: float = 0.7) -> bool:
        """
        Check if an image contains NSFW content.
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence score to consider as NSFW (0.0-1.0)
            
        Returns:
            True if NSFW content is detected, False otherwise
        """
        try:
            # Load and process the image
            img = Image.open(image_path).convert('RGB')
            
            # Run classification
            results = self.classifier(img)
            
            # Check results - look for 'nsfw' label with high confidence
            for result in results:
                if result['label'].lower() == 'nsfw' and result['score'] >= confidence_threshold:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False


def calculate_image_hash(image_path: str) -> str:
    """
    Calculate SHA256 hash of an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        SHA256 hash as hexadecimal string
    """
    try:
        with open(image_path, 'rb') as f:
            file_hash = hashlib.sha256()
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
            return file_hash.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {image_path}: {e}")
        return ""


def get_image_files(directory: str) -> List[str]:
    """
    Recursively find all image files in a directory.
    
    Args:
        directory: Root directory to search
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []
    
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if Path(file_path).suffix.lower() in image_extensions:
                    image_files.append(file_path)
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")
    
    return image_files


def scan_directory_for_nsfw(directory: str, output_csv: str = "nsfw_results.csv", 
                           confidence_threshold: float = 0.7) -> None:
    """
    Scan a directory recursively for images containing NSFW content and save results to CSV.
    
    Args:
        directory: Directory to scan
        output_csv: Output CSV file path
        confidence_threshold: Minimum confidence score for NSFW detection
    """
    # Initialize the NSFW checker
    checker = NSFWChecker()
    
    # Get all image files
    print(f"Scanning directory: {directory}")
    image_files = get_image_files(directory)
    print(f"Found {len(image_files)} image files")
    
    if not image_files:
        print("No image files found in the specified directory.")
        return
    
    # Prepare CSV output
    nsfw_detected_files = []
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Check for NSFW content
        has_nsfw = checker.check_image_nsfw(image_path, confidence_threshold)
        
        if has_nsfw:
            # Calculate hash for images with NSFW content
            image_hash = calculate_image_hash(image_path)
            nsfw_detected_files.append({
                'hash': image_hash,
                'path': image_path,
                'filename': os.path.basename(image_path)
            })
            print(f"  🚨 NSFW content detected in: {image_path}")
    
    # Write results to CSV
    if nsfw_detected_files:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['hash', 'path', 'filename']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for file_info in nsfw_detected_files:
                writer.writerow(file_info)
        
        print(f"\n✅ Results saved to: {output_csv}")
        print(f"📊 Total images processed: {len(image_files)}")
        print(f"🚨 Images with NSFW content detected: {len(nsfw_detected_files)}")
    else:
        print(f"\n✅ No NSFW content detected in any of the {len(image_files)} images processed.")
        # Still create an empty CSV with headers
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['hash', 'path', 'filename']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        print(f"📄 Empty results file created: {output_csv}")


def main():
    """Main function to handle command line arguments and execute the scan."""
    parser = argparse.ArgumentParser(
        description="Scan directory for images containing NSFW content using Hugging Face ViT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nudity_detector.py /path/to/images
  python nudity_detector.py /path/to/images --output results.csv
  python nudity_detector.py /path/to/images --confidence 0.8
        """
    )
    
    parser.add_argument(
        'directory',
        help='Directory to scan for images (will scan recursively)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='nsfw_results.csv',
        help='Output CSV file path (default: nsfw_results.csv)'
    )
    
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.7,
        help='Confidence threshold for NSFW detection (0.0-1.0, default: 0.7)'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist or is not a directory.")
        sys.exit(1)
    
    # Validate confidence threshold
    if not 0.0 <= args.confidence <= 1.0:
        print("Error: Confidence threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    try:
        scan_directory_for_nsfw(args.directory, args.output, args.confidence)
    except KeyboardInterrupt:
        print("\n\n⏹️  Scan interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()