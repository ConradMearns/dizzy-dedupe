#!/usr/bin/env python3
"""
Standalone YOLO Object Detection and Segmentation Script
Processes all images in the objects/ directory and saves results to CSV
"""

import os
import csv
import glob
from pathlib import Path
from typing import List, Dict, Any
from ultralytics import YOLO
import cv2
import numpy as np


def setup_model() -> YOLO:
    """Initialize YOLO model for object detection and segmentation."""
    print("Loading YOLO model...")
    # Use YOLOv8n-seg for segmentation (includes detection + segmentation masks)
    model = YOLO('yolov8n-seg.pt')  # Will auto-download if not present
    print("Model loaded successfully!")
    return model


def process_image(model: YOLO, image_path: str) -> List[Dict[str, Any]]:
    """
    Process a single image and return detection results.
    
    Args:
        model: YOLO model instance
        image_path: Path to the image file
        
    Returns:
        List of detection dictionaries
    """
    print(f"Processing: {os.path.basename(image_path)}")
    
    # Run inference
    results = model(image_path)
    
    detections = []
    
    for result in results:
        # Get image dimensions
        img_height, img_width = result.orig_shape
        
        # Process each detection
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
            
            # Process segmentation masks if available
            masks = None
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box
                class_name = model.names[int(class_id)]
                
                # Calculate additional metrics
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                box_area_ratio = box_area / (img_width * img_height)
                
                detection = {
                    'image_name': os.path.basename(image_path),
                    'image_path': image_path,
                    'image_width': img_width,
                    'image_height': img_height,
                    'detection_id': i,
                    'class_name': class_name,
                    'class_id': int(class_id),
                    'confidence': float(conf),
                    'bbox_x1': float(x1),
                    'bbox_y1': float(y1),
                    'bbox_x2': float(x2),
                    'bbox_y2': float(y2),
                    'bbox_width': float(box_width),
                    'bbox_height': float(box_height),
                    'bbox_area': float(box_area),
                    'bbox_area_ratio': float(box_area_ratio),
                    'bbox_center_x': float((x1 + x2) / 2),
                    'bbox_center_y': float((y1 + y2) / 2),
                }
                
                # Add segmentation info if available
                if masks is not None and i < len(masks):
                    mask = masks[i]
                    mask_area = np.sum(mask)
                    detection.update({
                        'has_segmentation': True,
                        'mask_area': float(mask_area),
                        'mask_area_ratio': float(mask_area / (img_width * img_height)),
                    })
                else:
                    detection.update({
                        'has_segmentation': False,
                        'mask_area': None,
                        'mask_area_ratio': None,
                    })
                
                detections.append(detection)
    
    return detections


def get_image_files(directory: str) -> List[str]:
    """Get all image files from the specified directory."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(directory, ext)
        image_files.extend(glob.glob(pattern, recursive=False))
        # Also check uppercase extensions
        pattern = os.path.join(directory, ext.upper())
        image_files.extend(glob.glob(pattern, recursive=False))
    
    return sorted(image_files)


def save_results_to_csv(detections: List[Dict[str, Any]], output_file: str):
    """Save detection results to CSV file."""
    headers = [
        'image_name', 'image_path', 'image_width', 'image_height',
        'detection_id', 'class_name', 'class_id', 'confidence',
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
        'bbox_width', 'bbox_height', 'bbox_area', 'bbox_area_ratio',
        'bbox_center_x', 'bbox_center_y',
        'has_segmentation', 'mask_area', 'mask_area_ratio'
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        if detections:
            writer.writerows(detections)
    
    print(f"Results saved to: {output_file}")
    
    if not detections:
        print("No detections found.")
        return
    
    # Print summary statistics using basic Python
    print(f"\nSummary:")
    unique_images = set(d['image_name'] for d in detections)
    print(f"Total images processed: {len(unique_images)}")
    print(f"Total detections: {len(detections)}")
    
    confidences = [d['confidence'] for d in detections]
    avg_confidence = sum(confidences) / len(confidences)
    print(f"Average confidence: {avg_confidence:.3f}")
    
    # Count classes
    class_counts = {}
    for d in detections:
        class_name = d['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"Classes detected: {sorted(class_counts.keys())}")
    print(f"Detections per class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")


def main():
    """Main function to run object detection on all images in objects directory."""
    print("YOLO Object Detection and Segmentation Script")
    print("=" * 50)
    
    # Configuration
    objects_dir = "objects"
    output_csv = "object_detection_results.csv"
    
    # Check if objects directory exists
    if not os.path.exists(objects_dir):
        print(f"Error: Directory '{objects_dir}' not found!")
        return
    
    # Get all image files
    image_files = get_image_files(objects_dir)
    
    if not image_files:
        print(f"No image files found in '{objects_dir}' directory!")
        return
    
    print(f"Found {len(image_files)} image files to process")
    
    # Setup YOLO model
    try:
        model = setup_model()
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    # Process all images
    all_detections = []
    
    for image_path in image_files:
        try:
            detections = process_image(model, image_path)
            all_detections.extend(detections)
            print(f"  Found {len(detections)} objects")
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            continue
    
    # Save results
    print(f"\nSaving results to {output_csv}...")
    save_results_to_csv(all_detections, output_csv)
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()