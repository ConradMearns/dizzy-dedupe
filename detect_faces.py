#!/usr/bin/env python3
"""
Face Detection and Cropping Script

Scans images to detect faces, logs metadata (bounding boxes, detection confidence, landmarks)
to a CSV file, and optionally crops detected faces to a separate directory.
Uses SCRFD model via insightface for efficient face detection.

Usage:
    python detect_faces.py --input_dir /path/to/images --output_csv faces_detected.csv
    python detect_faces.py --input_dir /path/to/images --output_csv faces_detected.csv --crop_dir /path/to/cropped_faces
    python detect_faces.py --input_video /path/to/video.mp4 --output_csv faces_detected.csv

Output CSV columns:
    - source_file: image filename or video filename
    - frame_id: frame number (0 for images, frame number for videos)
    - face_id: unique face ID within the source
    - bbox_x1, bbox_y1, bbox_x2, bbox_y2: bounding box coordinates
    - confidence: detection confidence score
    - landmark_points: facial landmarks as JSON string (optional)
    - cropped_face_path: path to cropped face image (if cropping enabled)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from rich.console import Console
from rich.progress import Progress, TaskID


console = Console()


class FaceDetector:
    def __init__(self, model_name: str = 'buffalo_l', confidence_threshold: float = 0.5, crop_dir: Optional[str] = None):
        """
        Initialize face detector with SCRFD model.
        
        Args:
            model_name: InsightFace model name (buffalo_l, buffalo_m, buffalo_s)
            confidence_threshold: Minimum confidence for face detection
            crop_dir: Directory to save cropped face images (optional)
        """
        self.confidence_threshold = confidence_threshold
        self.crop_dir = crop_dir
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Create crop directory if specified
        if self.crop_dir:
            os.makedirs(self.crop_dir, exist_ok=True)
            console.print(f"[blue]Cropped faces will be saved to: {self.crop_dir}[/blue]")
        
    def detect_faces_in_image(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect faces in a single image and optionally crop them.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of face detection results
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                console.print(f"[red]Error: Could not read image {image_path}[/red]")
                return []
            
            # Detect faces
            faces = self.app.get(img)
            
            results = []
            for face_idx, face in enumerate(faces):
                if face.det_score >= self.confidence_threshold:
                    bbox = face.bbox.astype(int)
                    landmarks = face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else None
                    
                    # Crop face if crop directory is specified
                    cropped_face_path = None
                    if self.crop_dir:
                        cropped_face_path = self._crop_and_save_face(
                            img, bbox, image_path, face_idx
                        )
                    
                    result = {
                        'source_file': os.path.basename(image_path),
                        'frame_id': 0,
                        'face_id': face_idx,
                        'bbox_x1': int(bbox[0]),
                        'bbox_y1': int(bbox[1]),
                        'bbox_x2': int(bbox[2]),
                        'bbox_y2': int(bbox[3]),
                        'confidence': float(face.det_score),
                        'landmark_points': json.dumps(landmarks.tolist()) if landmarks is not None else None,
                        'cropped_face_path': cropped_face_path
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            console.print(f"[red]Error processing {image_path}: {e}[/red]")
            return []
    
    def _crop_and_save_face(self, img: np.ndarray, bbox: np.ndarray, source_path: str, face_idx: int) -> Optional[str]:
        """
        Crop a face from an image and save it to the crop directory.
        
        Args:
            img: Source image as numpy array
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            source_path: Path to the source image
            face_idx: Index of the face in the image
            
        Returns:
            Path to the saved cropped face image, or None if failed
        """
        try:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = bbox
            
            # Add some padding around the face (10% of face size)
            face_width = x2 - x1
            face_height = y2 - y1
            padding_x = int(face_width * 0.1)
            padding_y = int(face_height * 0.1)
            
            # Apply padding while staying within image boundaries
            img_height, img_width = img.shape[:2]
            x1_padded = max(0, x1 - padding_x)
            y1_padded = max(0, y1 - padding_y)
            x2_padded = min(img_width, x2 + padding_x)
            y2_padded = min(img_height, y2 + padding_y)
            
            # Crop the face
            cropped_face = img[y1_padded:y2_padded, x1_padded:x2_padded]
            
            # Generate output filename
            source_filename = Path(source_path).stem
            source_ext = Path(source_path).suffix
            output_filename = f"{source_filename}_face_{face_idx:02d}{source_ext}"
            output_path = os.path.join(self.crop_dir, output_filename)
            
            # Save the cropped face
            success = cv2.imwrite(output_path, cropped_face)
            if success:
                return output_path
            else:
                console.print(f"[yellow]Warning: Failed to save cropped face to {output_path}[/yellow]")
                return None
                
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to crop face from {source_path}: {e}[/yellow]")
            return None
    
    def detect_faces_in_video(self, video_path: str, frame_skip: int = 1) -> List[Dict[str, Any]]:
        """
        Detect faces in video frames.
        
        Args:
            video_path: Path to the video file
            frame_skip: Process every Nth frame (1 = every frame)
            
        Returns:
            List of face detection results across all frames
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                console.print(f"[red]Error: Could not open video {video_path}[/red]")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            results = []
            
            with Progress() as progress:
                task = progress.add_task(f"Processing {os.path.basename(video_path)}", total=total_frames)
                
                frame_id = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Skip frames if specified
                    if frame_id % frame_skip == 0:
                        faces = self.app.get(frame)
                        
                        for face_idx, face in enumerate(faces):
                            if face.det_score >= self.confidence_threshold:
                                bbox = face.bbox.astype(int)
                                landmarks = face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else None
                                
                                result = {
                                    'source_file': os.path.basename(video_path),
                                    'frame_id': frame_id,
                                    'face_id': face_idx,
                                    'bbox_x1': int(bbox[0]),
                                    'bbox_y1': int(bbox[1]),
                                    'bbox_x2': int(bbox[2]),
                                    'bbox_y2': int(bbox[3]),
                                    'confidence': float(face.det_score),
                                    'landmark_points': json.dumps(landmarks.tolist()) if landmarks is not None else None
                                }
                                results.append(result)
                    
                    frame_id += 1
                    progress.update(task, advance=1)
            
            cap.release()
            return results
            
        except Exception as e:
            console.print(f"[red]Error processing video {video_path}: {e}[/red]")
            return []


def get_image_files(directory: str) -> List[str]:
    """Get all image files from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)


def save_results_to_csv(results: List[Dict[str, Any]], output_path: str):
    """Save detection results to CSV file."""
    if not results:
        console.print("[yellow]No faces detected. Creating empty CSV file.[/yellow]")
        # Create empty CSV with headers
        df = pd.DataFrame(columns=[
            'source_file', 'frame_id', 'face_id', 'bbox_x1', 'bbox_y1',
            'bbox_x2', 'bbox_y2', 'confidence', 'landmark_points', 'cropped_face_path'
        ])
    else:
        df = pd.DataFrame(results)
    
    df.to_csv(output_path, index=False)
    console.print(f"[green]Results saved to {output_path}[/green]")
    console.print(f"[blue]Total faces detected: {len(results)}[/blue]")
    
    # Count cropped faces if any
    if results:
        cropped_count = sum(1 for r in results if r.get('cropped_face_path'))
        if cropped_count > 0:
            console.print(f"[blue]Total faces cropped and saved: {cropped_count}[/blue]")


def main():
    parser = argparse.ArgumentParser(description="Detect faces in images or video and optionally crop them")
    parser.add_argument('--input_dir', type=str, help='Directory containing images')
    parser.add_argument('--input_video', type=str, help='Path to video file')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--crop_dir', type=str, help='Directory to save cropped face images (optional)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--model', type=str, default='buffalo_l',
                       choices=['buffalo_l', 'buffalo_m', 'buffalo_s'],
                       help='InsightFace model to use (default: buffalo_l)')
    parser.add_argument('--frame_skip', type=int, default=1,
                       help='For videos: process every Nth frame (default: 1)')
    
    args = parser.parse_args()
    
    # Validate input arguments
    if not args.input_dir and not args.input_video:
        console.print("[red]Error: Must specify either --input_dir or --input_video[/red]")
        sys.exit(1)
    
    if args.input_dir and args.input_video:
        console.print("[red]Error: Cannot specify both --input_dir and --input_video[/red]")
        sys.exit(1)
    
    # Initialize face detector
    console.print(f"[blue]Initializing face detector with model: {args.model}[/blue]")
    detector = FaceDetector(model_name=args.model, confidence_threshold=args.confidence, crop_dir=args.crop_dir)
    
    all_results = []
    
    if args.input_dir:
        # Process images in directory
        if not os.path.exists(args.input_dir):
            console.print(f"[red]Error: Directory {args.input_dir} does not exist[/red]")
            sys.exit(1)
        
        image_files = get_image_files(args.input_dir)
        if not image_files:
            console.print(f"[yellow]No image files found in {args.input_dir}[/yellow]")
            sys.exit(0)
        
        console.print(f"[blue]Found {len(image_files)} image files[/blue]")
        
        with Progress() as progress:
            task = progress.add_task("Processing images", total=len(image_files))
            
            for image_path in image_files:
                results = detector.detect_faces_in_image(image_path)
                all_results.extend(results)
                progress.update(task, advance=1)
    
    elif args.input_video:
        # Process video
        if not os.path.exists(args.input_video):
            console.print(f"[red]Error: Video file {args.input_video} does not exist[/red]")
            sys.exit(1)
        
        console.print(f"[blue]Processing video: {args.input_video}[/blue]")
        results = detector.detect_faces_in_video(args.input_video, frame_skip=args.frame_skip)
        all_results.extend(results)
    
    # Save results
    save_results_to_csv(all_results, args.output_csv)


if __name__ == "__main__":
    main()