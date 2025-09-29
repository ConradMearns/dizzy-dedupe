#!/usr/bin/env python3
"""
Face Clustering Script

Takes face detection results and cropped face images, generates embeddings using InsightFace,
and clusters similar faces together using DBSCAN clustering. Stores results in DuckDB.

Usage:
    python cluster_faces.py --csv face_detection_results.csv --faces_dir ./cropped_faces --output_db faces_clusters.db
    python cluster_faces.py --csv face_detection_results.csv --faces_dir ./cropped_faces --output_db faces_clusters.db --eps 0.5 --min_samples 2

Output:
    - DuckDB database with face embeddings and cluster assignments
    - Summary statistics of clustering results
    - Optional visualization of cluster results
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import uuid

import cv2
import numpy as np
import pandas as pd
import duckdb
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from insightface.app import FaceAnalysis
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

console = Console()


class FaceClusterer:
    def __init__(self, model_name: str = 'buffalo_l', db_path: str = 'faces_clusters.db'):
        """
        Initialize face clustering system.
        
        Args:
            model_name: InsightFace model name for embeddings
            db_path: Path to DuckDB database file
        """
        self.model_name = model_name
        self.db_path = db_path
        
        # Initialize face analysis model
        console.print(f"[blue]Loading InsightFace model: {model_name}[/blue]")
        self.app = FaceAnalysis(name=model_name)
        # Use smaller detection size and lower threshold for cropped faces
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        
        # Lower detection threshold for cropped faces
        for model in self.app.models.values():
            if hasattr(model, 'det_thresh'):
                model.det_thresh = 0.3  # Lower threshold for better detection of cropped faces
        
        # Initialize database
        self.conn = duckdb.connect(db_path)
        self._setup_database()
        
    def _setup_database(self):
        """Create database tables for storing embeddings and clusters."""
        console.print(f"[blue]Setting up database: {self.db_path}[/blue]")
        
        # Create embeddings table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id VARCHAR PRIMARY KEY,
                source_file VARCHAR,
                frame_id INTEGER,
                face_id INTEGER,
                cropped_face_path VARCHAR,
                embedding FLOAT[512],
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                confidence FLOAT,
                landmark_points TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create clusters table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS face_clusters (
                cluster_id INTEGER,
                face_embedding_id VARCHAR,
                cluster_size INTEGER,
                is_noise BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (face_embedding_id) REFERENCES face_embeddings(id)
            )
        """)
        
        # Create cluster summary table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cluster_summary (
                cluster_id INTEGER PRIMARY KEY,
                cluster_size INTEGER,
                representative_face_id VARCHAR,
                avg_confidence FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        console.print("[green]Database tables created successfully[/green]")
    
    def generate_embeddings(self, csv_path: str, faces_dir: str) -> int:
        """
        Generate embeddings for all cropped faces and store in database.
        
        Args:
            csv_path: Path to face detection results CSV
            faces_dir: Directory containing cropped face images
            
        Returns:
            Number of embeddings generated
        """
        console.print(f"[blue]Loading face detection results from: {csv_path}[/blue]")
        df = pd.read_csv(csv_path)
        
        # Filter rows that have cropped face paths
        df_with_faces = df[df['cropped_face_path'].notna()].copy()
        console.print(f"[blue]Found {len(df_with_faces)} faces with cropped images[/blue]")
        
        embeddings_generated = 0
        
        with Progress() as progress:
            task = progress.add_task("Generating embeddings", total=len(df_with_faces))
            
            for idx, row in df_with_faces.iterrows():
                face_path = row['cropped_face_path']
                
                # Check if file exists
                if not os.path.exists(face_path):
                    console.print(f"[yellow]Warning: Face image not found: {face_path}[/yellow]")
                    progress.update(task, advance=1)
                    continue
                
                # Generate embedding
                embedding = self._get_face_embedding(face_path, row)
                if embedding is not None:
                    # Store in database
                    self._store_embedding(row, embedding)
                    embeddings_generated += 1
                
                progress.update(task, advance=1)
        
        console.print(f"[green]Generated {embeddings_generated} face embeddings[/green]")
        return embeddings_generated
    
    def _get_face_embedding(self, face_path: str, fallback_data: Optional[pd.Series] = None) -> Optional[np.ndarray]:
        """
        Extract embedding from a cropped face image.
        
        Args:
            face_path: Path to cropped face image
            fallback_data: Original detection data for fallback approach
            
        Returns:
            Face embedding vector or None if failed
        """
        try:
            img = cv2.imread(face_path)
            if img is None:
                return None
            
            # Resize image if it's too small (common issue with cropped faces)
            h, w = img.shape[:2]
            if h < 112 or w < 112:
                # Resize to minimum size while maintaining aspect ratio
                scale = max(112/h, 112/w)
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Get face analysis (should be just one face since it's cropped)
            faces = self.app.get(img)
            if len(faces) == 0:
                # Fallback: try to extract embedding directly from the cropped image
                # by treating the entire image as a face region
                if fallback_data is not None:
                    return self._extract_embedding_direct(img)
                else:
                    console.print(f"[yellow]No face detected in cropped image: {face_path}[/yellow]")
                    return None
            
            # Use the largest face (in case multiple are detected)
            if len(faces) > 1:
                faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            
            face = faces[0]
            return face.embedding
            
        except Exception as e:
            console.print(f"[red]Error processing {face_path}: {e}[/red]")
            return None
    
    def _extract_embedding_direct(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding directly from a face image without detection.
        This assumes the entire image is a face region.
        """
        try:
            # Resize to standard face recognition size
            face_img_resized = cv2.resize(face_img, (112, 112))
            
            # Normalize the image
            face_img_normalized = face_img_resized.astype(np.float32) / 255.0
            face_img_normalized = (face_img_normalized - 0.5) / 0.5
            
            # Transpose to CHW format and add batch dimension
            face_img_input = np.transpose(face_img_normalized, (2, 0, 1))
            face_img_input = np.expand_dims(face_img_input, axis=0)
            
            # Get the recognition model
            rec_model = None
            for model in self.app.models.values():
                if hasattr(model, 'get_feat'):
                    rec_model = model
                    break
            
            if rec_model is None:
                return None
            
            # Extract embedding
            embedding = rec_model.get_feat(face_img_input)
            return embedding.flatten()
            
        except Exception as e:
            console.print(f"[yellow]Direct embedding extraction failed: {e}[/yellow]")
            return None
    
    def _store_embedding(self, row: pd.Series, embedding: np.ndarray):
        """Store face embedding and metadata in database."""
        face_id = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO face_embeddings (
                id, source_file, frame_id, face_id, cropped_face_path, embedding,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, landmark_points
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            face_id,
            row['source_file'],
            int(row['frame_id']),
            int(row['face_id']),
            row['cropped_face_path'],
            embedding.tolist(),
            int(row['bbox_x1']),
            int(row['bbox_y1']),
            int(row['bbox_x2']),
            int(row['bbox_y2']),
            float(row['confidence']),
            row.get('landmark_points')
        ])
    
    def cluster_faces(self, eps: float = 0.5, min_samples: int = 2) -> Tuple[int, int]:
        """
        Cluster faces based on their embeddings using DBSCAN.
        
        Args:
            eps: Maximum distance between samples for clustering
            min_samples: Minimum samples in a neighborhood for a core point
            
        Returns:
            Tuple of (number of clusters, number of noise points)
        """
        console.print(f"[blue]Clustering faces with eps={eps}, min_samples={min_samples}[/blue]")
        
        # Get all embeddings from database
        result = self.conn.execute("""
            SELECT id, embedding FROM face_embeddings ORDER BY id
        """).fetchall()
        
        if len(result) == 0:
            console.print("[red]No embeddings found in database[/red]")
            return 0, 0
        
        # Prepare data for clustering
        face_ids = [row[0] for row in result]
        embeddings = np.array([row[1] for row in result])
        
        console.print(f"[blue]Clustering {len(embeddings)} face embeddings[/blue]")
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Count clusters and noise
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        console.print(f"[green]Found {n_clusters} clusters and {n_noise} noise points[/green]")
        
        # Calculate silhouette score if we have clusters
        if n_clusters > 1:
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            console.print(f"[blue]Silhouette score: {silhouette_avg:.3f}[/blue]")
        
        # Store clustering results
        self._store_clustering_results(face_ids, cluster_labels)
        
        return n_clusters, n_noise
    
    def _store_clustering_results(self, face_ids: List[str], cluster_labels: np.ndarray):
        """Store clustering results in database."""
        console.print("[blue]Storing clustering results in database[/blue]")
        
        # Clear previous clustering results
        self.conn.execute("DELETE FROM face_clusters")
        self.conn.execute("DELETE FROM cluster_summary")
        
        # Store individual cluster assignments
        for face_id, cluster_label in zip(face_ids, cluster_labels):
            is_noise = bool(cluster_label == -1)  # Convert numpy bool to Python bool
            cluster_id = None if is_noise else int(cluster_label)
            
            self.conn.execute("""
                INSERT INTO face_clusters (cluster_id, face_embedding_id, is_noise)
                VALUES (?, ?, ?)
            """, [cluster_id, face_id, is_noise])
        
        # Generate cluster summary
        unique_clusters = set(cluster_labels) - {-1}  # Exclude noise
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_face_ids = [face_ids[i] for i in range(len(face_ids)) if cluster_mask[i]]
            
            # Get cluster statistics
            cluster_stats = self.conn.execute("""
                SELECT 
                    COUNT(*) as cluster_size,
                    AVG(confidence) as avg_confidence,
                    (SELECT id FROM face_embeddings
                     WHERE id IN ({})
                     ORDER BY confidence DESC LIMIT 1) as representative_face_id
                FROM face_embeddings
                WHERE id IN ({})
            """.format(
                ','.join([f"'{fid}'" for fid in cluster_face_ids]),
                ','.join([f"'{fid}'" for fid in cluster_face_ids])
            )).fetchone()
            
            self.conn.execute("""
                INSERT INTO cluster_summary (cluster_id, cluster_size, representative_face_id, avg_confidence)
                VALUES (?, ?, ?, ?)
            """, [
                int(cluster_id),
                cluster_stats[0],
                cluster_stats[2],
                cluster_stats[1]
            ])
    
    def get_clustering_summary(self) -> Dict[str, Any]:
        """Get summary statistics of clustering results."""
        # Get overall statistics
        total_faces = self.conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
        total_clusters = self.conn.execute("SELECT COUNT(*) FROM cluster_summary").fetchone()[0]
        noise_points = self.conn.execute("SELECT COUNT(*) FROM face_clusters WHERE is_noise = true").fetchone()[0]
        
        # Get cluster size distribution
        cluster_sizes = self.conn.execute("""
            SELECT cluster_size, COUNT(*) as count 
            FROM cluster_summary 
            GROUP BY cluster_size 
            ORDER BY cluster_size
        """).fetchall()
        
        # Get largest clusters
        largest_clusters = self.conn.execute("""
            SELECT cs.cluster_id, cs.cluster_size, cs.avg_confidence, fe.cropped_face_path
            FROM cluster_summary cs
            JOIN face_embeddings fe ON cs.representative_face_id = fe.id
            ORDER BY cs.cluster_size DESC
            LIMIT 10
        """).fetchall()
        
        return {
            'total_faces': total_faces,
            'total_clusters': total_clusters,
            'noise_points': noise_points,
            'cluster_sizes': cluster_sizes,
            'largest_clusters': largest_clusters
        }
    
    def print_clustering_summary(self):
        """Print a formatted summary of clustering results."""
        summary = self.get_clustering_summary()
        
        console.print("\n[bold green]Face Clustering Summary[/bold green]")
        console.print(f"Total faces processed: {summary['total_faces']}")
        console.print(f"Number of clusters found: {summary['total_clusters']}")
        console.print(f"Noise points (unclustered): {summary['noise_points']}")
        
        if summary['cluster_sizes']:
            console.print("\n[bold]Cluster Size Distribution:[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Cluster Size", style="dim")
            table.add_column("Number of Clusters", justify="right")
            
            for size, count in summary['cluster_sizes']:
                table.add_row(str(size), str(count))
            
            console.print(table)
        
        if summary['largest_clusters']:
            console.print("\n[bold]Largest Clusters:[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Cluster ID", style="dim")
            table.add_column("Size", justify="right")
            table.add_column("Avg Confidence", justify="right")
            table.add_column("Representative Face", style="dim")
            
            for cluster_id, size, avg_conf, face_path in summary['largest_clusters']:
                face_name = os.path.basename(face_path) if face_path else "N/A"
                table.add_row(
                    str(cluster_id), 
                    str(size), 
                    f"{avg_conf:.3f}", 
                    face_name
                )
            
            console.print(table)
    
    def export_clusters(self, output_dir: str):
        """
        Export clustered faces to organized directories.
        
        Args:
            output_dir: Directory to create cluster subdirectories
        """
        console.print(f"[blue]Exporting clusters to: {output_dir}[/blue]")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all clusters
        clusters = self.conn.execute("""
            SELECT 
                fc.cluster_id,
                fe.cropped_face_path,
                fe.source_file,
                fe.confidence
            FROM face_clusters fc
            JOIN face_embeddings fe ON fc.face_embedding_id = fe.id
            WHERE fc.cluster_id IS NOT NULL
            ORDER BY fc.cluster_id, fe.confidence DESC
        """).fetchall()
        
        # Group by cluster and copy files
        current_cluster = None
        cluster_dir = None
        
        for cluster_id, face_path, source_file, confidence in clusters:
            if cluster_id != current_cluster:
                current_cluster = cluster_id
                cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id:03d}")
                os.makedirs(cluster_dir, exist_ok=True)
            
            # Copy face image to cluster directory
            if os.path.exists(face_path):
                face_filename = os.path.basename(face_path)
                dest_path = os.path.join(cluster_dir, face_filename)
                
                # Copy file
                import shutil
                shutil.copy2(face_path, dest_path)
        
        # Export noise points
        noise_faces = self.conn.execute("""
            SELECT fe.cropped_face_path
            FROM face_clusters fc
            JOIN face_embeddings fe ON fc.face_embedding_id = fe.id
            WHERE fc.is_noise = true
        """).fetchall()
        
        if noise_faces:
            noise_dir = os.path.join(output_dir, "noise")
            os.makedirs(noise_dir, exist_ok=True)
            
            for (face_path,) in noise_faces:
                if os.path.exists(face_path):
                    face_filename = os.path.basename(face_path)
                    dest_path = os.path.join(noise_dir, face_filename)
                    import shutil
                    shutil.copy2(face_path, dest_path)
        
        console.print(f"[green]Clusters exported to {output_dir}[/green]")
    
    def query_similar_faces(self, face_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find faces similar to a given face image.
        
        Args:
            face_path: Path to query face image
            top_k: Number of similar faces to return
            
        Returns:
            List of similar faces with similarity scores
        """
        # Generate embedding for query face
        query_embedding = self._get_face_embedding(face_path)
        if query_embedding is None:
            return []
        
        # Get all embeddings from database
        all_embeddings = self.conn.execute("""
            SELECT id, embedding, cropped_face_path, source_file, confidence
            FROM face_embeddings
        """).fetchall()
        
        similarities = []
        for face_id, embedding, face_path_db, source_file, confidence in all_embeddings:
            # Calculate cosine similarity
            embedding_array = np.array(embedding)
            similarity = np.dot(query_embedding, embedding_array) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding_array)
            )
            
            similarities.append({
                'face_id': face_id,
                'similarity': float(similarity),
                'face_path': face_path_db,
                'source_file': source_file,
                'confidence': confidence
            })
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(description="Cluster faces using embeddings and store in DuckDB")
    parser.add_argument('--csv', type=str, required=True, help='Path to face detection results CSV')
    parser.add_argument('--faces_dir', type=str, required=True, help='Directory containing cropped face images')
    parser.add_argument('--output_db', type=str, default='faces_clusters.db', help='Output DuckDB database path')
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN eps parameter (default: 0.5)')
    parser.add_argument('--min_samples', type=int, default=2, help='DBSCAN min_samples parameter (default: 2)')
    parser.add_argument('--model', type=str, default='buffalo_l', 
                       choices=['buffalo_l', 'buffalo_m', 'buffalo_s'],
                       help='InsightFace model to use (default: buffalo_l)')
    parser.add_argument('--export_clusters', type=str, help='Export clustered faces to directory')
    parser.add_argument('--query_face', type=str, help='Find similar faces to this image')
    parser.add_argument('--top_k', type=int, default=5, help='Number of similar faces to return (default: 5)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.csv):
        console.print(f"[red]Error: CSV file not found: {args.csv}[/red]")
        sys.exit(1)
    
    if not os.path.exists(args.faces_dir):
        console.print(f"[red]Error: Faces directory not found: {args.faces_dir}[/red]")
        sys.exit(1)
    
    # Initialize clusterer
    clusterer = FaceClusterer(model_name=args.model, db_path=args.output_db)
    
    try:
        # Check if we need to generate embeddings
        existing_embeddings = clusterer.conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
        
        if existing_embeddings == 0:
            # Generate embeddings
            embeddings_count = clusterer.generate_embeddings(args.csv, args.faces_dir)
            if embeddings_count == 0:
                console.print("[red]No embeddings generated. Exiting.[/red]")
                sys.exit(1)
        else:
            console.print(f"[blue]Using existing {existing_embeddings} embeddings from database[/blue]")
        
        # Handle query mode
        if args.query_face:
            if not os.path.exists(args.query_face):
                console.print(f"[red]Query face image not found: {args.query_face}[/red]")
                sys.exit(1)
            
            console.print(f"[blue]Finding faces similar to: {args.query_face}[/blue]")
            similar_faces = clusterer.query_similar_faces(args.query_face, args.top_k)
            
            if similar_faces:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Rank", style="dim")
                table.add_column("Similarity", justify="right")
                table.add_column("Face Image", style="dim")
                table.add_column("Source File", style="dim")
                table.add_column("Confidence", justify="right")
                
                for i, face in enumerate(similar_faces, 1):
                    table.add_row(
                        str(i),
                        f"{face['similarity']:.3f}",
                        os.path.basename(face['face_path']),
                        face['source_file'],
                        f"{face['confidence']:.3f}"
                    )
                
                console.print(table)
            else:
                console.print("[yellow]No similar faces found[/yellow]")
        
        else:
            # Perform clustering
            n_clusters, n_noise = clusterer.cluster_faces(args.eps, args.min_samples)
            
            # Print summary
            clusterer.print_clustering_summary()
            
            # Export clusters if requested
            if args.export_clusters:
                clusterer.export_clusters(args.export_clusters)
    
    finally:
        clusterer.close()


if __name__ == "__main__":
    main()