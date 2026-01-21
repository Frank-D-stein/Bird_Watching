"""
Data logging module for bird observations.
"""
import os
import json
import csv
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLogger:
    """Logs bird sightings and related data."""
    
    def __init__(self, logs_dir='/app/data/logs', images_dir='/app/data/images'):
        """
        Initialize the data logger.
        
        Args:
            logs_dir: Directory for log files
            images_dir: Directory for captured images
        """
        self.logs_dir = Path(logs_dir)
        self.images_dir = Path(images_dir)
        
        # Create directories if they don't exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV log file
        self.csv_log_path = self.logs_dir / 'bird_sightings.csv'
        self._initialize_csv()
        
        logger.info(f"DataLogger initialized - logs: {logs_dir}, images: {images_dir}")
    
    def _initialize_csv(self):
        """Initialize CSV log file with headers if it doesn't exist."""
        if not self.csv_log_path.exists():
            with open(self.csv_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'species', 'confidence', 'size', 
                    'temperature', 'weather_conditions', 'image_path', 'notes'
                ])
            logger.info(f"Created new CSV log: {self.csv_log_path}")
    
    def log_sighting(self, detection_info, species_info, weather_info, image_path=None):
        """
        Log a bird sighting.
        
        Args:
            detection_info: Detection information from BirdDetector
            species_info: Species information from SpeciesIdentifier
            weather_info: Weather information from WeatherMonitor
            image_path: Path to captured image (optional)
            
        Returns:
            dict: Complete sighting record
        """
        timestamp = datetime.now()
        
        # Create complete record
        record = {
            'timestamp': timestamp.isoformat(),
            'detection': detection_info,
            'species': species_info,
            'weather': weather_info,
            'image_path': str(image_path) if image_path else None
        }
        
        # Save JSON record
        json_filename = f"sighting_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        json_path = self.logs_dir / json_filename
        
        with open(json_path, 'w') as f:
            json.dump(record, f, indent=2)
        
        # Append to CSV log
        with open(self.csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp.isoformat(),
                species_info.get('species', 'Unknown'),
                species_info.get('confidence', 0.0),
                species_info.get('characteristics', {}).get('size', 'Unknown'),
                weather_info.get('temperature', ''),
                weather_info.get('conditions', ''),
                image_path or '',
                species_info.get('notes', '')
            ])
        
        logger.info(f"Logged sighting: {species_info.get('species', 'Unknown')} "
                   f"at {timestamp.isoformat()}")
        
        return record
    
    def save_image(self, image, prefix='bird'):
        """
        Save a captured image.
        
        Args:
            image: Image array to save
            prefix: Filename prefix
            
        Returns:
            Path: Path to saved image
        """
        import cv2
        
        timestamp = datetime.now()
        filename = f"{prefix}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = self.images_dir / filename
        
        cv2.imwrite(str(filepath), image)
        logger.info(f"Saved image: {filepath}")
        
        return filepath
    
    def get_statistics(self):
        """
        Get statistics from logged sightings.
        
        Returns:
            dict: Statistics about bird sightings
        """
        if not self.csv_log_path.exists():
            return {'total_sightings': 0}
        
        with open(self.csv_log_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        species_count = {}
        for row in rows:
            species = row['species']
            species_count[species] = species_count.get(species, 0) + 1
        
        return {
            'total_sightings': len(rows),
            'species_count': species_count,
            'most_common_species': max(species_count.items(), key=lambda x: x[1])[0] if species_count else None
        }
