"""
Data logging module for bird observations.
"""
import os
import json
import csv
import logging
import wave
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DataLogger:
    """Logs bird sightings and related data."""
    
    def __init__(self, logs_dir='/app/data/logs', images_dir='/app/data/images', audio_dir='/app/data/audio'):
        """
        Initialize the data logger.
        
        Args:
            logs_dir: Directory for log files
            images_dir: Directory for captured images
            audio_dir: Directory for captured audio
        """
        self.logs_dir = Path(logs_dir)
        self.images_dir = Path(images_dir)
        self.audio_dir = Path(audio_dir)
        
        # Create directories if they don't exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV log file
        self.csv_log_path = self.logs_dir / 'bird_sightings.csv'
        self.csv_headers = [
            'timestamp', 'camera_id', 'species', 'confidence', 'size',
            'temperature', 'weather_conditions', 'image_path', 'audio_path',
            'audio_species', 'audio_confidence', 'is_rare', 'notes'
        ]
        self._initialize_csv()
        
        logger.info(
            f"DataLogger initialized - logs: {logs_dir}, images: {images_dir}, audio: {audio_dir}"
        )
    
    def _initialize_csv(self):
        """Initialize CSV log file with headers if it doesn't exist."""
        if not self.csv_log_path.exists():
            with open(self.csv_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)
            logger.info(f"Created new CSV log: {self.csv_log_path}")
        else:
            with open(self.csv_log_path, 'r') as f:
                reader = csv.reader(f)
                existing = next(reader, [])
            if existing != self.csv_headers:
                logger.warning("CSV headers differ from expected schema; new fields may be missing.")
    
    def log_sighting(self, detection_info, species_info, weather_info, image_path=None,
                     audio_path=None, audio_info=None, is_rare=False):
        """
        Log a bird sighting.
        
        Args:
            detection_info: Detection information from BirdDetector
            species_info: Species information from SpeciesIdentifier
            weather_info: Weather information from WeatherMonitor
            image_path: Path to captured image (optional)
            audio_path: Path to captured audio (optional)
            audio_info: Audio analysis info (optional)
            is_rare: Whether species is rare
            
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
            'image_path': str(image_path) if image_path else None,
            'audio_path': str(audio_path) if audio_path else None,
            'audio': audio_info or {},
            'is_rare': is_rare
        }
        
        # Save JSON record
        json_filename = f"sighting_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        json_path = self.logs_dir / json_filename
        
        with open(json_path, 'w') as f:
            json.dump(record, f, indent=2)
        
        # Append to CSV log
        audio_info = audio_info or {}
        row = {
            'timestamp': timestamp.isoformat(),
            'camera_id': detection_info.get('camera_id', ''),
            'species': species_info.get('species', 'Unknown'),
            'confidence': species_info.get('confidence', 0.0),
            'size': species_info.get('characteristics', {}).get('size', 'Unknown'),
            'temperature': weather_info.get('temperature', ''),
            'weather_conditions': weather_info.get('conditions', ''),
            'image_path': image_path or '',
            'audio_path': audio_path or '',
            'audio_species': audio_info.get('species', ''),
            'audio_confidence': audio_info.get('confidence', ''),
            'is_rare': str(bool(is_rare)),
            'notes': species_info.get('notes', '')
        }

        with open(self.csv_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writerow(row)
        
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
        timestamp = datetime.now()
        filename = f"{prefix}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = self.images_dir / filename
        
        cv2.imwrite(str(filepath), image)
        logger.info(f"Saved image: {filepath}")
        
        return filepath

    def save_audio(self, audio, sample_rate, prefix='song'):
        """
        Save a captured audio clip as WAV.

        Args:
            audio: 1D numpy array
            sample_rate: Audio sample rate
            prefix: Filename prefix

        Returns:
            Path: Path to saved audio file
        """
        timestamp = datetime.now()
        filename = f"{prefix}_{timestamp.strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = self.audio_dir / filename
        scaled = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(scaled.tobytes())
        logger.info(f"Saved audio: {filepath}")
        return filepath

    def get_recent_sightings(self, limit=10):
        """
        Get recent sightings from CSV.

        Args:
            limit: Number of records to return

        Returns:
            list: Recent sightings
        """
        if not self.csv_log_path.exists():
            return []
        with open(self.csv_log_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        rows = rows[-limit:] if limit else rows
        return [
            {
                'timestamp': row.get('timestamp'),
                'camera_id': row.get('camera_id'),
                'species': row.get('species'),
                'confidence': row.get('confidence'),
                'image_path': row.get('image_path'),
                'audio_path': row.get('audio_path'),
                'audio_species': row.get('audio_species'),
                'audio_confidence': row.get('audio_confidence'),
                'is_rare': row.get('is_rare') == 'True'
            }
            for row in rows
        ]
    
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
        
        most_common = None
        if species_count:
            most_common = max(species_count.items(), key=lambda x: x[1])[0]
        
        return {
            'total_sightings': len(rows),
            'species_count': species_count,
            'most_common_species': most_common
        }
