#!/usr/bin/env python3
"""
Bird Watching Application - Main Entry Point

This application monitors a bird feeder using a webcam, detects birds,
attempts to identify species, and logs observations along with weather data.
"""
import cv2
import time
import logging
import signal
import sys
from datetime import datetime

import config
from bird_detector import BirdDetector
from species_identifier import SpeciesIdentifier
from weather_monitor import WeatherMonitor
from data_logger import DataLogger


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'{config.DATA_DIR}/app.log')
    ]
)
logger = logging.getLogger(__name__)


class BirdWatchingApp:
    """Main application for bird watching."""
    
    def __init__(self):
        """Initialize the bird watching application."""
        logger.info("Initializing Bird Watching Application...")
        
        # Initialize components
        self.detector = BirdDetector(
            motion_threshold=config.MOTION_THRESHOLD,
            min_contour_area=config.MIN_CONTOUR_AREA
        )
        self.identifier = SpeciesIdentifier()
        self.weather_monitor = WeatherMonitor(
            api_key=config.WEATHER_API_KEY,
            api_url=config.WEATHER_API_URL,
            latitude=config.LOCATION_LAT,
            longitude=config.LOCATION_LON
        )
        self.data_logger = DataLogger(
            logs_dir=config.LOGS_DIR,
            images_dir=config.IMAGES_DIR
        )
        
        # Camera
        self.camera = None
        self.running = False
        
        logger.info("Bird Watching Application initialized successfully")
    
    def setup_camera(self):
        """Initialize the camera."""
        logger.info(f"Setting up camera (index: {config.CAMERA_INDEX})...")
        self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not self.camera.isOpened():
            logger.error("Failed to open camera")
            return False
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.IMAGE_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.IMAGE_HEIGHT)
        
        logger.info(f"Camera initialized: {config.IMAGE_WIDTH}x{config.IMAGE_HEIGHT}")
        return True
    
    def capture_frame(self):
        """
        Capture a frame from the camera.
        
        Returns:
            Frame from camera or None if failed
        """
        if self.camera is None or not self.camera.isOpened():
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            logger.warning("Failed to capture frame")
            return None
        
        return frame
    
    def process_detection(self, frame, contours, detection_info):
        """
        Process a bird detection.
        
        Args:
            frame: Camera frame
            contours: Detected bird contours
            detection_info: Detection metadata
        """
        # Get weather data
        weather_info = self.weather_monitor.get_current_weather()
        
        # Identify species (using first contour for now)
        contour = contours[0] if contours else None
        species_info = self.identifier.identify(frame, contour)
        
        # Save annotated image
        annotated_frame = self.detector.draw_detections(frame, contours)
        image_path = self.data_logger.save_image(annotated_frame)
        
        # Log the sighting
        self.data_logger.log_sighting(
            detection_info, species_info, weather_info, image_path
        )
        
        logger.info("=" * 60)
        logger.info(f"BIRD SIGHTING LOGGED")
        logger.info(f"Species: {species_info['species']}")
        logger.info(f"Temperature: {weather_info.get('temperature', 'N/A')}°C")
        logger.info(f"Weather: {weather_info.get('conditions', 'N/A')}")
        logger.info(f"Image: {image_path}")
        logger.info("=" * 60)
    
    def run(self):
        """Run the main application loop."""
        logger.info("Starting Bird Watching Application...")
        
        # Setup camera
        if not self.setup_camera():
            logger.error("Cannot start without camera")
            return
        
        self.running = True
        frame_count = 0
        last_detection_time = 0
        
        try:
            while self.running:
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    time.sleep(1)
                    continue
                
                frame_count += 1
                
                # Detect birds
                detected, contours, detection_info = self.detector.detect(frame)
                
                # Process detection with cooldown to avoid duplicate logs
                current_time = time.time()
                if detected and (current_time - last_detection_time) > config.CAPTURE_INTERVAL:
                    self.process_detection(frame, contours, detection_info)
                    last_detection_time = current_time
                
                # Log frame processing
                if frame_count % 30 == 0:  # Every 30 frames
                    logger.debug(f"Processed {frame_count} frames")
                
                # Small delay to reduce CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        self.running = False
        
        if self.camera is not None:
            self.camera.release()
        
        # Print statistics
        stats = self.data_logger.get_statistics()
        logger.info("=" * 60)
        logger.info("SESSION STATISTICS")
        logger.info(f"Total sightings: {stats['total_sightings']}")
        if stats.get('species_count'):
            logger.info("Species observed:")
            for species, count in stats['species_count'].items():
                logger.info(f"  - {species}: {count}")
        logger.info("=" * 60)
        
        logger.info("Application shutdown complete")


def signal_handler(sig, frame):
    """Handle termination signals."""
    logger.info("Received termination signal")
    sys.exit(0)


def main():
    """Main entry point."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run application
    app = BirdWatchingApp()
    app.run()


if __name__ == '__main__':
    main()
