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
from alerts import AlertManager
from audio_analyzer import AudioAnalyzer
from bird_detector import BirdDetector
from dashboard_server import DashboardServer
from data_logger import DataLogger
from object_detector import ObjectDetector, create_simple_detector
from species_identifier import SpeciesIdentifier
from weather_monitor import WeatherMonitor


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
        self.identifier = SpeciesIdentifier(
            model_path=config.ML_MODEL_PATH,
            labels_path=config.ML_LABELS_PATH
        )
        self.weather_monitor = WeatherMonitor(
            api_key=config.WEATHER_API_KEY,
            api_url=config.WEATHER_API_URL,
            latitude=config.LOCATION_LAT,
            longitude=config.LOCATION_LON
        )
        self.data_logger = DataLogger(
            logs_dir=config.LOGS_DIR,
            images_dir=config.IMAGES_DIR,
            audio_dir=config.AUDIOS_DIR
        )
        self.alerts = AlertManager(
            webhook_url=config.ALERT_WEBHOOK_URL,
            cooldown_seconds=config.ALERT_COOLDOWN_SECONDS,
            min_confidence=config.ALERT_MIN_CONFIDENCE
        )
        self.audio_analyzer = self._initialize_audio_analyzer()
        self.object_detector = self._initialize_object_detector()
        self.rare_species = self._parse_rare_species()
        
        # Camera
        self.cameras = []
        self.last_detection_time = {}
        self.running = False
        self.dashboard = None
        self.dashboard_thread = None
        
        logger.info("Bird Watching Application initialized successfully")

    def get_cameras(self):
        """Return camera list for dashboard streaming."""
        return self.cameras

    def _parse_camera_indexes(self):
        if config.CAMERA_INDEXES:
            indexes = [idx.strip() for idx in config.CAMERA_INDEXES.split(',') if idx.strip()]
            parsed = []
            for idx in indexes:
                if idx.isdigit():
                    parsed.append(int(idx))
            return parsed or [config.CAMERA_INDEX]
        return [config.CAMERA_INDEX]

    def _parse_rare_species(self):
        if not config.RARE_SPECIES:
            return set()
        return {item.strip() for item in config.RARE_SPECIES.split(',') if item.strip()}

    def _initialize_audio_analyzer(self):
        if not (config.AUDIO_ENABLED or config.AUDIO_MODEL_PATH):
            return None
        labels = self._load_labels(config.AUDIO_LABELS_PATH)
        return AudioAnalyzer(
            model_path=config.AUDIO_MODEL_PATH,
            labels=labels,
            sample_rate=config.AUDIO_SAMPLE_RATE,
            sample_seconds=config.AUDIO_SAMPLE_SECONDS,
            device_index=config.AUDIO_DEVICE_INDEX
        )

    def _initialize_object_detector(self):
        """Initialize the general object detector for humans, animals, etc."""
        if not config.OBJECT_DETECTION_ENABLED:
            return None
        
        # If an ONNX model path is provided, use it
        if config.OBJECT_MODEL_PATH:
            detector = ObjectDetector(
                model_path=config.OBJECT_MODEL_PATH,
                confidence_threshold=config.OBJECT_CONFIDENCE_THRESHOLD,
                nms_threshold=config.OBJECT_NMS_THRESHOLD
            )
            if detector.is_ready():
                logger.info("Object detector initialized with ONNX model")
                return detector
        
        # Fall back to simple motion-based detector
        logger.info("Object detector initialized (motion-based fallback)")
        return create_simple_detector()

    def _load_labels(self, labels_path):
        if not labels_path:
            return []
        try:
            with open(labels_path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception:
            logger.warning(f"Could not load labels from {labels_path}")
            return []

    def setup_cameras(self):
        """Initialize cameras (multi-camera support)."""
        camera_indexes = self._parse_camera_indexes()
        logger.info(f"Setting up cameras: {camera_indexes}")
        self.cameras = []

        for camera_id in camera_indexes:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id}")
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.IMAGE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.IMAGE_HEIGHT)
            detector = BirdDetector(
                motion_threshold=config.MOTION_THRESHOLD,
                min_contour_area=config.MIN_CONTOUR_AREA
            )
            self.cameras.append({'id': camera_id, 'cap': cap, 'detector': detector})
            self.last_detection_time[camera_id] = 0
            logger.info(f"Camera {camera_id} initialized: {config.IMAGE_WIDTH}x{config.IMAGE_HEIGHT}")

        return len(self.cameras) > 0
    
    def capture_frame(self, camera):
        """
        Capture a frame from the camera.
        
        Returns:
            Frame from camera or None if failed
        """
        if camera is None or not camera.isOpened():
            return None
        
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to capture frame")
            return None
        
        return frame
    
    def process_detection(self, frame, contours, detection_info, detector, camera_id):
        """
        Process a bird detection.
        
        Args:
            frame: Camera frame
            contours: Detected bird contours
            detection_info: Detection metadata
            detector: Detector used for drawing
            camera_id: Camera identifier
        """
        # Get weather data
        weather_info = self.weather_monitor.get_current_weather()
        
        # Identify species (using first contour for now)
        contour = contours[0] if contours else None
        species_info = self.identifier.identify(frame, contour)

        # Analyze audio if available
        audio_info = None
        audio_path = None
        if self.audio_analyzer:
            audio_data = None
            sample_rate = config.AUDIO_SAMPLE_RATE
            if config.AUDIO_SAMPLE_PATH:
                audio_data, sample_rate = self.audio_analyzer.load_wav(config.AUDIO_SAMPLE_PATH)
            else:
                audio_data = self.audio_analyzer.record_clip()
            if audio_data is not None:
                audio_path = self.data_logger.save_audio(audio_data, sample_rate, prefix=f"song_cam{camera_id}")
                audio_info = self.audio_analyzer.analyze(audio_data, sample_rate)
                # Update dashboard with audio info
                if self.dashboard:
                    self.dashboard.update_audio(audio_info)
        
        # Save annotated image
        annotated_frame = detector.draw_detections(frame, contours)
        image_path = self.data_logger.save_image(annotated_frame, prefix=f"bird_cam{camera_id}")

        is_rare = species_info.get('species') in self.rare_species
        if is_rare:
            self.alerts.send_alert(species_info, detection_info, image_path=image_path, camera_id=camera_id)
        
        # Log the sighting
        self.data_logger.log_sighting(
            detection_info, species_info, weather_info, image_path,
            audio_path=audio_path, audio_info=audio_info, is_rare=is_rare
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
        
        # Setup cameras
        if not self.setup_cameras():
            logger.error("Cannot start without camera")
            return

        if config.DASHBOARD_ENABLED:
            self.dashboard = DashboardServer(
                self.data_logger,
                camera_provider=self.get_cameras,
                weather_monitor=self.weather_monitor,
                object_detector=self.object_detector
            )
            self.dashboard_thread = self.dashboard.start()
        
        self.running = True
        frame_count = 0
        
        try:
            while self.running:
                for camera in self.cameras:
                    frame = self.capture_frame(camera['cap'])
                    if frame is None:
                        continue

                    frame_count += 1
                    detected, contours, detection_info = camera['detector'].detect(frame)
                    detection_info['camera_id'] = camera['id']
                    current_time = time.time()

                    last_time = self.last_detection_time.get(camera['id'], 0)
                    if detected and (current_time - last_time) > config.CAPTURE_INTERVAL:
                        self.process_detection(frame, contours, detection_info, camera['detector'], camera['id'])
                        self.last_detection_time[camera['id']] = current_time
                
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
        
        for camera in self.cameras:
            camera['cap'].release()
        
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
