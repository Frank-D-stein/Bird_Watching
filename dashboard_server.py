"""
Web dashboard for monitoring sightings and statistics.
Includes live camera feed streaming with multi-class object detection.
"""
import io
import logging
import threading
import time

from flask import Flask, Response, jsonify, render_template, request

import config
from migration_tracker import MigrationTracker

logger = logging.getLogger(__name__)

# Detection category colors (BGR format for OpenCV)
CATEGORY_COLORS = {
    'human': (0, 0, 255),      # Red for humans
    'bird': (0, 255, 0),       # Green for birds
    'animal': (0, 165, 255),   # Orange for other animals
    'vehicle': (255, 0, 255),  # Magenta for vehicles
    'object': (255, 255, 0),   # Cyan for general objects
}


class DashboardServer:
    """Flask dashboard server running in a background thread."""

    def __init__(self, data_logger, camera_provider=None, audio_provider=None, 
                 weather_monitor=None, object_detector=None):
        """
        Initialize the dashboard server.

        Args:
            data_logger: DataLogger instance for sighting data
            camera_provider: Callable that returns list of camera dicts with 'id' and 'cap'
            audio_provider: Callable that returns latest audio analysis info
            weather_monitor: WeatherMonitor instance for current conditions
            object_detector: ObjectDetector instance for general object detection
        """
        self.data_logger = data_logger
        self.camera_provider = camera_provider
        self.audio_provider = audio_provider
        self.weather_monitor = weather_monitor
        self.object_detector = object_detector
        self.app = Flask(__name__, template_folder='templates')
        self.migration = MigrationTracker(self.data_logger.csv_log_path)
        self.latest_audio = {}
        self.latest_detections = {}
        self.audio_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        self._setup_routes()

    def update_audio(self, audio_info):
        """Update the latest audio analysis info."""
        with self.audio_lock:
            self.latest_audio = audio_info or {}

    def update_detections(self, camera_id, detections):
        """Update the latest detections for a camera."""
        with self.detection_lock:
            self.latest_detections[camera_id] = {
                'timestamp': time.time(),
                'detections': detections or []
            }

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            cameras = []
            if self.camera_provider:
                cams = self.camera_provider()
                cameras = [{'id': c['id']} for c in cams] if cams else []
            return render_template('dashboard.html', cameras=cameras)

        @self.app.route('/api/stats')
        def stats():
            limit = int(request.args.get('limit', config.DASHBOARD_RECENT_LIMIT))
            stats = self.data_logger.get_statistics()
            recent = self.data_logger.get_recent_sightings(limit=limit)
            migration = self.migration.get_summary()
            return jsonify({
                'stats': stats,
                'recent': recent,
                'migration': migration
            })

        @self.app.route('/api/sightings')
        def sightings():
            limit = int(request.args.get('limit', config.DASHBOARD_RECENT_LIMIT))
            return jsonify(self.data_logger.get_recent_sightings(limit=limit))

        @self.app.route('/api/audio')
        def audio_status():
            with self.audio_lock:
                return jsonify(self.latest_audio)

        @self.app.route('/api/weather')
        def weather():
            """Get current weather conditions."""
            if self.weather_monitor:
                weather_data = self.weather_monitor.get_current_weather()
                return jsonify(weather_data)
            return jsonify({
                'error': 'Weather monitoring not configured',
                'conditions': 'Unknown',
                'temperature': None
            })

        @self.app.route('/api/detections')
        def detections():
            """Get current live detections from all cameras."""
            with self.detection_lock:
                result = {}
                for cam_id, data in self.latest_detections.items():
                    # Only return recent detections (< 2 seconds old)
                    if time.time() - data['timestamp'] < 2:
                        result[cam_id] = {
                            'count': len(data['detections']),
                            'detections': [
                                {
                                    'class': d.get('class_name', 'unknown'),
                                    'category': d.get('category', 'object'),
                                    'confidence': d.get('confidence', 0)
                                }
                                for d in data['detections']
                            ]
                        }
                return jsonify(result)

        @self.app.route('/video_feed/<int:camera_id>')
        def video_feed(camera_id):
            """MJPEG stream for the specified camera."""
            return Response(
                self._generate_frames(camera_id),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

    def _generate_frames(self, camera_id):
        """Generator that yields JPEG frames for MJPEG streaming with detection overlays."""
        import cv2

        while True:
            frame = None
            bird_detector = None
            
            if self.camera_provider:
                cameras = self.camera_provider()
                for cam in cameras:
                    if cam['id'] == camera_id:
                        cap = cam.get('cap')
                        bird_detector = cam.get('detector')
                        if cap and cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                frame = None
                        break

            if frame is None:
                # Generate a placeholder frame
                frame = self._placeholder_frame(camera_id)
            else:
                # Run detection and draw overlays on the frame
                frame = self._draw_detection_overlay(frame, bird_detector, camera_id)

            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.1)  # ~10 FPS

    def _draw_detection_overlay(self, frame, bird_detector, camera_id):
        """Draw detection overlays on the frame with category-based colors."""
        import cv2
        from datetime import datetime

        annotated = frame.copy()
        all_detections = []
        
        # Add timestamp overlay
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            annotated, timestamp, (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        
        # Try object detector first (for humans, objects, etc.)
        if self.object_detector and self.object_detector.is_ready():
            try:
                obj_detections = self.object_detector.detect(frame)
                for det in obj_detections:
                    all_detections.append(det)
            except Exception as e:
                logger.debug(f"Object detection error: {e}")
        
        # Also run bird detector for motion-based bird detection
        if bird_detector:
            try:
                detected, contours, _ = bird_detector.detect(frame)
                if detected and contours:
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        # Check if this detection overlaps with an object detection
                        is_duplicate = False
                        for det in all_detections:
                            bx1, by1, bx2, by2 = det['bbox']
                            # Check for significant overlap
                            if (abs(x - bx1) < 50 and abs(y - by1) < 50):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            all_detections.append({
                                'bbox': (x, y, x + w, y + h),
                                'confidence': 0.5,
                                'class_name': 'motion',
                                'category': 'bird',  # Assume motion is bird
                                'contour': contour
                            })
            except Exception as e:
                logger.debug(f"Bird detection error: {e}")
        
        # Update latest detections for API
        self.update_detections(camera_id, all_detections)
        
        # Draw all detections
        if all_detections:
            # Count by category
            category_counts = {}
            
            for det in all_detections:
                category = det.get('category', 'object')
                class_name = det.get('class_name', 'unknown')
                confidence = det.get('confidence', 0)
                x1, y1, x2, y2 = det['bbox']
                w, h = x2 - x1, y2 - y1
                
                # Get color for this category
                color = CATEGORY_COLORS.get(category, CATEGORY_COLORS['object'])
                
                # Count categories
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw corner accents
                corner_len = min(15, w // 5, h // 5)
                if corner_len > 3:
                    # Brighter version of color for corners
                    corner_color = tuple(min(255, c + 50) for c in color)
                    
                    # Top-left corner
                    cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), corner_color, 2)
                    cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), corner_color, 2)
                    
                    # Top-right corner
                    cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), corner_color, 2)
                    cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), corner_color, 2)
                    
                    # Bottom-left corner
                    cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), corner_color, 2)
                    cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), corner_color, 2)
                    
                    # Bottom-right corner
                    cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), corner_color, 2)
                    cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), corner_color, 2)
                
                # Draw contour if available
                if 'contour' in det:
                    cv2.drawContours(annotated, [det['contour']], -1, color, 1)
                
                # Draw label background
                label = f"{class_name.upper()}"
                if confidence > 0:
                    label += f" {confidence:.0%}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                
                # Label background with category color
                label_bg_color = tuple(c // 2 for c in color)  # Darker version
                cv2.rectangle(annotated, (x1, y1 - label_h - 6), (x1 + label_w + 6, y1), label_bg_color, -1)
                cv2.rectangle(annotated, (x1, y1 - label_h - 6), (x1 + label_w + 6, y1), color, 1)
                
                # Label text
                cv2.putText(
                    annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                )
            
            # Draw detection summary panel
            panel_height = 20 + len(category_counts) * 18
            cv2.rectangle(annotated, (10, 10), (180, 10 + panel_height), (0, 0, 0), -1)
            cv2.rectangle(annotated, (10, 10), (180, 10 + panel_height), (100, 100, 100), 1)
            
            # Title
            cv2.putText(
                annotated, f"DETECTIONS: {len(all_detections)}", (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )
            
            # Category breakdown
            y_offset = 44
            for cat, count in sorted(category_counts.items()):
                color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS['object'])
                cv2.circle(annotated, (22, y_offset - 4), 5, color, -1)
                cv2.putText(
                    annotated, f"{cat}: {count}", (32, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
                )
                y_offset += 18
        else:
            # Show "scanning" indicator when no detection
            cv2.rectangle(annotated, (10, 10), (120, 35), (0, 0, 0), -1)
            cv2.putText(
                annotated, "Scanning...", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA
            )
        
        return annotated

    def _placeholder_frame(self, camera_id):
        """Create a placeholder frame when camera is unavailable."""
        import cv2
        import numpy as np

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (48, 48, 48)  # Dark gray
        cv2.putText(
            frame, f"Camera {camera_id}", (200, 220),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 128), 2
        )
        cv2.putText(
            frame, "No Signal", (230, 270),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2
        )
        return frame

    def start(self):
        thread = threading.Thread(
            target=self.app.run,
            kwargs={
                'host': config.DASHBOARD_HOST,
                'port': config.DASHBOARD_PORT,
                'debug': False,
                'use_reloader': False,
                'threaded': True
            },
            daemon=True
        )
        thread.start()
        logger.info(f"Dashboard running on http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
        return thread
