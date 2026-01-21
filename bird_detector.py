"""
Bird detection module using computer vision techniques.
"""
import cv2
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BirdDetector:
    """Detects bird movement using motion detection."""
    
    def __init__(self, motion_threshold=25, min_contour_area=500):
        """
        Initialize the bird detector.
        
        Args:
            motion_threshold: Threshold for detecting motion
            min_contour_area: Minimum contour area to consider as a bird
        """
        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        logger.info(f"BirdDetector initialized with threshold={motion_threshold}, "
                   f"min_area={min_contour_area}")
    
    def detect(self, frame):
        """
        Detect birds in the given frame.
        
        Args:
            frame: Input image frame from camera
            
        Returns:
            tuple: (detected: bool, contours: list, detection_info: dict)
        """
        if frame is None:
            return False, [], {}
        
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Remove shadows and noise
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area
        bird_contours = [
            cnt for cnt in contours 
            if cv2.contourArea(cnt) > self.min_contour_area
        ]
        
        detected = len(bird_contours) > 0
        
        detection_info = {
            'timestamp': datetime.now().isoformat(),
            'num_detections': len(bird_contours),
            'contour_areas': [cv2.contourArea(cnt) for cnt in bird_contours]
        }
        
        if detected:
            logger.info(f"Bird detected! {len(bird_contours)} object(s) found")
        
        return detected, bird_contours, detection_info
    
    def draw_detections(self, frame, contours):
        """
        Draw bounding boxes around detected birds.
        
        Args:
            frame: Input image frame
            contours: List of contours to draw
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            label = f"Bird ({w}x{h})"
            cv2.putText(
                annotated, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        return annotated
