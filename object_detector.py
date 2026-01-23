"""
General object and human detection module.
Uses a lightweight model for detecting people, animals, and common objects.
"""
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None

logger = logging.getLogger(__name__)

# COCO class labels (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Categories for grouping detections
CATEGORY_HUMAN = ['person']
CATEGORY_BIRD = ['bird']
CATEGORY_ANIMAL = ['cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
CATEGORY_VEHICLE = ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']

# Colors for different categories (BGR format)
COLORS = {
    'human': (0, 0, 255),      # Red for humans
    'bird': (0, 255, 0),       # Green for birds
    'animal': (255, 165, 0),   # Orange for other animals
    'vehicle': (255, 0, 255),  # Magenta for vehicles
    'object': (255, 255, 0),   # Cyan for general objects
}


class ObjectDetector:
    """
    General-purpose object detector using ONNX runtime.
    Detects humans, birds, animals, and common objects.
    """
    
    def __init__(self, model_path='', confidence_threshold=0.4, nms_threshold=0.5):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to ONNX model file
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
        """
        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.session = None
        self.input_name = None
        self.input_size = (640, 640)
        self.classes = COCO_CLASSES
        
        self._load()
    
    def _load(self):
        """Load the ONNX model."""
        if not self.model_path or not self.model_path.exists():
            logger.info("Object detection model not configured")
            return
        if ort is None:
            logger.warning("onnxruntime not installed - object detection disabled")
            return
        
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=['CPUExecutionProvider']
            )
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            shape = input_info.shape
            if len(shape) >= 4:
                h = shape[2] if isinstance(shape[2], int) else 640
                w = shape[3] if isinstance(shape[3], int) else 640
                self.input_size = (w, h)
            logger.info(f"Loaded object detection model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load object model: {e}")
            self.session = None
    
    def is_ready(self):
        return self.session is not None
    
    def _preprocess(self, image):
        """Preprocess image for model input."""
        # Resize maintaining aspect ratio with padding
        h, w = image.shape[:2]
        scale = min(self.input_size[0] / w, self.input_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.full((self.input_size[1], self.input_size[0], 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # NCHW format
        tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        
        return tensor, scale, (w, h)
    
    def _postprocess(self, outputs, scale, original_size):
        """Process model outputs to get detections."""
        detections = []
        
        # Handle different output formats
        if len(outputs) == 1:
            output = outputs[0]
            # Standard YOLO format: [batch, num_detections, 85] or transposed
            if output.ndim == 3:
                if output.shape[2] > output.shape[1]:
                    output = output.transpose(0, 2, 1)
                output = output[0]  # Remove batch dim
            
            for detection in output:
                if len(detection) >= 5:
                    # Format: [x, y, w, h, conf, class_scores...]
                    x, y, w, h = detection[:4]
                    
                    if len(detection) > 5:
                        # Class scores follow
                        class_scores = detection[5:]
                        class_id = int(np.argmax(class_scores))
                        confidence = float(detection[4] * class_scores[class_id])
                    else:
                        confidence = float(detection[4])
                        class_id = 0
                    
                    if confidence >= self.confidence_threshold:
                        # Convert to corner format and scale back
                        x1 = int((x - w / 2) / scale)
                        y1 = int((y - h / 2) / scale)
                        x2 = int((x + w / 2) / scale)
                        y2 = int((y + h / 2) / scale)
                        
                        # Clamp to image bounds
                        x1 = max(0, min(x1, original_size[0]))
                        y1 = max(0, min(y1, original_size[1]))
                        x2 = max(0, min(x2, original_size[0]))
                        y2 = max(0, min(y2, original_size[1]))
                        
                        if x2 > x1 and y2 > y1:
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self.classes[class_id] if class_id < len(self.classes) else f'class_{class_id}'
                            })
        
        # Apply NMS
        if detections:
            detections = self._nms(detections)
        
        return detections
    
    def _nms(self, detections):
        """Apply non-maximum suppression."""
        if not detections:
            return []
        
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        return []
    
    def detect(self, image):
        """
        Detect objects in the image.
        
        Args:
            image: BGR image array
            
        Returns:
            list: List of detection dicts with bbox, confidence, class_id, class_name, category
        """
        if not self.is_ready() or image is None:
            return []
        
        try:
            tensor, scale, original_size = self._preprocess(image)
            outputs = self.session.run(None, {self.input_name: tensor})
            detections = self._postprocess(outputs, scale, original_size)
            
            # Add category to each detection
            for det in detections:
                det['category'] = self._get_category(det['class_name'])
            
            return detections
        except Exception as e:
            logger.debug(f"Object detection error: {e}")
            return []
    
    def _get_category(self, class_name):
        """Get the category for a class name."""
        if class_name in CATEGORY_HUMAN:
            return 'human'
        elif class_name in CATEGORY_BIRD:
            return 'bird'
        elif class_name in CATEGORY_ANIMAL:
            return 'animal'
        elif class_name in CATEGORY_VEHICLE:
            return 'vehicle'
        else:
            return 'object'
    
    def draw_detections(self, image, detections):
        """
        Draw detection boxes on the image.
        
        Args:
            image: BGR image array
            detections: List of detection dicts
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            category = det.get('category', 'object')
            class_name = det['class_name']
            confidence = det['confidence']
            color = COLORS.get(category, COLORS['object'])
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name}: {confidence:.0%}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 8), (x1 + label_w + 4, y1), color, -1)
            
            # Draw label text
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
        
        return annotated


def create_simple_detector():
    """
    Create a simple motion-based detector that categorizes detections.
    Used when no ML model is available.
    """
    class SimpleObjectDetector:
        """Fallback detector using motion detection and size heuristics."""
        
        def __init__(self):
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=True
            )
            self.min_area = 500
        
        def is_ready(self):
            return True
        
        def detect(self, image):
            if image is None:
                return []
            
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(image)
            _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 1
                
                # Simple heuristics for categorization
                # Tall aspect ratio might be human, small area might be bird
                if aspect_ratio > 1.5 and area > 5000:
                    category = 'human'
                    class_name = 'person'
                elif area < 3000:
                    category = 'bird'
                    class_name = 'bird'
                elif aspect_ratio < 0.7:
                    category = 'vehicle'
                    class_name = 'vehicle'
                else:
                    category = 'object'
                    class_name = 'motion'
                
                detections.append({
                    'bbox': (x, y, x + w, y + h),
                    'confidence': 0.5,  # Placeholder confidence
                    'class_id': 0,
                    'class_name': class_name,
                    'category': category,
                    'contour': contour
                })
            
            return detections
        
        def draw_detections(self, image, detections):
            annotated = image.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                category = det.get('category', 'object')
                color = COLORS.get(category, COLORS['object'])
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                label = f"{det['class_name']}"
                cv2.putText(
                    annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                )
                
                # Draw contour if available
                if 'contour' in det:
                    cv2.drawContours(annotated, [det['contour']], -1, color, 1)
            
            return annotated
    
    return SimpleObjectDetector()
