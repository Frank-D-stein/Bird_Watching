"""
Bird species identification module.
Supports neural network inference via ONNX for southeastern US bird species.
"""
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

logger = logging.getLogger(__name__)

# ImageNet normalization values (used by most pre-trained models)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class NeuralNetSpeciesModel:
    """ONNX runtime wrapper for bird species classification."""

    def __init__(self, model_path, labels=None, min_confidence=0.1):
        self.model_path = Path(model_path) if model_path else None
        self.labels = labels or []
        self.min_confidence = min_confidence
        self.session = None
        self.input_name = None
        self.input_size = (224, 224)

        self._load()

    def _load(self):
        if not self.model_path or not self.model_path.exists():
            logger.info("Neural model not configured or missing")
            return
        if ort is None:
            logger.warning("onnxruntime not installed - ML model disabled")
            return

        try:
            self.session = ort.InferenceSession(
                str(self.model_path), 
                providers=["CPUExecutionProvider"]
            )
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            input_shape = input_info.shape
            
            # Extract input dimensions (handle dynamic shapes)
            if len(input_shape) >= 4:
                h = input_shape[2] if isinstance(input_shape[2], int) else 224
                w = input_shape[3] if isinstance(input_shape[3], int) else 224
                self.input_size = (w, h)

            logger.info(f"Loaded ML model: {self.model_path} (input: {self.input_size})")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.session = None

    def is_ready(self):
        return self.session is not None

    def _preprocess(self, image, contour=None):
        """Preprocess image for model inference."""
        # Crop to bird region if contour provided
        if contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            # Add padding around the detection
            pad = int(max(w, h) * 0.1)
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(image.shape[1] - x, w + 2 * pad)
            h = min(image.shape[0] - y, h + 2 * pad)
            crop = image[y:y + h, x:x + w]
        else:
            crop = image

        if crop is None or crop.size == 0:
            crop = image

        # Resize to model input size
        resized = cv2.resize(crop, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] then apply ImageNet normalization
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
        
        # Convert HWC to NCHW format
        tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        return tensor

    def predict(self, image, contour=None, top_k=3):
        """
        Run inference and return top predictions.
        
        Returns:
            list: Top-k predictions as (species, confidence) tuples
        """
        if not self.is_ready():
            return None

        try:
            input_tensor = self._preprocess(image, contour)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            scores = outputs[0][0]
            
            # Softmax
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            
            # Get top-k predictions
            top_indices = np.argsort(probs)[::-1][:top_k]
            predictions = []
            
            for idx in top_indices:
                confidence = float(probs[idx])
                if confidence >= self.min_confidence:
                    species = self.labels[idx] if self.labels and idx < len(self.labels) else f"Class {idx}"
                    predictions.append((species, confidence))
            
            return predictions if predictions else None
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None


class SpeciesIdentifier:
    """Identifies bird species from images using neural network classification."""
    
    # Southeastern US bird species (default fallback)
    SE_US_BIRDS = [
        "Northern Cardinal",
        "Blue Jay", 
        "American Robin",
        "Carolina Chickadee",
        "Carolina Wren",
        "Tufted Titmouse",
        "Eastern Bluebird",
        "Mourning Dove",
        "Red-bellied Woodpecker",
        "Downy Woodpecker",
        "Unknown Bird"
    ]
    
    def __init__(self, model_path='', labels_path='', min_confidence=0.1):
        """
        Initialize the species identifier.
        
        Args:
            model_path: Path to ONNX model file
            labels_path: Path to labels text file (one label per line)
            min_confidence: Minimum confidence threshold for predictions
        """
        self.labels = self._load_labels(labels_path)
        self.model = NeuralNetSpeciesModel(
            model_path, 
            labels=self.labels,
            min_confidence=min_confidence
        )
        self.species_database = self.labels or self.SE_US_BIRDS
        
        if self.model.is_ready():
            logger.info(f"SpeciesIdentifier initialized with {len(self.labels)} species")
        else:
            logger.info("SpeciesIdentifier initialized (no model - using fallback)")
    
    def identify(self, image, contour=None):
        """
        Identify bird species from image.
        
        Args:
            image: Input image containing the bird (BGR format)
            contour: Optional contour information for the bird region
            
        Returns:
            dict: Species information including name, confidence, characteristics
        """
        timestamp = datetime.now().isoformat()
        size_estimate = self._estimate_size(contour) if contour is not None else 'Unknown'
        colors = self._estimate_colors(image, contour)
        
        if self.model.is_ready():
            predictions = self.model.predict(image, contour, top_k=3)
            
            if predictions:
                top_species, top_confidence = predictions[0]
                
                # Build alternative predictions string
                alternatives = []
                for species, conf in predictions[1:]:
                    alternatives.append(f"{species} ({conf:.1%})")
                
                species_info = {
                    'timestamp': timestamp,
                    'species': top_species,
                    'confidence': top_confidence,
                    'alternatives': predictions[1:] if len(predictions) > 1 else [],
                    'characteristics': {
                        'size': size_estimate,
                        'colors': colors,
                        'behavior': 'Visiting feeder'
                    },
                    'notes': f"ML confidence: {top_confidence:.1%}" + 
                            (f" | Also possible: {', '.join(alternatives)}" if alternatives else "")
                }
                
                logger.info(f"Species identified: {top_species} ({top_confidence:.1%})")
                return species_info

        # Fallback when model not available
        species_info = {
            'timestamp': timestamp,
            'species': 'Unknown Bird',
            'confidence': 0.0,
            'alternatives': [],
            'characteristics': {
                'size': size_estimate,
                'colors': colors,
                'behavior': 'Visiting feeder'
            },
            'notes': 'ML model required for species identification'
        }
        
        logger.info("Species: Unknown (no model)")
        return species_info
    
    def _estimate_size(self, contour):
        """
        Estimate bird size category from contour area.
        
        Small: Hummingbirds, Chickadees, Wrens
        Medium: Cardinals, Blue Jays, Robins
        Large: Woodpeckers, Doves, Crows
        """
        if contour is None:
            return 'Unknown'
            
        area = cv2.contourArea(contour)
        
        if area < 1500:
            return 'Small'
        elif area < 6000:
            return 'Medium'
        else:
            return 'Large'
    
    def _estimate_colors(self, image, contour):
        """
        Estimate dominant colors in the bird region.
        
        Returns:
            str: Description of dominant colors
        """
        if contour is None or image is None:
            return 'Unknown'
        
        try:
            # Get bird region
            x, y, w, h = cv2.boundingRect(contour)
            if w < 10 or h < 10:
                return 'Unknown'
                
            crop = image[y:y + h, x:x + w]
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # Define color ranges (HSV)
            colors_found = []
            
            # Red (wraps around in HSV)
            red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
            if cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2) > w * h * 0.1:
                colors_found.append('Red')
            
            # Blue
            blue_mask = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
            if cv2.countNonZero(blue_mask) > w * h * 0.1:
                colors_found.append('Blue')
            
            # Yellow/Orange
            yellow_mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
            if cv2.countNonZero(yellow_mask) > w * h * 0.1:
                colors_found.append('Yellow')
            
            # Green
            green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
            if cv2.countNonZero(green_mask) > w * h * 0.08:
                colors_found.append('Green')
            
            # Brown (low saturation orange/yellow)
            brown_mask = cv2.inRange(hsv, np.array([10, 30, 50]), np.array([25, 150, 180]))
            if cv2.countNonZero(brown_mask) > w * h * 0.15:
                colors_found.append('Brown')
            
            # Gray/Black (low saturation, variable value)
            gray_mask = cv2.inRange(hsv, np.array([0, 0, 30]), np.array([180, 50, 150]))
            if cv2.countNonZero(gray_mask) > w * h * 0.2:
                colors_found.append('Gray')
            
            # White (high value, low saturation)
            white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
            if cv2.countNonZero(white_mask) > w * h * 0.1:
                colors_found.append('White')
            
            # Black (very low value)
            black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 40]))
            if cv2.countNonZero(black_mask) > w * h * 0.15:
                colors_found.append('Black')
            
            if colors_found:
                return ', '.join(colors_found[:3])  # Top 3 colors
            return 'Unknown'
            
        except Exception as e:
            logger.debug(f"Color estimation failed: {e}")
            return 'Unknown'

    def _load_labels(self, labels_path):
        """Load species labels from file."""
        if not labels_path:
            return []
        path = Path(labels_path)
        if not path.exists():
            logger.warning(f"Labels file not found: {labels_path}")
            return []
        with open(path, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(labels)} species labels from {labels_path}")
        return labels
