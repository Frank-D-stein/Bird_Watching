"""
Configuration settings for the Bird Watching application.
"""
import os

# Camera settings
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
CAMERA_INDEXES = os.getenv('CAMERA_INDEXES', '')
CAPTURE_INTERVAL = int(os.getenv('CAPTURE_INTERVAL', '60'))  # seconds
IMAGE_WIDTH = int(os.getenv('IMAGE_WIDTH', '1280'))
IMAGE_HEIGHT = int(os.getenv('IMAGE_HEIGHT', '720'))

# Detection settings
MOTION_THRESHOLD = int(os.getenv('MOTION_THRESHOLD', '25'))
MIN_CONTOUR_AREA = int(os.getenv('MIN_CONTOUR_AREA', '500'))

# Storage settings
DATA_DIR = os.getenv('DATA_DIR', '/app/data')
LOGS_DIR = os.path.join(DATA_DIR, 'logs')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
AUDIOS_DIR = os.path.join(DATA_DIR, 'audio')

# Weather API settings (optional - use OpenWeatherMap or similar)
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', '')
WEATHER_API_URL = os.getenv('WEATHER_API_URL', 'https://api.openweathermap.org/data/2.5/weather')
LOCATION_LAT = os.getenv('LOCATION_LAT', '')
LOCATION_LON = os.getenv('LOCATION_LON', '')

# ML model settings (bird species classifier)
ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', '')
ML_LABELS_PATH = os.getenv('ML_LABELS_PATH', '')
ML_MIN_CONFIDENCE = float(os.getenv('ML_MIN_CONFIDENCE', '0.5'))

# Object detection settings (humans, animals, objects)
OBJECT_DETECTION_ENABLED = os.getenv('OBJECT_DETECTION_ENABLED', '1') == '1'
OBJECT_MODEL_PATH = os.getenv('OBJECT_MODEL_PATH', '')
OBJECT_CONFIDENCE_THRESHOLD = float(os.getenv('OBJECT_CONFIDENCE_THRESHOLD', '0.4'))
OBJECT_NMS_THRESHOLD = float(os.getenv('OBJECT_NMS_THRESHOLD', '0.5'))

# Alerts settings
RARE_SPECIES = os.getenv('RARE_SPECIES', '')
ALERT_WEBHOOK_URL = os.getenv('ALERT_WEBHOOK_URL', '')
ALERT_COOLDOWN_SECONDS = int(os.getenv('ALERT_COOLDOWN_SECONDS', '300'))
ALERT_MIN_CONFIDENCE = float(os.getenv('ALERT_MIN_CONFIDENCE', '0.7'))

# Audio analysis settings
AUDIO_ENABLED = os.getenv('AUDIO_ENABLED', '0') == '1'
AUDIO_SAMPLE_PATH = os.getenv('AUDIO_SAMPLE_PATH', '')
AUDIO_MODEL_PATH = os.getenv('AUDIO_MODEL_PATH', '')
AUDIO_LABELS_PATH = os.getenv('AUDIO_LABELS_PATH', '')
AUDIO_SAMPLE_SECONDS = int(os.getenv('AUDIO_SAMPLE_SECONDS', '4'))
AUDIO_SAMPLE_RATE = int(os.getenv('AUDIO_SAMPLE_RATE', '16000'))
AUDIO_DEVICE_INDEX = os.getenv('AUDIO_DEVICE_INDEX', '')

# Dashboard settings
DASHBOARD_ENABLED = os.getenv('DASHBOARD_ENABLED', '1') == '1'
DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', '0.0.0.0')
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '5000'))
DASHBOARD_RECENT_LIMIT = int(os.getenv('DASHBOARD_RECENT_LIMIT', '10'))

# Logging settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
