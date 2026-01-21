"""
Configuration settings for the Bird Watching application.
"""
import os

# Camera settings
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
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

# Weather API settings (optional - use OpenWeatherMap or similar)
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', '')
WEATHER_API_URL = os.getenv('WEATHER_API_URL', 'https://api.openweathermap.org/data/2.5/weather')
LOCATION_LAT = os.getenv('LOCATION_LAT', '')
LOCATION_LON = os.getenv('LOCATION_LON', '')

# Logging settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
