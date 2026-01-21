"""
Weather monitoring module to track environmental conditions.
"""
import logging
import requests
from datetime import datetime

logger = logging.getLogger(__name__)


class WeatherMonitor:
    """Monitors weather conditions at the bird feeder location."""
    
    def __init__(self, api_key='', api_url='', latitude='', longitude=''):
        """
        Initialize the weather monitor.
        
        Args:
            api_key: API key for weather service
            api_url: Weather API endpoint URL
            latitude: Location latitude
            longitude: Location longitude
        """
        self.api_key = api_key
        self.api_url = api_url
        self.latitude = latitude
        self.longitude = longitude
        self.enabled = bool(api_key and latitude and longitude)
        
        if self.enabled:
            logger.info(f"WeatherMonitor initialized for location ({latitude}, {longitude})")
        else:
            logger.info("WeatherMonitor initialized (disabled - no API key or location)")
    
    def get_current_weather(self):
        """
        Get current weather conditions.
        
        Returns:
            dict: Weather information including temperature, conditions, etc.
        """
        if not self.enabled:
            return self._get_default_weather()
        
        try:
            params = {
                'lat': self.latitude,
                'lon': self.longitude,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Safely extract weather data with fallbacks
            main_data = data.get('main', {})
            weather_data = data.get('weather', [{}])[0]
            wind_data = data.get('wind', {})
            clouds_data = data.get('clouds', {})
            
            weather_info = {
                'timestamp': datetime.now().isoformat(),
                'temperature': main_data.get('temp'),
                'feels_like': main_data.get('feels_like'),
                'humidity': main_data.get('humidity'),
                'pressure': main_data.get('pressure'),
                'conditions': weather_data.get('description', 'Unknown'),
                'wind_speed': wind_data.get('speed'),
                'clouds': clouds_data.get('all')
            }
            
            logger.info(f"Weather: {weather_info['temperature']}°C, "
                       f"{weather_info['conditions']}")
            return weather_info
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._get_default_weather()
    
    def _get_default_weather(self):
        """
        Get default weather info when API is not available.
        
        Returns:
            dict: Default weather information
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'temperature': None,
            'feels_like': None,
            'humidity': None,
            'pressure': None,
            'conditions': 'Unknown',
            'wind_speed': None,
            'clouds': None,
            'note': 'Weather API not configured'
        }
