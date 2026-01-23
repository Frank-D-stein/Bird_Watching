"""
Alerts module for rare species notifications.
"""
import logging
import time
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class AlertManager:
    """Sends notifications when rare species are detected."""

    def __init__(self, webhook_url='', cooldown_seconds=300, min_confidence=0.7):
        self.webhook_url = webhook_url
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence
        self._last_sent = {}

    def _should_send(self, species, confidence):
        if confidence < self.min_confidence:
            return False
        last_sent = self._last_sent.get(species, 0)
        return (time.time() - last_sent) >= self.cooldown_seconds

    def send_alert(self, species_info, detection_info, image_path=None, camera_id=None):
        if not self.webhook_url:
            logger.info("Alerts disabled - no webhook configured")
            return False

        species = species_info.get('species', 'Unknown')
        confidence = float(species_info.get('confidence', 0.0))
        if not self._should_send(species, confidence):
            return False

        payload = {
            'species': species,
            'confidence': confidence,
            'timestamp': detection_info.get('timestamp'),
            'camera_id': camera_id,
            'image_path': str(image_path) if image_path else None
        }

        try:
            parsed = urlparse(self.webhook_url)
            if not parsed.scheme.startswith('http'):
                logger.warning("Invalid webhook URL scheme")
                return False

            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            self._last_sent[species] = time.time()
            logger.info(f"Alert sent for rare species: {species}")
            return True
        except Exception as exc:
            logger.error(f"Failed to send alert: {exc}")
            return False
