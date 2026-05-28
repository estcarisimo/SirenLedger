"""Broadcast utility for sending real-time updates to web dashboard."""

import logging
import requests
from typing import Optional

from ..models.events import SirenDetection, SirenEvent

logger = logging.getLogger(__name__)


class EventBroadcaster:
    """Broadcasts detection events to the web dashboard."""
    
    def __init__(self, web_url: str = "http://localhost:5555"):
        """Initialize broadcaster.
        
        Parameters
        ----------
        web_url : str
            Base URL of the web service.
        """
        self.web_url = web_url
        self.enabled = True
    
    def broadcast_detection(self, detection: SirenDetection) -> None:
        """Broadcast a new detection to the web service.
        
        Parameters
        ----------
        detection : SirenDetection
            Detection to broadcast.
        """
        if not self.enabled:
            return
            
        try:
            # For now, we'll use HTTP POST to notify the web service
            # The web service will then broadcast via WebSocket
            url = f"{self.web_url}/api/v1/broadcast/detection"
            data = detection.dict()
            data['timestamp'] = detection.timestamp.isoformat()
            
            response = requests.post(url, json=data, timeout=1.0)
            if response.status_code != 200:
                logger.warning(f"Failed to broadcast detection: {response.status_code}")
        except Exception as e:
            logger.debug(f"Could not broadcast detection: {e}")
            # Don't fail the main service if broadcasting fails
    
    def broadcast_event(self, event: SirenEvent) -> None:
        """Broadcast an event update to the web service.
        
        Parameters
        ----------
        event : SirenEvent
            Event to broadcast.
        """
        if not self.enabled:
            return
            
        try:
            url = f"{self.web_url}/api/v1/broadcast/event"
            data = event.dict()
            data['start_time'] = event.start_time.isoformat()
            if event.end_time:
                data['end_time'] = event.end_time.isoformat()
            
            response = requests.post(url, json=data, timeout=1.0)
            if response.status_code != 200:
                logger.warning(f"Failed to broadcast event: {response.status_code}")
        except Exception as e:
            logger.debug(f"Could not broadcast event: {e}")