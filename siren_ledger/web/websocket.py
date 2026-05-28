"""WebSocket support for real-time updates."""

import logging
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any

from flask_socketio import SocketIO, emit
from sqlalchemy.orm import Session

from ..models.events import SirenDetection, SirenEvent
from ..storage.database import Database

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self, socketio: SocketIO, database: Database):
        """Initialize WebSocket manager.
        
        Parameters
        ----------
        socketio : SocketIO
            Flask-SocketIO instance.
        database : Database
            Database instance for monitoring changes.
        """
        self.socketio = socketio
        self.database = database
        self.last_event_id: Optional[int] = None
        self.last_detection_id: Optional[int] = None
        
        # Register event handlers
        self.socketio.on_event('connect', self.on_connect)
        self.socketio.on_event('disconnect', self.on_disconnect)
        self.socketio.on_event('request_update', self.on_request_update)
    
    def on_connect(self) -> None:
        """Handle client connection."""
        logger.info("Client connected via WebSocket")
        # Send initial data
        self.send_full_update()
    
    def on_disconnect(self) -> None:
        """Handle client disconnection."""
        logger.info("Client disconnected from WebSocket")
    
    def on_request_update(self) -> None:
        """Handle manual update request from client."""
        self.send_full_update()
    
    def send_full_update(self) -> None:
        """Send complete dashboard update to all clients."""
        try:
            # Get latest statistics
            stats = self._get_summary_stats()
            emit('stats_update', stats, broadcast=True)
            
            # Get latest reports
            reports = self.database.get_daily_reports(limit=30)
            reports_data = []
            for report in reports:
                report_dict = report.dict()
                report_dict['date'] = report.date.isoformat()
                reports_data.append(report_dict)
            emit('reports_update', {'reports': reports_data}, broadcast=True)
            
            # Get hourly distribution
            hourly_data = self._get_hourly_distribution()
            emit('hourly_update', hourly_data, broadcast=True)
            
            # Get recent events (last 7 days)
            end_date = date.today()
            start_date = end_date - timedelta(days=7)
            events = self.database.get_events_by_date_range(start_date, end_date)
            events_data = []
            for event in events:
                event_dict = event.dict()
                event_dict['start_time'] = event.start_time.isoformat()
                if event.end_time:
                    event_dict['end_time'] = event.end_time.isoformat()
                events_data.append(event_dict)
            emit('events_update', {'events': events_data}, broadcast=True)
            
        except Exception as e:
            logger.error(f"Failed to send full update: {e}")
    
    def broadcast_new_detection(self, detection: SirenDetection) -> None:
        """Broadcast new detection to all connected clients.
        
        Parameters
        ----------
        detection : SirenDetection
            New detection to broadcast.
        """
        try:
            detection_data = detection.dict()
            detection_data['timestamp'] = detection.timestamp.isoformat()
            self.socketio.emit('new_detection', detection_data)
            logger.debug(f"Broadcasted new detection: {detection.detected_class}")
        except Exception as e:
            logger.error(f"Failed to broadcast detection: {e}")
    
    def broadcast_event_update(self, event: SirenEvent) -> None:
        """Broadcast event update to all connected clients.
        
        Parameters
        ----------
        event : SirenEvent
            Updated event to broadcast.
        """
        try:
            event_data = event.dict()
            event_data['start_time'] = event.start_time.isoformat()
            if event.end_time:
                event_data['end_time'] = event.end_time.isoformat()
            self.socketio.emit('event_update', event_data)
            logger.debug(f"Broadcasted event update: {event.id}")
        except Exception as e:
            logger.error(f"Failed to broadcast event: {e}")
    
    def _get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        recent_reports = self.database.get_daily_reports(limit=30)
        
        if not recent_reports:
            return {
                'total_events': 0,
                'total_detections': 0,
                'total_duration_hours': 0.0,
                'avg_events_per_day': 0.0,
                'last_activity_date': None,
                'data_range_days': 0
            }
        
        total_events = sum(r.total_events for r in recent_reports)
        total_detections = sum(r.total_detections for r in recent_reports)
        total_duration_hours = sum(r.total_duration_minutes for r in recent_reports) / 60.0
        
        days_with_data = len([r for r in recent_reports if r.total_events > 0])
        avg_events_per_day = total_events / max(days_with_data, 1)
        
        last_activity_date = None
        for report in recent_reports:
            if report.total_events > 0:
                last_activity_date = report.date.isoformat()
                break
        
        return {
            'total_events': total_events,
            'total_detections': total_detections,
            'total_duration_hours': round(total_duration_hours, 2),
            'avg_events_per_day': round(avg_events_per_day, 1),
            'last_activity_date': last_activity_date,
            'data_range_days': len(recent_reports),
            'days_with_activity': days_with_data
        }
    
    def _get_hourly_distribution(self) -> Dict[str, Any]:
        """Get hourly distribution data."""
        reports = self.database.get_daily_reports(limit=30)
        
        if not reports:
            return {'hourly_distribution': {'hours': list(range(24)), 'totals': [0]*24, 'averages': [0.0]*24}}
        
        hourly_totals = [0] * 24
        total_days = len(reports)
        
        for report in reports:
            if report.events_by_hour and len(report.events_by_hour) == 24:
                for hour, count in enumerate(report.events_by_hour):
                    hourly_totals[hour] += count
        
        hourly_averages = [total / max(total_days, 1) for total in hourly_totals]
        
        return {
            'hourly_distribution': {
                'hours': list(range(24)),
                'totals': hourly_totals,
                'averages': [round(avg, 2) for avg in hourly_averages]
            },
            'metadata': {
                'days_analyzed': total_days,
                'total_events': sum(hourly_totals)
            }
        }