"""REST API endpoints for SirenLedger data access."""

import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional

from flask import Blueprint, request, jsonify, current_app
from pydantic import ValidationError

from ..models.events import DailyReport
from ..storage import Database

logger = logging.getLogger(__name__)


def create_api_blueprint() -> Blueprint:
    """Create API blueprint with all endpoints.
    
    Returns
    -------
    Blueprint
        Flask blueprint for API routes.
    """
    api = Blueprint('api', __name__)
    
    @api.route('/health')
    def health_check():
        """Health check endpoint."""
        try:
            database = current_app.config['DATABASE']
            # Simple database connectivity test
            from sqlalchemy import text
            with database.get_session() as session:
                session.execute(text("SELECT 1"))
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'database': 'connected'
            })
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }), 503
    
    @api.route('/reports/daily')
    def get_daily_reports():
        """Get daily activity reports.
        
        Query Parameters:
        - start_date: Start date (YYYY-MM-DD)
        - end_date: End date (YYYY-MM-DD)
        - limit: Maximum number of reports (default: 30)
        """
        try:
            # Parse query parameters
            start_date_str = request.args.get('start_date')
            end_date_str = request.args.get('end_date')
            limit = request.args.get('limit', 30, type=int)
            
            start_date = None
            end_date = None
            
            if start_date_str:
                try:
                    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                except ValueError:
                    return jsonify({'error': 'Invalid start_date format. Use YYYY-MM-DD'}), 400
            
            if end_date_str:
                try:
                    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
                except ValueError:
                    return jsonify({'error': 'Invalid end_date format. Use YYYY-MM-DD'}), 400
            
            # Validate date range
            if start_date and end_date and start_date > end_date:
                return jsonify({'error': 'start_date must be before end_date'}), 400
            
            if limit < 1 or limit > 365:
                return jsonify({'error': 'limit must be between 1 and 365'}), 400
            
            # Get reports from database
            database = current_app.config['DATABASE']
            reports = database.get_daily_reports(start_date, end_date, limit)
            
            # Convert to JSON-serializable format
            reports_data = []
            for report in reports:
                report_dict = report.dict()
                report_dict['date'] = report.date.isoformat()
                reports_data.append(report_dict)
            
            return jsonify({
                'reports': reports_data,
                'count': len(reports_data),
                'query': {
                    'start_date': start_date.isoformat() if start_date else None,
                    'end_date': end_date.isoformat() if end_date else None,
                    'limit': limit
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to get daily reports: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @api.route('/reports/daily/<date_str>')
    def get_daily_report(date_str: str):
        """Get daily report for specific date.
        
        Parameters:
        - date_str: Date in YYYY-MM-DD format
        """
        try:
            # Parse date
            try:
                target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
            
            # Get or generate report
            database = current_app.config['DATABASE']
            report = database.generate_daily_report(target_date)
            
            if report.total_events == 0 and report.total_detections == 0:
                return jsonify({'error': 'No data found for specified date'}), 404
            
            # Convert to JSON-serializable format
            report_dict = report.dict()
            report_dict['date'] = report.date.isoformat()
            
            return jsonify(report_dict)
            
        except Exception as e:
            logger.error(f"Failed to get daily report for {date_str}: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @api.route('/reports/daily/<string:date_str>/generate', methods=['POST'])
    def generate_daily_report(date_str: str):
        """Manually generate and store daily report for a specific date."""
        try:
            # Parse date
            try:
                target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
            
            # Generate and store report
            database = current_app.config['DATABASE']
            report = database.generate_daily_report(target_date)
            database.store_daily_report(report)
            
            # Convert to JSON-serializable format
            report_dict = report.dict()
            report_dict['date'] = report.date.isoformat()
            
            return jsonify({
                'message': f'Daily report generated for {target_date}',
                'report': report_dict
            })
            
        except Exception as e:
            logger.error(f"Failed to generate daily report for {date_str}: {e}")
            return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500
    
    @api.route('/events')
    def get_events():
        """Get siren events.
        
        Query Parameters:
        - start_date: Start date (YYYY-MM-DD)
        - end_date: End date (YYYY-MM-DD)
        - limit: Maximum number of events (default: 100)
        """
        try:
            # Parse query parameters
            start_date_str = request.args.get('start_date')
            end_date_str = request.args.get('end_date')
            limit = request.args.get('limit', 100, type=int)
            
            # Default to last 7 days if no dates specified
            if not start_date_str and not end_date_str:
                end_date = date.today()
                start_date = end_date - timedelta(days=7)
            else:
                start_date = None
                end_date = None
                
                if start_date_str:
                    try:
                        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                    except ValueError:
                        return jsonify({'error': 'Invalid start_date format. Use YYYY-MM-DD'}), 400
                
                if end_date_str:
                    try:
                        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
                    except ValueError:
                        return jsonify({'error': 'Invalid end_date format. Use YYYY-MM-DD'}), 400
            
            if limit < 1 or limit > 1000:
                return jsonify({'error': 'limit must be between 1 and 1000'}), 400
            
            # Get events from database
            database = current_app.config['DATABASE']
            events = database.get_events_by_date_range(start_date, end_date)
            
            # Apply limit
            if len(events) > limit:
                events = events[:limit]
            
            # Convert to JSON-serializable format
            events_data = []
            for event in events:
                event_dict = event.dict()
                event_dict['start_time'] = event.start_time.isoformat()
                if event.end_time:
                    event_dict['end_time'] = event.end_time.isoformat()
                    event_dict['duration_seconds'] = event.duration
                events_data.append(event_dict)
            
            return jsonify({
                'events': events_data,
                'count': len(events_data),
                'query': {
                    'start_date': start_date.isoformat() if start_date else None,
                    'end_date': end_date.isoformat() if end_date else None,
                    'limit': limit
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @api.route('/events/<int:event_id>/detections')
    def get_event_detections(event_id: int):
        """Get all detections for a specific event.
        
        Parameters:
        - event_id: Database ID of the event
        """
        try:
            database = current_app.config['DATABASE']
            detections = database.get_detections_by_event(event_id)
            
            if not detections:
                return jsonify({'error': 'Event not found or no detections'}), 404
            
            # Convert to JSON-serializable format
            detections_data = []
            for detection in detections:
                detection_dict = detection.dict()
                detection_dict['timestamp'] = detection.timestamp.isoformat()
                detections_data.append(detection_dict)
            
            return jsonify({
                'detections': detections_data,
                'count': len(detections_data),
                'event_id': event_id
            })
            
        except Exception as e:
            logger.error(f"Failed to get detections for event {event_id}: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @api.route('/stats/summary')
    def get_summary_stats():
        """Get summary statistics across all data."""
        try:
            database = current_app.config['DATABASE']
            
            # Get recent reports for statistics
            recent_reports = database.get_daily_reports(limit=30)
            
            if not recent_reports:
                return jsonify({
                    'total_events': 0,
                    'total_detections': 0,
                    'total_duration_hours': 0.0,
                    'avg_events_per_day': 0.0,
                    'last_activity_date': None,
                    'data_range_days': 0
                })
            
            # Calculate summary statistics
            total_events = sum(r.total_events for r in recent_reports)
            total_detections = sum(r.total_detections for r in recent_reports)
            total_duration_hours = sum(r.total_duration_minutes for r in recent_reports) / 60.0
            
            days_with_data = len([r for r in recent_reports if r.total_events > 0])
            avg_events_per_day = total_events / max(days_with_data, 1)
            
            last_activity_date = None
            for report in recent_reports:
                if report.total_events > 0:
                    last_activity_date = report.date
                    break
            
            return jsonify({
                'total_events': total_events,
                'total_detections': total_detections,
                'total_duration_hours': round(total_duration_hours, 2),
                'avg_events_per_day': round(avg_events_per_day, 1),
                'last_activity_date': last_activity_date.isoformat() if last_activity_date else None,
                'data_range_days': len(recent_reports),
                'days_with_activity': days_with_data
            })
            
        except Exception as e:
            logger.error(f"Failed to get summary stats: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @api.route('/stats/hourly')
    def get_hourly_distribution():
        """Get hourly distribution of siren events."""
        try:
            # Parse query parameters
            days = request.args.get('days', 30, type=int)
            
            if days < 1 or days > 365:
                return jsonify({'error': 'days must be between 1 and 365'}), 400
            
            database = current_app.config['DATABASE']
            reports = database.get_daily_reports(limit=days)
            
            if not reports:
                return jsonify({'error': 'No data available'}), 404
            
            # Aggregate hourly data across all reports
            hourly_totals = [0] * 24
            total_days = len(reports)
            
            for report in reports:
                if report.events_by_hour and len(report.events_by_hour) == 24:
                    for hour, count in enumerate(report.events_by_hour):
                        hourly_totals[hour] += count
            
            # Calculate averages
            hourly_averages = [total / max(total_days, 1) for total in hourly_totals]
            
            return jsonify({
                'hourly_distribution': {
                    'hours': list(range(24)),
                    'totals': hourly_totals,
                    'averages': [round(avg, 2) for avg in hourly_averages]
                },
                'metadata': {
                    'days_analyzed': total_days,
                    'total_events': sum(hourly_totals)
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to get hourly distribution: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @api.route('/config')
    def get_config():
        """Get current configuration (sanitized)."""
        try:
            config = current_app.config['SIREN_CONFIG']
            
            # Return sanitized configuration (no sensitive data)
            config_dict = {
                'audio': {
                    'sample_rate': config.audio.sample_rate,
                    'samples_per_window': config.audio.samples_per_window,
                    'channels': config.audio.channels,
                    'window_duration': config.audio.window_duration
                },
                'detection': {
                    'confidence_threshold': config.confidence_threshold,
                    'siren_keywords': config.siren_keywords
                },
                'model': {
                    'model_path': str(config.model_path),
                    'labels_path': str(config.labels_path)
                }
            }
            
            return jsonify(config_dict)
            
        except Exception as e:
            logger.error(f"Failed to get config: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @api.route('/broadcast/detection', methods=['POST'])
    def broadcast_detection():
        """Receive detection broadcast from detector service."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Broadcast via WebSocket if available
            if 'WEBSOCKET_MANAGER' in current_app.config:
                from ..models.events import SirenDetection
                detection = SirenDetection(**data)
                current_app.config['WEBSOCKET_MANAGER'].broadcast_new_detection(detection)
            
            return jsonify({'status': 'broadcasted'}), 200
            
        except Exception as e:
            logger.error(f"Failed to broadcast detection: {e}")
            return jsonify({'error': 'Broadcast failed'}), 500
    
    @api.route('/broadcast/event', methods=['POST'])
    def broadcast_event():
        """Receive event broadcast from detector service."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Broadcast via WebSocket if available
            if 'WEBSOCKET_MANAGER' in current_app.config:
                from ..models.events import SirenEvent
                event = SirenEvent(**data)
                current_app.config['WEBSOCKET_MANAGER'].broadcast_event_update(event)
            
            return jsonify({'status': 'broadcasted'}), 200
            
        except Exception as e:
            logger.error(f"Failed to broadcast event: {e}")
            return jsonify({'error': 'Broadcast failed'}), 500
    
    return api