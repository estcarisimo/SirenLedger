"""Database operations and management for siren event storage."""

import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from sqlalchemy import create_engine, func, desc, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from ..models.config import DatabaseConfig
from ..models.events import SirenEvent, SirenDetection, DailyReport
from .models import Base, SirenEventDB, SirenDetectionDB, DailyReportDB, SystemStatusDB

logger = logging.getLogger(__name__)


class Database:
    """Database manager for siren event storage and retrieval.
    
    Provides high-level interface for storing and querying siren detection data.
    
    Parameters
    ----------
    config : DatabaseConfig
        Database configuration including URL and options.
        
    Attributes
    ----------
    engine : sqlalchemy.Engine
        Database engine for connections.
    SessionLocal : sessionmaker
        Session factory for database operations.
    """
    
    def __init__(self, config: DatabaseConfig) -> None:
        """Initialize database connection.
        
        Parameters
        ----------
        config : DatabaseConfig
            Database configuration.
        """
        self.config = config
        self.engine = create_engine(
            config.url,
            echo=config.echo,
            # SQLite-specific optimizations
            connect_args={"check_same_thread": False} if "sqlite" in config.url else {}
        )
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables if they don't exist
        self.create_tables()
        
        logger.info(f"Database initialized: {config.url}")
    
    def create_tables(self) -> None:
        """Create database tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.debug("Database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session.
        
        Returns
        -------
        Session
            SQLAlchemy session for database operations.
        """
        return self.SessionLocal()
    
    def store_detection(self, detection: SirenDetection) -> int:
        """Store a siren detection in the database.
        
        Parameters
        ----------
        detection : SirenDetection
            Detection event to store.
            
        Returns
        -------
        int
            Database ID of the stored detection.
            
        Raises
        ------
        SQLAlchemyError
            If database operation fails.
        """
        with self.get_session() as session:
            try:
                db_detection = SirenDetectionDB(
                    timestamp=detection.timestamp,
                    confidence=detection.confidence,
                    detected_class=detection.detected_class,
                    class_index=detection.class_index,
                    duration_estimate=detection.duration_estimate
                )
                
                session.add(db_detection)
                session.commit()
                session.refresh(db_detection)
                
                logger.debug(f"Stored detection: {db_detection.id}")
                return db_detection.id
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to store detection: {e}")
                raise
    
    def store_event(self, event: SirenEvent) -> int:
        """Store a siren event in the database.
        
        Parameters
        ----------
        event : SirenEvent
            Event to store.
            
        Returns
        -------
        int
            Database ID of the stored event.
        """
        with self.get_session() as session:
            try:
                db_event = SirenEventDB(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    max_confidence=event.max_confidence,
                    avg_confidence=event.avg_confidence,
                    detection_count=event.detection_count,
                    dominant_class=event.dominant_class
                )
                
                session.add(db_event)
                session.commit()
                session.refresh(db_event)
                
                logger.debug(f"Stored event: {db_event.id}")
                return db_event.id
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to store event: {e}")
                raise
    
    def get_events_by_date_range(
        self, 
        start_date: date, 
        end_date: date
    ) -> List[SirenEvent]:
        """Retrieve events within a date range.
        
        Parameters
        ----------
        start_date : date
            Start date (inclusive).
        end_date : date
            End date (inclusive).
            
        Returns
        -------
        List[SirenEvent]
            List of siren events in the date range.
        """
        with self.get_session() as session:
            try:
                start_datetime = datetime.combine(start_date, datetime.min.time())
                end_datetime = datetime.combine(end_date, datetime.max.time())
                
                db_events = session.query(SirenEventDB).filter(
                    and_(
                        SirenEventDB.start_time >= start_datetime,
                        SirenEventDB.start_time <= end_datetime
                    )
                ).order_by(SirenEventDB.start_time).all()
                
                events = []
                for db_event in db_events:
                    event = SirenEvent(
                        id=db_event.id,
                        start_time=db_event.start_time,
                        end_time=db_event.end_time,
                        max_confidence=db_event.max_confidence,
                        avg_confidence=db_event.avg_confidence,
                        detection_count=db_event.detection_count,
                        dominant_class=db_event.dominant_class
                    )
                    events.append(event)
                
                return events
                
            except Exception as e:
                logger.error(f"Failed to get events by date range: {e}")
                raise
    
    def get_detections_by_event(self, event_id: int) -> List[SirenDetection]:
        """Get all detections for a specific event.
        
        Parameters
        ----------
        event_id : int
            Event database ID.
            
        Returns
        -------
        List[SirenDetection]
            List of detections for the event.
        """
        with self.get_session() as session:
            try:
                db_detections = session.query(SirenDetectionDB).filter(
                    SirenDetectionDB.event_id == event_id
                ).order_by(SirenDetectionDB.timestamp).all()
                
                detections = []
                for db_detection in db_detections:
                    detection = SirenDetection(
                        timestamp=db_detection.timestamp,
                        confidence=db_detection.confidence,
                        detected_class=db_detection.detected_class,
                        class_index=db_detection.class_index,
                        duration_estimate=db_detection.duration_estimate
                    )
                    detections.append(detection)
                
                return detections
                
            except Exception as e:
                logger.error(f"Failed to get detections for event {event_id}: {e}")
                raise
    
    def generate_daily_report(self, target_date: date) -> DailyReport:
        """Generate daily report for a specific date.
        
        Parameters
        ----------
        target_date : date
            Date to generate report for.
            
        Returns
        -------
        DailyReport
            Daily activity report.
        """
        with self.get_session() as session:
            try:
                start_datetime = datetime.combine(target_date, datetime.min.time())
                end_datetime = datetime.combine(target_date, datetime.max.time())
                
                # Get all events for the day
                events = session.query(SirenEventDB).filter(
                    and_(
                        SirenEventDB.start_time >= start_datetime,
                        SirenEventDB.start_time <= end_datetime
                    )
                ).all()
                
                # Get all detections for the day
                detections = session.query(SirenDetectionDB).filter(
                    and_(
                        SirenDetectionDB.timestamp >= start_datetime,
                        SirenDetectionDB.timestamp <= end_datetime
                    )
                ).all()
                
                # Calculate statistics
                total_events = len(events)
                total_detections = len(detections)
                
                if events:
                    # Duration calculations
                    durations = [
                        (e.end_time - e.start_time).total_seconds() / 60
                        for e in events if e.end_time is not None
                    ]
                    total_duration_minutes = sum(durations)
                    longest_event_duration = max(durations) if durations else 0.0
                    
                    # Average confidence
                    avg_confidence = sum(e.avg_confidence for e in events) / len(events)
                    
                    # Hourly distribution
                    events_by_hour = [0] * 24
                    for event in events:
                        hour = event.start_time.hour
                        events_by_hour[hour] += 1
                    
                    # Dominant classes
                    class_counts = {}
                    for event in events:
                        class_counts[event.dominant_class] = class_counts.get(event.dominant_class, 0) + 1
                    
                    dominant_classes = sorted(
                        class_counts.keys(), 
                        key=lambda x: class_counts[x], 
                        reverse=True
                    )
                else:
                    total_duration_minutes = 0.0
                    longest_event_duration = 0.0
                    avg_confidence = 0.0
                    events_by_hour = [0] * 24
                    dominant_classes = []
                
                report = DailyReport(
                    date=target_date,
                    total_events=total_events,
                    total_detections=total_detections,
                    total_duration_minutes=total_duration_minutes,
                    longest_event_duration=longest_event_duration,
                    avg_confidence=avg_confidence,
                    events_by_hour=events_by_hour,
                    dominant_classes=dominant_classes
                )
                
                return report
                
            except Exception as e:
                logger.error(f"Failed to generate daily report for {target_date}: {e}")
                raise
    
    def store_daily_report(self, report: DailyReport) -> None:
        """Store daily report in database.
        
        Parameters
        ----------
        report : DailyReport
            Report to store.
        """
        with self.get_session() as session:
            try:
                # Check if report already exists
                existing = session.query(DailyReportDB).filter(
                    DailyReportDB.date == report.date
                ).first()
                
                if existing:
                    # Update existing report
                    existing.total_events = report.total_events
                    existing.total_detections = report.total_detections
                    existing.total_duration_minutes = report.total_duration_minutes
                    existing.longest_event_duration = report.longest_event_duration
                    existing.avg_confidence = report.avg_confidence
                    existing.events_by_hour = json.dumps(report.events_by_hour)
                    existing.dominant_classes = json.dumps(report.dominant_classes)
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new report
                    db_report = DailyReportDB(
                        date=report.date,
                        total_events=report.total_events,
                        total_detections=report.total_detections,
                        total_duration_minutes=report.total_duration_minutes,
                        longest_event_duration=report.longest_event_duration,
                        avg_confidence=report.avg_confidence,
                        events_by_hour=json.dumps(report.events_by_hour),
                        dominant_classes=json.dumps(report.dominant_classes)
                    )
                    session.add(db_report)
                
                session.commit()
                logger.debug(f"Stored daily report for {report.date}")
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to store daily report: {e}")
                raise
    
    def get_daily_reports(
        self, 
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> List[DailyReport]:
        """Get daily reports within date range.
        
        Parameters
        ----------
        start_date : Optional[date]
            Start date (inclusive). If None, no start limit.
        end_date : Optional[date]
            End date (inclusive). If None, no end limit.
        limit : Optional[int]
            Maximum number of reports to return.
            
        Returns
        -------
        List[DailyReport]
            List of daily reports.
        """
        with self.get_session() as session:
            try:
                query = session.query(DailyReportDB)
                
                if start_date:
                    query = query.filter(DailyReportDB.date >= start_date)
                if end_date:
                    query = query.filter(DailyReportDB.date <= end_date)
                
                query = query.order_by(desc(DailyReportDB.date))
                
                if limit:
                    query = query.limit(limit)
                
                db_reports = query.all()
                
                reports = []
                for db_report in db_reports:
                    report = DailyReport(
                        date=db_report.date,
                        total_events=db_report.total_events,
                        total_detections=db_report.total_detections,
                        total_duration_minutes=db_report.total_duration_minutes,
                        longest_event_duration=db_report.longest_event_duration,
                        avg_confidence=db_report.avg_confidence,
                        events_by_hour=json.loads(db_report.events_by_hour),
                        dominant_classes=json.loads(db_report.dominant_classes)
                    )
                    reports.append(report)
                
                return reports
                
            except Exception as e:
                logger.error(f"Failed to get daily reports: {e}")
                raise
    
    def migrate_csv_data(self, csv_path: Path) -> int:
        """Migrate data from legacy CSV format.
        
        Parameters
        ----------
        csv_path : Path
            Path to siren_daily_counts.csv file.
            
        Returns
        -------
        int
            Number of records migrated.
        """
        if not csv_path.exists():
            logger.warning(f"CSV file not found: {csv_path}")
            return 0
        
        imported_count = 0
        
        try:
            with open(csv_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        date_str, count_str = line.split(',')
                        report_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        count = int(count_str)
                        
                        # Create minimal daily report (only count available from CSV)
                        report = DailyReport(
                            date=report_date,
                            total_events=count,
                            total_detections=count,  # Assume 1 detection per event
                            total_duration_minutes=0.0,  # Unknown from CSV
                            longest_event_duration=0.0,  # Unknown from CSV
                            avg_confidence=0.0,  # Unknown from CSV
                            events_by_hour=[0] * 24,  # Unknown from CSV
                            dominant_classes=[]  # Unknown from CSV
                        )
                        
                        self.store_daily_report(report)
                        imported_count += 1
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping invalid CSV line: {line} ({e})")
                        continue
            
            logger.info(f"Migrated {imported_count} records from CSV")
            return imported_count
            
        except Exception as e:
            logger.error(f"Failed to migrate CSV data: {e}")
            raise
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> Tuple[int, int]:
        """Clean up old data beyond retention period.
        
        Parameters
        ----------
        days_to_keep : int
            Number of days of data to retain.
            
        Returns
        -------
        Tuple[int, int]
            Number of (events, detections) deleted.
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        with self.get_session() as session:
            try:
                # Delete old detections
                deleted_detections = session.query(SirenDetectionDB).filter(
                    SirenDetectionDB.timestamp < cutoff_date
                ).delete()
                
                # Delete old events
                deleted_events = session.query(SirenEventDB).filter(
                    SirenEventDB.start_time < cutoff_date
                ).delete()
                
                session.commit()
                
                logger.info(f"Cleaned up {deleted_events} events and {deleted_detections} detections")
                return deleted_events, deleted_detections
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to cleanup old data: {e}")
                raise