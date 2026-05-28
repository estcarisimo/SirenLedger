"""SQLAlchemy database models for siren event storage."""

from datetime import datetime, date
from typing import Optional, List

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Date, Boolean,
    ForeignKey, Text, JSON, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped

Base = declarative_base()


class SirenEventDB(Base):
    """Database model for siren events.
    
    Represents aggregated siren events with start/end times and statistics.
    """
    
    __tablename__ = "siren_events"
    
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    start_time: Mapped[datetime] = Column(DateTime, nullable=False, index=True)
    end_time: Mapped[Optional[datetime]] = Column(DateTime, nullable=True)
    max_confidence: Mapped[float] = Column(Float, nullable=False)
    avg_confidence: Mapped[float] = Column(Float, nullable=False)
    detection_count: Mapped[int] = Column(Integer, nullable=False)
    dominant_class: Mapped[str] = Column(String(100), nullable=False)
    
    # Relationship to individual detections
    detections: Mapped[List["SirenDetectionDB"]] = relationship(
        "SirenDetectionDB", back_populates="event", cascade="all, delete-orphan"
    )
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate event duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()
    
    def __repr__(self) -> str:
        return (f"<SirenEventDB(id={self.id}, start={self.start_time}, "
                f"confidence={self.max_confidence:.2f}, class='{self.dominant_class}')>")


class SirenDetectionDB(Base):
    """Database model for individual siren detections.
    
    Represents raw detection events from YAMNet classifier.
    """
    
    __tablename__ = "siren_detections"
    
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = Column(DateTime, nullable=False, index=True)
    confidence: Mapped[float] = Column(Float, nullable=False)
    detected_class: Mapped[str] = Column(String(100), nullable=False)
    class_index: Mapped[int] = Column(Integer, nullable=False)
    duration_estimate: Mapped[Optional[float]] = Column(Float, nullable=True)
    
    # Foreign key to parent event
    event_id: Mapped[Optional[int]] = Column(
        Integer, ForeignKey("siren_events.id"), nullable=True
    )
    event: Mapped[Optional[SirenEventDB]] = relationship(
        "SirenEventDB", back_populates="detections"
    )
    
    def __repr__(self) -> str:
        return (f"<SirenDetectionDB(id={self.id}, timestamp={self.timestamp}, "
                f"confidence={self.confidence:.2f}, class='{self.detected_class}')>")


class DailyReportDB(Base):
    """Database model for daily siren activity reports."""
    
    __tablename__ = "daily_reports"
    
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = Column(Date, nullable=False, unique=True, index=True)
    total_events: Mapped[int] = Column(Integer, nullable=False, default=0)
    total_detections: Mapped[int] = Column(Integer, nullable=False, default=0)
    total_duration_minutes: Mapped[float] = Column(Float, nullable=False, default=0.0)
    longest_event_duration: Mapped[float] = Column(Float, nullable=False, default=0.0)
    avg_confidence: Mapped[float] = Column(Float, nullable=False, default=0.0)
    
    # JSON field for hourly distribution (24 hours)
    events_by_hour: Mapped[str] = Column(JSON, nullable=False, default="[]")
    
    # JSON field for class distribution
    dominant_classes: Mapped[str] = Column(JSON, nullable=False, default="[]")
    
    # Metadata
    created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    
    def __repr__(self) -> str:
        return (f"<DailyReportDB(date={self.date}, events={self.total_events}, "
                f"duration={self.total_duration_minutes:.1f}min)>")


class SystemStatusDB(Base):
    """Database model for system status and health monitoring."""
    
    __tablename__ = "system_status"
    
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = Column(DateTime, nullable=False, index=True)
    
    # System health metrics
    is_audio_active: Mapped[bool] = Column(Boolean, nullable=False)
    audio_device_id: Mapped[Optional[int]] = Column(Integer, nullable=True)
    audio_rms_level: Mapped[Optional[float]] = Column(Float, nullable=True)
    
    # Processing metrics
    total_windows_processed: Mapped[int] = Column(Integer, nullable=False, default=0)
    avg_inference_time_ms: Mapped[Optional[float]] = Column(Float, nullable=True)
    queue_size: Mapped[Optional[int]] = Column(Integer, nullable=True)
    
    # Error tracking
    last_error: Mapped[Optional[str]] = Column(Text, nullable=True)
    error_count: Mapped[int] = Column(Integer, nullable=False, default=0)
    
    # System info
    cpu_usage: Mapped[Optional[float]] = Column(Float, nullable=True)
    memory_usage: Mapped[Optional[float]] = Column(Float, nullable=True)
    disk_usage: Mapped[Optional[float]] = Column(Float, nullable=True)
    
    def __repr__(self) -> str:
        return (f"<SystemStatusDB(timestamp={self.timestamp}, "
                f"audio_active={self.is_audio_active}, "
                f"windows_processed={self.total_windows_processed})>")


# Create indexes for performance
Index("idx_siren_events_start_time", SirenEventDB.start_time)
Index("idx_siren_detections_timestamp", SirenDetectionDB.timestamp)
Index("idx_siren_detections_event_id", SirenDetectionDB.event_id)
Index("idx_daily_reports_date", DailyReportDB.date)
Index("idx_system_status_timestamp", SystemStatusDB.timestamp)