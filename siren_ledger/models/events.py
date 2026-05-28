"""Event models for siren detection data."""

from datetime import datetime, date
from typing import Optional, List

from pydantic import BaseModel, Field


class SirenDetection(BaseModel):
    """Individual siren detection event.
    
    Parameters
    ----------
    timestamp : datetime
        When the siren was detected.
    confidence : float
        Detection confidence score (0.0 to 1.0).
    detected_class : str
        YAMNet class that triggered the detection.
    class_index : int
        YAMNet class index.
    duration_estimate : Optional[float], default=None
        Estimated duration of the siren in seconds.
    """
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(ge=0.0, le=1.0)
    detected_class: str
    class_index: int = Field(ge=0)
    duration_estimate: Optional[float] = Field(default=None, ge=0.0)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class SirenEvent(BaseModel):
    """Aggregated siren event with start/end times.
    
    Parameters
    ----------
    id : Optional[int], default=None
        Database primary key.
    start_time : datetime
        Event start timestamp.
    end_time : Optional[datetime], default=None
        Event end timestamp (None for ongoing events).
    max_confidence : float
        Maximum confidence during the event.
    avg_confidence : float
        Average confidence during the event.
    detection_count : int
        Number of individual detections in this event.
    dominant_class : str
        Most frequently detected class during the event.
    """
    
    id: Optional[int] = Field(default=None)
    start_time: datetime
    end_time: Optional[datetime] = Field(default=None)
    max_confidence: float = Field(ge=0.0, le=1.0)
    avg_confidence: float = Field(ge=0.0, le=1.0)
    detection_count: int = Field(ge=1)
    dominant_class: str
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate event duration in seconds.
        
        Returns
        -------
        Optional[float]
            Duration in seconds, or None if event is ongoing.
        """
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class DailyReport(BaseModel):
    """Daily summary of siren activity.
    
    Parameters
    ----------
    date : date
        Report date.
    total_events : int
        Total number of siren events.
    total_detections : int
        Total number of individual detections.
    total_duration_minutes : float
        Total duration of all events in minutes.
    longest_event_duration : float
        Duration of longest single event in minutes.
    avg_confidence : float
        Average confidence across all detections.
    events_by_hour : List[int]
        Number of events per hour (24 elements, index 0 = midnight-1am).
    dominant_classes : List[str]
        Most frequently detected classes in order.
    """
    
    date: date
    total_events: int = Field(ge=0)
    total_detections: int = Field(ge=0)
    total_duration_minutes: float = Field(ge=0.0)
    longest_event_duration: float = Field(ge=0.0)
    avg_confidence: float = Field(ge=0.0, le=1.0)
    events_by_hour: List[int] = Field(min_items=24, max_items=24)
    dominant_classes: List[str] = Field(default_factory=list)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            date: lambda v: v.isoformat(),
        }