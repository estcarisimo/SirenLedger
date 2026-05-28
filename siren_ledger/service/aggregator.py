"""Event aggregation service for grouping related siren detections."""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..models.events import SirenDetection, SirenEvent
from ..storage import Database

logger = logging.getLogger(__name__)


class EventAggregator:
    """Aggregates individual detections into coherent siren events.
    
    Groups nearby detections in time to create meaningful events with
    start/end times and aggregate statistics.
    
    Parameters
    ----------
    database : Database
        Database interface for storing events.
    max_gap_seconds : float, default=5.0
        Maximum gap between detections to consider them part of same event.
    min_detections : int, default=2
        Minimum detections required to create an event.
        
    Attributes
    ----------
    database : Database
        Database interface.
    max_gap : timedelta
        Maximum gap between detections.
    min_detections : int
        Minimum detections per event.
    pending_events : Dict[str, List[SirenDetection]]
        Events currently being built, keyed by class name.
    """
    
    def __init__(
        self, 
        database: Database,
        max_gap_seconds: float = 5.0,
        min_detections: int = 2
    ) -> None:
        """Initialize event aggregator.
        
        Parameters
        ----------
        database : Database
            Database interface.
        max_gap_seconds : float
            Maximum gap between detections in seconds.
        min_detections : int
            Minimum detections to form an event.
        """
        self.database = database
        self.max_gap = timedelta(seconds=max_gap_seconds)
        self.min_detections = min_detections
        
        # Track pending events by class
        self.pending_events: Dict[str, List[SirenDetection]] = defaultdict(list)
        
        logger.info(f"EventAggregator initialized: max_gap={max_gap_seconds}s, "
                   f"min_detections={min_detections}")
    
    def add_detection(self, detection: SirenDetection) -> Optional[SirenEvent]:
        """Add a detection and potentially create/update events.
        
        Parameters
        ----------
        detection : SirenDetection
            New detection to process.
            
        Returns
        -------
        Optional[SirenEvent]
            Completed event if one was finalized, None otherwise.
        """
        class_name = detection.detected_class
        
        # Check if this detection extends an existing pending event
        if class_name in self.pending_events:
            last_detection = self.pending_events[class_name][-1]
            time_gap = detection.timestamp - last_detection.timestamp
            
            if time_gap <= self.max_gap:
                # Extend existing event
                self.pending_events[class_name].append(detection)
                logger.debug(f"Extended pending event for {class_name}: "
                           f"{len(self.pending_events[class_name])} detections")
                return None
            else:
                # Gap too large - finalize existing event and start new one
                completed_event = self._finalize_event(class_name)
                self.pending_events[class_name] = [detection]
                logger.debug(f"Started new event for {class_name} after gap of {time_gap}")
                return completed_event
        else:
            # Start new event for this class
            self.pending_events[class_name] = [detection]
            logger.debug(f"Started first event for {class_name}")
            return None
    
    def _finalize_event(self, class_name: str) -> Optional[SirenEvent]:
        """Finalize a pending event and store it in database.
        
        Parameters
        ----------
        class_name : str
            Class name of the event to finalize.
            
        Returns
        -------
        Optional[SirenEvent]
            Finalized event if it met minimum requirements, None otherwise.
        """
        detections = self.pending_events[class_name]
        
        if len(detections) < self.min_detections:
            logger.debug(f"Discarding event for {class_name}: "
                        f"only {len(detections)} detections (min: {self.min_detections})")
            return None
        
        # Calculate event statistics
        start_time = detections[0].timestamp
        end_time = detections[-1].timestamp
        confidences = [d.confidence for d in detections]
        max_confidence = max(confidences)
        avg_confidence = sum(confidences) / len(confidences)
        
        # Create event object
        event = SirenEvent(
            start_time=start_time,
            end_time=end_time,
            max_confidence=max_confidence,
            avg_confidence=avg_confidence,
            detection_count=len(detections),
            dominant_class=class_name
        )
        
        try:
            # Store in database
            event_id = self.database.store_event(event)
            event.id = event_id
            
            # Update detection records to link to this event
            self._link_detections_to_event(detections, event_id)
            
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Finalized event {event_id}: {class_name} "
                       f"({len(detections)} detections, {duration:.1f}s, "
                       f"max_conf={max_confidence:.2f})")
            
            return event
            
        except Exception as e:
            logger.error(f"Failed to store event for {class_name}: {e}")
            return None
    
    def _link_detections_to_event(self, detections: List[SirenDetection], event_id: int) -> None:
        """Link individual detections to their parent event in database.
        
        Parameters
        ----------
        detections : List[SirenDetection]
            Detections to link.
        event_id : int
            Database ID of the parent event.
        """
        # This would require updating the detection records in the database
        # to set their event_id foreign key. For now, we'll just log it.
        # In a full implementation, we'd need to add this functionality
        # to the Database class.
        logger.debug(f"Would link {len(detections)} detections to event {event_id}")
    
    def finalize_pending_events(self) -> List[SirenEvent]:
        """Finalize all pending events.
        
        Called at end of day or service shutdown to process any
        incomplete events.
        
        Returns
        -------
        List[SirenEvent]
            List of finalized events.
        """
        finalized_events = []
        
        for class_name in list(self.pending_events.keys()):
            event = self._finalize_event(class_name)
            if event:
                finalized_events.append(event)
            del self.pending_events[class_name]
        
        if finalized_events:
            logger.info(f"Finalized {len(finalized_events)} pending events")
        
        return finalized_events
    
    def check_stale_events(self, max_age_seconds: float = 300.0) -> List[SirenEvent]:
        """Check for and finalize stale pending events.
        
        Events that haven't received new detections for a while
        are considered complete and should be finalized.
        
        Parameters
        ----------
        max_age_seconds : float
            Maximum age of last detection before considering event stale.
            
        Returns
        -------
        List[SirenEvent]
            List of finalized stale events.
        """
        current_time = datetime.utcnow()
        max_age = timedelta(seconds=max_age_seconds)
        finalized_events = []
        
        for class_name in list(self.pending_events.keys()):
            if not self.pending_events[class_name]:
                continue
            
            last_detection = self.pending_events[class_name][-1]
            age = current_time - last_detection.timestamp
            
            if age > max_age:
                logger.debug(f"Finalizing stale event for {class_name} "
                           f"(age: {age.total_seconds():.1f}s)")
                event = self._finalize_event(class_name)
                if event:
                    finalized_events.append(event)
                del self.pending_events[class_name]
        
        return finalized_events
    
    def get_pending_summary(self) -> Dict[str, int]:
        """Get summary of pending events.
        
        Returns
        -------
        Dict[str, int]
            Dictionary mapping class names to detection counts.
        """
        return {
            class_name: len(detections)
            for class_name, detections in self.pending_events.items()
        }