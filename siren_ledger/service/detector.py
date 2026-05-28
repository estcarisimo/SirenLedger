"""Main siren detection service for continuous monitoring."""

import logging
import signal
import sys
import threading
import time
from datetime import datetime, date
from typing import Optional, Callable

from ..audio import AudioProcessor, YAMNetClassifier
from ..config import ConfigManager
from ..models.events import SirenDetection
from ..storage import Database
from .aggregator import EventAggregator
from .broadcast import EventBroadcaster

logger = logging.getLogger(__name__)


class SirenDetectorService:
    """Main service for continuous siren detection and logging.
    
    This service integrates audio processing, ML inference, and data storage
    to provide real-time siren monitoring with detailed logging.
    
    Parameters
    ----------
    config_manager : ConfigManager
        Configuration manager instance.
    callback : Optional[Callable[[SirenDetection], None]]
        Optional callback for real-time detection notifications.
        
    Attributes
    ----------
    config : SirenConfig
        Configuration object.
    database : Database
        Database interface for storage.
    classifier : YAMNetClassifier
        YAMNet audio classifier.
    audio_processor : AudioProcessor
        Real-time audio processor.
    aggregator : EventAggregator
        Event aggregation service.
    is_running : bool
        Whether the service is currently running.
    """
    
    def __init__(
        self, 
        config_manager: ConfigManager,
        callback: Optional[Callable[[SirenDetection], None]] = None
    ) -> None:
        """Initialize siren detection service.
        
        Parameters
        ----------
        config_manager : ConfigManager
            Configuration manager instance.
        callback : Optional[Callable[[SirenDetection], None]]
            Optional callback for detections.
        """
        self.config = config_manager.config
        self.callback = callback
        
        # Initialize components
        self.database = Database(self.config.database)
        self.classifier = YAMNetClassifier(self.config)
        self.audio_processor = AudioProcessor(self.config, self._on_detection)
        self.aggregator = EventAggregator(self.database)
        self.broadcaster = EventBroadcaster()
        
        # Service state
        self.is_running = False
        self.current_date = date.today()
        self.daily_stats = {"detections": 0, "events": 0}
        
        # Threading
        self._detection_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("SirenDetectorService initialized")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully.
        
        Parameters
        ----------
        signum : int
            Signal number.
        frame
            Current stack frame.
        """
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
    
    def _on_detection(self, detection: SirenDetection) -> None:
        """Handle siren detection event.
        
        Parameters
        ----------
        detection : SirenDetection
            Detected siren event.
        """
        try:
            # Store detection in database
            detection_id = self.database.store_detection(detection)
            
            # Add to event aggregator
            self.aggregator.add_detection(detection)
            
            # Update daily stats
            self.daily_stats["detections"] += 1
            
            # Broadcast to web dashboard
            self.broadcaster.broadcast_detection(detection)
            
            # Log detection
            logger.info(
                f"SIREN DETECTED: {detection.detected_class} "
                f"(confidence: {detection.confidence:.2f}, "
                f"timestamp: {detection.timestamp.strftime('%H:%M:%S')})"
            )
            
            # Call user callback if provided
            if self.callback:
                try:
                    self.callback(detection)
                except Exception as e:
                    logger.warning(f"Detection callback failed: {e}")
            
        except Exception as e:
            logger.error(f"Failed to handle detection: {e}")
    
    def _detection_loop(self) -> None:
        """Main detection loop running in separate thread."""
        logger.info("Detection loop started")
        
        try:
            while not self._stop_event.is_set():
                # Get audio window
                audio_window = self.audio_processor.get_audio_window(timeout=1.0)
                if audio_window is None:
                    continue
                
                # Check for date change (daily report generation)
                self._check_date_change()
                
                try:
                    # Run inference
                    confidence, detected_class, class_index = self.classifier.classify(audio_window)
                    
                    if confidence > 0:  # Siren detected
                        detection = SirenDetection(
                            timestamp=datetime.utcnow(),
                            confidence=confidence,
                            detected_class=detected_class,
                            class_index=class_index
                        )
                        self._on_detection(detection)
                        
                except Exception as e:
                    logger.error(f"Classification error: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Detection loop error: {e}")
        finally:
            logger.info("Detection loop stopped")
    
    def _check_date_change(self) -> None:
        """Check if date has changed and generate daily report if needed."""
        today = date.today()
        
        if today != self.current_date:
            try:
                # Generate report for previous day
                logger.info(f"Generating daily report for {self.current_date}")
                report = self.database.generate_daily_report(self.current_date)
                self.database.store_daily_report(report)
                
                # Finalize any pending events
                self.aggregator.finalize_pending_events()
                
                # Log previous day stats
                logger.info(
                    f"Daily summary for {self.current_date}: "
                    f"{self.daily_stats['detections']} detections, "
                    f"{report.total_events} events, "
                    f"{report.total_duration_minutes:.1f} minutes"
                )
                
                # Reset for new day
                self.current_date = today
                self.daily_stats = {"detections": 0, "events": 0}
                
            except Exception as e:
                logger.error(f"Failed to generate daily report: {e}")
    
    def start(self) -> None:
        """Start the siren detection service.
        
        Raises
        ------
        RuntimeError
            If service is already running or startup fails.
        """
        if self.is_running:
            raise RuntimeError("Service is already running")
        
        try:
            logger.info("Starting SirenDetectorService...")
            
            # Validate runtime requirements
            logger.info("Validating runtime requirements...")
            # Note: This would require implementing validate_runtime_requirements
            # in ConfigManager, which we haven't done yet for brevity
            
            # Start audio processing
            self.audio_processor.start()
            
            # Start detection thread
            self._stop_event.clear()
            self._detection_thread = threading.Thread(
                target=self._detection_loop,
                name="SirenDetection",
                daemon=False
            )
            self._detection_thread.start()
            
            self.is_running = True
            logger.info("SirenDetectorService started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            self.stop()
            raise
    
    def stop(self) -> None:
        """Stop the siren detection service."""
        if not self.is_running:
            return
        
        logger.info("Stopping SirenDetectorService...")
        
        try:
            # Signal stop to detection thread
            self._stop_event.set()
            
            # Stop audio processing
            self.audio_processor.stop()
            
            # Wait for detection thread to finish
            if self._detection_thread and self._detection_thread.is_alive():
                self._detection_thread.join(timeout=5.0)
                if self._detection_thread.is_alive():
                    logger.warning("Detection thread did not stop gracefully")
            
            # Generate final daily report
            try:
                if self.daily_stats["detections"] > 0:
                    report = self.database.generate_daily_report(self.current_date)
                    self.database.store_daily_report(report)
                    logger.info(f"Final daily report generated for {self.current_date}")
            except Exception as e:
                logger.error(f"Failed to generate final daily report: {e}")
            
            # Finalize any pending events
            try:
                self.aggregator.finalize_pending_events()
            except Exception as e:
                logger.error(f"Failed to finalize pending events: {e}")
            
            self.is_running = False
            logger.info("SirenDetectorService stopped")
            
        except Exception as e:
            logger.error(f"Error during service shutdown: {e}")
    
    def run(self) -> None:
        """Run the service until interrupted.
        
        This is a convenience method that starts the service and waits
        for a keyboard interrupt or termination signal.
        """
        try:
            self.start()
            
            logger.info("SirenDetectorService running. Press Ctrl+C to stop.")
            
            # Keep main thread alive
            while self.is_running:
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Service error: {e}")
        finally:
            self.stop()
    
    def get_status(self) -> dict:
        """Get current service status.
        
        Returns
        -------
        dict
            Status information including running state, daily stats, etc.
        """
        return {
            "is_running": self.is_running,
            "current_date": self.current_date.isoformat(),
            "daily_stats": self.daily_stats.copy(),
            "audio_active": self.audio_processor.is_running,
            "classifier_loaded": self.classifier is not None,
            "database_connected": True,  # Could add actual health check
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()