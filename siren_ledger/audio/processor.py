"""Real-time audio processing for siren detection."""

import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from scipy import signal

from ..models.config import SirenConfig, AudioConfig
from ..models.events import SirenDetection

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Real-time audio processor for continuous siren monitoring.
    
    This class handles audio capture from microphone and provides
    windowed audio data for classification.
    
    Parameters
    ----------
    config : SirenConfig
        Configuration for audio processing and detection.
    callback : Optional[Callable[[SirenDetection], None]]
        Optional callback function called when siren is detected.
        
    Attributes
    ----------
    audio_queue : queue.Queue
        Thread-safe queue for audio data.
    stream : Optional[sd.InputStream]
        Sound device input stream.
    is_running : bool
        Whether audio processing is active.
    """
    
    def __init__(
        self, 
        config: SirenConfig,
        callback: Optional[Callable[[SirenDetection], None]] = None
    ) -> None:
        """Initialize audio processor.
        
        Parameters
        ----------
        config : SirenConfig
            Configuration object.
        callback : Optional[Callable[[SirenDetection], None]]
            Optional callback for siren detections.
        """
        self.config = config
        self.audio_config = config.audio
        self.callback = callback
        
        # YAMNet always expects 16kHz mono
        self.yamnet_sample_rate = 16000
        self.yamnet_samples_per_window = 15600  # ~0.975 seconds at 16kHz
        
        self.audio_queue: queue.Queue = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self.is_running = False
        
        # Calculate resampling ratio
        self.resample_ratio = self.yamnet_sample_rate / self.audio_config.sample_rate
        self.capture_samples_per_window = int(self.yamnet_samples_per_window / self.resample_ratio)
        
        logger.info(f"Audio processor initialized: {self.audio_config.sample_rate}Hz -> {self.yamnet_sample_rate}Hz, "
                   f"{self.audio_config.channels} channel(s)")
    
    def _audio_callback(
        self, 
        indata: np.ndarray, 
        frames: int, 
        time_info: dict, 
        status: sd.CallbackFlags
    ) -> None:
        """Callback function for real-time audio capture.
        
        This runs in the audio thread and should be fast to avoid dropouts.
        
        Parameters
        ----------
        indata : np.ndarray
            Audio data from microphone.
        frames : int
            Number of frames captured.
        time_info : dict
            Timing information.
        status : sd.CallbackFlags
            Status flags indicating any issues.
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Copy data to avoid issues with the audio buffer
        try:
            self.audio_queue.put(indata.copy(), block=False)
        except queue.Full:
            logger.warning("Audio queue full, dropping frame")
    
    def start(self) -> None:
        """Start audio capture stream.
        
        Raises
        ------
        RuntimeError
            If stream is already running or cannot be started.
        """
        if self.is_running:
            raise RuntimeError("Audio processor is already running")
        
        try:
            self.stream = sd.InputStream(
                device=self.audio_config.device_id,
                channels=self.audio_config.channels,
                samplerate=self.audio_config.sample_rate,
                blocksize=self.capture_samples_per_window,
                dtype=np.float32,
                callback=self._audio_callback
            )
            
            self.stream.start()
            self.is_running = True
            logger.info("Audio capture started")
            
        except Exception as e:
            raise RuntimeError(f"Failed to start audio stream: {e}") from e
    
    def stop(self) -> None:
        """Stop audio capture stream."""
        if not self.is_running:
            return
        
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            self.is_running = False
            logger.info("Audio capture stopped")
            
        except Exception as e:
            logger.error(f"Error stopping audio stream: {e}")
    
    def get_audio_window(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Get next audio window from the queue.
        
        Parameters
        ----------
        timeout : Optional[float]
            Timeout in seconds. If None, blocks indefinitely.
            
        Returns
        -------
        Optional[np.ndarray]
            Audio window as float32 array with shape (samples_per_window,).
            Returns None if timeout occurs or queue is empty.
        """
        try:
            chunk = self.audio_queue.get(timeout=timeout)
            
            # Convert to mono if needed (mix stereo to mono)
            if chunk.ndim > 1 and chunk.shape[1] > 1:
                mono = np.mean(chunk, axis=1)
            else:
                mono = np.squeeze(chunk)
            
            # Resample to 16kHz if needed
            if self.audio_config.sample_rate != self.yamnet_sample_rate:
                # Use scipy.signal.resample for high-quality resampling
                resampled_length = int(len(mono) * self.resample_ratio)
                mono = signal.resample(mono, resampled_length)
            
            # Ensure correct length for YAMNet (15600 samples at 16kHz)
            if mono.shape[0] != self.yamnet_samples_per_window:
                if mono.shape[0] > self.yamnet_samples_per_window:
                    # Truncate if too long
                    mono = mono[:self.yamnet_samples_per_window]
                else:
                    # Pad with zeros if too short
                    padding = self.yamnet_samples_per_window - mono.shape[0]
                    mono = np.pad(mono, (0, padding), mode='constant')
                    
                    if mono.shape[0] < self.yamnet_samples_per_window * 0.8:
                        logger.warning(f"Short audio window: {mono.shape[0]} samples")
            
            return mono.astype(np.float32)
            
        except queue.Empty:
            return None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    @staticmethod
    def list_audio_devices() -> None:
        """Print available audio devices to help with configuration."""
        print("Available audio devices:")
        print(sd.query_devices())
    
    @staticmethod
    def test_audio_device(device_id: Optional[int] = None, duration: float = 5.0) -> bool:
        """Test if audio device is working.
        
        Parameters
        ----------
        device_id : Optional[int]
            Device ID to test. If None, uses default.
        duration : float
            Test duration in seconds.
            
        Returns
        -------
        bool
            True if device works, False otherwise.
        """
        try:
            print(f"Testing audio device {device_id} for {duration} seconds...")
            
            recording = sd.rec(
                int(duration * 16000),
                samplerate=16000,
                channels=1,
                device=device_id,
                dtype=np.float32
            )
            sd.wait()
            
            # Check if we got reasonable audio levels
            rms = np.sqrt(np.mean(recording**2))
            print(f"RMS level: {rms:.6f}")
            
            if rms > 1e-6:
                print("✓ Audio device is working")
                return True
            else:
                print("⚠ Very low audio levels - check microphone")
                return False
                
        except Exception as e:
            print(f"✗ Audio device test failed: {e}")
            return False