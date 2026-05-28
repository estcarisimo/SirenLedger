"""Configuration models using Pydantic for type safety and validation."""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class AudioConfig(BaseModel):
    """Configuration for audio capture and processing.
    
    Parameters
    ----------
    sample_rate : int, default=16000
        Audio sample rate in Hz. YAMNet requires 16kHz.
    samples_per_window : int, default=15600
        Number of samples per inference window. YAMNet requires exactly 15,600.
    channels : int, default=1
        Number of audio channels. YAMNet requires mono (1 channel).
    device_id : Optional[int], default=None
        Audio device ID. If None, uses system default.
    """
    
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    samples_per_window: int = Field(default=15600, ge=1000)
    channels: int = Field(default=1, ge=1, le=2)
    device_id: Optional[int] = Field(default=None, ge=0)
    
    @property
    def window_duration(self) -> float:
        """Calculate window duration in seconds.
        
        Returns
        -------
        float
            Window duration in seconds.
        """
        return self.samples_per_window / self.sample_rate


class DatabaseConfig(BaseModel):
    """Configuration for database storage.
    
    Parameters
    ----------
    url : str, default="sqlite:///siren_ledger.db"
        Database connection URL.
    echo : bool, default=False
        Whether to echo SQL queries for debugging.
    """
    
    url: str = Field(default="sqlite:///siren_ledger.db")
    echo: bool = Field(default=False)


class SirenConfig(BaseModel):
    """Main configuration for siren detection system.
    
    Parameters
    ----------
    model_path : Path, default="yamnet.tflite"
        Path to YAMNet TensorFlow Lite model file.
    labels_path : Path, default="yamnet_label_list.txt"
        Path to YAMNet class labels file.
    confidence_threshold : float, default=0.20
        Minimum confidence threshold for siren detection.
    siren_keywords : List[str]
        Keywords used to identify siren-related classes in YAMNet labels.
    audio : AudioConfig
        Audio processing configuration.
    database : DatabaseConfig
        Database configuration.
    log_level : str, default="INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    
    model_path: Path = Field(default=Path("yamnet.tflite"))
    labels_path: Path = Field(default=Path("yamnet_label_list.txt"))
    confidence_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    siren_keywords: List[str] = Field(
        default=["siren", "police car", "ambulance", "fire engine", "fire truck"]
    )
    audio: AudioConfig = Field(default_factory=AudioConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    log_level: str = Field(default="INFO")
    
    @validator("model_path", "labels_path")
    def validate_file_exists(cls, v: Path) -> Path:
        """Validate that required files exist.
        
        Parameters
        ----------
        v : Path
            File path to validate.
            
        Returns
        -------
        Path
            Validated file path.
            
        Raises
        ------
        ValueError
            If file does not exist.
        """
        if not v.exists():
            raise ValueError(f"File not found: {v}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is supported.
        
        Parameters
        ----------
        v : str
            Log level string.
            
        Returns
        -------
        str
            Validated log level.
            
        Raises
        ------
        ValueError
            If log level is not supported.
        """
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True