"""Configuration management with support for files and environment variables."""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from pydantic import ValidationError

from ..models.config import SirenConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Configuration manager for loading and validating SirenLedger configuration.
    
    Supports loading configuration from:
    - JSON files
    - YAML files  
    - Environment variables
    - Default values
    
    Parameters
    ----------
    config_path : Optional[Path]
        Path to configuration file. If None, uses default locations.
        
    Attributes
    ----------
    config : SirenConfig
        Validated configuration object.
    """
    
    DEFAULT_CONFIG_PATHS = [
        Path("siren_config.yaml"),
        Path("siren_config.json"),
        Path("config/siren.yaml"),
        Path("config/siren.json"),
        Path.home() / ".config" / "siren_ledger" / "config.yaml",
        Path("/etc/siren_ledger/config.yaml"),
    ]
    
    ENV_PREFIX = "SIREN_"
    
    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize configuration manager.
        
        Parameters
        ----------
        config_path : Optional[Path]
            Explicit path to configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file using default search paths.
        
        Returns
        -------
        Optional[Path]
            Path to configuration file, or None if not found.
        """
        if self.config_path and self.config_path.exists():
            return self.config_path
        
        for path in self.DEFAULT_CONFIG_PATHS:
            if path.exists():
                logger.info(f"Found configuration file: {path}")
                return path
        
        logger.debug("No configuration file found, using defaults")
        return None
    
    def _load_from_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file.
        
        Parameters
        ----------
        config_path : Path
            Path to configuration file.
            
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary.
            
        Raises
        ------
        ValueError
            If file format is not supported or invalid.
        """
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            logger.info(f"Loaded configuration from {config_path}")
            return config_dict or {}
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}") from e
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables.
        
        Environment variables should be prefixed with SIREN_ and use
        double underscores for nested keys (e.g., SIREN_AUDIO__SAMPLE_RATE).
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary from environment.
        """
        config_dict = {}
        
        for key, value in os.environ.items():
            if not key.startswith(self.ENV_PREFIX):
                continue
            
            # Remove prefix and convert to lowercase
            config_key = key[len(self.ENV_PREFIX):].lower()
            
            # Handle nested keys (double underscore)
            key_parts = config_key.split('__')
            
            # Convert string values to appropriate types
            try:
                # Try to parse as JSON first (handles booleans, numbers, lists)
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                # Fall back to string
                parsed_value = value
            
            # Build nested dictionary
            current_dict = config_dict
            for part in key_parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            
            current_dict[key_parts[-1]] = parsed_value
        
        if config_dict:
            logger.debug(f"Loaded configuration from environment: {list(config_dict.keys())}")
        
        return config_dict
    
    def _merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries.
        
        Later configs override earlier ones for conflicting keys.
        
        Parameters
        ----------
        *configs : Dict[str, Any]
            Configuration dictionaries to merge.
            
        Returns
        -------
        Dict[str, Any]
            Merged configuration dictionary.
        """
        merged = {}
        
        for config in configs:
            if not config:
                continue
            
            for key, value in config.items():
                if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                    # Recursively merge nested dictionaries
                    merged[key] = self._merge_configs(merged[key], value)
                else:
                    merged[key] = value
        
        return merged
    
    def _load_config(self) -> SirenConfig:
        """Load and validate configuration from all sources.
        
        Returns
        -------
        SirenConfig
            Validated configuration object.
            
        Raises
        ------
        ValidationError
            If configuration validation fails.
        ValueError
            If configuration file is invalid.
        """
        # Start with defaults (handled by Pydantic)
        config_dict = {}
        
        # Load from file if available
        config_file = self._find_config_file()
        if config_file:
            file_config = self._load_from_file(config_file)
            config_dict = self._merge_configs(config_dict, file_config)
        
        # Load from environment (highest priority)
        env_config = self._load_from_env()
        config_dict = self._merge_configs(config_dict, env_config)
        
        try:
            # Validate and create SirenConfig object
            config = SirenConfig(**config_dict)
            logger.info("Configuration loaded and validated successfully")
            return config
            
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def save_config(self, output_path: Path, format: str = "yaml") -> None:
        """Save current configuration to file.
        
        Parameters
        ----------
        output_path : Path
            Output file path.
        format : str
            Output format ("yaml" or "json").
            
        Raises
        ------
        ValueError
            If format is not supported.
        """
        config_dict = self.config.dict()
        
        try:
            with open(output_path, 'w') as f:
                if format.lower() == "yaml":
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == "json":
                    json.dump(config_dict, f, indent=2, default=str)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def reload(self) -> None:
        """Reload configuration from sources."""
        self.config = self._load_config()
        logger.info("Configuration reloaded")
    
    def validate_runtime_requirements(self) -> None:
        """Validate that runtime requirements are met.
        
        Raises
        ------
        FileNotFoundError
            If required model files are missing.
        RuntimeError
            If audio device is not available.
        """
        # Check model files
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"YAMNet model file not found: {self.config.model_path}\n"
                "Download with: wget -O yamnet.tflite "
                "\"https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite\""
            )
        
        if not self.config.labels_path.exists():
            raise FileNotFoundError(
                f"YAMNet labels file not found: {self.config.labels_path}\n"
                "Download with: wget -O yamnet_label_list.txt "
                "https://storage.googleapis.com/mediapipe-tasks/audio_classifier/yamnet_label_list.txt"
            )
        
        # Test audio device availability
        try:
            import sounddevice as sd
            
            if self.config.audio.device_id is not None:
                devices = sd.query_devices()
                if self.config.audio.device_id >= len(devices):
                    raise RuntimeError(
                        f"Audio device {self.config.audio.device_id} not found. "
                        f"Available devices: 0-{len(devices)-1}"
                    )
            
            # Quick test recording
            test_duration = 0.1  # 100ms test
            test_recording = sd.rec(
                int(test_duration * self.config.audio.sample_rate),
                samplerate=self.config.audio.sample_rate,
                channels=self.config.audio.channels,
                device=self.config.audio.device_id,
                dtype='float32'
            )
            sd.wait()
            
            logger.info("Audio device validation successful")
            
        except Exception as e:
            raise RuntimeError(f"Audio device validation failed: {e}") from e
    
    @classmethod
    def create_default_config(cls, output_path: Path) -> None:
        """Create a default configuration file.
        
        Parameters
        ----------
        output_path : Path
            Path where to save the default configuration.
        """
        default_config = SirenConfig()
        manager = cls()
        manager.config = default_config
        
        format = "yaml" if output_path.suffix.lower() in ['.yaml', '.yml'] else "json"
        manager.save_config(output_path, format)
        
        logger.info(f"Default configuration created at {output_path}")
    
    def get_summary(self) -> str:
        """Get a human-readable configuration summary.
        
        Returns
        -------
        str
            Configuration summary.
        """
        config = self.config
        
        summary = f"""SirenLedger Configuration Summary:
        
Audio:
  - Sample Rate: {config.audio.sample_rate} Hz
  - Window Size: {config.audio.samples_per_window} samples ({config.audio.window_duration:.3f}s)
  - Channels: {config.audio.channels}
  - Device ID: {config.audio.device_id or 'default'}

Detection:
  - Model: {config.model_path}
  - Labels: {config.labels_path}
  - Confidence Threshold: {config.confidence_threshold}
  - Siren Classes: {len(config.siren_keywords)} keywords

Database:
  - URL: {config.database.url}
  - Echo SQL: {config.database.echo}

Logging:
  - Level: {config.log_level}
"""
        return summary