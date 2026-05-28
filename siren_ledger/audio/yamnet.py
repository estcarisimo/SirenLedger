"""YAMNet TensorFlow Lite classifier for audio event detection."""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tflite_runtime.interpreter import Interpreter

from ..models.config import SirenConfig

logger = logging.getLogger(__name__)


class YAMNetClassifier:
    """YAMNet-based audio classifier for siren detection.
    
    This class wraps the YAMNet TensorFlow Lite model to provide
    siren detection functionality with configurable thresholds.
    
    Parameters
    ----------
    config : SirenConfig
        Configuration containing model paths and detection parameters.
        
    Attributes
    ----------
    interpreter : Interpreter
        TensorFlow Lite interpreter for the YAMNet model.
    labels : List[str]
        Class labels from YAMNet (521 classes).
    siren_indices : List[int]
        Indices of siren-related classes in the label list.
    """
    
    def __init__(self, config: SirenConfig) -> None:
        """Initialize YAMNet classifier.
        
        Parameters
        ----------
        config : SirenConfig
            Configuration object with model paths and parameters.
            
        Raises
        ------
        FileNotFoundError
            If model or labels file is not found.
        RuntimeError
            If TensorFlow Lite model cannot be loaded.
        """
        self.config = config
        self._load_model()
        self._load_labels()
        self._find_siren_indices()
        
        logger.info(f"YAMNet classifier initialized with {len(self.siren_indices)} siren classes")
        logger.debug(f"Tracking labels: {[self.labels[i] for i in self.siren_indices]}")
    
    def _load_model(self) -> None:
        """Load and initialize TensorFlow Lite model.
        
        Raises
        ------
        RuntimeError
            If model cannot be loaded or initialized.
        """
        try:
            self.interpreter = Interpreter(model_path=str(self.config.model_path))
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Validate input shape
            expected_shape = (self.config.audio.samples_per_window,)
            input_shape = self.input_details[0]['shape'][1:]
            if input_shape != expected_shape:
                logger.warning(
                    f"Model input shape {input_shape} != expected {expected_shape}"
                )
                
        except Exception as e:
            raise RuntimeError(f"Failed to load YAMNet model: {e}") from e
    
    def _load_labels(self) -> None:
        """Load YAMNet class labels from file.
        
        Raises
        ------
        FileNotFoundError
            If labels file is not found.
        IOError
            If labels file cannot be read.
        """
        try:
            with open(self.config.labels_path, "r") as f:
                self.labels = [line.strip() for line in f.readlines()]
            
            if len(self.labels) != 521:
                logger.warning(
                    f"Expected 521 YAMNet labels, got {len(self.labels)}"
                )
                
        except Exception as e:
            raise IOError(f"Failed to load labels file: {e}") from e
    
    def _find_siren_indices(self) -> None:
        """Find indices of siren-related classes in the label list.
        
        Uses keywords from config to identify relevant classes.
        """
        self.siren_indices = []
        for i, label in enumerate(self.labels):
            if any(keyword in label.lower() for keyword in self.config.siren_keywords):
                self.siren_indices.append(i)
        
        if not self.siren_indices:
            logger.warning("No siren-related classes found in YAMNet labels")
    
    def classify(self, audio_window: np.ndarray) -> Tuple[float, str, int]:
        """Classify audio window and return siren detection result.
        
        Parameters
        ----------
        audio_window : np.ndarray
            Audio samples as float32 array with shape (samples_per_window,).
            Values should be normalized to [-1, 1] range.
            
        Returns
        -------
        Tuple[float, str, int]
            Tuple of (confidence, detected_class, class_index).
            Confidence is the maximum score among siren classes.
            If no siren detected (confidence < threshold), returns (0.0, "", -1).
            
        Raises
        ------
        ValueError
            If audio_window has incorrect shape or dtype.
        RuntimeError
            If inference fails.
        """
        # Validate input
        expected_shape = (self.config.audio.samples_per_window,)
        if audio_window.shape != expected_shape:
            raise ValueError(
                f"Audio window shape {audio_window.shape} != expected {expected_shape}"
            )
        
        if audio_window.dtype != np.float32:
            audio_window = audio_window.astype(np.float32)
        
        try:
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], audio_window)
            self.interpreter.invoke()
            scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            # Find maximum confidence among siren classes
            if self.siren_indices:
                siren_scores = scores[self.siren_indices]
                max_idx = np.argmax(siren_scores)
                max_confidence = siren_scores[max_idx]
                class_idx = self.siren_indices[max_idx]
                
                if max_confidence >= self.config.confidence_threshold:
                    return max_confidence, self.labels[class_idx], class_idx
            
            return 0.0, "", -1
            
        except Exception as e:
            raise RuntimeError(f"YAMNet inference failed: {e}") from e
    
    def get_all_scores(self, audio_window: np.ndarray) -> np.ndarray:
        """Get full prediction scores for all 521 classes.
        
        Parameters
        ----------
        audio_window : np.ndarray
            Audio samples as float32 array.
            
        Returns
        -------
        np.ndarray
            Prediction scores for all classes, shape (521,).
            
        Raises
        ------
        ValueError
            If audio_window has incorrect shape.
        RuntimeError
            If inference fails.
        """
        expected_shape = (self.config.audio.samples_per_window,)
        if audio_window.shape != expected_shape:
            raise ValueError(
                f"Audio window shape {audio_window.shape} != expected {expected_shape}"
            )
        
        if audio_window.dtype != np.float32:
            audio_window = audio_window.astype(np.float32)
        
        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], audio_window)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
        except Exception as e:
            raise RuntimeError(f"YAMNet inference failed: {e}") from e