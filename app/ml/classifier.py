"""
Model loader and runtime manager for PhishLens.
"""

import json
import logging
import os
from typing import Optional, Dict, Any
import joblib

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loaded classifier, vectorizer, and metadata."""

    def __init__(self):
        self.classifier = None
        self.vectorizer = None
        self.metadata: Dict[str, Any] = {}
        self.optimal_threshold: float = 0.75
        self.is_loaded: bool = False

    def load_artifacts(
        self,
        classifier_path: str,
        vectorizer_path: str,
        metadata_path: str
    ) -> bool:
        """Load trained model artifacts into memory."""
        try:
            if not os.path.exists(classifier_path) or not os.path.exists(vectorizer_path):
                logger.warning("Model artifacts not found at %s or %s", classifier_path, vectorizer_path)
                return False

            self.classifier = joblib.load(classifier_path)
            self.vectorizer = joblib.load(vectorizer_path)

            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                self.optimal_threshold = self.metadata.get("optimal_threshold", 0.75)
            else:
                self.optimal_threshold = 0.75

            self.is_loaded = True
            logger.info("PhishLens ML artifacts loaded successfully (Threshold: %.4f)", self.optimal_threshold)
            return True
        except Exception as e:
            logger.error("Failed to load model artifacts: %s", e)
            self.is_loaded = False
            return False


# Global singleton instance
model_manager = ModelManager()
