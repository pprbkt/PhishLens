"""
Configuration and settings management for PhishLens.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PhishLens"
    app_version: str = "1.0.0"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # Model & Artifact Paths
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path: str = os.path.join(base_dir, "models", "classifier.joblib")
    vectorizer_path: str = os.path.join(base_dir, "models", "vectorizer.joblib")
    metadata_path: str = os.path.join(base_dir, "models", "metadata.json")

    # Threshold and Constraints
    scam_threshold: float = 0.75
    scam_precision_target: float = 0.90
    high_risk_threshold: float = 0.85
    low_risk_threshold: float = 0.35

    # API and Upload Limits
    max_text_length: int = 10000
    max_image_size_mb: int = 10
    max_batch_size: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
