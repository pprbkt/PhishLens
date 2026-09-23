"""
Pydantic schemas for PhishLens API endpoints.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ok"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    model_version: Optional[str] = Field("phishlens-v1", json_schema_extra={"example": "phishlens-v1"})
    operating_threshold: Optional[float] = Field(0.75, json_schema_extra={"example": 0.42})


class TextPredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="SMS or message content to inspect", json_schema_extra={"example": "URGENT: Your account has been suspended. Click http://scam-link.top to verify."})
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional custom decision threshold override", json_schema_extra={"example": 0.75})


class TextPredictionResponse(BaseModel):
    prediction: Literal["SCAM", "NOT_SCAM"] = Field(..., json_schema_extra={"example": "SCAM"})
    scam_probability: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.975})
    threshold: float = Field(..., json_schema_extra={"example": 0.42})
    risk_level: Literal["HIGH_RISK", "UNCERTAIN", "LOW_RISK"] = Field(..., json_schema_extra={"example": "HIGH_RISK"})
    cleaned_text: str = Field(..., json_schema_extra={"example": "urgent : your account has been suspended . click <url> to verify ."})
    signals: List[str] = Field(default_factory=list, description="Identified risk signals & contributing terms", json_schema_extra={"example": ["Suspicious link/URL: http://scam-link.top", "High-risk indicator: 'urgent'"]})
    model_version: str = Field("phishlens-v1", json_schema_extra={"example": "phishlens-v1"})


class ImagePredictionResponse(BaseModel):
    prediction: Literal["SCAM", "NOT_SCAM"] = Field(..., json_schema_extra={"example": "SCAM"})
    scam_probability: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.962})
    threshold: float = Field(..., json_schema_extra={"example": 0.42})
    risk_level: Literal["HIGH_RISK", "UNCERTAIN", "LOW_RISK"] = Field(..., json_schema_extra={"example": "HIGH_RISK"})
    extracted_text: str = Field(..., json_schema_extra={"example": "Your package is delayed. Pay $1.99 redelivery fee at http://fedx-pkg.top"})
    ocr_confidence: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.93})
    signals: List[str] = Field(default_factory=list, json_schema_extra={"example": ["Suspicious link/URL: http://fedx-pkg.top", "High-risk indicator: 'redelivery fee'"]})
    model_version: str = Field("phishlens-v1", json_schema_extra={"example": "phishlens-v1"})


class BatchTextPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of messages to classify (up to 100)")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional custom decision threshold override")


class BatchTextPredictionResponse(BaseModel):
    total_processed: int = Field(..., json_schema_extra={"example": 2})
    scam_count: int = Field(..., json_schema_extra={"example": 1})
    not_scam_count: int = Field(..., json_schema_extra={"example": 1})
    results: List[Dict[str, Any]]
