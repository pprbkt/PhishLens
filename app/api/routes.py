"""
FastAPI route handlers for PhishLens scam detection services.
"""

import logging
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.api.schemas import (
    BatchTextPredictionRequest,
    BatchTextPredictionResponse,
    HealthResponse,
    ImagePredictionResponse,
    TextPredictionRequest,
    TextPredictionResponse,
)
from app.config import settings
from app.ml.classifier import model_manager
from app.ml.inference import predict_batch_texts, predict_single_text
from app.ocr.extractor import ocr_engine

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}


@router.get("/health", response_model=HealthResponse, summary="Service Health & Model Status")
async def health_check():
    """Check API service health and verify loaded model status."""
    return HealthResponse(
        status="ok",
        model_loaded=model_manager.is_loaded,
        model_version=model_manager.metadata.get("model_version", "phishlens-v1"),
        operating_threshold=model_manager.optimal_threshold
    )


@router.post("/predict/text", response_model=TextPredictionResponse, summary="Detect Scam from SMS / Text")
async def predict_text(payload: TextPredictionRequest):
    """
    Analyze SMS or raw text message to determine if it is SCAM or NOT_SCAM.
    Returns calibrated probability, operating threshold, risk category, and explainable risk signals.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable or training artifacts are missing."
        )

    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text payload cannot be empty or whitespace only."
        )

    if len(payload.text) > settings.max_text_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Text length ({len(payload.text)}) exceeds maximum allowed length of {settings.max_text_length} characters."
        )

    try:
        result = predict_single_text(payload.text, custom_threshold=payload.threshold)
        return TextPredictionResponse(**result)
    except Exception as e:
        logger.error("Error during text prediction: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction analysis."
        )


@router.post("/predict/image", response_model=ImagePredictionResponse, summary="Detect Scam from Screenshot (OCR)")
async def predict_image(
    file: UploadFile = File(..., description="Screenshot image file (PNG, JPG, WEBP)"),
    threshold: Optional[float] = Form(None, description="Optional custom decision threshold override")
):
    """
    Accept an uploaded screenshot image, extract text via RapidOCR, and run scam classification.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable or training artifacts are missing."
        )

    # 1. Validate MIME type
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_MIME_TYPES:
        filename_ext = file.filename.split(".")[-1].lower() if file.filename and "." in file.filename else ""
        if filename_ext not in {"png", "jpg", "jpeg", "webp", "bmp"}:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type '{content_type}'. Please upload PNG, JPG, or WEBP images."
            )

    # 2. Read and validate file size
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded image: {str(e)}"
        )

    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size exceeds maximum limit of {settings.max_image_size_mb} MB."
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded image file is empty."
        )

    # 3. Perform OCR Extraction
    ocr_result = ocr_engine.extract_text_from_bytes(image_bytes)
    if ocr_result.get("error") or not ocr_result.get("extracted_text"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ocr_result.get("error", "Unable to extract readable text from image.")
        )

    extracted_text = ocr_result["extracted_text"]
    ocr_confidence = ocr_result["ocr_confidence"]

    # 4. Predict Scam on Extracted Text
    try:
        prediction_res = predict_single_text(extracted_text, custom_threshold=threshold)
        return ImagePredictionResponse(
            prediction=prediction_res["prediction"],
            scam_probability=prediction_res["scam_probability"],
            threshold=prediction_res["threshold"],
            risk_level=prediction_res["risk_level"],
            extracted_text=extracted_text,
            ocr_confidence=ocr_confidence,
            signals=prediction_res["signals"],
            model_version=prediction_res["model_version"]
        )
    except Exception as e:
        logger.error("Error during image text prediction: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error evaluating extracted text from screenshot."
        )


@router.post("/predict/batch", response_model=BatchTextPredictionResponse, summary="Batch Scam Prediction")
async def predict_batch(payload: BatchTextPredictionRequest):
    """Classify multiple SMS messages in a single batch request."""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable or training artifacts are missing."
        )

    if not payload.texts or len(payload.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size must be between 1 and {settings.max_batch_size} messages."
        )

    try:
        results = predict_batch_texts(payload.texts, custom_threshold=payload.threshold)
        scam_count = sum(1 for r in results if r["prediction"] == "SCAM")
        return BatchTextPredictionResponse(
            total_processed=len(results),
            scam_count=scam_count,
            not_scam_count=len(results) - scam_count,
            results=results
        )
    except Exception as e:
        logger.error("Error during batch prediction: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing batch predictions."
        )


@router.get("/metrics", summary="Model Performance Metrics & Metadata")
async def get_metrics():
    """Retrieve current model metrics, threshold configuration, and dataset distribution."""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is unavailable."
        )
    return model_manager.metadata
