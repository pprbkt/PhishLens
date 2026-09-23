"""
Unit tests for image preprocessing and OCR extraction.
"""

import io
import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.ocr.extractor import ocr_engine
from app.ocr.preprocess_image import load_image_bytes, preprocess_for_ocr


def create_test_text_image(text: str = "URGENT: Click here to verify your account") -> bytes:
    """Create a synthetic image containing clean text for OCR testing."""
    img = Image.new("RGB", (450, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 35), text, fill=(0, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_valid_image():
    img_bytes = create_test_text_image("Test")
    arr = load_image_bytes(img_bytes)
    assert arr is not None
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 100
    assert arr.shape[1] == 450


def test_load_invalid_image():
    corrupt_bytes = b"not_an_image_file_content"
    arr = load_image_bytes(corrupt_bytes)
    assert arr is None


def test_preprocess_for_ocr():
    img_bytes = create_test_text_image("Test Preprocessing")
    arr = load_image_bytes(img_bytes)
    processed = preprocess_for_ocr(arr)
    assert processed is not None
    assert processed.shape == arr.shape


def test_ocr_extraction_valid_image():
    img_bytes = create_test_text_image("URGENT: Click here")
    result = ocr_engine.extract_text_from_bytes(img_bytes)
    assert result["error"] is None
    assert "urgent" in result["extracted_text"].lower()
    assert result["ocr_confidence"] > 0.50


def test_ocr_extraction_empty_or_corrupt_payload():
    empty_result = ocr_engine.extract_text_from_bytes(b"")
    assert empty_result["error"] is not None

    corrupt_result = ocr_engine.extract_text_from_bytes(b"bad_bytes")
    assert corrupt_result["error"] is not None
