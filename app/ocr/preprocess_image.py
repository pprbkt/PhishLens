"""
Image preprocessing module for PhishLens OCR pipeline.
Uses OpenCV to optimize screenshot contrast, denoise, and handle light/dark mode UI screenshots.
"""

import io
import logging
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Safely decode image bytes into a numpy BGR/RGB array."""
    try:
        # Try PIL first for broadest format support (PNG, JPG, WEBP, BMP)
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = pil_img.convert("RGB")
        img_np = np.array(pil_img)
        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning("Failed to decode image using PIL: %s. Trying cv2.imdecode...", e)
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e2:
            logger.error("All image decode methods failed: %s", e2)
            return None


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Enhance screenshot image for optical character recognition.
    - Converts to grayscale
    - Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    - Applies light Gaussian blur to reduce compression artifacts
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Invalid or empty image array.")

    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect if image is mostly dark (Dark Mode screenshot)
    mean_val = np.mean(gray)
    if mean_val < 90:
        # Invert dark mode to black text on white background for higher OCR contrast
        gray = cv2.bitwise_not(gray)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Convert back to 3-channel for OCR engines expecting RGB/BGR
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
