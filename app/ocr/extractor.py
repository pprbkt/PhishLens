"""
OCR Extraction module for PhishLens.
Wraps RapidOCR engine with OpenCV preprocessing to extract text and confidence scores.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

from app.ocr.preprocess_image import load_image_bytes, preprocess_for_ocr

logger = logging.getLogger(__name__)


class OCREngine:
    """Wrapper for RapidOCR engine with lazy initialization."""

    def __init__(self):
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._engine = RapidOCR()
                logger.info("RapidOCR initialized successfully.")
            except Exception as e:
                logger.error("Failed to initialize RapidOCR: %s", e)
                raise RuntimeError(f"OCR Engine initialization failure: {e}")
        return self._engine

    def extract_text_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract readable text and confidence score from raw image bytes.
        
        Returns:
            Dict containing 'extracted_text', 'ocr_confidence', 'raw_lines', and 'error' (if any)
        """
        if not image_bytes:
            return {"error": "Empty image payload provided", "extracted_text": "", "ocr_confidence": 0.0}

        img_bgr = load_image_bytes(image_bytes)
        if img_bgr is None:
            return {"error": "Invalid or unsupported image file format", "extracted_text": "", "ocr_confidence": 0.0}

        try:
            # 1. Preprocess image for OCR
            processed_img = preprocess_for_ocr(img_bgr)

            # 2. Run OCR
            result, _ = self.engine(processed_img)

            if not result:
                # Try raw image if preprocessed produced nothing
                result, _ = self.engine(img_bgr)

            if not result:
                return {
                    "extracted_text": "",
                    "ocr_confidence": 0.0,
                    "error": "Unable to extract readable text from image"
                }

            # 3. Parse lines and confidence
            lines = []
            confidences = []
            for item in result:
                # Format: [box, text, score]
                if len(item) >= 3:
                    text_segment = str(item[1]).strip()
                    conf = float(item[2])
                    if text_segment:
                        lines.append(text_segment)
                        confidences.append(conf)

            full_text = " ".join(lines).strip()
            avg_confidence = round(float(np.mean(confidences)), 4) if confidences else 0.0

            if not full_text:
                return {
                    "extracted_text": "",
                    "ocr_confidence": 0.0,
                    "error": "Unable to extract readable text from image"
                }

            return {
                "extracted_text": full_text,
                "ocr_confidence": avg_confidence,
                "line_count": len(lines),
                "error": None
            }

        except Exception as e:
            logger.error("Error during OCR execution: %s", e)
            return {
                "extracted_text": "",
                "ocr_confidence": 0.0,
                "error": f"OCR processing failed: {str(e)}"
            }


ocr_engine = OCREngine()
