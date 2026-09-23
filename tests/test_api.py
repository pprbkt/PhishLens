"""
Integration tests for FastAPI endpoints.
"""

import io
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest

from app.main import app
from app.config import settings
from app.ml.classifier import model_manager

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    model_manager.load_artifacts(
        settings.model_path,
        settings.vectorizer_path,
        settings.metadata_path
    )


def create_sample_image(text: str = "URGENT: Click http://bank.pw to verify") -> io.BytesIO:
    img = Image.new("RGB", (400, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((15, 25), text, fill=(0, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "model_version" in data


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "optimal_threshold" in data


def test_predict_text_scam():
    payload = {
        "text": "URGENT: Your bank account will be blocked due to pending KYC. Click http://sbi-kyc.top to verify."
    }
    response = client.post("/predict/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "SCAM"
    assert data["scam_probability"] > 0.40
    assert isinstance(data["signals"], list)
    assert len(data["signals"]) > 0


def test_predict_text_benign():
    payload = {
        "text": "Your one-time password for HDFC Netbanking is 582910. Do not share it with anyone."
    }
    response = client.post("/predict/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "scam_probability" in data


def test_predict_text_empty_error():
    payload = {"text": "   "}
    response = client.post("/predict/text", json=payload)
    assert response.status_code in [400, 422]


def test_predict_image_endpoint():
    img_io = create_sample_image("URGENT: Your account is suspended. Verify at http://fake.com")
    response = client.post(
        "/predict/image",
        files={"file": ("screenshot.png", img_io, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "scam_probability" in data
    assert "extracted_text" in data
    assert "ocr_confidence" in data
    assert len(data["extracted_text"]) > 0


def test_predict_image_unsupported_type():
    fake_txt = io.BytesIO(b"not an image")
    response = client.post(
        "/predict/image",
        files={"file": ("test.txt", fake_txt, "text/plain")}
    )
    assert response.status_code == 415


def test_predict_batch_endpoint():
    payload = {
        "texts": [
            "URGENT: Your account is suspended. Click http://bank-kyc.top",
            "Hey, let me know when you are free for the call."
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2
