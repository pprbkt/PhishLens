"""
Unit tests for classifier inference, threshold handling, and model loading.
"""

import pytest
from app.config import settings
from app.ml.classifier import model_manager
from app.ml.inference import predict_single_text, predict_batch_texts
from app.ml.threshold import apply_threshold, categorize_risk_region


@pytest.fixture(autouse=True)
def ensure_model_loaded():
    if not model_manager.is_loaded:
        model_manager.load_artifacts(
            settings.model_path,
            settings.vectorizer_path,
            settings.metadata_path
        )
    assert model_manager.is_loaded, "Model artifacts must be loaded for tests"


def test_model_loading():
    assert model_manager.classifier is not None
    assert model_manager.vectorizer is not None
    assert 0.0 < model_manager.optimal_threshold < 1.0


def test_threshold_logic():
    assert apply_threshold(0.85, 0.50) == "SCAM"
    assert apply_threshold(0.40, 0.50) == "NOT_SCAM"
    assert apply_threshold(0.50, 0.50) == "SCAM"


def test_risk_region_categorization():
    assert categorize_risk_region(0.92) == "HIGH_RISK"
    assert categorize_risk_region(0.15) == "LOW_RISK"
    assert categorize_risk_region(0.55) == "UNCERTAIN"


def test_predict_scam_message():
    scam_text = "URGENT: Your bank account will be blocked today due to pending KYC. Click http://bank-kyc-verify.top immediately."
    res = predict_single_text(scam_text)
    assert res["prediction"] == "SCAM"
    assert res["scam_probability"] >= 0.50
    assert len(res["signals"]) > 0


def test_predict_benign_message():
    benign_text = "Hey, are we still meeting for lunch at 1 PM today?"
    res = predict_single_text(benign_text)
    assert res["prediction"] == "NOT_SCAM"
    assert res["scam_probability"] < 0.50


def test_batch_prediction():
    texts = [
        "URGENT: Your SBI account has been locked. Verify at http://sbi-fake.top",
        "Your flight confirmation is 6E-204 from Delhi to Mumbai.",
        "Congratulations! You won ₹1,00,000 cash prize. Call +919876543210 now."
    ]
    results = predict_batch_texts(texts)
    assert len(results) == 3
    assert results[0]["prediction"] == "SCAM"
    assert results[1]["prediction"] == "NOT_SCAM"
    assert results[2]["prediction"] == "SCAM"
