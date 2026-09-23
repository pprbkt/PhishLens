"""
High-level inference engine for PhishLens text inputs.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from app.config import settings
from app.ml.classifier import model_manager
from app.ml.explain import get_risk_signals
from app.ml.preprocess import clean_text
from app.ml.threshold import apply_threshold, categorize_risk_region


def predict_single_text(text: str, custom_threshold: Optional[float] = None) -> Dict[str, Any]:
    """
    Run scam detection pipeline on a single text string.
    
    Args:
        text: Raw SMS / extracted message text
        custom_threshold: Optional override for operating threshold
        
    Returns:
        Structured prediction dictionary
    """
    if not model_manager.is_loaded:
        raise RuntimeError("ML model artifacts are not loaded.")

    cleaned = clean_text(text)
    if not cleaned:
        # Fallback for empty/unintelligible text
        return {
            "prediction": "NOT_SCAM",
            "scam_probability": 0.0,
            "risk_level": "LOW_RISK",
            "threshold": custom_threshold or model_manager.optimal_threshold,
            "cleaned_text": "",
            "signals": [],
            "model_version": model_manager.metadata.get("model_version", "phishlens-v1")
        }

    # Vectorize and predict probability
    X_vec = model_manager.vectorizer.transform([cleaned])
    proba = float(model_manager.classifier.predict_proba(X_vec)[0, 1])

    threshold = custom_threshold if custom_threshold is not None else model_manager.optimal_threshold
    prediction = apply_threshold(proba, threshold)
    risk_level = categorize_risk_region(proba, settings.high_risk_threshold, settings.low_risk_threshold)
    signals = get_risk_signals(text, cleaned, model_manager.vectorizer, model_manager.classifier)

    return {
        "prediction": prediction,
        "scam_probability": round(proba, 4),
        "risk_level": risk_level,
        "threshold": round(threshold, 4),
        "cleaned_text": cleaned,
        "signals": signals,
        "model_version": model_manager.metadata.get("model_version", "phishlens-v1")
    }


def predict_batch_texts(texts: List[str], custom_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """Run scam detection pipeline across a list of text inputs."""
    if not model_manager.is_loaded:
        raise RuntimeError("ML model artifacts are not loaded.")

    threshold = custom_threshold if custom_threshold is not None else model_manager.optimal_threshold
    results: List[Dict[str, Any]] = []

    cleaned_texts = [clean_text(t) for t in texts]
    non_empty_indices = [i for i, c in enumerate(cleaned_texts) if c]

    # Pre-populate defaults
    for i, t in enumerate(texts):
        results.append({
            "text": t,
            "prediction": "NOT_SCAM",
            "scam_probability": 0.0,
            "risk_level": "LOW_RISK",
            "threshold": round(threshold, 4),
            "signals": [],
            "model_version": model_manager.metadata.get("model_version", "phishlens-v1")
        })

    if non_empty_indices:
        valid_cleaned = [cleaned_texts[i] for i in non_empty_indices]
        X_vec = model_manager.vectorizer.transform(valid_cleaned)
        probas = model_manager.classifier.predict_proba(X_vec)[:, 1]

        for idx_in_valid, orig_idx in enumerate(non_empty_indices):
            p = float(probas[idx_in_valid])
            raw = texts[orig_idx]
            cleaned = cleaned_texts[orig_idx]

            results[orig_idx]["scam_probability"] = round(p, 4)
            results[orig_idx]["prediction"] = apply_threshold(p, threshold)
            results[orig_idx]["risk_level"] = categorize_risk_region(p, settings.high_risk_threshold, settings.low_risk_threshold)
            results[orig_idx]["signals"] = get_risk_signals(raw, cleaned, model_manager.vectorizer, model_manager.classifier)

    return results
