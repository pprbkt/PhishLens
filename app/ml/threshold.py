"""
Decision threshold and risk band classification module for PhishLens.
"""

from typing import Literal

RiskLevel = Literal["HIGH_RISK", "UNCERTAIN", "LOW_RISK"]
PredictionLabel = Literal["SCAM", "NOT_SCAM"]


def apply_threshold(probability: float, threshold: float) -> PredictionLabel:
    """Classify as SCAM if probability meets or exceeds threshold."""
    return "SCAM" if probability >= threshold else "NOT_SCAM"


def categorize_risk_region(
    probability: float,
    high_risk_cutoff: float = 0.80,
    low_risk_cutoff: float = 0.35
) -> RiskLevel:
    """
    Categorize model probability into interpretative risk bands:
    - HIGH_RISK: Substantial likelihood of phishing/fraud.
    - UNCERTAIN: Borderline features or mixed signals; warrants careful inspection.
    - LOW_RISK: Standard legitimate or routine communication.
    """
    if probability >= high_risk_cutoff:
        return "HIGH_RISK"
    elif probability <= low_risk_cutoff:
        return "LOW_RISK"
    else:
        return "UNCERTAIN"
