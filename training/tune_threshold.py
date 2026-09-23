"""
Precision-oriented decision threshold optimization for PhishLens.
Finds the optimal decision threshold on validation data to achieve target scam precision.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def find_optimal_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    min_precision: float = 0.90,
    default_threshold: float = 0.70
) -> Tuple[float, pd.DataFrame]:
    """
    Search thresholds from 0.10 to 0.95 to find the optimal operating point.
    Prioritizes achieving >= min_precision for the SCAM class while maximizing recall.
    
    Args:
        y_true: Ground truth binary labels (1=scam, 0=not_scam)
        probabilities: Predicted scam probability (class 1)
        min_precision: Target scam precision constraint
        default_threshold: Fallback threshold if constraint cannot be strictly satisfied
        
    Returns:
        optimal_threshold (float), analysis_dataframe (pd.DataFrame)
    """
    thresholds = np.linspace(0.10, 0.95, 86)
    records: List[Dict[str, float]] = []

    best_threshold = default_threshold
    best_f1 = -1.0
    satisfied_best_recall = -1.0

    for th in thresholds:
        preds = (probabilities >= th).astype(int)
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, pos_label=1, zero_division=0)
        rec = recall_score(y_true, preds, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)

        records.append({
            "threshold": round(float(th), 4),
            "accuracy": round(float(acc), 4),
            "scam_precision": round(float(prec), 4),
            "scam_recall": round(float(rec), 4),
            "scam_f1": round(float(f1), 4)
        })

        # Selection strategy: Must satisfy min_precision; among those, pick highest recall / F1
        if prec >= min_precision and rec > 0:
            if rec > satisfied_best_recall or (rec == satisfied_best_recall and f1 > best_f1):
                satisfied_best_recall = rec
                best_f1 = f1
                best_threshold = th
        elif satisfied_best_recall == -1.0 and f1 > best_f1:
            # Fallback if no threshold meets precision constraint
            best_f1 = f1
            best_threshold = th

    df_analysis = pd.DataFrame(records)
    return round(float(best_threshold), 4), df_analysis
