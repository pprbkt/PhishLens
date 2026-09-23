"""
Comprehensive model evaluation and report generation module for PhishLens.
Computes test metrics, classification reports, confusion matrices, and PR curves.
"""

import json
import os
from typing import Dict, Any
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc
)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")


def evaluate_and_generate_reports(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    threshold_df: pd.DataFrame = None,
    output_dir: str = REPORTS_DIR
) -> Dict[str, Any]:
    """
    Perform final evaluation on held-out test set using tuned threshold.
    Saves visual and tabular reports to reports/ directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    y_pred = (probabilities >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_true, probabilities)

    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, probabilities)
    pr_auc = auc(recalls, precisions)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": round(float(acc), 4),
        "scam_precision": round(float(prec), 4),
        "scam_recall": round(float(rec), 4),
        "scam_f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "operating_threshold": round(float(threshold), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "test_samples_total": len(y_true),
        "test_scam_samples": int(np.sum(y_true == 1)),
        "test_not_scam_samples": int(np.sum(y_true == 0))
    }

    # 1. Save metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 2. Save classification_report.txt
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=["NOT_SCAM", "SCAM"],
        digits=4
    )
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== PhishLens Test Evaluation Report ===\n\n")
        f.write(f"Operating Threshold: {threshold:.4f}\n\n")
        f.write(report_text)
        f.write(f"\nConfusion Matrix:\nTN: {tn} | FP: {fp}\nFN: {fn} | TP: {tp}\n")

    # 3. Save threshold_analysis.csv
    if threshold_df is not None:
        csv_path = os.path.join(output_dir, "threshold_analysis.csv")
        threshold_df.to_csv(csv_path, index=False)

    # 4. Generate Confusion Matrix Plot (Neo-Brutalist inspired aesthetic: clean crisp contrast)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    plt.colorbar(cax, fraction=0.046, pad=0.04)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, f"{val}\n({val/len(y_true)*100:.1f}%)",
                    ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white",
                    fontsize=12, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["NOT_SCAM", "SCAM"], fontweight="bold", fontsize=10)
    ax.set_yticklabels(["NOT_SCAM", "SCAM"], fontweight="bold", fontsize=10)
    ax.set_xlabel("Predicted Label", fontweight="bold", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label", fontweight="bold", fontsize=11, labelpad=10)
    ax.set_title(f"Confusion Matrix (Threshold = {threshold:.2f})", fontweight="bold", fontsize=12, pad=15)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # 5. Generate Precision-Recall Curve Plot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(recalls, precisions, color="#FF5C00", lw=3, label=f"PR Curve (AUC = {pr_auc:.3f})")
    ax.scatter([rec], [prec], color="#000000", s=80, zorder=5, label=f"Operating Point (Th={threshold:.2f})")
    ax.axhline(y=prec, color="gray", linestyle="--", alpha=0.6)
    ax.axvline(x=rec, color="gray", linestyle="--", alpha=0.6)

    ax.set_xlabel("SCAM Recall", fontweight="bold", fontsize=11)
    ax.set_ylabel("SCAM Precision", fontweight="bold", fontsize=11)
    ax.set_title("Precision-Recall Curve (SCAM Detection)", fontweight="bold", fontsize=12)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    pr_path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(pr_path)
    plt.close()

    return metrics
