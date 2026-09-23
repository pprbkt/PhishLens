"""
End-to-end training pipeline for PhishLens.
Executes stratified data splitting, leakage-free vectorization, model benchmarking,
precision-targeted threshold optimization, test evaluation, and model artifact persistence.
"""

import datetime
import json
import logging
import os
import sys
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure root workspace directory is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.ml.preprocess import clean_text
from training.compare_models import build_vectorizer, compare_models
from training.tune_threshold import find_optimal_threshold
from training.evaluate import evaluate_and_generate_reports

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "sms_dataset.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def run_training_pipeline(
    raw_data_path: str = RAW_DATA_PATH,
    precision_target: float = 0.90,
    random_state: int = 42
) -> Dict[str, Any]:
    """Execute complete reproducible training pipeline."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load and Validate Dataset
    if not os.path.exists(raw_data_path):
        logger.error("Raw dataset not found at %s. Please run scripts/download_data.py first.", raw_data_path)
        raise FileNotFoundError(f"Missing dataset: {raw_data_path}")

    logger.info("Loading dataset from %s...", raw_data_path)
    df = pd.read_csv(raw_data_path)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["scam", "not_scam"])]
    df["binary_label"] = (df["label"] == "scam").astype(int)

    # Clean text
    logger.info("Cleaning and normalizing %d messages...", len(df))
    df["cleaned_text"] = df["text"].apply(clean_text)
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)

    # 2. Stratified Train / Val / Test Split (70% Train, 15% Val, 15% Test)
    logger.info("Splitting dataset into Train (70%%), Validation (15%%), Test (15%%)...")
    X = df["cleaned_text"].tolist()
    y = df["binary_label"].to_numpy()

    # First split: 70% Train, 30% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    # Second split: Split 30% temp equally into Val (15%) and Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )

    # Save processed splits
    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"), index=False)
    pd.DataFrame({"text": X_val, "label": y_val}).to_csv(os.path.join(PROCESSED_DATA_DIR, "val.csv"), index=False)
    pd.DataFrame({"text": X_test, "label": y_test}).to_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"), index=False)
    logger.info("Dataset splits saved to %s", PROCESSED_DATA_DIR)

    # 3. Model Comparison on Validation Data
    logger.info("Benchmarking candidate models on validation split...")
    comparison_results, best_model_name, _ = compare_models(X_train, y_train, X_val, y_val)
    logger.info("Validation Comparison Results:\n%s", json.dumps(comparison_results, indent=2))
    logger.info("Selected Primary Model: %s", best_model_name)

    # 4. Fit Final Vectorizer & Model on Training Split Only
    logger.info("Fitting TF-IDF Vectorizer and %s on training data...", best_model_name)
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    # Fit final Logistic Regression classifier for calibrated probabilities
    from sklearn.linear_model import LogisticRegression
    final_model = LogisticRegression(C=3.0, max_iter=1000, class_weight="balanced", random_state=random_state)
    final_model.fit(X_train_vec, y_train)

    # 5. Tune Threshold on Validation Split ONLY
    logger.info("Optimizing decision threshold on validation set (Target SCAM Precision: %.2f)...", precision_target)
    X_val_vec = vectorizer.transform(X_val)
    val_probabilities = final_model.predict_proba(X_val_vec)[:, 1]
    optimal_threshold, threshold_df = find_optimal_threshold(y_val, val_probabilities, min_precision=precision_target)
    logger.info("Optimized Decision Threshold: %.4f", optimal_threshold)

    # 6. Final Evaluation on Held-out Test Set (Evaluated ONCE)
    logger.info("Running final evaluation on held-out test split...")
    X_test_vec = vectorizer.transform(X_test)
    test_probabilities = final_model.predict_proba(X_test_vec)[:, 1]

    metrics = evaluate_and_generate_reports(
        y_true=y_test,
        probabilities=test_probabilities,
        threshold=optimal_threshold,
        threshold_df=threshold_df,
        output_dir=REPORTS_DIR
    )
    logger.info("Final Test Metrics:\n%s", json.dumps(metrics, indent=2))

    # 7. Save Model Artifacts & Metadata
    classifier_path = os.path.join(MODELS_DIR, "classifier.joblib")
    vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.joblib")
    metadata_path = os.path.join(MODELS_DIR, "metadata.json")

    joblib.dump(final_model, classifier_path)
    joblib.dump(vectorizer, vectorizer_path)

    metadata = {
        "model_version": "phishlens-v1",
        "model_type": best_model_name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "optimal_threshold": optimal_threshold,
        "precision_target": precision_target,
        "metrics": metrics,
        "validation_comparison": comparison_results,
        "dataset_info": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "scam_ratio": float(round(np.mean(y), 4))
        }
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model artifacts and metadata saved successfully to %s", MODELS_DIR)
    return metadata


if __name__ == "__main__":
    run_training_pipeline()
