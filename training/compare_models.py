"""
Model comparison script for PhishLens.
Evaluates TF-IDF + Logistic Regression, Linear SVM, and Multinomial Naive Bayes on validation data.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def build_vectorizer() -> FeatureUnion:
    """
    Build combined word and character n-gram TF-IDF vectorizer.
    Captures vocabulary words and obfuscated characters, URLs, and symbols.
    """
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_df=0.98,
            sublinear_tf=True
        ))
    ])


def get_candidate_models() -> Dict[str, Any]:
    """Define candidate classical ML models for comparison."""
    return {
        "Logistic_Regression": LogisticRegression(
            C=3.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Linear_SVM": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=42),
            method="sigmoid",
            cv=3
        ),
        "Naive_Bayes": MultinomialNB(alpha=0.1)
    }


def compare_models(
    X_train: list,
    y_train: np.ndarray,
    X_val: list,
    y_val: np.ndarray
) -> Tuple[Dict[str, Dict[str, float]], str, Pipeline]:
    """
    Train and evaluate candidate models on validation data.
    
    Returns:
        results_dict, best_model_name, best_pipeline
    """
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    results: Dict[str, Dict[str, float]] = {}
    best_f1 = -1.0
    best_name = "Logistic_Regression"
    best_model = None

    candidates = get_candidate_models()
    for name, model in candidates.items():
        model.fit(X_train_vec, y_train)
        y_val_pred = model.predict(X_val_vec)

        acc = accuracy_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_val, y_val_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_val, y_val_pred, pos_label=1, zero_division=0)

        results[name] = {
            "accuracy": round(float(acc), 4),
            "scam_precision": round(float(prec), 4),
            "scam_recall": round(float(rec), 4),
            "scam_f1": round(float(f1), 4)
        }

        # Select model balancing precision with high F1
        score = prec * 0.6 + f1 * 0.4
        if score > best_f1:
            best_f1 = score
            best_name = name
            best_model = model

    # If Logistic Regression is close, prefer it for interpretability and direct calibrated probabilities
    if "Logistic_Regression" in candidates and results["Logistic_Regression"]["scam_f1"] >= 0.90:
        best_name = "Logistic_Regression"
        best_model = candidates["Logistic_Regression"]

    return results, best_name, best_model
