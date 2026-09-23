"""
Explainability and risk signal extraction module for PhishLens.
Identifies salient TF-IDF features and semantic risk triggers contributing to scam probability.
"""

from typing import List, Tuple, Any
import numpy as np
from app.ml.preprocess import find_risk_signals_in_text


def extract_salient_terms(
    cleaned_text: str,
    vectorizer: Any,
    classifier: Any,
    top_k: int = 5
) -> List[str]:
    """
    Extract the top TF-IDF features with positive weights that are active in the input text.
    """
    try:
        if not hasattr(classifier, "coef_"):
            return []

        # Vectorize single input
        vec = vectorizer.transform([cleaned_text])
        if vec.nnz == 0:
            return []

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        coefs = classifier.coef_[0]

        # Find non-zero indices in the vector
        _, col_indices = vec.nonzero()
        contributions: List[Tuple[str, float]] = []

        for idx in col_indices:
            weight = coefs[idx] * vec[0, idx]
            if weight > 0:  # Positively contributing to SCAM class
                raw_feature = feature_names[idx]
                # Clean up feature union prefix (e.g. 'word_tfidf__claim' -> 'claim')
                clean_feat = raw_feature.split("__")[-1]
                contributions.append((clean_feat, float(weight)))

        # Sort descending by positive contribution
        contributions.sort(key=lambda x: x[1], reverse=True)
        return [term for term, _ in contributions[:top_k]]
    except Exception:
        return []


def get_risk_signals(
    raw_text: str,
    cleaned_text: str,
    vectorizer: Any = None,
    classifier: Any = None
) -> List[str]:
    """
    Combine domain pattern matches and top model feature weights to produce explainable risk signals.
    """
    signals = find_risk_signals_in_text(raw_text)

    if vectorizer is not None and classifier is not None:
        salient_terms = extract_salient_terms(cleaned_text, vectorizer, classifier, top_k=4)
        for term in salient_terms:
            if not term.startswith("<") and not any(term in s.lower() for s in signals):
                signals.append(f"Model term weight: '{term}'")

    return list(dict.fromkeys(signals))[:6]
