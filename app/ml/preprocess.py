"""
Text preprocessing and entity normalization module for PhishLens.
Normalizes text while preserving and tagging critical scam/phishing indicators.
"""

import re
import unicodedata
from typing import Dict, List, Tuple


# Regex patterns for entity normalization
URL_REGEX = re.compile(
    r"(?:https?:\/\/|www\.)[^\s/$.?#].[^\s]*|bit\.ly\/[^\s]+|tinyurl\.com\/[^\s]+|[a-zA-Z0-9-]+\.(?:top|xyz|site|cc|pw|biz|co|org|net|info|online|live|shop|club|vip)\b[^\s]*",
    re.IGNORECASE
)
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b(?:\+91|0)?[6-9]\d{9}\b|\b\d{5}[-\s]?\d{5}\b")
AMOUNT_REGEX = re.compile(r"(?:[\$€£₹]|(?:USD|INR|EUR|GBP|RS\.?|INR))\s*[\d,]+(?:\.\d{1,2})?|\b[\d,]+(?:\.\d{1,2})?\s*(?:dollars|rupees|pounds|euros|inr|usd)\b", re.IGNORECASE)
REPEATED_CHAR_REGEX = re.compile(r"(.)\1{2,}")

# Domain-specific high-risk scam triggers for explainability & feature preservation
RISK_KEYWORDS = [
    "urgent", "immediately", "action required", "account blocked", "account suspended",
    "verify", "verification", "kyc", "pan card", "aadhaar", "otp", "one-time password",
    "claim prize", "lottery", "congratulations", "winner", "reward", "cashback", "refund",
    "unauthorized", "suspicious activity", "dispute", "redelivery fee", "parcel", "package",
    "electricity disconnected", "sim deactivated", "pre-approved", "instant loan",
    "crypto", "giveaway", "free bitcoin", "restricted", "unlock account", "click here"
]


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters using NFKC format."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


def collapse_repeated_chars(text: str) -> str:
    """Collapse 3 or more repeated characters down to 2 (e.g., 'hurrrrry' -> 'hurry')."""
    return REPEATED_CHAR_REGEX.sub(r"\1\1", text)


def normalize_whitespace(text: str) -> str:
    """Trim and replace multi-whitespaces with a single space."""
    return re.sub(r"\s+", " ", text).strip()


def extract_raw_entities(text: str) -> Dict[str, List[str]]:
    """Extract raw detected entities before normalization (useful for explainability)."""
    return {
        "urls": URL_REGEX.findall(text),
        "emails": EMAIL_REGEX.findall(text),
        "phones": PHONE_REGEX.findall(text),
        "amounts": AMOUNT_REGEX.findall(text),
    }


def clean_text(text: str, replace_entities: bool = True) -> str:
    """
    Complete text cleaning pipeline.
    
    Args:
        text: Raw input string
        replace_entities: If True, replaces URLs, phones, emails, and amounts with tokens.
        
    Returns:
        Cleaned, normalized string ready for vectorization.
    """
    if not text or not isinstance(text, str):
        return ""

    cleaned = normalize_unicode(text)
    cleaned = collapse_repeated_chars(cleaned)

    if replace_entities:
        cleaned = URL_REGEX.sub(" <URL> ", cleaned)
        cleaned = EMAIL_REGEX.sub(" <EMAIL> ", cleaned)
        cleaned = PHONE_REGEX.sub(" <PHONE> ", cleaned)
        cleaned = AMOUNT_REGEX.sub(" <AMOUNT> ", cleaned)

    cleaned = normalize_whitespace(cleaned)
    return cleaned.lower()


def find_risk_signals_in_text(raw_text: str) -> List[str]:
    """Identify explainable risk indicators and suspicious tokens from raw text."""
    signals: List[str] = []
    lowered = raw_text.lower()

    # Check for entity presence
    entities = extract_raw_entities(raw_text)
    if entities["urls"]:
        signals.append(f"Suspicious link/URL: {entities['urls'][0]}")
    if entities["amounts"]:
        signals.append(f"Financial amount requested/mentioned: {entities['amounts'][0]}")
    if entities["phones"]:
        signals.append(f"Phone contact prompt: {entities['phones'][0]}")

    # Check keyword triggers
    for kw in RISK_KEYWORDS:
        if kw in lowered:
            signals.append(f"High-risk indicator: '{kw}'")

    return list(dict.fromkeys(signals))[:6]  # Deduplicate and return top 6
