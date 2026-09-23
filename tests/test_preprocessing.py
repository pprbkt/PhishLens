"""
Unit tests for text preprocessing, entity normalization, and keyword extraction.
"""

import pytest
from app.ml.preprocess import (
    clean_text,
    collapse_repeated_chars,
    extract_raw_entities,
    find_risk_signals_in_text,
    normalize_unicode,
    normalize_whitespace,
)


def test_unicode_and_whitespace_normalization():
    raw = "  URGENT:   Account suspended! \n\n  "
    cleaned = clean_text(raw, replace_entities=False)
    assert "urgent: account suspended!" == cleaned


def test_repeated_characters():
    text = "Huuuuurry upppp nooow"
    result = collapse_repeated_chars(text)
    assert result == "Huurry upp noow"


def test_url_detection_and_replacement():
    text = "Visit http://phish-target.com/login or www.bank-secure.top now"
    entities = extract_raw_entities(text)
    assert len(entities["urls"]) >= 1

    cleaned = clean_text(text, replace_entities=True)
    assert "<url>" in cleaned


def test_phone_and_amount_replacement():
    text = "Call +919876543210 and send $500 or ₹50,000"
    entities = extract_raw_entities(text)
    assert len(entities["phones"]) >= 1
    assert len(entities["amounts"]) >= 1

    cleaned = clean_text(text, replace_entities=True)
    assert "<phone>" in cleaned
    assert "<amount>" in cleaned


def test_empty_and_none_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   ") == ""


def test_find_risk_signals():
    text = "URGENT! Your account is blocked due to KYC. Verify at http://bank.pw"
    signals = find_risk_signals_in_text(text)
    assert any("urgent" in s.lower() for s in signals)
    assert any("kyc" in s.lower() for s in signals)
    assert any("http://" in s.lower() for s in signals)
