"""Crash-proofing audit: every public function should degrade to a clear
message on bad input or a corrupted/missing artifact — never raise an
unhandled exception or hang.
"""

from pathlib import Path

from src.pipeline.engine import NXTSightEngine, Task
from src.pipeline.ocr import extract_text_from_image
from src.scam_detector import classifier as scam_classifier
from src.scam_detector.classifier import classify_scam
from src.spend_categorizer import categorizer as spend_categorizer
from src.spend_categorizer.categorizer import categorize_transactions


# --- ocr.py -----------------------------------------------------------


def test_ocr_rejects_none_path_without_crashing():
    result = extract_text_from_image(None)
    assert result.startswith("Error")


def test_ocr_rejects_non_path_types_without_crashing():
    for bad_path in [123, 3.14, ["not", "a", "path"], {"path": "x"}, object()]:
        result = extract_text_from_image(bad_path)
        assert result.startswith("Error")


def test_ocr_rejects_empty_string_path_without_crashing():
    result = extract_text_from_image("")
    assert result.startswith("Error")


# --- engine.py: a task whose artifacts are missing/corrupted ----------


def test_engine_reports_missing_artifacts_instead_of_crashing(tmp_path):
    broken_engine = NXTSightEngine()
    broken_engine.register_task(Task(name="broken", artifacts_dir=tmp_path))

    result = broken_engine.analyze("broken", "this is a perfectly normal English sentence")

    assert isinstance(result, str)
    assert "unavailable" in result.lower()


def test_engine_reports_corrupted_onnx_instead_of_crashing(tmp_path):
    from joblib import dump
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    vectorizer.fit(["hello world"])
    dump(vectorizer, tmp_path / "vectorizer.joblib")
    (tmp_path / "classifier.onnx").write_bytes(b"not a real onnx file")

    broken_engine = NXTSightEngine()
    broken_engine.register_task(Task(name="broken", artifacts_dir=tmp_path))

    result = broken_engine.analyze("broken", "this is a perfectly normal English sentence")

    assert isinstance(result, str)
    assert "unavailable" in result.lower()


def test_engine_vocabulary_terms_returns_empty_set_not_crash(tmp_path):
    broken_engine = NXTSightEngine()
    broken_engine.register_task(Task(name="broken", artifacts_dir=tmp_path))

    assert broken_engine.vocabulary_terms("broken", "some text") == set()


# --- classifier.py: degraded top_terms.json ----------------------------


def test_classify_scam_degrades_gracefully_without_top_terms(monkeypatch, tmp_path):
    monkeypatch.setattr(scam_classifier, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(scam_classifier, "_top_terms", None)

    result = scam_classifier._load_top_terms()

    assert result == {"scam": [], "legit": []}


def test_classify_scam_still_works_end_to_end():
    # Sanity check the real (non-broken) artifacts still work after the
    # hardening changes above.
    result = classify_scam("Click here now to verify your account or it will be suspended.")
    assert isinstance(result["is_scam"], bool)
    assert isinstance(result["reason"], str) and result["reason"]


# --- categorizer.py: degraded categories.json ---------------------------


def test_categorize_transactions_degrades_gracefully_without_categories(monkeypatch, tmp_path):
    monkeypatch.setattr(spend_categorizer, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(spend_categorizer, "_categories", None)

    result = spend_categorizer._load_categories()

    assert result is None


def test_categorize_transactions_still_works_end_to_end():
    result = categorize_transactions(["Rs 450.00 debited from A/c XX1234 at SWIGGY BANGALORE on 12-Sep-25."])
    assert result["categorized"][0]["category"] == "Food & Dining"


# --- categorize_transactions: malformed / hostile batch input ----------


def test_categorize_transactions_handles_wildly_mixed_batch_without_crashing():
    batch = [
        "Rs 450.00 debited from A/c XX1234 at SWIGGY BANGALORE on 12-Sep-25.",
        None,
        "",
        12345,
        3.14,
        ["nested", "list"],
        {"not": "a string"},
        "asdkjf qpwoei zxcvbn",
        "Bonjour, comment allez-vous?",
        "x" * 10000,
        "\x00\x01\x02 control characters",
    ]
    result = categorize_transactions(batch)
    assert len(result["categorized"]) == len(batch)
    assert isinstance(result["insight"], str) and result["insight"]


def test_classify_scam_handles_control_characters_and_weird_unicode():
    for text in ["\x00\x01\x02", "😀😀😀😀😀😀😀😀", "​​​" * 20, "​​​​​​​​​​"]:
        result = classify_scam(text)
        assert isinstance(result["is_scam"], bool)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["reason"], str) and result["reason"]
