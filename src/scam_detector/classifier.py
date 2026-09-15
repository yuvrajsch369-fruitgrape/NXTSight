"""Scam-classification task, running on NXTSight's shared on-device engine.

classify_scam(text) -> {"is_scam": bool, "confidence": float, "reason": str}

The model itself is a small TF-IDF + Logistic Regression classifier
exported to ONNX (train_classifier.py builds it — 60 labeled examples,
~15KB). This module doesn't load it or run inference directly — it
registers a Task with src.pipeline.engine.engine and calls engine.analyze(),
the same shared object the spend categorizer calls through. Everything
below is scam-specific interpretation of the engine's raw Prediction: the
binary is_scam flag and the human-readable `reason` built from the model's
own learned vocabulary.
"""

import json
from pathlib import Path

from src.pipeline.engine import Task, engine

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TASK_NAME = "scam_detection"
REASON_TERMS_SHOWN = 4

engine.register_task(Task(name=TASK_NAME, artifacts_dir=ARTIFACTS_DIR, max_chars=4000))

_top_terms = None


def _load_top_terms():
    global _top_terms
    if _top_terms is None:
        _top_terms = json.loads((ARTIFACTS_DIR / "top_terms.json").read_text())
    return _top_terms


def _unanalyzable(reason: str) -> dict:
    return {"is_scam": False, "confidence": 0.0, "reason": f"Couldn't analyze this: {reason}."}


def classify_scam(text) -> dict:
    result = engine.analyze(TASK_NAME, text)
    if isinstance(result, str):
        return _unanalyzable(result)

    is_scam = bool(result.label_id)

    top_terms = _load_top_terms()
    present_terms = engine.vocabulary_terms(TASK_NAME, result.text)
    candidate_pool = top_terms["scam"] if is_scam else top_terms["legit"]
    matched = [term for term in candidate_pool if term in present_terms][:REASON_TERMS_SHOWN]

    if matched:
        quoted = ", ".join(f"'{term}'" for term in matched)
        if is_scam:
            reason = f"Contains phrases commonly seen in scams: {quoted}."
        else:
            reason = f"Contains routine-message phrasing: {quoted}; no strong scam indicators."
    else:
        reason = (
            "Flagged as likely scam based on overall wording, though no single standout phrase."
            if is_scam
            else "No strong scam indicators found; looks like a routine message."
        )

    return {
        "is_scam": is_scam,
        "confidence": round(result.confidence, 3),
        "reason": reason,
    }
