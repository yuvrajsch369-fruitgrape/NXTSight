"""Scam-classification stage of the NXTSight pipeline.

classify_scam(text) -> {"is_scam": bool, "confidence": float, "reason": str}

Backed by a small TF-IDF + Logistic Regression model exported to ONNX
(train_classifier.py builds it — 60 labeled examples, ~15KB model) and run
through src/pipeline/runtime.py — the same QNN-aware session creator used
for OCR, so once this exact .onnx file is compiled for Snapdragon via AI
Hub, it runs on the NPU with no code change here.
"""

import json
from pathlib import Path

import numpy as np
from joblib import load

from src.pipeline.runtime import create_inference_session
from src.pipeline.text_guard import unanalyzable_reason

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MAX_CHARS = 4000  # generous upper bound for a screenshot's worth of text
REASON_TERMS_SHOWN = 4

_vectorizer = None
_session = None
_top_terms = None


def _load_artifacts():
    global _vectorizer, _session, _top_terms
    if _vectorizer is None:
        _vectorizer = load(ARTIFACTS_DIR / "vectorizer.joblib")
    if _session is None:
        _session, _ = create_inference_session(str(ARTIFACTS_DIR / "classifier.onnx"))
    if _top_terms is None:
        _top_terms = json.loads((ARTIFACTS_DIR / "top_terms.json").read_text())
    return _vectorizer, _session, _top_terms


def _unanalyzable(reason: str) -> dict:
    return {"is_scam": False, "confidence": 0.0, "reason": f"Couldn't analyze this: {reason}."}


def classify_scam(text) -> dict:
    reason = unanalyzable_reason(text)
    if reason:
        return _unanalyzable(reason)

    truncated = text.strip()[:MAX_CHARS]

    vectorizer, session, top_terms = _load_artifacts()
    vector = vectorizer.transform([truncated]).toarray().astype(np.float32)

    input_name = session.get_inputs()[0].name
    labels, probabilities = session.run(["label", "probabilities"], {input_name: vector})
    is_scam = bool(labels[0])
    result_confidence = float(probabilities[0][1] if is_scam else probabilities[0][0])

    present_terms = set(vectorizer.build_analyzer()(truncated))
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
        "confidence": round(result_confidence, 3),
        "reason": reason,
    }
