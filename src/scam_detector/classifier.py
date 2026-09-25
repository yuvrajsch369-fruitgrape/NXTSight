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

# label_id 0 = legit, 1 = scam — must match how train_classifier.py encodes
# labels (`1 if is_scam else 0`), since engine._maybe_escalate() maps the
# LLM's chosen label back to a label_id via this list's index.
LLM_LABELS = ["legit", "scam"]
LLM_SYSTEM_PROMPT = (
    "You are a fraud-detection classifier for Indian bank/UPI text messages. "
    "Decide if a message is a SCAM or LEGIT.\n\n"
    "Common scam patterns: fake KYC-update threats, requests to share an OTP "
    "or PIN, fake lottery/prize wins, digital-arrest or police-impersonation "
    "threats, fake courier/customs fees, urgent account-suspension threats "
    "with a link, too-good-to-be-true investment offers, fake tech support.\n\n"
    "Legit messages: routine bank/UPI transaction confirmations, real OTPs "
    "sent by a bank for a purchase the user is actively making (not "
    "requested by the message itself), calendar/appointment reminders, "
    "everyday personal messages.\n\n"
    "confidence is a decimal between 0.0 and 1.0."
)

engine.register_task(
    Task(
        name=TASK_NAME,
        artifacts_dir=ARTIFACTS_DIR,
        max_chars=4000,
        llm_labels=LLM_LABELS,
        llm_system_prompt=LLM_SYSTEM_PROMPT,
        # 0.35, not the Task default of 0.3 — calibrated against real
        # examples: this classifier is generally very peaked (most real
        # messages land above 0.5 margin even when "borderline-sounding"),
        # but a genuinely ambiguous legit message (a refund notice) landed
        # at 0.327 — 0.35 reliably catches that band without escalating on
        # every message. See tests/test_llm_escalation.py.
        llm_escalation_margin=0.35,
    )
)

_top_terms = None


def _load_top_terms():
    """Load the reason-building vocabulary; {"scam": [], "legit": []} if it
    can't be read — a missing/corrupted top_terms.json degrades the quality
    of the `reason` text, it shouldn't take down classification itself."""
    global _top_terms
    if _top_terms is None:
        try:
            _top_terms = json.loads((ARTIFACTS_DIR / "top_terms.json").read_text())
        except Exception:
            _top_terms = {"scam": [], "legit": []}
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
