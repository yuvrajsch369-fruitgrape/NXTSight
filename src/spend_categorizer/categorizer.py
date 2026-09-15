"""Spend-categorization stage of the NXTSight pipeline.

categorize_transactions(list_of_texts) -> {"categorized": [...], "insight": str}

Reuses the same local-model pattern as the scam classifier: TF-IDF +
Logistic Regression trained on labeled examples (train_classifier.py),
exported to ONNX via skl2onnx, and run through the same QNN-aware
runtime.py used by both other stages — one on-device model-serving layer
for scam detection and spend categorization alike. Also reuses the same
text_guard.unanalyzable_reason() gibberish/non-English/empty-text check
that the scam classifier uses, rather than reimplementing it.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import load

from src.pipeline.runtime import create_inference_session
from src.pipeline.text_guard import unanalyzable_reason

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MAX_CHARS = 2000  # a transaction SMS is short; this is a generous ceiling
# With ~8 examples per category spread across 11 classes, softmax probability
# mass is naturally diffuse (random baseline is ~0.09) — 0.20 is comfortably
# above chance without demanding near-certainty from a small model.
MIN_CATEGORY_CONFIDENCE = 0.20

AMOUNT_PATTERN = re.compile(r"(?:rs\.?|inr|₹)\s?([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE)
DEBIT_KEYWORDS = ["debited", "spent", "paid", "withdrawn", "withdrawal", "sent", "invested", "transferred"]
CREDIT_KEYWORDS = ["credited", "received", "deposited", "refund", "cashback", "bonus"]

_vectorizer = None
_session = None
_categories = None


def _load_artifacts():
    global _vectorizer, _session, _categories
    if _vectorizer is None:
        _vectorizer = load(ARTIFACTS_DIR / "vectorizer.joblib")
    if _session is None:
        _session, _ = create_inference_session(str(ARTIFACTS_DIR / "classifier.onnx"))
    if _categories is None:
        _categories = json.loads((ARTIFACTS_DIR / "categories.json").read_text())
    return _vectorizer, _session, _categories


def _extract_amount(text: str):
    match = AMOUNT_PATTERN.search(text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _extract_direction(text: str):
    lowered = text.lower()
    debit_pos = min((lowered.find(w) for w in DEBIT_KEYWORDS if w in lowered), default=-1)
    credit_pos = min((lowered.find(w) for w in CREDIT_KEYWORDS if w in lowered), default=-1)
    if debit_pos == -1 and credit_pos == -1:
        return None
    if debit_pos == -1:
        return "credit"
    if credit_pos == -1:
        return "debit"
    return "debit" if debit_pos < credit_pos else "credit"


def _unrecognized_item(raw) -> dict:
    text = raw if isinstance(raw, str) else str(raw)
    return {"text": text, "category": "Unrecognized", "confidence": 0.0, "amount": None, "direction": None}


def _classify_one(raw) -> dict:
    reason = unanalyzable_reason(raw)
    if reason:
        return _unrecognized_item(raw)

    cleaned = raw.strip()[:MAX_CHARS]
    vectorizer, session, categories = _load_artifacts()
    vector = vectorizer.transform([cleaned]).toarray().astype(np.float32)

    input_name = session.get_inputs()[0].name
    labels, probabilities = session.run(["label", "probabilities"], {input_name: vector})
    label_id = int(labels[0])
    confidence = float(probabilities[0][label_id])
    amount = _extract_amount(cleaned)
    direction = _extract_direction(cleaned)

    # Confidence alone isn't a reliable "is this a transaction" gate — a
    # message like "Happy birthday!" can still land a mid-confidence category
    # guess. Require an actual amount or debit/credit cue too: a real
    # transaction SMS always has at least one.
    looks_like_a_transaction = amount is not None or direction is not None

    if confidence < MIN_CATEGORY_CONFIDENCE or not looks_like_a_transaction:
        item = _unrecognized_item(cleaned)
        item["confidence"] = round(confidence, 3)
        return item

    return {
        "text": cleaned,
        "category": categories[label_id],
        "confidence": round(confidence, 3),
        "amount": amount,
        "direction": direction,
    }


def _build_insight(categorized: list) -> str:
    recognized = [item for item in categorized if item["category"] != "Unrecognized"]
    if not recognized:
        return "Couldn't analyze this: none of the provided transactions could be understood."

    debit_by_category = defaultdict(float)
    count_by_category = defaultdict(int)
    total_debit = 0.0

    for item in recognized:
        count_by_category[item["category"]] += 1
        if item["direction"] == "debit" and item["amount"] is not None:
            debit_by_category[item["category"]] += item["amount"]
            total_debit += item["amount"]

    if total_debit > 0:
        top_category, top_amount = max(debit_by_category.items(), key=lambda kv: kv[1])
        pct = round(100 * top_amount / total_debit)
        return (
            f"Your top spending category was {top_category} "
            f"(₹{top_amount:,.0f} of ₹{total_debit:,.0f} total spent, {pct}%), "
            f"across {len(recognized)} recognized transactions."
        )

    top_category, top_count = max(count_by_category.items(), key=lambda kv: kv[1])
    return (
        f"Couldn't total the amounts, but most transactions ({top_count} of {len(recognized)}) "
        f"were in {top_category}."
    )


def categorize_transactions(list_of_texts) -> dict:
    if list_of_texts is None or not isinstance(list_of_texts, (list, tuple)) or len(list_of_texts) == 0:
        return {"categorized": [], "insight": "Couldn't analyze this: no transactions provided."}

    categorized = [_classify_one(raw) for raw in list_of_texts]
    insight = _build_insight(categorized)
    return {"categorized": categorized, "insight": insight}
