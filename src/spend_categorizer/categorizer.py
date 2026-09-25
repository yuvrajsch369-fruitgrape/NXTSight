"""Spend-categorization task, running on NXTSight's shared on-device engine.

categorize_transactions(list_of_texts) -> {"categorized": [...], "insight": str}

Same shared-engine pattern as the scam classifier: this module registers a
Task with src.pipeline.engine.engine and calls engine.analyze() — it never
touches onnxruntime or loads a model file itself. Everything below is
spend-specific interpretation of the engine's raw Prediction: turning a
category id into a label, pulling amount/direction out with a small regex
(independent of the ML model), and rolling per-transaction results up into
one plain-language insight.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from src.pipeline.engine import Task, engine

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TASK_NAME = "spend_categorization"
# With ~8 examples per category spread across 11 classes, softmax probability
# mass is naturally diffuse (random baseline is ~0.09) — 0.20 is comfortably
# above chance without demanding near-certainty from a small model.
MIN_CATEGORY_CONFIDENCE = 0.20

# Read directly (not through the lazy _load_categories() below) because
# Task registration happens once, here, at import time — categories.json
# may not exist yet in a from-scratch checkout before train_classifier.py
# has ever run, so this degrades to no LLM escalation (llm_labels=None)
# rather than failing the import. label_id order must match this file
# exactly, since engine._maybe_escalate() maps the LLM's chosen label back
# to a label_id via this same list's index.
try:
    _LLM_CATEGORY_LABELS = json.loads((ARTIFACTS_DIR / "categories.json").read_text())
except Exception:
    _LLM_CATEGORY_LABELS = None

LLM_SYSTEM_PROMPT = (
    "You categorize Indian bank/UPI transaction SMS messages into exactly "
    "one spending category.\n\nCategories and what belongs in them:\n"
    "- Bills & Utilities: electricity, water, gas, phone/internet bills, insurance premiums\n"
    "- Cash Withdrawal: ATM withdrawals\n"
    "- Entertainment: movies, streaming subscriptions, games, events\n"
    "- Food & Dining: restaurants, food delivery, cafes\n"
    "- Groceries: supermarkets, kirana stores, grocery delivery\n"
    "- Healthcare: hospitals, clinics, doctors, pharmacies, medical tests\n"
    "- Income & Refunds: salary credits, refunds, cashback received\n"
    "- Investment & Savings: mutual funds, SIPs, stocks, fixed deposits, PPF\n"
    "- Shopping: e-commerce, clothing, electronics, general retail\n"
    "- Transfers & UPI P2P: person-to-person transfers, rent, sending money to family/friends\n"
    "- Transport: cabs, fuel, metro, bus, parking, tolls\n\n"
    "confidence is a decimal between 0.0 and 1.0."
)

engine.register_task(
    Task(
        name=TASK_NAME,
        artifacts_dir=ARTIFACTS_DIR,
        max_chars=2000,
        llm_labels=_LLM_CATEGORY_LABELS,
        llm_system_prompt=LLM_SYSTEM_PROMPT if _LLM_CATEGORY_LABELS else None,
        # Much stricter than scam_detection's 0.35: with 11 categories and
        # only ~8 training examples each, softmax mass is naturally diffuse
        # even on a confidently *correct* call — measured margins of
        # 0.3-1.0 on transactions this classifier gets right, so a cutoff
        # tuned for the binary scam task would escalate on nearly every
        # transaction here. Even at 0.15, real batches of genuinely novel
        # (not-in-training-data) merchant text still escalate often — this
        # classifier is frequently uncertain on inputs it hasn't seen a
        # close match for, which is exactly the case LLM escalation exists
        # for, but it's a real, honest latency cost: a batch of several
        # transactions can take tens of seconds, not milliseconds, when
        # several of them are genuinely ambiguous. Not a free upgrade.
        llm_escalation_margin=0.15,
    )
)

AMOUNT_PATTERN = re.compile(r"(?:rs\.?|inr|₹)\s?([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE)
DEBIT_KEYWORDS = ["debited", "spent", "paid", "withdrawn", "withdrawal", "sent", "invested", "transferred"]
CREDIT_KEYWORDS = ["credited", "received", "deposited", "refund", "cashback", "bonus"]

_categories = None


def _load_categories():
    """Load the label_id -> category-name list, or None if it can't be read.

    Unlike the scam classifier's top_terms (cosmetic), this is required to
    turn a raw label_id into a real category name — if it's missing or
    corrupted, _classify_one() below treats every result as unrecognized
    rather than crashing on an out-of-range list index.
    """
    global _categories
    if _categories is None:
        try:
            _categories = json.loads((ARTIFACTS_DIR / "categories.json").read_text())
        except Exception:
            _categories = None
    return _categories


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


def _normalize_item(raw):
    """Accepts a plain string (source defaults to "sms") or a
    {"text": ..., "source": ...} dict (e.g. source="screenshot" from a
    receipt scan). Returns (text, source); malformed dicts (missing/non-
    string "text") pass a non-string through unchanged — the existing
    engine.analyze() -> text_guard path already rejects those cleanly.
    """
    if isinstance(raw, dict):
        return raw.get("text"), raw.get("source", "sms")
    return raw, "sms"


def _unrecognized_item(raw, source="sms") -> dict:
    text = raw if isinstance(raw, str) else str(raw)
    return {
        "text": text,
        "category": "Unrecognized",
        "confidence": 0.0,
        "amount": None,
        "direction": None,
        "source": source,
    }


def _classify_one(raw, source="sms") -> dict:
    result = engine.analyze(TASK_NAME, raw)
    if isinstance(result, str):
        return _unrecognized_item(raw, source)

    categories = _load_categories()
    amount = _extract_amount(result.text)
    direction = _extract_direction(result.text)

    # Confidence alone isn't a reliable "is this a transaction" gate — a
    # message like "Happy birthday!" can still land a mid-confidence category
    # guess. Require an actual amount or debit/credit cue too: a real
    # transaction SMS always has at least one.
    looks_like_a_transaction = amount is not None or direction is not None
    category_available = categories is not None and 0 <= result.label_id < len(categories)

    if (
        result.confidence < MIN_CATEGORY_CONFIDENCE
        or not looks_like_a_transaction
        or not category_available
    ):
        item = _unrecognized_item(result.text, source)
        item["confidence"] = round(result.confidence, 3)
        return item

    return {
        "text": result.text,
        "category": categories[result.label_id],
        "confidence": round(result.confidence, 3),
        "amount": amount,
        "direction": direction,
        "source": source,
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
    """Each item may be a plain string (SMS text, source defaults to "sms")
    or a {"text": str, "source": str} dict — e.g. {"text": ..., "source":
    "screenshot"} for a transaction extracted from a receipt photo via
    receipt_parser.py. Both shapes run through the exact same
    classification logic; "source" just rides along into the result so
    screenshot-derived and SMS-derived entries can be told apart.
    """
    if list_of_texts is None or not isinstance(list_of_texts, (list, tuple)) or len(list_of_texts) == 0:
        return {"categorized": [], "insight": "Couldn't analyze this: no transactions provided."}

    categorized = [_classify_one(*_normalize_item(raw)) for raw in list_of_texts]
    insight = _build_insight(categorized)
    return {"categorized": categorized, "insight": insight}
