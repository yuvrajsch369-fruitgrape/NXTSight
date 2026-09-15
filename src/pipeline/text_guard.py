"""Shared "is this text analyzable" guard for the pipeline's text stages.

Both classify_scam() and categorize_transactions() need to reject the same
kinds of bad input before it ever reaches a model: empty text, gibberish,
non-English text. Centralized here so the two stages share one definition
instead of drifting apart, and so a Snapdragon-side language model (if one
replaces langdetect later) only needs to change in one place.
"""

from langdetect import LangDetectException, detect_langs

MIN_CHARS = 3
MIN_LETTER_RATIO = 0.3  # below this, text is mostly symbols/numbers, not language
MIN_LANGDETECT_CONFIDENCE = 0.70


def unanalyzable_reason(text, allowed_langs=("en",)) -> "str | None":
    """Return a reason string if `text` can't be analyzed, else None.

    `allowed_langs` defaults to English only, matching both models today.
    """
    if text is None or not isinstance(text, str):
        return "no text provided"

    cleaned = text.strip()
    if not cleaned:
        return "no text provided"

    if len(cleaned) < MIN_CHARS:
        return "text is too short to judge"

    letter_ratio = sum(c.isalpha() for c in cleaned) / len(cleaned)
    if letter_ratio < MIN_LETTER_RATIO:
        return "text doesn't look like readable language (too few letters)"

    try:
        candidates = detect_langs(cleaned)
    except LangDetectException:
        return "text doesn't look like readable language (gibberish)"
    if not candidates:
        return "text doesn't look like readable language (gibberish)"

    top = candidates[0]
    if top.prob < MIN_LANGDETECT_CONFIDENCE:
        return "couldn't confidently identify the language (possibly gibberish)"
    if top.lang not in allowed_langs:
        return f"detected language '{top.lang}' — this model only supports English"

    return None
