"""Turns a receipt / payment-confirmation screenshot into a spend-insight entry.

Reuses extract_text_from_image() (src/pipeline/ocr.py) for the OCR step —
no separate OCR path. But receipt and payment-app layouts don't read like
bank/UPI SMS text at all (no "debited"/"credited" boilerplate, merchant
name and amount scattered across lines rather than one sentence), so
feeding raw OCR'd receipt text straight into the SMS-trained spend
classifier would be unreliable. Instead: pull out (merchant, amount, date)
with layout-aware heuristics, then build a normalized sentence in the
same shape the classifier actually knows — "Rs X debited via UPI to
MERCHANT on DATE." — and hand that to categorize_transactions(), tagged
with source="screenshot" so it's reused exactly, not duplicated.
"""

import re

from src.pipeline.ocr import extract_text_from_image

# [ \t]? (not \s?) deliberately: \s also matches a literal newline, which
# let a currency marker on one OCR'd line falsely bind to an unrelated
# number on the *next* line (e.g. "Rs" then, on the following line, a
# date "15/09/2025" was misread as "Rs 15"). Currency marker and amount
# must be on the same line, as intended from the start.
AMOUNT_PATTERN = re.compile(r"(?:rs\.?|inr|₹)[ \t]?([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE)
# A line that's JUST a number, nothing else — how a payment-app's amount
# often OCRs when its ₹/currency glyph isn't captured as text. Receipts and
# bills don't produce lines like this: their numbers always sit next to an
# item name or a label, so this is a fairly safe, narrow signal, used only
# as a last resort below.
STANDALONE_NUMBER_LINE = re.compile(r"^[^\d]{0,2}(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)[^\d]{0,2}$")
# Two independent guards on that fallback, since a single stray digit OCR
# hallucinates from image noise (e.g. a blurry photo reading as just "2")
# must never be trusted as a real amount:
#  - it has to be at least this large (a genuine payment is rarely <10), and
#  - the text has to actually look like a payment screen somewhere, not just
#    contain one isolated number with nothing else recognizable around it.
MIN_BARE_AMOUNT = 10
# Widened to include receipt/bill words, not just payment-app words — a
# printed bill saying "total"/"gst"/"grand total" is just as strong a
# signal this is a real purchase document as "paid"/"successful" is for a
# payment-app screen. Still a real requirement, not a rubber stamp: text
# with none of these words present gets no bare-number fallback at all.
PAYMENT_CONTEXT_KEYWORDS = [
    "paid", "payment", "successful", "transaction", "upi", "amount", "sent", "received",
    "total", "bill", "gst", "grand",
]
# A decimal point OCR sometimes reads with stray spaces around it
# ("845 . 00" instead of "845.00") — collapse those before any amount
# regex runs, rather than trying to make every pattern whitespace-tolerant.
_SPACED_DECIMAL = re.compile(r"(\d)\s*\.\s*(\d{1,2})(?!\d)")
TOTAL_KEYWORDS = ["grand total", "total amount", "amount paid", "total", "paid", "amount"]
DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"),
    re.compile(
        r"\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})\b",
        re.IGNORECASE,
    ),
]
MIN_MERCHANT_LETTERS = 2
MAX_MERCHANT_LEN = 40
MERCHANT_MERGE_MAX_LEN = 6  # a merchant line this short is often an OCR-split header (e.g. "BIG" / "BAZAAR")
MERCHANT_CONTINUATION_MAX_LEN = 20
# Header/chrome/label text that shows up on payment-app screens and
# receipts but isn't a merchant name — skip these when picking the
# merchant line.
MERCHANT_LINE_STOPWORDS = {
    "payment successful", "payment successfull", "transaction successful",
    "successful", "payment complete", "paid successfully", "success",
    "google pay", "phonepe", "paytm", "gpay", "upi", "transaction details",
    "transaction id", "utr", "receipt", "invoice", "bill", "cash memo",
    "paid to", "sent to", "pay to", "paid by", "sent by", "from", "to",
    "date & time", "date and time", "date", "time", "amount", "amount paid",
}


def _extract_all_amounts(text):
    amounts = []
    for match in AMOUNT_PATTERN.finditer(text):
        try:
            amounts.append(float(match.group(1).replace(",", "")))
        except ValueError:
            continue
    return amounts


def _pick_bare_amount(full_text, lines):
    if not any(keyword in full_text.lower() for keyword in PAYMENT_CONTEXT_KEYWORDS):
        return None
    # Same reasoning as tier 2 (largest currency-marked amount): with
    # multiple standalone-number lines and no currency tag to disambiguate
    # which one is the real total, the largest is the best single guess —
    # not just whichever line happened to come first (a bill's line items
    # are individually smaller than its total).
    candidates = []
    for line in lines:
        match = STANDALONE_NUMBER_LINE.match(line.strip())
        if not match:
            continue
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if MIN_BARE_AMOUNT <= value <= 10_000_000:
            candidates.append(value)
    return max(candidates) if candidates else None


def _pick_amount(full_text, lines):
    # 1. Prefer a currency-marked amount on a line that also mentions a
    #    total/paid keyword — avoids picking a per-item price off a
    #    multi-line-item bill.
    for line in lines:
        if any(keyword in line.lower() for keyword in TOTAL_KEYWORDS):
            amounts = _extract_all_amounts(line)
            if amounts:
                return max(amounts)
    # 2. Otherwise: the largest currency-marked amount anywhere. On both
    #    confirmation screens (usually one prominent amount) and itemized
    #    bills (line items are individually smaller than the total), the
    #    largest number is the best single-value guess without a real
    #    layout/vision model.
    all_amounts = _extract_all_amounts(full_text)
    if all_amounts:
        return max(all_amounts)
    # 3. Last resort: a bare number alone on its own line — how a payment
    #    app's amount often reads when its currency glyph isn't OCR'd as
    #    text (a real, common case: this is the *only* way GPay/PhonePe-
    #    style "Payment Successful" screens show the amount at all).
    return _pick_bare_amount(full_text, lines)


def _pick_date(full_text):
    for pattern in DATE_PATTERNS:
        match = pattern.search(full_text)
        if match:
            return match.group(1)
    return None


def _is_plausible_merchant_line(text):
    return (
        bool(text)
        and text.lower() not in MERCHANT_LINE_STOPWORDS
        and not AMOUNT_PATTERN.search(text)
        and sum(c.isalpha() for c in text) >= MIN_MERCHANT_LETTERS
    )


def _pick_merchant(lines):
    for i, line in enumerate(lines):
        cleaned = line.strip()
        if not cleaned or not _is_plausible_merchant_line(cleaned):
            continue

        merged = cleaned
        # A short header is often split across two OCR lines (e.g. "BIG" /
        # "BAZAAR", "CAFE" / "COFFEE DAY") — stitch in the next line if it
        # looks like a plausible continuation rather than an unrelated label.
        if len(cleaned) <= MERCHANT_MERGE_MAX_LEN and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if (
                _is_plausible_merchant_line(nxt)
                and len(nxt) <= MERCHANT_CONTINUATION_MAX_LEN
            ):
                merged = f"{cleaned} {nxt}"

        return merged[:MAX_MERCHANT_LEN]
    return None


def parse_receipt(raw_text: str) -> dict:
    """Extract {merchant, amount, date} from OCR'd receipt/confirmation text.

    Returns {"merchant": str|None, "amount": float|None, "date": str|None,
    "problem": str|None}. `problem` is set — with a plain-language reason —
    when the amount (the one field everything downstream depends on)
    couldn't be found; merchant/date are best-effort and may be None even
    when there's no `problem`.
    """
    if not raw_text or not raw_text.strip():
        return {"merchant": None, "amount": None, "date": None, "problem": "no text was detected in this image"}

    # Collapse "845 . 00" -> "845.00" before anything else touches the
    # text — some OCR passes read a decimal point with stray spaces
    # around it, which would otherwise break every amount pattern below.
    raw_text = _SPACED_DECIMAL.sub(r"\1.\2", raw_text)

    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    amount = _pick_amount(raw_text, lines)
    merchant = _pick_merchant(lines)
    date = _pick_date(raw_text)

    if amount is None:
        return {
            "merchant": merchant,
            "amount": None,
            "date": date,
            "problem": (
                "couldn't find a clear amount in this image — it may be too "
                "blurry, cropped, or in a layout this can't read yet"
            ),
        }

    return {"merchant": merchant, "amount": amount, "date": date, "problem": None}


def build_transaction_text(parsed: dict) -> str:
    """Turn parsed receipt fields into a sentence shaped like the bank/UPI
    SMS text the spend classifier was actually trained on, so it can be
    categorized by the exact same model rather than a separate one."""
    merchant = (parsed["merchant"] or "UNKNOWN MERCHANT").upper()
    amount = parsed["amount"]
    date_part = f" on {parsed['date']}" if parsed.get("date") else ""
    return f"Rs {amount:,.2f} debited via UPI to {merchant}{date_part}."


def process_receipt_screenshot(image_path) -> dict:
    """OCR a receipt/payment-confirmation screenshot and extract a
    ready-to-categorize transaction. Never raises — always returns a dict
    with "ok" telling the caller whether it succeeded.

    On success: {"ok": True, "transaction_text": str, "merchant": ...,
    "amount": ..., "date": ..., "raw_text": str}
    On failure: {"ok": False, "message": str, "raw_text": str|None} — message
    is a clear, human-readable reason, never a raw exception.
    """
    text = extract_text_from_image(image_path)
    if text.startswith("Error"):
        return {"ok": False, "message": text[len("Error: "):].capitalize(), "raw_text": None}

    parsed = parse_receipt(text)
    if parsed["problem"]:
        return {"ok": False, "message": parsed["problem"].capitalize(), "raw_text": text}

    return {
        "ok": True,
        "transaction_text": build_transaction_text(parsed),
        "merchant": parsed["merchant"],
        "amount": parsed["amount"],
        "date": parsed["date"],
        "raw_text": text,
    }
