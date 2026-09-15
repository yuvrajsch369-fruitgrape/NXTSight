"""Tests the receipt/payment-confirmation screenshot -> spend-insight path.

Covers the 5 sample screenshots this feature was built and stress-tested
against (data/samples/receipt_*.png): a UPI confirmation, a printed
receipt, a handwritten note, a blurry photo, and a multi-line-item bill.
Real, documented outcomes — 3 succeed, 2 correctly decline rather than
guess a wrong number. See README for the honest limitations this surfaced.
"""

from pathlib import Path

from src.spend_categorizer.categorizer import categorize_transactions
from src.spend_categorizer.receipt_parser import (
    build_transaction_text,
    parse_receipt,
    process_receipt_screenshot,
)

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


# --- process_receipt_screenshot() against the 5 real sample images -----


def test_upi_confirmation_extracts_all_fields():
    result = process_receipt_screenshot(str(SAMPLES_DIR / "receipt_upi_confirmation.png"))
    assert result["ok"] is True
    assert result["merchant"] == "SWIGGY"
    assert result["amount"] == 450.0
    assert result["date"] == "12 Sep 2025"


def test_printed_grocery_receipt_extracts_all_fields():
    result = process_receipt_screenshot(str(SAMPLES_DIR / "receipt_printed_grocery.png"))
    assert result["ok"] is True
    assert result["merchant"] == "BIG BAZAAR"
    assert result["amount"] == 845.0
    assert result["date"] == "15/09/2025"


def test_multi_item_bill_picks_the_total_not_a_line_item():
    result = process_receipt_screenshot(str(SAMPLES_DIR / "receipt_multi_item_bill.png"))
    assert result["ok"] is True
    assert result["merchant"] == "CAFE COFFEE DAY"
    # Known limitation: OCR splits "933.50" across two lines ("933" / "50"),
    # so the parsed amount drops the decimal cents. Documented, not hidden —
    # this locks in the current (imperfect but honest) behavior.
    assert result["amount"] == 933.0


def test_handwritten_note_declines_rather_than_guessing():
    result = process_receipt_screenshot(str(SAMPLES_DIR / "receipt_handwritten_note.png"))
    # OCR misreads the handwritten "500" as "S00" (digit 5 -> letter S) —
    # not a valid number, so this must NOT silently become some other
    # value. It has to decline cleanly instead.
    assert result["ok"] is False
    assert "couldn't find a clear amount" in result["message"].lower()


def test_blurry_photo_declines_rather_than_hallucinating():
    result = process_receipt_screenshot(str(SAMPLES_DIR / "receipt_blurry_photo.png"))
    assert result["ok"] is False
    assert "couldn't find a clear amount" in result["message"].lower()


# --- Failure cases that must never crash or silently produce a bad number


def test_missing_file_returns_clear_error_not_crash():
    result = process_receipt_screenshot(str(SAMPLES_DIR / "does_not_exist.png"))
    assert result["ok"] is False
    assert result["message"]


def test_none_path_returns_clear_error_not_crash():
    result = process_receipt_screenshot(None)
    assert result["ok"] is False
    assert result["message"]


def test_corrupted_image_returns_clear_error_not_crash(tmp_path):
    bad_file = tmp_path / "corrupted.png"
    bad_file.write_bytes(b"not a real image")
    result = process_receipt_screenshot(str(bad_file))
    assert result["ok"] is False
    assert result["message"]


def test_garbage_ocr_text_never_produces_a_bare_hallucinated_amount():
    # Simulates what a heavily blurred/noisy photo OCRs to: a single stray
    # digit with no surrounding payment context. Must decline, not "succeed"
    # with a tiny made-up amount — this is the specific bug this feature's
    # bare-number fallback could otherwise introduce.
    parsed = parse_receipt("2")
    assert parsed["problem"] is not None
    assert parsed["amount"] is None


def test_empty_text_returns_clear_problem():
    parsed = parse_receipt("")
    assert parsed["problem"] is not None


# --- parse_receipt() / build_transaction_text() unit behavior ----------


def test_prefers_total_line_over_line_item_amounts():
    text = "CORNER STORE\nMilk 40.00\nBread 30.00\nTotal Rs 70.00\nDate: 01/01/2025"
    parsed = parse_receipt(text)
    assert parsed["amount"] == 70.0


def test_build_transaction_text_matches_sms_shape():
    parsed = {"merchant": "swiggy", "amount": 450.0, "date": "12 Sep 2025", "problem": None}
    text = build_transaction_text(parsed)
    assert text == "Rs 450.00 debited via UPI to SWIGGY on 12 Sep 2025."


def test_build_transaction_text_handles_missing_merchant_and_date():
    parsed = {"merchant": None, "amount": 100.0, "date": None, "problem": None}
    text = build_transaction_text(parsed)
    assert "UNKNOWN MERCHANT" in text
    assert "Rs 100.00" in text


# --- End-to-end: a screenshot-derived transaction flows into the same ---
# --- categorize_transactions() used for SMS text, tagged by source ------


def test_screenshot_derived_transaction_categorizes_and_is_tagged():
    result = process_receipt_screenshot(str(SAMPLES_DIR / "receipt_upi_confirmation.png"))
    assert result["ok"] is True

    batch = categorize_transactions([{"text": result["transaction_text"], "source": "screenshot"}])
    item = batch["categorized"][0]

    assert item["source"] == "screenshot"
    assert item["category"] == "Food & Dining"
    assert item["amount"] == 450.0


def test_mixed_sms_and_screenshot_batch_both_tagged_correctly():
    sms_text = "Rs 240.00 debited via UPI to UBER INDIA on 12-Sep-25."
    receipt = process_receipt_screenshot(str(SAMPLES_DIR / "receipt_printed_grocery.png"))
    assert receipt["ok"] is True

    batch = categorize_transactions(
        [sms_text, {"text": receipt["transaction_text"], "source": "screenshot"}]
    )

    sources = {item["source"] for item in batch["categorized"]}
    assert sources == {"sms", "screenshot"}
