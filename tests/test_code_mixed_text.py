"""Hindi-English code-mixed ("Hinglish") text — what real Indian scam and
transaction messages actually look like, not just clean English. Added
after measuring real, sometimes-uncomfortable results directly (see
PROJECT_DESCRIPTION.md / README's hardening notes): a large fraction of
realistic Hinglish text never reaches either classifier at all, rejected
by src/pipeline/text_guard.py's `langdetect`-based English-only gate —
and that gate is non-deterministic (no fixed seed), so the *same* message
can be accepted on one call and rejected on the next.

These tests are split into three honest groups:

1. Messages that were empirically stable (rejected 0/8 times across 8
   repeated trials during measurement) — parametrized as regular
   assertions, since a regression here is a real regression.
2. `test_text_guard_rejects_most_hinglish_messages_nondeterministically` —
   documents the rejection-rate finding directly, as a measured range
   rather than a single flaky assertion.
3. `test_known_false_positive_hinglish_refund_message` — an `xfail`
   documenting a confirmed, confidently-wrong classification (not an
   ambiguous one the LLM tier would catch) on a legit Hinglish refund
   message. This should start passing (and the xfail should be removed)
   the day this is actually fixed, not silently forgotten.
"""

import pytest

from src.pipeline.text_guard import unanalyzable_reason
from src.scam_detector.classifier import classify_scam
from src.spend_categorizer.categorizer import categorize_transactions

# Only messages that passed src/pipeline/text_guard.py's langdetect gate on
# every one of 8 repeated trials during measurement — deterministic enough
# to assert on directly. (Plenty of other realistic Hinglish phrasings did
# NOT clear this bar — see the rejection-rate test below.)
SCAM_HINGLISH_STABLE = [
    "URGENT: Aapke Aadhaar card se ek fraud case register hua hai. Turant is number par call karein warna warrant issue ho jayega.",
    "Yeh HDFC Bank se hai. Aapke account mein suspicious activity dekhi gayi hai. Apna OTP share karein verification ke liye.",
    "Aapko ek work from home job offer mili hai, Rs 40,000 per week. Sirf Rs 999 registration fee bhejein activate karne ke liye.",
    "Your Aadhaar-linked account mein Rs 49,000 ka suspicious transaction hua hai. Turant is toll-free number par call back karein.",
]

LEGIT_HINGLISH_STABLE = [
    "Class 5B ka parent-teacher meeting Friday 4 baje school auditorium mein hai.",
]

# Hinglish variants of SAMPLE_BATCH-style transactions (tests/test_categorizer.py).
# Unlike SCAM_HINGLISH_STABLE/LEGIT_HINGLISH_STABLE above, neither of these
# passed text_guard on all 8 measurement trials (1/8 and 2/8 rejected,
# respectively) — the categorizer got every trial that DID pass correct,
# but "passed every single time" wasn't true for any spend example tested,
# unlike several scam ones. test_categorizes_stable_hinglish_transactions
# below retries a few times to test the categorizer's own judgment rather
# than getting incidentally skipped by the separately-documented text_guard
# flakiness (see test_text_guard_rejects_most_hinglish_messages_nondeterministically).
SPEND_HINGLISH_STABLE = [
    ("Aapke account se Rs 380 kat gaye Swiggy se Behrouz Biryani order karne par.", "Food & Dining"),
    ("Rs 899 Sony Liv subscription ke liye UPI se kat gaye.", "Entertainment"),
]


@pytest.mark.parametrize("text", SCAM_HINGLISH_STABLE)
def test_flags_stable_hinglish_scam_examples(text):
    result = classify_scam(text)
    assert result["is_scam"] is True
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.parametrize("text", LEGIT_HINGLISH_STABLE)
def test_clears_stable_hinglish_legit_examples(text):
    result = classify_scam(text)
    assert result["is_scam"] is False
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.parametrize("text,expected_category", SPEND_HINGLISH_STABLE)
def test_categorizes_stable_hinglish_transactions(text, expected_category):
    # Retries a few times: text_guard rejects this specific text roughly
    # 1-in-4 to 1-in-8 calls (measured), and that flakiness is already its
    # own documented finding below — this test exists to check the
    # categorizer's judgment once the text actually reaches it, not to
    # re-prove the gate is flaky.
    for attempt in range(5):
        item = categorize_transactions([text])["categorized"][0]
        # text_guard rejection shows up as Unrecognized + confidence 0.0 +
        # no extracted amount — a low-confidence-but-reached-the-model
        # Unrecognized result would have a non-None amount instead.
        text_guard_rejected = item["category"] == "Unrecognized" and item["amount"] is None and item["confidence"] == 0.0
        if not text_guard_rejected:
            break
    else:
        pytest.skip("text_guard rejected this Hinglish text on every retry — see the rejection-rate test below")
    assert item["category"] == expected_category


def test_text_guard_rejects_most_hinglish_messages_nondeterministically():
    """Measured directly (not assumed): across 8 repeated calls on 10
    realistic Hinglish scam messages and 10 realistic Hinglish legit/
    transaction messages, text_guard's langdetect-based gate rejected
    46% of scam calls and 72% of legit calls as "not English" — most
    often misidentifying the code-mixed text as Indonesian ('id'),
    Estonian, or Turkish (short-text n-gram collisions with romanized
    Hindi), and inconsistently: the identical string can pass on one
    call and get rejected on the next, since langdetect ships with no
    fixed random seed (confirmed: the same string produces different
    confidence distributions across repeated calls with nothing else
    changed). This test locks in that this is a known, measured
    limitation — not a silent one — by asserting the rejection rate
    stays roughly in the range actually observed, so a change that
    makes it dramatically worse is caught, without asserting an exact
    single-call outcome that would make this test itself flaky.
    """
    hinglish_samples = SCAM_HINGLISH_STABLE + [
        "Aapka SBI account 24 ghante mein block ho jayega. Turant apna KYC verify karein is link par: bit.ly/sbi-kyc-verify",
        "Aapka parcel customs mein atka hai. Rs 149 ka duty pay karein is link se warna parcel return ho jayega: http://parcel-fee.net",
        "Aapke SBI a/c se Rs 500 debit hua hai Swiggy Bangalore par 12-Sep ko. Balance: Rs 4,200.",
        "Kal 1 baje lunch pe milte hain? Agar reschedule karna ho toh batao.",
        "Dr. Mehta ke saath aapka dental appointment kal 11:30 AM confirm hai.",
    ]
    trials = 6
    rejected = sum(
        1 for _ in range(trials) for text in hinglish_samples if unanalyzable_reason(text) is not None
    )
    total = trials * len(hinglish_samples)
    rejection_rate = rejected / total
    # Observed range during measurement was roughly 30-80% depending on the
    # exact sample set; this is a wide, deliberately loose band that only
    # fails if the gate becomes either suspiciously lenient (a real-language
    # check that never rejects anything isn't checking anything) or even
    # worse than what was already measured as a real problem.
    assert 0.2 <= rejection_rate <= 0.95, (
        f"Hinglish rejection rate {rejection_rate:.0%} is outside the previously-measured "
        "range — re-investigate rather than assuming this is still the same known gap."
    )


@pytest.mark.xfail(
    reason=(
        "Confirmed, repeatable false positive: this legit Hinglish refund message is "
        "classified as scam with 0.934 confidence (not an ambiguous ~0.5 call the LLM "
        "escalation tier would catch — see engine.py's margin-based escalation, which "
        "never fires here because the fast classifier is confidently wrong, not unsure). "
        "The equivalent English phrasing ('A refund of Rs 2,340 has been initiated...') "
        "is a known *ambiguous* case that correctly escalates (see "
        "tests/test_network_guard.py) — the Hinglish phrasing shifts the same semantic "
        "message from 'appropriately uncertain' to 'confidently wrong' for the fast "
        "MiniLM+head classifier. Left failing on purpose so this is caught and fixed "
        "deliberately (more Hinglish training data, or a lower escalation margin), not "
        "silently forgotten."
    ),
    strict=True,
)
def test_known_false_positive_hinglish_refund_message():
    result = classify_scam(
        "Rs 2,340 ka refund aapke cancelled order ke liye initiate ho gaya hai, 3-5 business days mein reflect hoga."
    )
    assert result["is_scam"] is False
