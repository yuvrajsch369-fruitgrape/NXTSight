import pytest

from src.scam_detector.classifier import classify_scam

# Deliberately NOT copied from src/scam_detector/data.py (the training set) —
# these are different sentences testing whether the model generalizes,
# rather than just memorizing training examples.
SCAM_HOLDOUT = [
    "URGENT: Your account will be suspended in 2 hours unless you verify your details now. Tap here to avoid permanent closure: bit.ly/verify-acc-2847",
    "Congratulations! You have been selected to receive a guaranteed 300% return on a one-time investment of Rs 10,000. Reply INVEST now to secure your spot before it closes tonight.",
    "This is HDFC Bank security. We noticed unusual login activity. Please share the OTP sent to your phone so we can verify and protect your account immediately.",
    "Your package could not be delivered due to an unpaid customs duty of Rs 149. Settle the payment within 3 hours at http://parcel-duty-clear.net or your item will be discarded.",
    "Great news! You're pre-approved for a work-from-home job paying Rs 45,000/week. Just pay a one-time registration fee of Rs 999 to activate your account and start today.",
]

LEGIT_HOLDOUT = [
    "Your OTP for the login attempt is 738291. It is valid for 5 minutes. Never share this code with anyone, including bank staff.",
    "Hey, are we still on for lunch at 1pm tomorrow? Let me know if you need to reschedule.",
    "Your Amazon order #402-1928374 has shipped and is expected to arrive on Thursday. Track your package in the app.",
    "Reminder: your electricity bill of Rs 1,240 is due on the 28th. Pay via the MyUtility app to avoid a late fee.",
    "Thanks for your purchase! Your receipt for $42.50 at Blue Bottle Coffee has been emailed to you.",
]


@pytest.mark.parametrize("text", SCAM_HOLDOUT)
def test_flags_holdout_scam_examples(text):
    result = classify_scam(text)
    assert result["is_scam"] is True
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["reason"]


@pytest.mark.parametrize("text", LEGIT_HOLDOUT)
def test_clears_holdout_legit_examples(text):
    result = classify_scam(text)
    assert result["is_scam"] is False
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["reason"]


def test_empty_text_is_unanalyzable():
    result = classify_scam("")
    assert result == {"is_scam": False, "confidence": 0.0, "reason": result["reason"]}
    assert "couldn't analyze" in result["reason"].lower()


def test_none_text_is_unanalyzable():
    result = classify_scam(None)
    assert "couldn't analyze" in result["reason"].lower()


def test_whitespace_only_text_is_unanalyzable():
    result = classify_scam("   \n\t  ")
    assert "couldn't analyze" in result["reason"].lower()


def test_gibberish_text_is_unanalyzable():
    result = classify_scam("asdkjf qpwoei zxcvbn qqqqq zzzzz xkjq")
    assert "couldn't analyze" in result["reason"].lower()


def test_symbols_only_text_is_unanalyzable():
    result = classify_scam("!!!! $$$$ #### 12345 6789 ****")
    assert "couldn't analyze" in result["reason"].lower()


def test_non_english_text_is_unanalyzable():
    result = classify_scam("Bonjour, comment allez-vous aujourd'hui? J'espère que tout va bien pour vous.")
    assert "couldn't analyze" in result["reason"].lower()


def test_extremely_long_text_does_not_crash():
    long_text = "URGENT click here to verify your account now bit.ly/scam " * 500
    result = classify_scam(long_text)
    assert isinstance(result["is_scam"], bool)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["reason"]
