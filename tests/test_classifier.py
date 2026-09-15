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

# A second, larger stress-test batch — deliberately covering scam sub-types
# and legit categories thin in both training data and SCAM_HOLDOUT above:
# romance/pig-butchering, tech support, fake charity, tax refund, SIM-swap,
# insurance renewal, fake legal threats, crypto phishing, code-forwarding,
# subscription scares, fake loans/jobs/health offers, and — on the legit
# side — school/appointment/calendar notices, security alerts, government
# status updates, and everyday personal/work correspondence.
SCAM_STRESS = [
    "Hi dear, I know we just met but I feel a real connection. I'm stuck at customs and need $800 to release my cargo shipment, I'll pay you back double once it clears, please help me now.",
    "This is Microsoft Support. We've detected a virus on your computer. Do not turn it off. Call this number immediately and allow remote access so our technician can fix it for you.",
    "On behalf of Global Relief Foundation, we're collecting emergency donations for flood victims. Send whatever you can via UPI to this number right now, every minute counts.",
    "Income Tax Department: You are eligible for a refund of Rs 15,750. Click the link and enter your bank details within 24 hours to receive it before the window closes.",
    "URGENT: Your SIM will be blocked in 2 hours due to KYC mismatch. Forward the OTP you just received to this number to keep your number active.",
    "Your car insurance policy has lapsed. Renew immediately by clicking here and entering your card details to avoid a coverage gap and legal penalty.",
    "This is an official notice from the Cyber Crime Cell. A case has been filed against your Aadhaar number for money laundering. Call immediately or a warrant will be issued.",
    "Your crypto wallet has been flagged for suspicious activity. Verify your seed phrase at this link within 30 minutes to prevent permanent account freeze.",
    "Forward the 6-digit code you received to activate your WhatsApp on this new device, otherwise your account will be permanently disabled.",
    "Your Netflix payment failed. Update your card details within 12 hours at this link or your subscription and saved data will be deleted.",
    "Pre-approved: Rs 5,00,000 personal loan at 0% interest, no documents needed. Just pay a Rs 1,499 processing fee to get the amount disbursed today.",
    "Congratulations, you've been shortlisted for a software engineer position in Canada with a $95,000 salary. Pay a Rs 12,000 visa processing fee to confirm your offer letter.",
    "New vaccine booster available exclusively for you. Register now and pay a Rs 199 slot-booking fee before all appointments are taken.",
    "We could not verify your identity for your recent parcel. Confirm your date of birth, address and card number here within 6 hours or the parcel will be destroyed.",
    "Your Instagram account has been reported for violating community guidelines and will be permanently deleted in 24 hours. Verify your identity now by logging in at this link.",
]

LEGIT_STRESS = [
    "Reminder: Parent-teacher meeting for Class 5B is scheduled for Friday at 4 PM in the school auditorium.",
    "Your dentist appointment with Dr. Mehta is confirmed for tomorrow at 11:30 AM. Reply STOP to cancel.",
    "You've been invited to 'Q3 Planning Sync' on Thursday 10:00-11:00 AM. Accept or decline in your calendar.",
    "Weather alert: Heavy rainfall expected in your area today between 3 PM and 7 PM. Carry an umbrella.",
    "A sign-in attempt was made on your Google account from a new Windows device in Mumbai. If this was you, no action is needed.",
    "Hi, following up on your support ticket #48213 — we've resolved the issue with your printer driver. Let us know if you need anything else.",
    "Your Blue Dart shipment AWB 88213456 is out for delivery and should arrive by 6 PM today.",
    "Hey, can you review the Q3 budget doc before our call tomorrow morning? I've left a few comments.",
    "Your passport renewal application (file no. PP2024981) has been received and is under processing. Track status on the official portal.",
    "Thank you for subscribing to The Weekly Digest. You'll receive your first issue this Sunday.",
    "Your table for 4 at Olive Garden is confirmed for 7:30 PM tonight. See you soon!",
    "Reminder: Your car service is due next week. Call the service center to book a slot at your convenience.",
    "Your electricity meter reading has been recorded. This month's estimated bill is Rs 1,180, final bill will be generated on the 3rd.",
    "The document you requested, 'Employee Handbook 2025', has been shared with you on the company drive.",
    "Your gym membership renews automatically on the 1st of next month at the same rate as before. No action needed.",
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


@pytest.mark.parametrize("text", SCAM_STRESS)
def test_flags_stress_scam_examples(text):
    result = classify_scam(text)
    assert result["is_scam"] is True
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.parametrize("text", LEGIT_STRESS)
def test_clears_stress_legit_examples(text):
    result = classify_scam(text)
    assert result["is_scam"] is False
    assert 0.0 <= result["confidence"] <= 1.0


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
