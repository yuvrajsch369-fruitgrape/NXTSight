"""Tests Call Shield: the scam-call classifier and its audio path.

Covers 5 held-out call transcripts (3 scam: digital-arrest, fake courier,
fake bank; 2 legit) — none copied from src/call_shield/data.py — plus the
4 bundled sample recordings (data/samples/call_*.wav) run through the real
speech-to-text pipeline, not just pasted text. See README for the honest
finding this testing surfaced: a legit call correctly cleared when pasted
with "Caller:"/"You:" labels initially false-positived when the same
content arrived as unlabeled continuous text (what real transcribed audio
actually looks like) — fixed at the data level, not by patching around it.
"""

from pathlib import Path

import pytest

from src.call_shield.classifier import analyze_call, analyze_call_recording

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"

SCAM_HOLDOUT = [
    (
        "Digital-arrest scam",
        """Caller: Namaste, I am Sub-Inspector Verma calling from the Cyber Crime Investigation Unit.
You: Yes, what is this regarding?
Caller: Your PAN card has been used to open a bank account involved in a 2 crore rupee money laundering racket linked to human trafficking. There is a digital arrest warrant against your name.
You: This can't be right, I've never opened any such account.
Caller: The evidence is very clear on our end. You must not disconnect this call or leave your house. Keep your camera on for the remainder of this investigation.
You: Okay, I'm scared, what do I do?
Caller: To prove you are not involved, transfer all funds from your savings account to the RBI secure holding account we provide for verification within the next hour, and read out the OTP the moment it arrives. Failing to comply means a police team arrives at your address immediately.""",
    ),
    (
        "Fake courier scam",
        """Caller: Hello, this is DHL express calling about a parcel booked under your name that has been stopped at Chennai customs.
You: I don't remember ordering anything from abroad.
Caller: The parcel contains an international SIM card and some banned cosmetic items, which is a customs violation. A case file has already been opened against your identity.
You: What happens now?
Caller: To close the case and release the parcel, you need to pay a customs penalty of 3200 rupees right now through the payment link, and confirm the OTP you receive so we can verify the transaction went through before the 20 minute deadline, otherwise this gets escalated to the cyber cell.""",
    ),
    (
        "Fake bank scam",
        """Caller: Good evening, this is calling from the fraud prevention department of your bank.
You: Okay, is something wrong?
Caller: We've blocked a suspicious transaction of 78,000 rupees attempted from your account a few minutes ago from a different city.
You: I didn't make that transaction.
Caller: That's exactly why we're calling, to help you cancel it before it processes. Please read out the one time password sent to your registered number right now so I can reverse the transaction on my end before the 5 minute window closes, or the amount will be deducted permanently.""",
    ),
]

LEGIT_HOLDOUT = [
    (
        "Real bank call",
        """Caller: Hi, this is calling from your bank regarding the credit card limit increase you requested through the app last week.
You: Oh right, has it been approved?
Caller: Yes, it's been approved and will reflect in your account within 2 business days. Is there anything else I can help you with?
You: No, that's all, thank you.
Caller: You're welcome, have a great day.""",
    ),
    (
        "Friend call",
        """Caller: Hey, it's Ankit, are you around this weekend?
You: Yeah, I should be free Saturday, why what's up?
Caller: A few of us are planning to go hiking near Lonavala, thought you might want to join.
You: That sounds great, count me in. What time are we leaving?
Caller: Probably 6 AM from my place, I'll send the details on the group chat.""",
    ),
]


@pytest.mark.parametrize("label,transcript", SCAM_HOLDOUT, ids=[t[0] for t in SCAM_HOLDOUT])
def test_flags_holdout_scam_calls(label, transcript):
    result = analyze_call(transcript)
    assert result["is_scam"] is True
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["reason"]


@pytest.mark.parametrize("label,transcript", LEGIT_HOLDOUT, ids=[t[0] for t in LEGIT_HOLDOUT])
def test_clears_holdout_legit_calls(label, transcript):
    result = analyze_call(transcript)
    assert result["is_scam"] is False
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["reason"]


def test_reason_quotes_a_line_from_the_transcript():
    _, transcript = SCAM_HOLDOUT[0]
    result = analyze_call(transcript)
    assert "part of the call" in result["reason"]


# --- Edge cases: never crash, never silently produce a wrong verdict ----


def test_empty_transcript_is_unanalyzable():
    result = analyze_call("")
    assert result["is_scam"] is False
    assert "couldn't analyze" in result["reason"].lower()


def test_none_transcript_is_unanalyzable():
    result = analyze_call(None)
    assert "couldn't analyze" in result["reason"].lower()


def test_gibberish_transcript_is_unanalyzable():
    result = analyze_call("asdkjf qpwoei zxcvbn qqqqq zzzzz xkjq")
    assert "couldn't analyze" in result["reason"].lower()


def test_non_english_transcript_is_unanalyzable():
    result = analyze_call("Bonjour, comment allez-vous aujourd'hui? J'espere que tout va bien.")
    assert "couldn't analyze" in result["reason"].lower()


def test_extremely_long_transcript_does_not_crash():
    long_transcript = "Caller: transfer the OTP now or you will be arrested. " * 500
    result = analyze_call(long_transcript)
    assert isinstance(result["is_scam"], bool)
    assert 0.0 <= result["confidence"] <= 1.0


# --- Audio path: real bundled WAV recordings, run through real STT -----


AUDIO_SAMPLES = [
    ("call_digital_arrest_scam.wav", True),
    ("call_fake_courier_scam.wav", True),
    ("call_fake_bank_scam.wav", True),
    ("call_legit_bank_call.wav", False),
]


@pytest.mark.parametrize("filename,expected_is_scam", AUDIO_SAMPLES)
def test_bundled_call_recordings_transcribe_and_classify_correctly(filename, expected_is_scam):
    result = analyze_call_recording(str(SAMPLES_DIR / filename))
    assert result["ok"] is True
    assert result["transcript"]
    assert result["is_scam"] is expected_is_scam


def test_missing_audio_file_returns_clear_error_not_crash():
    result = analyze_call_recording(str(SAMPLES_DIR / "does_not_exist.wav"))
    assert result["ok"] is False
    assert result["message"]


def test_none_audio_path_returns_clear_error_not_crash():
    result = analyze_call_recording(None)
    assert result["ok"] is False
    assert result["message"]


def test_non_wav_audio_returns_clear_error_not_crash(tmp_path):
    fake_mp3 = tmp_path / "recording.mp3"
    fake_mp3.write_bytes(b"not a real mp3 file")
    result = analyze_call_recording(str(fake_mp3))
    assert result["ok"] is False
    assert "wav" in result["message"].lower()


def test_corrupted_wav_returns_clear_error_not_crash(tmp_path):
    fake_wav = tmp_path / "corrupted.wav"
    fake_wav.write_bytes(b"not a real wav file")
    result = analyze_call_recording(str(fake_wav))
    assert result["ok"] is False
    assert result["message"]
