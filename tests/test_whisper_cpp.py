"""Tests for src/pipeline/whisper_cpp.py — the optional whisper.cpp/GGUF
speech-to-text tier. `pywhispercpp` is optional, not in the base
requirements (see requirements.txt), so these tests branch on
whisper_cpp.status() the same way tests/test_receipt_parser.py branches
on ocr_qai_hub.status() — honestly correct whether or not it's installed,
rather than assuming one environment.
"""

from pathlib import Path

import numpy as np
import pytest

from src.pipeline import whisper_cpp
from src.pipeline.stt import _load_wav_as_float32

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_status_returns_a_real_tuple_either_way():
    active, description = whisper_cpp.status()
    assert isinstance(active, bool)
    assert isinstance(description, str) and description


def test_wrong_sample_rate_rejected_without_crashing():
    audio = np.zeros(8000, dtype=np.float32)
    with pytest.raises(ValueError, match="16kHz"):
        whisper_cpp.transcribe_whisper_cpp(audio, 8000)


def test_transcribes_real_sample_when_active():
    active, _ = whisper_cpp.status()
    if not active:
        pytest.skip("pywhispercpp not installed in this environment — see requirements.txt")

    audio = _load_wav_as_float32(str(SAMPLES_DIR / "call_digital_arrest_scam.wav"))
    text = whisper_cpp.transcribe_whisper_cpp(audio, 16000)

    assert text
    lowered = text.lower()
    assert "arrest" in lowered or "otp" in lowered


def test_classifier_reaches_the_same_verdict_on_all_four_samples_when_active():
    """Not byte-for-byte identical to the PyTorch/ONNX baseline — this is
    a different model artifact (GGUF-quantized), so exact transcript
    equality isn't the right bar. What matters is that Call Shield's
    classifier still reaches the correct verdict on every bundled
    sample — verified here, not assumed."""
    active, _ = whisper_cpp.status()
    if not active:
        pytest.skip("pywhispercpp not installed in this environment — see requirements.txt")

    from src.call_shield.classifier import analyze_call

    expected_scam = {
        "call_digital_arrest_scam.wav": True,
        "call_fake_courier_scam.wav": True,
        "call_fake_bank_scam.wav": True,
        "call_legit_bank_call.wav": False,
    }
    for filename, expected in expected_scam.items():
        audio = _load_wav_as_float32(str(SAMPLES_DIR / filename))
        text = whisper_cpp.transcribe_whisper_cpp(audio, 16000)
        result = analyze_call(text)
        assert result["is_scam"] is expected, f"{filename}: got {result}"


def test_import_error_reported_gracefully_not_raised(monkeypatch):
    """Simulates pywhispercpp genuinely missing — status() must degrade
    to (False, "not active (...)"), never raise."""
    whisper_cpp._model = None
    whisper_cpp._model_load_error = "pywhispercpp not installed (simulated for this test)"
    try:
        active, description = whisper_cpp.status()
        assert active is False
        assert "not active" in description
    finally:
        whisper_cpp._model_load_error = None
