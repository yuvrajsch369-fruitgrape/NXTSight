from pathlib import Path

from src.pipeline.stt import extract_text_from_audio

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_transcribes_sample_call_recording():
    text = extract_text_from_audio(str(SAMPLES_DIR / "call_digital_arrest_scam.wav"))
    assert not text.startswith("Error")
    assert "arrest" in text.lower() or "otp" in text.lower()


def test_rejects_none_path_without_crashing():
    result = extract_text_from_audio(None)
    assert result.startswith("Error")


def test_rejects_non_path_types_without_crashing():
    for bad_path in [123, 3.14, ["not", "a", "path"], object()]:
        result = extract_text_from_audio(bad_path)
        assert result.startswith("Error")


def test_missing_file_returns_error_not_crash():
    result = extract_text_from_audio(str(SAMPLES_DIR / "does_not_exist.wav"))
    assert result.startswith("Error")


def test_non_wav_extension_returns_clear_error(tmp_path):
    fake = tmp_path / "recording.mp3"
    fake.write_bytes(b"not real audio")
    result = extract_text_from_audio(str(fake))
    assert result.startswith("Error")
    assert "wav" in result.lower()


def test_corrupted_wav_returns_error_not_crash(tmp_path):
    fake = tmp_path / "corrupted.wav"
    fake.write_bytes(b"not a real wav file")
    result = extract_text_from_audio(str(fake))
    assert result.startswith("Error")


def test_silent_wav_returns_no_speech_error(tmp_path):
    import wave

    silent_path = tmp_path / "silent.wav"
    with wave.open(str(silent_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)  # 1 second of digital silence

    result = extract_text_from_audio(str(silent_path))
    assert result.startswith("Error")
