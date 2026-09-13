from pathlib import Path

from src.pipeline.ocr import extract_text_from_image

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_extracts_text_from_sample_screenshot():
    text = extract_text_from_image(str(SAMPLES_DIR / "scam_lottery_win.png"))
    assert "Error" not in text
    assert "KBC" in text or "prize" in text.lower()


def test_missing_file_returns_error_not_crash():
    result = extract_text_from_image(str(SAMPLES_DIR / "does_not_exist.png"))
    assert result.startswith("Error")


def test_corrupted_file_returns_error_not_crash(tmp_path):
    bad_file = tmp_path / "corrupted.png"
    bad_file.write_bytes(b"not a real image")
    result = extract_text_from_image(str(bad_file))
    assert result.startswith("Error")


def test_blank_image_returns_no_text_error(tmp_path):
    from PIL import Image

    blank_path = tmp_path / "blank.png"
    Image.new("RGB", (200, 100), (255, 255, 255)).save(blank_path)
    result = extract_text_from_image(str(blank_path))
    assert result.startswith("Error")
