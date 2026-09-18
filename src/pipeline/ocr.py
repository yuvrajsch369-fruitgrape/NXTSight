"""OCR stage of the NXTSight pipeline: turn an image file into raw text.

Prefers running through Qualcomm AI Hub's compiled EasyOCR detector +
recognizer (src/pipeline/ocr_qai_hub.py) — the same ONNX models already
genuinely compiled and profiled on real Snapdragon X Elite hardware, run
through the shared QNN/CPU-aware runtime.create_inference_session(), same
as every other model in this app. Falls back to EasyOCR's own PyTorch
inference (pure-Python, needs no system OCR binary, runs anywhere) if
that path isn't available for any reason — qai_hub_models not installed,
the compiled .onnx files not present, or any other failure. Either way
extract_text_from_image()'s contract never changes: real text, or a clear
"Error: ..." string, never a crash.
"""

import logging
import os
from pathlib import Path

import easyocr
from PIL import Image, UnidentifiedImageError

from src.pipeline import ocr_qai_hub

logger = logging.getLogger("nxtsight")

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def extract_text_from_image(path) -> str:
    if not isinstance(path, (str, os.PathLike)):
        return f"Error: expected a file path, got {type(path).__name__}"

    if not Path(path).is_file():
        return f"Error: no image found at '{path}'"

    try:
        with Image.open(path) as image:
            image.verify()
    except UnidentifiedImageError:
        return f"Error: '{path}' is not a readable image (corrupted or unsupported format)"
    except Exception as e:
        return f"Error: could not open '{path}' ({e})"

    text = None
    try:
        with Image.open(path) as image:
            text = ocr_qai_hub.extract_text_qai_hub(image.convert("RGB"))
        if not text:
            raise ValueError("AI Hub OCR path returned no text")
    except Exception as e:
        logger.warning(
            "AI Hub OCR path unavailable (%s: %s) — falling back to local EasyOCR/PyTorch inference.",
            type(e).__name__,
            e,
        )
        try:
            reader = _get_reader()
            lines = reader.readtext(path, detail=0)
        except Exception as e:
            return f"Error: OCR failed on '{path}' ({e})"
        text = "\n".join(line.strip() for line in lines if line.strip()).strip()

    if not text:
        return f"Error: no text detected in '{path}'"

    return text
