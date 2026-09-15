"""OCR stage of the NXTSight pipeline: turn an image file into raw text.

Backed by EasyOCR for now — it's pure-Python and needs no system OCR binary,
so it runs anywhere without extra setup. Once we have access to a Qualcomm
AI Hub OCR model compiled for the Snapdragon NPU, swap the implementation
inside extract_text_from_image() to call that instead; the function
signature is the pipeline's stable contract, not the OCR backend.
"""

import os
from pathlib import Path

import easyocr
from PIL import Image, UnidentifiedImageError

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

    try:
        reader = _get_reader()
        lines = reader.readtext(path, detail=0)
    except Exception as e:
        return f"Error: OCR failed on '{path}' ({e})"

    text = "\n".join(line.strip() for line in lines if line.strip()).strip()
    if not text:
        return f"Error: no text detected in '{path}'"

    return text
