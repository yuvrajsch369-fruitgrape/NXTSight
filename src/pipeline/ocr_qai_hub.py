"""EasyOCR, running through Qualcomm AI Hub's compiled detector/recognizer
ONNX models — the same models already genuinely compiled and profiled on
real Snapdragon X Elite hardware (see README's AI Hub section; job URLs
jp8em79kp/jgk2qynwg compile, jgly2kzj5/jgzlj7ko5 profile).

This module is intentionally optional and side-effect-free to import:
qai_hub_models is a heavier, more fragile dependency (it pulls plain
opencv-python, which conflicts with the opencv-python-headless the rest
of the app needs — see requirements.txt) than the core pipeline should
ever require. Every qai_hub_models import here is deferred into the
function that needs it, so a machine without qai_hub_models installed, or
without the two compiled .onnx files downloaded, never fails to import
ocr.py — it just can't use this path, and ocr.py falls back to the local
EasyOCR/PyTorch inference it already had.

The two ONNX sessions are created via runtime.create_inference_session()
— the exact same QNN/CPU auto-detecting session creation every other
model in this app uses. No second execution-provider-selection path.
"""

import logging
from pathlib import Path

import numpy as np
import torch

from src.pipeline.runtime import create_inference_session

logger = logging.getLogger("nxtsight")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DETECTOR_ONNX = _PROJECT_ROOT / "models" / "easyocr_detector" / "model.onnx"
RECOGNIZER_ONNX = _PROJECT_ROOT / "models" / "easyocr_recognizer" / "model.onnx"

DETECTOR_IMG_SHAPE = (608, 800)
RECOGNIZER_IMG_SHAPE = (64, 1000)

_app = None
_active_description = None


class _OnnxTensorCallable:
    """Wraps one onnxruntime session as a torch-tensor-in/torch-tensor-out
    callable — the interface EasyOCRApp expects in place of a
    torch.nn.Module. Session creation goes through the shared
    runtime.create_inference_session(), so this genuinely picks up QNN
    when available and CPU otherwise, logged exactly like every other
    model in the app.
    """

    def __init__(self, onnx_path: Path):
        self.session, self.description = create_inference_session(str(onnx_path))
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        array = x.detach().cpu().numpy().astype(np.float32)
        result = self.session.run([self._output_name], {self._input_name: array})[0]
        return torch.from_numpy(result)


def _build_app():
    """Build the EasyOCRApp once, lazily. Raises on any problem — missing
    qai_hub_models, missing .onnx files, a bad session — so the caller
    (ocr.py) can catch it and fall back, rather than this module ever
    crashing the app on its own.
    """
    global _app, _active_description

    if not DETECTOR_ONNX.exists() or not RECOGNIZER_ONNX.exists():
        raise FileNotFoundError(
            f"AI Hub EasyOCR models not found at {DETECTOR_ONNX} / {RECOGNIZER_ONNX}"
        )

    # Deferred import: qai_hub_models is optional, never required just to
    # run the app (see preflight.py's CORE_REQUIRED_MODULES, which does
    # not include it).
    from qai_hub_models.models.easyocr.app import EasyOCRApp

    detector = _OnnxTensorCallable(DETECTOR_ONNX)
    recognizer = _OnnxTensorCallable(RECOGNIZER_ONNX)

    _app = EasyOCRApp(
        detector=detector,
        recognizer=recognizer,
        detector_img_shape=DETECTOR_IMG_SHAPE,
        recognizer_img_shape=RECOGNIZER_IMG_SHAPE,
        lang_list=["en"],
    )
    _active_description = (
        f"EasyOCR detector: {detector.description} | EasyOCR recognizer: {recognizer.description}"
    )
    logger.info("AI Hub OCR path active — %s", _active_description)
    return _app


def _get_app():
    global _app
    if _app is None:
        _build_app()
    return _app


def extract_text_qai_hub(image) -> str:
    """Run a PIL Image through the AI-Hub-compiled EasyOCR pipeline.

    Raises on any failure — ocr.py is responsible for catching that and
    falling back to the local EasyOCR/PyTorch path. Never called with a
    file path; the caller has already opened and validated the image.
    """
    app = _get_app()
    _, texts, _confidences = app.predict_text_from_image(image)[0]
    return "\n".join(t.strip() for t in texts if t.strip()).strip()


def status() -> tuple:
    """(is_active, description) — whether the AI Hub OCR path is actually
    usable right now on this machine, without running inference. Used for
    the startup status log; never raises.
    """
    try:
        _get_app()
        return True, _active_description
    except Exception as e:
        return False, f"not active ({type(e).__name__}: {e})"
