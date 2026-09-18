"""Call Shield speech-to-text, running Whisper-tiny's encoder through
Qualcomm AI Hub's compiled ONNX model — decoding stays on qai_hub_models'
own, unmodified PyTorch decode loop.

Deliberately encoder-only, on purpose, not a shortcut: Whisper's decoder
is the autoregressive, KV-cache-dependent part — many tensor inputs
threaded through a token-by-token generation loop, genuinely risky to
hand-wire through ONNX Runtime with real confidence (get one cache index
or one mask value wrong and it can produce a transcript that reads as
plausible but is quietly wrong — worse than the honest local fallback).
The encoder is a single forward pass with no loop, the same shape as
MiniLM-v2's export. This module does not reimplement Whisper's decode
loop at all: it calls qai_hub_models' own HfWhisperApp with only the
encoder's execution backend swapped for an ONNX Runtime session (via the
shared runtime.create_inference_session() — same QNN/CPU auto-detect
every other model in this app uses) — everything autoregressive is still
Qualcomm's own, already-shipped, untouched code.

Verified numerically before wiring this in as the preferred path: the
ONNX-encoder + PyTorch-decoder hybrid produces byte-identical
transcriptions to the all-PyTorch baseline on all 4 bundled sample call
recordings — see the main README's Call Shield section for that
comparison. This module is intentionally optional and side-effect-free to
import — every qai_hub_models/transformers import is deferred, so a
machine without them installed, or without the exported .onnx present,
never fails to import this module, just can't use this path — Call
Shield's existing extract_text_from_audio() (OpenAI Whisper "tiny.en" via
plain PyTorch) is always available as the fallback.
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch

from src.pipeline.runtime import create_inference_session

logger = logging.getLogger("nxtsight")

# HuggingFace's from_pretrained() makes a real network call by default to
# check for a newer revision, even with everything cached locally — a
# confirmed, real violation of "runs fully offline" once network_guard is
# installed (the exact same issue found and fixed for MiniLM-v2). Unlike
# text_encoder.py, this module doesn't control qai_hub_models' own
# from_pretrained() calls directly, so the fix has to be the environment
# variable HuggingFace itself documents for fully-offline use, set before
# any transformers/qai_hub_models import happens.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENCODER_ONNX = _PROJECT_ROOT / "models" / "whisper_tiny_encoder" / "model.onnx"
HF_MODEL_ID = "openai/whisper-tiny"

_app = None
_active_description = None


class _OnnxEncoderCallable:
    """Wraps the encoder's ONNX session as the callable HfWhisperApp
    expects in place of a torch.nn.Module: input_features in, a tuple of
    per-layer cross-attention (k, v) tensors out — AI Hub's own monkey-
    patched encoder output shape, unchanged by running it through ONNX
    Runtime instead of PyTorch.
    """

    def __init__(self, onnx_path: Path, output_names: list):
        self.session, self.description = create_inference_session(str(onnx_path))
        self._input_name = self.session.get_inputs()[0].name
        self._output_names = output_names

    def __call__(self, input_features: torch.Tensor):
        array = input_features.detach().cpu().numpy().astype(np.float32)
        outputs = self.session.run(self._output_names, {self._input_name: array})
        return tuple(torch.from_numpy(o) for o in outputs)


def _build_app():
    """Build the HfWhisperApp once, lazily. Raises on any problem —
    missing qai_hub_models/transformers, missing .onnx, a bad session —
    so the caller (stt.py) can catch it and fall back, rather than this
    module ever crashing the app on its own.
    """
    global _app, _active_description

    if not ENCODER_ONNX.exists():
        raise FileNotFoundError(f"AI Hub Whisper encoder not found at {ENCODER_ONNX}")

    # Deferred imports: qai_hub_models/transformers are optional, never
    # required just to run the app (see preflight.py's CORE_REQUIRED_MODULES
    # — transformers is required for MiniLM-v2, but qai_hub_models isn't).
    from qai_hub_models.models._shared.hf_whisper.app import HfWhisperApp
    from qai_hub_models.models.whisper_tiny.model import WhisperTiny

    whisper = WhisperTiny.from_pretrained()  # loads the matching (unmodified) decoder too
    encoder = _OnnxEncoderCallable(ENCODER_ONNX, whisper.encoder.get_output_names())

    _app = HfWhisperApp(encoder, whisper.decoder, HF_MODEL_ID)
    _active_description = f"Whisper-tiny encoder: {encoder.description}"
    logger.info("AI Hub Whisper encoder path active — %s", _active_description)
    return _app


def _get_app():
    global _app
    if _app is None:
        _build_app()
    return _app


def transcribe_qai_hub(audio: np.ndarray, sample_rate: int) -> str:
    """Run a mono float32 audio array through the AI-Hub-encoder /
    PyTorch-decoder hybrid pipeline. Raises on any failure — stt.py is
    responsible for catching that and falling back to the local Whisper
    "tiny.en" path.
    """
    app = _get_app()
    return app.transcribe(audio, audio_sample_rate=sample_rate).strip()


def status() -> tuple:
    """(is_active, description) — whether the AI Hub Whisper encoder path
    is actually usable right now on this machine, without running
    inference. Used for the startup status log; never raises.
    """
    try:
        _get_app()
        return True, _active_description
    except Exception as e:
        return False, f"not active ({type(e).__name__}: {e})"
