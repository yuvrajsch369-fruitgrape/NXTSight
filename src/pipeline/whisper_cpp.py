"""Call Shield speech-to-text via whisper.cpp (GGUF) — an optional,
genuinely hardware-accelerated path this project didn't have before.

Unlike src/pipeline/whisper_qai_hub.py (Qualcomm AI Hub's compiled ONNX
encoder — the fastest path when actually on a Snapdragon NPU, but only
there), this module uses whisper.cpp's own hardware dispatch, via
`pywhispercpp` (a real Python binding to the upstream ggml-org/whisper.cpp
project, not a reimplementation). ggml — whisper.cpp's compute backend —
auto-detects Metal on Apple platforms, and can be built with CUDA or
Vulkan support elsewhere; none of that vendor-specific logic lives here,
this module just calls into whatever ggml was built with.

Priority in src/pipeline/stt.py, in order:
    1. AI Hub-compiled encoder (whisper_qai_hub) — fastest when genuinely
       on a Snapdragon NPU (27.8ms, measured — see README).
    2. This module — Metal/CUDA/Vulkan/CPU via whisper.cpp, real GPU
       acceleration on machines that have no Snapdragon NPU at all
       (this dev Mac included — previously pure CPU PyTorch, now
       genuinely running on Metal).
    3. Plain openai-whisper PyTorch (stt.py's own fallback) — the final,
       always-available safety net.

Import is deferred: `pywhispercpp` is optional, not in the base
requirements (see requirements.txt) — a machine without it installed
falls through to step 3 above without this module ever failing loudly.

Genuinely verified, not assumed: transcripts from this path were
diffed against the existing all-PyTorch baseline on all four bundled
sample recordings before this became a preferred path — see
tests/test_whisper_cpp.py and the README's Call Shield section.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("nxtsight")

MODEL_NAME = "tiny.en"

_model = None
_model_load_error: Optional[str] = None
_backend_description: Optional[str] = None


def _models_dir() -> Path:
    from platformdirs import user_data_dir

    return Path(user_data_dir("pywhispercpp")) / "models"


def _ensure_model():
    """Load (and cache) the whisper.cpp Model, downloading the GGUF
    weights on first use if they aren't already cached locally. Any
    failure — pywhispercpp not installed, no network for the one-time
    download, a corrupted cache — raises, caught by status()/the caller
    in stt.py, which falls back to the next path rather than crashing."""
    global _model, _model_load_error, _backend_description
    if _model is not None:
        return _model
    if _model_load_error is not None:
        raise RuntimeError(_model_load_error)

    # CONFIRMED, REPRODUCIBLE: pywhispercpp and llama-cpp-python (see
    # llm_classifier.py) each bundle their own independently-built copy of
    # libggml — same library names, different builds. macOS's dynamic
    # linker treats same-named dylibs as one shared identity: whichever
    # package's copy loads *first* in the process wins that identity, and
    # if the two builds' symbol sets don't fully agree, the second package
    # to load can fail with a real dlopen "Symbol not found" error — not a
    # hypothetical, reproduced directly: `import pywhispercpp; import
    # llama_cpp` raises `Symbol not found: (_ggml_dsv4_hc_comb)`, while the
    # reverse order works cleanly. This import, before pywhispercpp's own,
    # makes llama_cpp's copy win that race whenever it's installed — a
    # best-effort, silently-skipped-if-absent import specifically so this
    # module's own load order doesn't depend on which of the two a caller
    # happened to touch first. This is a real fix for the versions pinned
    # today (llama-cpp-python 0.3.35 / pywhispercpp 1.5.1), verified by
    # tests/test_llm_and_whisper_cpp_coexist.py — not guaranteed to survive
    # every future version bump of either package, since the actual root
    # cause (two independently-built ggml copies sharing dylib names) isn't
    # something this file can fix at the source.
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        pass

    try:
        from pywhispercpp.model import Model
        from pywhispercpp.utils import download_model
    except ImportError as e:
        _model_load_error = f"pywhispercpp not installed ({e})"
        raise RuntimeError(_model_load_error) from e

    try:
        models_dir = _models_dir()
        model_path = models_dir / f"ggml-{MODEL_NAME}.bin"
        if not model_path.exists():
            logger.info("whisper.cpp: no cached %s model — downloading once (~75MB)...", MODEL_NAME)
            download_model(MODEL_NAME, download_dir=str(models_dir))

        loaded = Model(
            MODEL_NAME,
            models_dir=str(models_dir),
            print_progress=False,
            print_realtime=False,
            redirect_whispercpp_logs_to=None,
        )
    except Exception as e:
        _model_load_error = f"whisper.cpp model failed to load ({type(e).__name__}: {e})"
        raise RuntimeError(_model_load_error) from e

    _model = loaded
    _backend_description = "whisper.cpp / GGUF (ggml hardware backend — Metal, CUDA, Vulkan, or CPU, auto-detected at build time)"
    logger.info("whisper.cpp Model ready: %s", _backend_description)
    return _model


def status() -> Tuple[bool, str]:
    """(active, description) — same shape as whisper_qai_hub.status(),
    for the "which real acceleration is active" reporting in app.py /
    backend/main.py."""
    try:
        _ensure_model()
        return True, _backend_description
    except Exception as e:
        return False, f"not active ({e})"


def transcribe_whisper_cpp(audio: np.ndarray, sample_rate: int) -> str:
    """audio: mono float32 PCM. whisper.cpp itself requires 16kHz —
    src/pipeline/stt.py already resamples to that before any transcribe
    path is called, so this never resamples twice."""
    if sample_rate != 16000:
        raise ValueError(f"whisper.cpp requires 16kHz audio, got {sample_rate}Hz")

    model = _ensure_model()
    segments = model.transcribe(audio.astype(np.float32))
    return " ".join(segment.text.strip() for segment in segments).strip()
