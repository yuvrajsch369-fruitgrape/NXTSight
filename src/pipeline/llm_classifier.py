"""A small local LLM (Qwen2.5-1.5B-Instruct, GGUF/Q4_K_M, via llama-cpp-python)
used as a second opinion when the fast classifier in engine.py is
genuinely unsure — not on every message.

Why this exists: the scam and spend classifiers (MiniLM embedding + a
small trained head) are fast (single-digit milliseconds) but were
trained on a few dozen to a couple hundred labeled examples each. A
1.5B-parameter instruction-tuned model, prompted with the task
description and a structured-output schema, brings broader world
knowledge to genuinely ambiguous cases at the cost of real latency
(measured on this dev machine: ~2.5-5.5s per call, even with Metal GPU
offload) — worth paying only when the fast path's own confidence says
it's actually unsure. See engine.py's `_maybe_escalate()` for exactly
when that is.

Deliberately generic, not scam- or spend-specific: `classify()` takes a
task's own labels and system prompt, so the same module serves both
registered tasks (and any future one) rather than duplicating the
llama-cpp-python plumbing per task.

Offline-safe: `Llama(model_path=...)` is given a literal local file
path, never a HuggingFace repo id — there is no download-on-first-use
behavior here to guard against (unlike the MiniLM/Whisper tokenizer
gotchas found earlier in this project), and this has been verified
against network_guard directly (see tests/test_llm_classifier.py).
"""

import atexit
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nxtsight")

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "llm" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_NAME = "Qwen2.5-1.5B-Instruct (Q4_K_M GGUF)"

_llm = None
_load_error: Optional[str] = None


def _ensure_loaded():
    """Load (and cache) the Llama instance. Any failure — llama-cpp-python
    not installed, the .gguf not downloaded, corrupted weights — raises,
    caught by status()/engine.py's escalation path, which just skips
    escalation rather than crashing a classification that already has a
    perfectly usable fast-path result."""
    global _llm, _load_error
    if _llm is not None:
        return _llm
    if _load_error is not None:
        raise RuntimeError(_load_error)

    if not MODEL_PATH.exists():
        _load_error = (
            f"model not downloaded — expected {MODEL_PATH} "
            "(see requirements.txt / README for the download command)"
        )
        raise RuntimeError(_load_error)

    try:
        from llama_cpp import Llama
    except ImportError as e:
        _load_error = f"llama-cpp-python not installed ({e})"
        raise RuntimeError(_load_error) from e

    try:
        loaded = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=1024,
            n_gpu_layers=-1,  # offload everything possible; ggml falls back to CPU layers on its own if no GPU backend
            verbose=False,
        )
    except Exception as e:
        _load_error = f"model failed to load ({type(e).__name__}: {e})"
        raise RuntimeError(_load_error) from e

    _llm = loaded
    # llama-cpp-python's own __del__ can fire after Python has already
    # torn down some of its C-extension state at interpreter shutdown,
    # printing a harmless but noisy "TypeError: 'NoneType' object is not
    # callable" traceback (a known llama-cpp-python quirk, not a bug in
    # this code) — closing explicitly, before that teardown starts,
    # avoids it.
    atexit.register(loaded.close)
    logger.info("LLM classifier ready: %s (local GGUF, no network)", MODEL_NAME)
    return _llm


def status() -> tuple:
    """(available, description) — same shape as whisper_cpp.status() /
    ocr_qai_hub.status(), but deliberately cheap: checks that
    llama-cpp-python imports and the .gguf file exists, without actually
    loading ~1GB of weights just to answer "is this available". Unlike
    OCR/MiniLM/Whisper (which every classification needs and are fast to
    load), the LLM is used rarely — only on genuinely ambiguous scam/spend
    calls — so paying a real ~5-25s load cost at every app startup just to
    report status, for a feature most sessions may never trigger, isn't
    the right tradeoff. The actual load happens lazily, once, inside
    classify() the first time escalation genuinely fires — see
    _ensure_loaded() above."""
    if _llm is not None:
        return True, f"{MODEL_NAME} — loaded, local, offline"
    if _load_error is not None:
        return False, f"not active ({_load_error})"

    try:
        import llama_cpp  # noqa: F401 — import check only, no model load
    except ImportError as e:
        return False, f"not active (llama-cpp-python not installed: {e})"

    if not MODEL_PATH.exists():
        return False, f"not active (model not downloaded — expected {MODEL_PATH})"

    return True, f"{MODEL_NAME} — available, loads on first genuinely ambiguous call"


def classify(text: str, labels: list, system_prompt: str, timeout_s: float = 20.0) -> dict:
    """Ask the LLM to pick one of `labels` for `text`, given `system_prompt`
    (the task's own description of the categories/decision). Returns
    {"label": one of `labels`, "confidence": float 0..1, "reason": str}.

    Raises on any failure (model unavailable, malformed/out-of-schema
    output, timeout) — the caller (engine.py) treats that as "escalation
    didn't work this time" and keeps the fast path's own result, never
    as a crash.
    """
    llm = _ensure_loaded()

    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": labels},
            "confidence": {
                "type": "number",
                "description": "a decimal between 0.0 and 1.0 — never a percentage",
            },
            "reason": {"type": "string", "description": "one short sentence"},
        },
        "required": ["label", "confidence", "reason"],
    }

    start = time.time()
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Message: "{text}"\n\nRespond with JSON only.'},
        ],
        response_format={"type": "json_object", "schema": schema},
        max_tokens=150,
        temperature=0.1,
    )
    elapsed = time.time() - start
    if elapsed > timeout_s:
        logger.warning("LLM classification took %.1fs (over the %.0fs soft budget)", elapsed, timeout_s)

    content = result["choices"][0]["message"]["content"]
    parsed = json.loads(content)  # malformed JSON raises here — caught by the caller

    if parsed["label"] not in labels:
        raise ValueError(f"LLM returned an out-of-schema label: {parsed['label']!r}")

    confidence = float(parsed["confidence"])
    if confidence > 1.0:  # models sometimes answer in 0-100 despite the prompt; normalize rather than reject
        confidence = confidence / 100.0
    confidence = max(0.0, min(1.0, confidence))

    logger.info("LLM classification: %r -> %s (%.2f, %.1fs)", text[:60], parsed["label"], confidence, elapsed)
    return {"label": parsed["label"], "confidence": confidence, "reason": str(parsed.get("reason", ""))}
