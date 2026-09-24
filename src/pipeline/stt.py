"""Speech-to-text stage of the NXTSight pipeline: turn a call recording into text.

Three-tier priority, fastest-and-most-specific first, always-available
last:

    1. Qualcomm AI Hub's compiled Whisper-tiny encoder
       (src/pipeline/whisper_qai_hub.py) — the fastest path, but only on
       a genuine Snapdragon NPU (27.8ms, measured — see README). Decoding
       stays on qai_hub_models' own unmodified PyTorch loop; nothing
       autoregressive runs through ONNX.
    2. whisper.cpp / GGUF (src/pipeline/whisper_cpp.py) — real hardware
       acceleration on whatever this machine actually has (Metal, CUDA,
       or Vulkan, auto-detected by ggml at build time), for every machine
       that isn't a Snapdragon NPU. Optional — not in the base
       requirements, see requirements.txt.
    3. OpenAI's Whisper ("tiny.en", fully on-device via PyTorch) — the
       final, always-available fallback, the same way OCR falls back to
       EasyOCR's own PyTorch inference.

Either way the contract never changes: a real transcript, or a clear
"Error: ..." string, never a crash.

Deliberately WAV-only: Whisper's own audio loader (either path) shells out
to a system `ffmpeg` binary for other formats, which isn't installed on
every machine (this dev Mac included) and would be a silent new system
dependency. WAV files are decoded here with the standard-library `wave`
module and resampled with plain numpy instead — no ffmpeg, no extra
dependency, and it covers the realistic case (most recording/voice-memo
apps, and macOS's own `say -o file.wav`, export WAV natively).
"""

import logging
import os
import wave
from pathlib import Path

import numpy as np

from src.pipeline import whisper_cpp, whisper_qai_hub

logger = logging.getLogger("nxtsight")

_model = None
TARGET_SAMPLE_RATE = 16000
MODEL_NAME = "tiny.en"


def _get_model():
    global _model
    if _model is None:
        import whisper

        _model = whisper.load_model(MODEL_NAME)
    return _model


def _load_wav_as_float32(path) -> np.ndarray:
    """Read a WAV file into a mono float32 array at TARGET_SAMPLE_RATE,
    using only the standard library and numpy — no ffmpeg."""
    with wave.open(str(path), "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        raw = wav_file.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(
            f"unsupported WAV sample width ({sample_width * 8}-bit) — only 16-bit PCM WAV is supported"
        )

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if frame_rate != TARGET_SAMPLE_RATE and len(audio) > 1:
        duration = len(audio) / frame_rate
        n_target = max(1, int(duration * TARGET_SAMPLE_RATE))
        x_old = np.linspace(0, duration, len(audio))
        x_new = np.linspace(0, duration, n_target)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)

    return audio


def extract_text_from_audio(path) -> str:
    if not isinstance(path, (str, os.PathLike)):
        return f"Error: expected a file path, got {type(path).__name__}"

    if not Path(path).is_file():
        return f"Error: no audio file found at '{path}'"

    if Path(path).suffix.lower() != ".wav":
        return (
            f"Error: '{path}' isn't a WAV file. This build reads WAV audio directly "
            "(no ffmpeg installed); convert the recording to .wav and try again."
        )

    try:
        audio = _load_wav_as_float32(path)
    except (wave.Error, EOFError) as e:
        # EOFError alongside wave.Error, not just the latter: the stdlib
        # `wave` module raises a bare EOFError (with an empty message) for
        # a file too short to even contain a valid header — a corrupted or
        # truncated WAV in practice, same user-facing story either way.
        detail = str(e) or type(e).__name__
        return f"Error: '{path}' is not a readable WAV file (corrupted or unsupported format) ({detail})"
    except Exception as e:
        detail = str(e) or type(e).__name__
        return f"Error: could not read '{path}' ({detail})"

    if len(audio) == 0:
        return f"Error: '{path}' contains no audio data"

    text = None
    try:
        text = whisper_qai_hub.transcribe_qai_hub(audio, TARGET_SAMPLE_RATE)
    except Exception as e:
        logger.info(
            "AI Hub Whisper encoder path unavailable (%s: %s) — trying whisper.cpp/GGUF next.",
            type(e).__name__,
            e,
        )
        try:
            text = whisper_cpp.transcribe_whisper_cpp(audio, TARGET_SAMPLE_RATE)
        except Exception as e:
            logger.warning(
                "whisper.cpp/GGUF path unavailable (%s: %s) — falling back to local "
                "Whisper 'tiny.en' (plain PyTorch) inference.",
                type(e).__name__,
                e,
            )
            try:
                model = _get_model()
                result = model.transcribe(audio, fp16=False)
            except Exception as e:
                return f"Error: speech-to-text failed on '{path}' ({e})"
            text = result.get("text", "").strip()

    if not text:
        return f"Error: no speech detected in '{path}'"

    return text
