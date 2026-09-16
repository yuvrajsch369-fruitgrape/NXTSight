"""Call Shield: scam-call detection, running on NXTSight's shared on-device engine.

analyze_call(transcript) -> {"is_scam": bool, "confidence": float, "reason": str}

Same shared-engine pattern as the scam-message classifier and spend
categorizer: registers a Task with src.pipeline.engine.engine and calls
engine.analyze() — never touches onnxruntime or loads a model file
directly. Its own model, trained on call-transcript-shaped data (see
data.py), since a phone call reads nothing like a single SMS: it's
multi-turn dialogue, much longer, and carries India-specific scam-call
patterns (digital-arrest threats, police/bank/courier impersonation) that
aren't represented in the SMS training data at all.

The one thing this stage does that the SMS classifier doesn't: cites
*which line* of the conversation drove the verdict, not just which words —
a transcript is long enough that "contains scam phrases" alone isn't
useful; knowing it was specifically the "you must stay on video call and
transfer your savings" line is what a person can actually act on.

IMPORTANT — read before demoing: this analyzes a transcript (pasted, or
produced by src/pipeline/stt.py from an uploaded WAV recording). It does
not, and cannot, listen to or intercept a live phone call — that would
require phone/telephony-level OS integration (call-audio access, a
dialer/carrier hook) far beyond what a local Python app can do. This is
explicitly a transcript-analysis demo, not a live call-blocking product.
"""

import json
from pathlib import Path

from src.pipeline.engine import Task, engine
from src.pipeline.stt import extract_text_from_audio

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TASK_NAME = "call_shield"
REASON_TERMS_SHOWN = 4

# Call transcripts run much longer than a single SMS or a receipt's OCR
# text — enough for a real multi-minute call script.
engine.register_task(Task(name=TASK_NAME, artifacts_dir=ARTIFACTS_DIR, max_chars=6000))

_top_terms = None


def _load_top_terms():
    global _top_terms
    if _top_terms is None:
        try:
            _top_terms = json.loads((ARTIFACTS_DIR / "top_terms.json").read_text())
        except Exception:
            _top_terms = {"scam": [], "legit": []}
    return _top_terms


def _unanalyzable(reason: str) -> dict:
    return {"is_scam": False, "confidence": 0.0, "reason": f"Couldn't analyze this: {reason}."}


def _find_triggering_line(transcript: str, candidate_pool: list) -> "tuple[str, list] | None":
    """Return (line, matched_terms) for the transcript line with the most
    overlap with `candidate_pool`, or None if no line matches anything."""
    lines = [line.strip() for line in transcript.split("\n") if line.strip()]
    best_line, best_matches = None, []
    for line in lines:
        present_terms = engine.vocabulary_terms(TASK_NAME, line)
        matched = [term for term in candidate_pool if term in present_terms]
        if len(matched) > len(best_matches):
            best_line, best_matches = line, matched
    if best_line is None or not best_matches:
        return None
    return best_line, best_matches[:REASON_TERMS_SHOWN]


def analyze_call(transcript) -> dict:
    result = engine.analyze(TASK_NAME, transcript)
    if isinstance(result, str):
        return _unanalyzable(result)

    is_scam = bool(result.label_id)

    top_terms = _load_top_terms()
    candidate_pool = top_terms["scam"] if is_scam else top_terms["legit"]
    trigger = _find_triggering_line(result.text, candidate_pool)

    if trigger:
        line, matched = trigger
        quoted_terms = ", ".join(f"'{term}'" for term in matched)
        shown_line = line if len(line) <= 160 else line[:157] + "..."
        if is_scam:
            reason = (
                f'Flagged because of this part of the call: "{shown_line}" '
                f"— contains phrases commonly seen in scam calls: {quoted_terms}."
            )
        else:
            reason = (
                f'No strong scam indicators found. Closest to routine call phrasing: "{shown_line}" '
                f"({quoted_terms})."
            )
    else:
        reason = (
            "Flagged as a likely scam call based on overall phrasing, though no single line stood out."
            if is_scam
            else "No strong scam indicators found; this reads like a routine call."
        )

    return {
        "is_scam": is_scam,
        "confidence": round(result.confidence, 3),
        "reason": reason,
    }


def analyze_call_recording(audio_path) -> dict:
    """Transcribe a WAV call recording (src/pipeline/stt.py) and run
    analyze_call() on the result. Never raises.

    On success: {"ok": True, "transcript": str, "is_scam": bool,
    "confidence": float, "reason": str}.
    On failure (unreadable/silent/unsupported audio): {"ok": False,
    "message": str} — a clear, human-readable reason, never a raw exception.
    """
    transcript = extract_text_from_audio(audio_path)
    if transcript.startswith("Error"):
        return {"ok": False, "message": transcript[len("Error: "):].capitalize()}

    return {"ok": True, "transcript": transcript, **analyze_call(transcript)}
