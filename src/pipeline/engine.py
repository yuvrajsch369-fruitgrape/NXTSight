"""NXTSight's shared on-device inference engine — one object, several jobs.

A single NXTSightEngine instance is the only thing that ever touches ONNX
Runtime, the Snapdragon NPU/QNN provider, MiniLM-v2, or a TF-IDF
vectorizer. The scam classifier, the spend categorizer, and the Call
Shield classifier all call through this same engine object; none of those
modules loads a model file or runs a session directly. What differs
between jobs is registered as a Task — its own trained artifacts, its own
confidence/format rules — not a second copy of the model-serving code.

This mirrors how you'd serve several prompts through one LLM: one engine,
one execution path (text_guard -> encode -> runtime.create_inference_session,
itself QNN-aware) with per-task logic layered on top of its output rather
than duplicated underneath it. Each task prefers its ONNX export (the
QNN/CPU-aware path) but falls back to calling its trained scikit-learn
model directly if that export is missing — see _ensure_loaded() below.

Classification runs on MiniLM-v2 sentence embeddings (src/pipeline/
text_encoder.py) — one shared, general-purpose encoder every task's
classification head sits on top of — not the TF-IDF vectors it used to.
Each task still fits and keeps its own TF-IDF vectorizer too, but purely
as an explanation signal now (vocabulary_terms() below, used to quote
"which words triggered this" in a reason string): a dense embedding
drives a better-informed decision, but it isn't the kind of thing you can
point at and say "these are the words that mattered" the way an
interpretable bag-of-words vector is. Two different jobs, two different
representations of the same text, on purpose.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from joblib import load

from src.pipeline import text_encoder
from src.pipeline.runtime import create_inference_session
from src.pipeline.text_guard import unanalyzable_reason

logger = logging.getLogger("nxtsight")


@dataclass
class Task:
    """One job the engine can run: a name plus where its trained artifacts live."""

    name: str
    artifacts_dir: Path
    max_chars: int = 4000


@dataclass
class Prediction:
    """Raw model output for one piece of text, before task-specific interpretation."""

    label_id: int
    confidence: float
    probabilities: np.ndarray
    text: str  # the cleaned/truncated text actually fed to the model


class NXTSightEngine:
    """One shared engine: text validation, TF-IDF vectorization, ONNX inference.

    Register a Task once per job with register_task(); every call after
    that just names the task. Each task's (vectorizer, ONNX session) pair
    is loaded lazily and cached independently — the two jobs never share
    model weights (a binary scam flag and an 11-way spending category are
    genuinely different output spaces) — but they share every line of code
    that loads, validates, and runs a model.
    """

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._vectorizers: dict[str, object] = {}
        self._sessions: dict[str, object] = {}

    def register_task(self, task: Task) -> None:
        self._tasks[task.name] = task

    def _ensure_loaded(self, task_name: str):
        """Load (vectorizer, (mode, model)) for a task, preferring ONNX.

        `mode` is "onnx" (the QNN/CPU-aware ONNX Runtime path — the
        provably Snapdragon-optimized one) whenever classifier.onnx
        exists. If it doesn't — the export step failed or wasn't run on
        this machine (e.g. the `onnx` package isn't installed; it's a
        training-only dependency, never required at runtime) — this falls
        back to `mode == "sklearn"`: the trained scikit-learn classifier,
        saved by every train_classifier.py alongside the ONNX export,
        loaded and called directly. Slower, no NPU acceleration, but the
        engine still serves real predictions instead of the whole task
        going dark over one missing conversion step.
        """
        task = self._tasks[task_name]
        if task_name not in self._vectorizers:
            self._vectorizers[task_name] = load(task.artifacts_dir / "vectorizer.joblib")
        if task_name not in self._sessions:
            onnx_path = task.artifacts_dir / "classifier.onnx"
            joblib_path = task.artifacts_dir / "classifier.joblib"
            if onnx_path.exists():
                session, _ = create_inference_session(str(onnx_path))
                self._sessions[task_name] = ("onnx", session)
            elif joblib_path.exists():
                logger.warning(
                    "No classifier.onnx for task '%s' — falling back to the trained "
                    "scikit-learn model directly (no ONNX Runtime / QNN acceleration "
                    "for this task until it's re-exported).",
                    task_name,
                )
                self._sessions[task_name] = ("sklearn", load(joblib_path))
            else:
                raise FileNotFoundError(
                    f"neither classifier.onnx nor classifier.joblib found for task '{task_name}' "
                    f"in {task.artifacts_dir}"
                )
        return self._vectorizers[task_name], self._sessions[task_name]

    def analyze(self, task_name: str, text) -> Union[str, Prediction]:
        """Run `text` through the named task's model.

        Returns a plain-language reason string if the text can't be
        analyzed — either because it fails text_guard (empty, gibberish,
        non-English — the same check for every task) or because the
        task's model artifacts are missing/corrupted (a bad deploy, not
        bad input, but the caller shouldn't crash over it either) —
        otherwise a Prediction for the task's own module to interpret.

        `task_name` itself is not validated: passing an unregistered name
        is a programming error in our own code, not user input, so it's
        allowed to raise KeyError loudly rather than being swallowed here.
        """
        reason = unanalyzable_reason(text)
        if reason:
            return reason

        task = self._tasks[task_name]
        try:
            _vectorizer, (mode, model) = self._ensure_loaded(task_name)
            cleaned = text.strip()[: task.max_chars]
            vector = text_encoder.encode(cleaned).reshape(1, -1).astype(np.float32)
            if mode == "onnx":
                input_name = model.get_inputs()[0].name
                labels, probabilities = model.run(["label", "probabilities"], {input_name: vector})
                label = labels[0]
                probs = probabilities[0]
            else:  # mode == "sklearn" — direct fallback, no ONNX Runtime involved
                label = model.predict(vector)[0]
                probs = model.predict_proba(vector)[0]
        except Exception as e:
            return f"the {task_name} model is unavailable right now ({type(e).__name__})"

        label_id = int(label)
        return Prediction(
            label_id=label_id,
            confidence=float(probs[label_id]),
            probabilities=probs,
            text=cleaned,
        )

    def vocabulary_terms(self, task_name: str, text: str) -> set:
        """Tokens/n-grams `text` produces under the task's vectorizer.

        Used to build human-readable explanations from a model's own
        learned vocabulary rather than a canned string. Returns an empty
        set (rather than raising) if the vectorizer can't be loaded —
        callers treat "no matched terms" as a normal, already-handled case.
        """
        try:
            vectorizer, _ = self._ensure_loaded(task_name)
            return set(vectorizer.build_analyzer()(text))
        except Exception:
            return set()


# One process-wide engine — every task module below registers itself with
# this same instance and calls through it, instead of importing onnxruntime
# or joblib directly.
engine = NXTSightEngine()
