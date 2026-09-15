"""NXTSight's shared on-device inference engine — one object, two jobs.

A single NXTSightEngine instance is the only thing that ever touches ONNX
Runtime, the Snapdragon NPU/QNN provider, or a TF-IDF vectorizer. Both the
scam classifier and the spend categorizer call through this same engine
object; neither module loads a model file or runs a session directly
anymore. What differs between the two jobs is registered as a Task — its
own trained artifacts, its own confidence/format rules — not a second copy
of the model-serving code.

This mirrors how you'd serve two prompts through one LLM: one engine, one
execution path (text_guard -> vectorize -> runtime.create_inference_session,
itself QNN-aware), and per-task logic layered on top of its output rather
than duplicated underneath it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from joblib import load

from src.pipeline.runtime import create_inference_session
from src.pipeline.text_guard import unanalyzable_reason


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
        task = self._tasks[task_name]
        if task_name not in self._vectorizers:
            self._vectorizers[task_name] = load(task.artifacts_dir / "vectorizer.joblib")
        if task_name not in self._sessions:
            self._sessions[task_name], _ = create_inference_session(
                str(task.artifacts_dir / "classifier.onnx")
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
            vectorizer, session = self._ensure_loaded(task_name)
            cleaned = text.strip()[: task.max_chars]
            vector = vectorizer.transform([cleaned]).toarray().astype(np.float32)
            input_name = session.get_inputs()[0].name
            labels, probabilities = session.run(["label", "probabilities"], {input_name: vector})
        except Exception as e:
            return f"the {task_name} model is unavailable right now ({type(e).__name__})"

        label_id = int(labels[0])
        return Prediction(
            label_id=label_id,
            confidence=float(probabilities[0][label_id]),
            probabilities=probabilities[0],
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
