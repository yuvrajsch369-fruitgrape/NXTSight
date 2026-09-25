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
from typing import Optional, Union

import numpy as np
from joblib import load

from src.pipeline import text_encoder
from src.pipeline.runtime import create_inference_session
from src.pipeline.text_guard import unanalyzable_reason

logger = logging.getLogger("nxtsight")


@dataclass
class Task:
    """One job the engine can run: a name plus where its trained artifacts live.

    `llm_labels` + `llm_system_prompt` are optional: set both to let this
    task escalate to the local LLM (src/pipeline/llm_classifier.py) when
    the fast classifier's own result is ambiguous (see
    NXTSightEngine._maybe_escalate below). Leave both None (the default)
    for a task that should only ever use the fast path — e.g. Call Shield,
    which isn't part of this escalation by design (see README).
    """

    name: str
    artifacts_dir: Path
    max_chars: int = 4000
    llm_labels: Optional[list] = None
    llm_system_prompt: Optional[str] = None
    llm_escalation_margin: float = 0.3


@dataclass
class Prediction:
    """Raw model output for one piece of text, before task-specific interpretation."""

    label_id: int
    confidence: float
    probabilities: np.ndarray
    text: str  # the cleaned/truncated text actually fed to the model
    source: str = "fast"  # "fast" (the trained MiniLM+head classifier) or "llm" (escalated)


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
        prediction = Prediction(
            label_id=label_id,
            confidence=float(probs[label_id]),
            probabilities=probs,
            text=cleaned,
        )
        return self._maybe_escalate(task, prediction)

    def _maybe_escalate(self, task: Task, prediction: Prediction) -> Prediction:
        """If `task` is configured for LLM escalation and the fast path's
        own result is genuinely ambiguous, ask the local LLM for a second
        opinion and use it. "Ambiguous" is the margin between the top two
        class probabilities, not raw confidence alone — the right measure
        for both a binary task (scam: margin = |p(scam) - p(legit)|) and a
        multi-class one (spend: how much the top guess actually beat the
        runner-up), whereas a fixed confidence cutoff only makes sense for
        binary. Any failure along this path — LLM not installed, model not
        downloaded, malformed output, anything — is caught and the
        original fast-path Prediction is returned unchanged: escalation is
        strictly additive, never a new way for classification to break.
        """
        if task.llm_labels is None or task.llm_system_prompt is None:
            return prediction

        sorted_probs = np.sort(prediction.probabilities)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
        if margin >= task.llm_escalation_margin:
            return prediction  # confident enough — not worth the LLM's latency

        try:
            from src.pipeline import llm_classifier

            result = llm_classifier.classify(
                prediction.text, labels=task.llm_labels, system_prompt=task.llm_system_prompt
            )
            label_id = task.llm_labels.index(result["label"])
            probabilities = np.zeros(len(task.llm_labels), dtype=np.float32)
            probabilities[label_id] = result["confidence"]
            logger.info(
                "Task '%s' escalated to LLM (fast-path margin %.2f < %.2f) -> %s (%.2f)",
                task.name,
                margin,
                task.llm_escalation_margin,
                result["label"],
                result["confidence"],
            )
            return Prediction(
                label_id=label_id,
                confidence=result["confidence"],
                probabilities=probabilities,
                text=prediction.text,
                source="llm",
            )
        except Exception as e:
            logger.info(
                "Task '%s': LLM escalation unavailable (%s: %s) — keeping the fast-path result.",
                task.name,
                type(e).__name__,
                e,
            )
            return prediction

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
