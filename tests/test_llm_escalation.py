"""Tests for the LLM-escalation hybrid: engine.py's _maybe_escalate() and
src/pipeline/llm_classifier.py. `llama-cpp-python` + the local GGUF model
are optional (see requirements.txt) — tests that need the LLM genuinely
active branch on llm_classifier.status(), the same pattern used
throughout this project for every other optional accelerated path
(ocr_qai_hub, whisper_cpp).
"""

import numpy as np
import pytest

import src.call_shield.classifier  # noqa: F401 — registers the task
import src.scam_detector.classifier  # noqa: F401 — registers the task
import src.spend_categorizer.categorizer  # noqa: F401 — registers the task
from src.pipeline import llm_classifier
from src.pipeline.engine import Prediction, Task, engine


def test_llm_status_returns_a_real_tuple_either_way():
    active, description = llm_classifier.status()
    assert isinstance(active, bool)
    assert isinstance(description, str) and description


def test_confident_fast_path_never_escalates(monkeypatch):
    """A high-margin Prediction must come back completely unchanged —
    no LLM call attempted at all. Verified by making any LLM call raise;
    if escalation were (wrongly) attempted, this test would fail with
    that exception instead of passing."""

    def _boom(*a, **k):
        raise AssertionError("LLM should not have been called for a confident result")

    monkeypatch.setattr("src.pipeline.llm_classifier.classify", _boom)

    task = Task(
        name="_test_confident",
        artifacts_dir=engine._tasks["scam_detection"].artifacts_dir,
        llm_labels=["legit", "scam"],
        llm_system_prompt="irrelevant for this test",
        llm_escalation_margin=0.3,
    )
    confident = Prediction(label_id=1, confidence=0.95, probabilities=np.array([0.05, 0.95]), text="x")
    result = engine._maybe_escalate(task, confident)
    assert result is confident  # unchanged, same object


def test_ambiguous_result_escalates_when_llm_available(monkeypatch):
    active, _ = llm_classifier.status()
    if not active:
        pytest.skip("LLM not installed/downloaded in this environment — see requirements.txt")

    monkeypatch.setattr(
        "src.pipeline.llm_classifier.classify",
        lambda text, labels, system_prompt: {"label": "scam", "confidence": 0.88, "reason": "mocked"},
    )

    task = Task(
        name="_test_ambiguous",
        artifacts_dir=engine._tasks["scam_detection"].artifacts_dir,
        llm_labels=["legit", "scam"],
        llm_system_prompt="irrelevant — classify() is mocked above",
        llm_escalation_margin=0.5,
    )
    ambiguous = Prediction(label_id=0, confidence=0.55, probabilities=np.array([0.55, 0.45]), text="x")
    result = engine._maybe_escalate(task, ambiguous)

    assert result.source == "llm"
    assert result.label_id == 1  # "scam" is index 1 in llm_labels
    assert result.confidence == 0.88


def test_llm_failure_keeps_the_fast_path_result_not_a_crash(monkeypatch):
    """Escalation must be strictly additive — any failure (model missing,
    malformed output, whatever) falls back to the original Prediction,
    never raises past engine.analyze()."""

    def _boom(*a, **k):
        raise RuntimeError("simulated: model not downloaded")

    monkeypatch.setattr("src.pipeline.llm_classifier.classify", _boom)

    task = Task(
        name="_test_failure",
        artifacts_dir=engine._tasks["scam_detection"].artifacts_dir,
        llm_labels=["legit", "scam"],
        llm_system_prompt="irrelevant",
        llm_escalation_margin=0.9,  # force escalation attempt on almost anything
    )
    ambiguous = Prediction(label_id=0, confidence=0.55, probabilities=np.array([0.55, 0.45]), text="x")
    result = engine._maybe_escalate(task, ambiguous)

    assert result is ambiguous
    assert result.source == "fast"


def test_task_without_llm_config_never_escalates():
    """Call Shield's Task (and any other task that doesn't set llm_labels)
    must be a pure no-op through _maybe_escalate — this is what keeps
    Call Shield entirely outside the scope of this escalation by design."""
    task = engine._tasks["call_shield"]
    assert task.llm_labels is None

    low_confidence = Prediction(label_id=0, confidence=0.51, probabilities=np.array([0.51, 0.49]), text="x")
    result = engine._maybe_escalate(task, low_confidence)
    assert result is low_confidence


def test_out_of_schema_label_raises_and_is_caught(monkeypatch):
    """If the LLM ever returns a label outside the task's own list (should
    be prevented by the JSON-schema enum constraint, but defense in depth
    matters), escalation must degrade to the fast-path result, not crash
    or silently accept an invalid label_id."""

    monkeypatch.setattr(
        "src.pipeline.llm_classifier.classify",
        lambda text, labels, system_prompt: {"label": "not_a_real_label", "confidence": 0.9, "reason": "bad"},
    )

    task = Task(
        name="_test_bad_label",
        artifacts_dir=engine._tasks["scam_detection"].artifacts_dir,
        llm_labels=["legit", "scam"],
        llm_system_prompt="irrelevant",
        llm_escalation_margin=0.9,
    )
    ambiguous = Prediction(label_id=0, confidence=0.55, probabilities=np.array([0.55, 0.45]), text="x")
    result = engine._maybe_escalate(task, ambiguous)

    assert result is ambiguous
    assert result.source == "fast"


def test_scam_and_spend_tasks_are_registered_for_escalation():
    """The interface refactor's actual point: classify_scam() and
    categorize_transactions() never mention hardware or the LLM at all —
    this is checked structurally, on the Task registration itself."""
    scam_task = engine._tasks["scam_detection"]
    spend_task = engine._tasks["spend_categorization"]

    assert scam_task.llm_labels == ["legit", "scam"]
    assert spend_task.llm_labels is not None and len(spend_task.llm_labels) == 11


def test_classify_scam_source_code_never_imports_hardware():
    """Structural check for the 'zero knowledge of hardware backend'
    requirement: the calling code must never *import* onnxruntime, a
    provider-selection module, or the LLM module directly — it only
    ever imports Task/engine from engine.py. (Checking actual imports,
    not raw text: both modules' own docstrings mention "onnxruntime" in
    prose explaining that they *don't* touch it, which a plain substring
    check would wrongly flag.)"""
    import ast
    import inspect

    from src.scam_detector import classifier as scam_classifier
    from src.spend_categorizer import categorizer as spend_categorizer

    forbidden_modules = {"onnxruntime", "src.pipeline.runtime", "src.pipeline.llm_classifier", "llama_cpp"}
    for module in (scam_classifier, spend_categorizer):
        tree = ast.parse(inspect.getsource(module))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        overlap = imported & forbidden_modules
        assert not overlap, f"{module.__name__} imports {overlap} — should be hardware-agnostic"
