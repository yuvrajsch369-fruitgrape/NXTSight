"""Proves the shared-engine architecture, not just that both features work.

classify_scam() and categorize_transactions() live in different modules,
but both must be calling through the exact same NXTSightEngine instance —
one on-device engine, two jobs — not two lookalike copies.
"""

import src.scam_detector.classifier as scam_module
import src.spend_categorizer.categorizer as spend_module
from src.pipeline.engine import engine


def test_both_task_modules_use_the_same_engine_instance():
    assert scam_module.engine is engine
    assert spend_module.engine is engine


def test_both_tasks_are_registered_on_the_shared_engine():
    assert "scam_detection" in engine._tasks
    assert "spend_categorization" in engine._tasks
    assert engine._tasks["scam_detection"].artifacts_dir == scam_module.ARTIFACTS_DIR
    assert engine._tasks["spend_categorization"].artifacts_dir == spend_module.ARTIFACTS_DIR


def test_tasks_load_independent_model_artifacts():
    # Run one prediction per task so both get lazily loaded, then confirm
    # they hold genuinely different (vectorizer, session) pairs — same
    # engine, same code path, but each task's own trained model.
    engine.analyze("scam_detection", "test")
    engine.analyze("spend_categorization", "test")

    assert engine._vectorizers["scam_detection"] is not engine._vectorizers["spend_categorization"]
    assert engine._sessions["scam_detection"] is not engine._sessions["spend_categorization"]
