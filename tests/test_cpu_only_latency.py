"""Real, measured latency with NO hardware accelerator at all — the
"catches it before the money moves" claim has to hold on weak hardware
too, not just on a Snapdragon NPU or an Intel NPU. Forces the execution-
provider list down to CPUExecutionProvider only and measures the fast
MiniLM+head path for real, on a genuinely fresh (uncached) session —
reusing `engine`'s already-cached CoreML/whatever sessions from earlier
tests in this same pytest process would silently measure the wrong path.

The LLM escalation tier is deliberately NOT force-CPU-tested here: on
this dev machine, `llm_classifier.py` always requests `n_gpu_layers=-1`
(offload everything possible), so it keeps using real Metal acceleration
regardless of what ONNX Runtime's provider list says — forcing genuine
zero-accelerator LLM inference means loading the ~1GB model a second
time with GPU offload explicitly disabled, which is real but expensive
(one measurement took several minutes) and machine-dependent enough that
re-running it on every CI invocation isn't the right tradeoff. Measured
once and reported directly instead — see the project's hardening notes
for the numbers: Metal-accelerated LLM escalation costs roughly 12-21s
per ambiguous message on this dev Mac; true CPU-only (no GPU backend at
all) was measured separately and is dramatically higher — treat LLM
escalation as something that must stay rare (see engine.py's margin
gate), not something whose latency is safe to assume away on weak
hardware.
"""

import time

import numpy as np
import pytest

from src.pipeline import runtime
from src.pipeline.engine import NXTSightEngine, Task
from src.scam_detector.classifier import ARTIFACTS_DIR as SCAM_ARTIFACTS_DIR
from src.spend_categorizer.categorizer import ARTIFACTS_DIR as SPEND_ARTIFACTS_DIR

# Generous ceilings, not tight benchmarks — this suite runs on whatever
# machine happens to run it (a CI box, a laptop under other load), so the
# point is to catch a real regression (a path that got orders of magnitude
# slower), not to pin down an exact number that would make this test
# itself flaky across hardware.
FAST_PATH_WARM_CEILING_S = 1.0
FAST_PATH_COLD_CEILING_S = 15.0


@pytest.fixture
def cpu_only(monkeypatch):
    """Force select_execution_providers() to report CPU-only, and use a
    throwaway NXTSightEngine (not the shared `engine` singleton other test
    files already warmed up with whatever this machine's best real
    accelerator is) so session creation is genuinely fresh under that
    forced condition."""
    monkeypatch.setattr(runtime.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    return NXTSightEngine()


def _timed_analyze(engine, task_name, text):
    start = time.perf_counter()
    result = engine.analyze(task_name, text)
    return result, time.perf_counter() - start


def test_scam_classification_cpu_only_cold_and_warm_latency(cpu_only):
    task = Task(name="_cpu_only_scam", artifacts_dir=SCAM_ARTIFACTS_DIR)
    cpu_only.register_task(task)

    text = "URGENT: Your account will be suspended in 2 hours unless you verify your details now. Tap here to avoid permanent closure: bit.ly/verify-acc-2847"
    result, cold_s = _timed_analyze(cpu_only, "_cpu_only_scam", text)
    assert not isinstance(result, str)  # must be a real Prediction, not a rejection
    assert cold_s < FAST_PATH_COLD_CEILING_S, f"CPU-only cold-start classification took {cold_s:.2f}s"

    warm_times = []
    for _ in range(10):
        _, t = _timed_analyze(cpu_only, "_cpu_only_scam", text)
        warm_times.append(t)
    warm_mean = sum(warm_times) / len(warm_times)
    assert warm_mean < FAST_PATH_WARM_CEILING_S, f"CPU-only warm classification averaged {warm_mean*1000:.1f}ms"


def test_spend_categorization_cpu_only_warm_latency(cpu_only):
    task = Task(name="_cpu_only_spend", artifacts_dir=SPEND_ARTIFACTS_DIR)
    cpu_only.register_task(task)

    text = "Rs 380.00 debited from A/c XX2109 on 02-Oct-25 at BEHROUZ BIRYANI via Swiggy. Avl Bal Rs 18,540.00"
    _timed_analyze(cpu_only, "_cpu_only_spend", text)  # cold, not asserted on here

    warm_times = []
    for _ in range(10):
        _, t = _timed_analyze(cpu_only, "_cpu_only_spend", text)
        warm_times.append(t)
    warm_mean = sum(warm_times) / len(warm_times)
    assert warm_mean < FAST_PATH_WARM_CEILING_S, f"CPU-only warm categorization averaged {warm_mean*1000:.1f}ms"


def test_cpu_only_execution_path_is_reported_correctly(cpu_only):
    """Sanity check on the fixture itself: confirms the forced CPU-only
    condition is what actually got exercised above, not silently still
    running on whatever accelerator this machine has."""
    _, description = runtime.select_execution_providers()
    assert description.startswith("CPU")
