"""Regression test for a real, reproducible bug: pywhispercpp and
llama-cpp-python each bundle their own independently-built copy of
libggml (same library names, different builds). macOS's dynamic linker
treats same-named dylibs as one shared identity — whichever package's
copy loads first in the process wins, and the two builds' symbol sets
don't fully agree, so loading the second one used to fail outright with
a real dlopen "Symbol not found: (_ggml_dsv4_hc_comb)" error.

Confirmed directly: `import pywhispercpp.model; import llama_cpp` raised
that exact error before the fix in src/pipeline/whisper_cpp.py
(_ensure_loaded() now imports llama_cpp, best-effort, before importing
pywhispercpp — see that file's comment for the full story). This matters
for real, not just in theory: backend/main.py imports both whisper_cpp
and llm_classifier in the same process, because a running server needs
Call Shield (whisper_cpp) and Scam Shield/Money Insight's LLM escalation
(llm_classifier) simultaneously — exactly the scenario that broke.

Each direction is run in its own fresh subprocess, deliberately — both
modules cache their loaded model in a module-level global, so testing
"order B" in the same process right after "order A" would just hit an
already-loaded cache and not genuinely re-exercise the dylib load at
all. A real regression test for an import-order bug needs a real fresh
interpreter per order, the same way this was originally verified by
hand.

Skips (never fails) if either optional dependency isn't installed —
same pattern as every other optional-path test in this project.
"""

import subprocess
import sys

import pytest

from src.pipeline import llm_classifier, whisper_cpp


def _both_available():
    whisper_active, _ = whisper_cpp.status()
    llm_active, _ = llm_classifier.status()
    return whisper_active and llm_active


def _run_in_fresh_process(load_order: str):
    """load_order: 'whisper_first' or 'llm_first'. Returns the completed
    subprocess so the caller can assert on returncode/stderr."""
    if load_order == "whisper_first":
        code = (
            "from src.pipeline import whisper_cpp, llm_classifier\n"
            "whisper_cpp._ensure_model()\n"
            "llm_classifier._ensure_loaded()\n"
            "print('OK')\n"
        )
    else:
        code = (
            "from src.pipeline import llm_classifier, whisper_cpp\n"
            "llm_classifier._ensure_loaded()\n"
            "whisper_cpp._ensure_model()\n"
            "print('OK')\n"
        )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_whisper_cpp_then_llm_both_load_in_a_fresh_process():
    if not _both_available():
        pytest.skip("whisper.cpp and/or the LLM not installed in this environment — see requirements.txt")

    result = _run_in_fresh_process("whisper_first")
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_llm_then_whisper_cpp_both_load_in_a_fresh_process():
    if not _both_available():
        pytest.skip("whisper.cpp and/or the LLM not installed in this environment — see requirements.txt")

    result = _run_in_fresh_process("llm_first")
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
