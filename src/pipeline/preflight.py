"""Environment preflight checks, shared by app.py and scripts/check_setup.py.

Turns "silent failure on someone else's machine" into a specific,
actionable message — which Python version is required, which package is
missing, and the exact command to fix it — instead of a raw ImportError
buried in a traceback, or a script that just hangs with no explanation.

Only uses the standard library: this module must itself always be
importable, even when every third-party dependency is missing, since it's
the thing that reports that they're missing.
"""

import importlib
import sys

MIN_PYTHON = (3, 9)  # engine.py's `dict[str, Task]` needs PEP 585 (3.9+)

# import name -> pip package name. The core set every entrypoint needs —
# anything that touches the shared engine, OCR, or speech-to-text — kept
# separate from UI-specific frameworks so backend/main.py isn't forced to
# require streamlit just because app.py does, and vice versa.
CORE_REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
    "easyocr": "easyocr",
    "torch": "torch",
    "sklearn": "scikit-learn",
    "langdetect": "langdetect",
    "onnxruntime": "onnxruntime",
    "joblib": "joblib",
    "transformers": "transformers",
}

# app.py's full requirement set: core + the Streamlit UI itself. Default
# for missing_dependencies() below, unchanged from before this split.
REQUIRED_MODULES = {**CORE_REQUIRED_MODULES, "streamlit": "streamlit"}


def python_version_problem():
    """Return a message if the running Python is too old, else None."""
    if sys.version_info < MIN_PYTHON:
        have = f"{sys.version_info[0]}.{sys.version_info[1]}"
        need = ".".join(str(p) for p in MIN_PYTHON)
        return (
            f"NXTSight requires Python {need}+ (found {have}). Install a "
            "newer Python (python.org, or `pyenv install`) and recreate "
            "the virtual environment: `python3 -m venv venv`."
        )
    return None


def missing_dependencies(required: dict = None) -> list:
    """Return the pip package names for any required module that fails to import.

    Defaults to REQUIRED_MODULES (app.py's full set, core + streamlit).
    Pass a different dict — e.g. CORE_REQUIRED_MODULES, or that plus
    backend/main.py's own {"fastapi": "fastapi"} — to check a different
    entrypoint's actual needs instead.
    """
    required = REQUIRED_MODULES if required is None else required
    missing = []
    for module_name, pip_name in required.items():
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(pip_name)
    return missing
