"""Environment doctor — run this if you're not sure your setup is complete.

    python scripts/check_setup.py

Checks the Python version and every required dependency, printing exactly
what's wrong and how to fix it — rather than letting `streamlit run app.py`
fail with a cryptic error partway through, or just hang.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import preflight  # noqa: E402


def main():
    problems = []

    version_problem = preflight.python_version_problem()
    if version_problem:
        problems.append(version_problem)

    missing = preflight.missing_dependencies()
    if missing:
        problems.append(
            "Missing dependencies: "
            + ", ".join(missing)
            + "\n   Fix: pip install -r requirements.txt"
        )

    if problems:
        print("NXTSight setup check: problems found\n")
        for i, problem in enumerate(problems, 1):
            print(f"{i}. {problem}\n")
        raise SystemExit(1)

    print(f"NXTSight setup check: all good (Python {sys.version.split()[0]}).")
    print("You can run: streamlit run app.py")


if __name__ == "__main__":
    main()
