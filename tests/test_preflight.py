from src.pipeline import preflight


def test_python_version_problem_is_none_on_this_interpreter():
    # We're running under pytest right now, so if this fails, every other
    # test in the suite would have failed to even collect.
    assert preflight.python_version_problem() is None


def test_python_version_problem_flags_old_python(monkeypatch):
    monkeypatch.setattr(preflight.sys, "version_info", (3, 7, 0))
    problem = preflight.python_version_problem()
    assert problem is not None
    assert "3.7" in problem
    assert "3.9" in problem


def test_missing_dependencies_is_empty_when_everything_is_installed():
    # This environment has requirements.txt installed, so nothing should
    # be reported missing.
    assert preflight.missing_dependencies() == []


def test_missing_dependencies_reports_a_real_missing_module(monkeypatch):
    original_required = preflight.REQUIRED_MODULES.copy()
    monkeypatch.setattr(
        preflight, "REQUIRED_MODULES", {**original_required, "definitely_not_a_real_module_xyz": "fake-package"}
    )
    assert "fake-package" in preflight.missing_dependencies()
