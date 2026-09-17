"""Tests the Payment Pause matching logic in isolation from Streamlit.

The rule under test is deliberately simple (see src/pipeline/payment_pause.py):
a flag counts as "recent" if it happened within WINDOW_MINUTES of the check,
regardless of what raised it or what the payment is for.
"""

from datetime import datetime, timedelta

from src.pipeline.payment_pause import Flag, add_flag, format_age, recent_flags


def test_add_flag_appends_with_current_timestamp():
    flags = []
    before = datetime.now()
    add_flag(flags, source="Scam Screenshot Scanner", reason="Fake KYC alert", confidence=0.91)
    after = datetime.now()

    assert len(flags) == 1
    flag = flags[0]
    assert flag.source == "Scam Screenshot Scanner"
    assert flag.reason == "Fake KYC alert"
    assert flag.confidence == 0.91
    assert before <= flag.at <= after


def test_recent_flags_excludes_old_flags():
    flags = [
        Flag(source="Call Shield", reason="old", confidence=0.8, at=datetime.now() - timedelta(minutes=25)),
        Flag(source="Scam Screenshot Scanner", reason="fresh", confidence=0.9, at=datetime.now() - timedelta(minutes=2)),
    ]

    result = recent_flags(flags, window_minutes=10)

    assert len(result) == 1
    assert result[0].reason == "fresh"


def test_recent_flags_boundary_is_inclusive_just_inside_window():
    flags = [Flag(source="Call Shield", reason="edge", confidence=0.7, at=datetime.now() - timedelta(minutes=9, seconds=59))]

    result = recent_flags(flags, window_minutes=10)

    assert len(result) == 1


def test_recent_flags_excludes_just_outside_window():
    flags = [Flag(source="Call Shield", reason="edge", confidence=0.7, at=datetime.now() - timedelta(minutes=10, seconds=1))]

    result = recent_flags(flags, window_minutes=10)

    assert result == []


def test_recent_flags_returns_newest_first():
    flags = [
        Flag(source="Call Shield", reason="older", confidence=0.7, at=datetime.now() - timedelta(minutes=5)),
        Flag(source="Scam Screenshot Scanner", reason="newer", confidence=0.9, at=datetime.now() - timedelta(minutes=1)),
    ]

    result = recent_flags(flags, window_minutes=10)

    assert [f.reason for f in result] == ["newer", "older"]


def test_recent_flags_empty_log_returns_empty_list():
    assert recent_flags([], window_minutes=10) == []


def test_recent_flags_ignores_source_and_confidence_for_matching():
    # The rule is intentionally blunt: any recent flag matches, regardless
    # of which feature raised it or how confident it was.
    flags = [Flag(source="Call Shield", reason="low confidence flag", confidence=0.05, at=datetime.now())]

    result = recent_flags(flags, window_minutes=10)

    assert len(result) == 1


def test_format_age_just_now_for_sub_minute():
    assert format_age(datetime.now() - timedelta(seconds=10)) == "just now"


def test_format_age_singular_minute():
    assert format_age(datetime.now() - timedelta(minutes=1, seconds=5)) == "1 minute ago"


def test_format_age_plural_minutes():
    assert format_age(datetime.now() - timedelta(minutes=7)) == "7 minutes ago"
