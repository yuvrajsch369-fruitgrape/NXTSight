"""Payment Pause: ties the Scam Screenshot Scanner and Call Shield together
at the moment a payment is being confirmed.

Whenever either feature flags something as a likely scam, that flag goes
into a short-lived, in-memory-only log (nothing written to disk, nothing
sent anywhere — it's just a Python list that lives as long as the demo
session does). When a payment is about to be confirmed, NXTSight checks
that log: if anything was flagged within the last WINDOW_MINUTES minutes,
the confirmation is interrupted with exactly what was flagged, when, and
why, instead of proceeding silently.

The matching rule is deliberately simple and stated plainly here, because
it needs to be explainable in a demo, not just effective: a flag counts as
"recent" if it happened within WINDOW_MINUTES minutes of the payment
attempt. No fuzzy correlation between the flagged message/call and the
payment itself — any recent flag pauses any payment attempt. That's a
blunt rule on purpose: a real product would narrow this (e.g. matching
amount, contact, or app context), but that narrowing needs signals this
prototype doesn't have access to.

IMPORTANT — read before demoing: the "Confirm Payment" screen this powers
is simulated. NXTSight cannot see or intercept a real payment being made
in an actual UPI/banking app — that would require integration with that
app (or the OS payments layer) far beyond what a local Python app can do.
This demonstrates the intervention logic and the interruption UX only.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

WINDOW_MINUTES = 10


@dataclass
class Flag:
    """One scam flag raised by Scam Screenshot Scanner or Call Shield."""

    source: str  # "Scam Screenshot Scanner" or "Call Shield"
    reason: str
    confidence: float
    at: datetime = field(default_factory=datetime.now)


def add_flag(flags: list, source: str, reason: str, confidence: float) -> None:
    """Append a new flag to `flags` (a plain list held in session state)."""
    flags.append(Flag(source=source, reason=reason, confidence=confidence, at=datetime.now()))


def recent_flags(flags: list, window_minutes: int = WINDOW_MINUTES) -> list:
    """Flags from `flags` raised within the last `window_minutes`, newest first."""
    cutoff = datetime.now() - timedelta(minutes=window_minutes)
    return sorted((f for f in flags if f.at >= cutoff), key=lambda f: f.at, reverse=True)


def format_age(at: datetime) -> str:
    """Human-readable elapsed time since `at`, e.g. 'just now' or '3 minutes ago'."""
    seconds = (datetime.now() - at).total_seconds()
    if seconds < 60:
        return "just now"
    minutes = int(seconds // 60)
    return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
