"""Blocks outbound network connections and proves it, at startup.

NXTSight's pitch is that inference runs fully on-device — turning off wifi
during a demo should change nothing. This module enforces that instead of
just asserting it: it patches socket.socket.connect() so any attempt to
reach anything other than localhost raises immediately, then makes a real
connection attempt against itself (verify_blocked()) to confirm the block
actually works before the app finishes starting up, rather than just
trusting that the patch was applied correctly.

Loopback connections (127.0.0.1 / localhost / ::1) are left untouched —
Streamlit's own local server needs those to talk to the browser.
"""

import socket

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

_original_connect = socket.socket.connect
_installed = False


class NetworkBlockedError(RuntimeError):
    """Raised when code attempts an outbound (non-loopback) network connection.

    NXTSight's inference pipeline is supposed to run fully offline once its
    models are on disk — this means something tried to reach the network
    when it shouldn't have. That's a bug to surface loudly, not swallow.
    """


def _is_loopback(address) -> bool:
    try:
        host = str(address[0])
    except (TypeError, IndexError, KeyError):
        return False
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


def _guarded_connect(self, address, *args, **kwargs):
    if _is_loopback(address):
        return _original_connect(self, address, *args, **kwargs)
    raise NetworkBlockedError(
        f"Blocked an outbound network connection to {address!r}. NXTSight's "
        "inference pipeline is supposed to run fully offline."
    )


def install() -> None:
    """Patch socket.socket.connect so only loopback connections succeed."""
    global _installed
    if _installed:
        return
    socket.socket.connect = _guarded_connect
    _installed = True


def uninstall() -> None:
    """Restore the real socket.socket.connect (used by tests)."""
    global _installed
    socket.socket.connect = _original_connect
    _installed = False


def is_installed() -> bool:
    return _installed


def verify_blocked() -> None:
    """Prove the guard actually works, not just that it's installed.

    Attempts a real outbound connection and confirms our own code rejects
    it. Raises RuntimeError if the block did not happen as expected — i.e.
    if this does NOT raise, offline isolation is not actually active.
    """
    if not _installed:
        raise RuntimeError("verify_blocked() called before install().")

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.connect(("8.8.8.8", 53))
    except NetworkBlockedError:
        return
    except OSError as e:
        raise RuntimeError(
            "Network guard did not intercept the test connection as expected "
            f"(got {type(e).__name__} instead) — offline isolation may not be active."
        ) from e
    else:
        raise RuntimeError(
            "Network guard failed: a real outbound connection succeeded. "
            "Offline isolation is NOT active."
        )
    finally:
        probe.close()
