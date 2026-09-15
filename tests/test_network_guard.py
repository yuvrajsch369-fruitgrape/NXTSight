import socket
import threading

import pytest

from src.pipeline import network_guard
from src.pipeline.ocr import extract_text_from_image
from src.scam_detector.classifier import classify_scam
from src.spend_categorizer.categorizer import categorize_transactions

SAMPLE_IMAGE = "data/samples/scam_lottery_win.png"


@pytest.fixture
def guarded():
    network_guard.install()
    yield
    network_guard.uninstall()


def test_outbound_connection_is_blocked(guarded):
    with pytest.raises(network_guard.NetworkBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))


def test_verify_blocked_passes_when_guard_is_active(guarded):
    network_guard.verify_blocked()  # should not raise


def test_verify_blocked_raises_if_guard_not_installed():
    network_guard.uninstall()
    with pytest.raises(RuntimeError):
        network_guard.verify_blocked()


def test_loopback_connections_still_work(guarded):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def accept_once():
        conn, _ = server.accept()
        conn.close()

    thread = threading.Thread(target=accept_once, daemon=True)
    thread.start()

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))  # must NOT raise — loopback is exempt
    client.close()
    thread.join(timeout=2)
    server.close()


def test_ocr_works_with_network_blocked(guarded):
    # Proves OCR doesn't secretly depend on network once models are cached —
    # this is the actual offline claim the demo needs to hold up.
    text = extract_text_from_image(SAMPLE_IMAGE)
    assert not text.startswith("Error")


def test_scam_classifier_works_with_network_blocked(guarded):
    result = classify_scam("Your account will be blocked, click here to verify now.")
    assert isinstance(result["is_scam"], bool)


def test_spend_categorizer_works_with_network_blocked(guarded):
    result = categorize_transactions(["Rs 450.00 debited from A/c XX1234 at SWIGGY BANGALORE on 12-Sep-25."])
    assert result["categorized"][0]["category"] == "Food & Dining"
