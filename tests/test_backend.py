"""Tests the FastAPI backend (backend/main.py) end to end, through real
HTTP requests via FastAPI's TestClient — not by calling the underlying
feature functions directly (those already have their own test files).
This file's job is to prove the HTTP layer itself works: routing,
file uploads, JSON bodies, status codes, and the Payment Pause flow tying
Scam Shield and Call Shield together through the same running process.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_status_reports_execution_path_and_network_isolation(client):
    response = client.get("/status")
    assert response.status_code == 200
    body = response.json()
    assert "execution_path" in body
    assert body["network_isolated"] is True


def test_scam_shield_text_flags_a_real_scam(client):
    response = client.post(
        "/scam-shield/text",
        json={
            "text": "Dear customer, your account will be BLOCKED today due to pending "
            "KYC update. Click here to update immediately and avoid suspension."
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_scam"] is True
    assert 0.0 < body["confidence"] <= 1.0


def test_scam_shield_text_clears_a_legit_message(client):
    response = client.post(
        "/scam-shield/text",
        json={"text": "Reminder: your dentist appointment is tomorrow at 4pm."},
    )
    assert response.status_code == 200
    assert response.json()["is_scam"] is False


def test_scam_shield_text_empty_input_does_not_crash(client):
    response = client.post("/scam-shield/text", json={"text": ""})
    assert response.status_code == 200
    body = response.json()
    assert body["confidence"] == 0.0
    assert "couldn't analyze" in body["reason"].lower()


def test_scam_shield_screenshot_extracts_real_text_and_flags_it(client):
    sample = SAMPLES_DIR / "scam_bank_kyc_alert.png"
    with open(sample, "rb") as f:
        response = client.post("/scam-shield/screenshot", files={"file": ("scam.png", f, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["is_scam"] is True
    assert "extracted_text" in body and len(body["extracted_text"]) > 0


def test_scam_shield_screenshot_corrupt_image_returns_422_not_a_crash(client, tmp_path):
    bad_file = tmp_path / "not_an_image.png"
    bad_file.write_text("this is not image data")
    with open(bad_file, "rb") as f:
        response = client.post("/scam-shield/screenshot", files={"file": ("bad.png", f, "image/png")})
    assert response.status_code == 422
    assert "not a readable image" in response.json()["detail"]


def test_money_insight_categorizes_real_transactions(client):
    response = client.post(
        "/money-insight/transactions",
        json={
            "transactions": [
                "Rs 450.00 debited from A/c XX1234 at SWIGGY BANGALORE on 12-Sep-25.",
                "Rs 3,499.00 debited from A/c XX1234 at AMAZON on 12-Sep-25.",
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["categorized"]) == 2
    assert body["categorized"][0]["category"] == "Food & Dining"
    assert "insight" in body


def test_spend_scanner_screenshot_parses_a_real_receipt(client):
    sample = SAMPLES_DIR / "receipt_upi_confirmation.png"
    with open(sample, "rb") as f:
        response = client.post("/spend-scanner/screenshot", files={"file": ("receipt.png", f, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["merchant"] == "SWIGGY"
    assert body["amount"] == 450.0


def test_call_shield_transcript_flags_a_real_scam_call(client):
    response = client.post(
        "/call-shield/transcript",
        json={
            "transcript": "Caller: This is the Cyber Crime unit. You must transfer all funds "
            "from your savings account right now and read out the OTP the moment it arrives."
        },
    )
    assert response.status_code == 200
    assert response.json()["is_scam"] is True


def test_call_shield_recording_transcribes_and_analyzes_real_audio(client):
    sample = SAMPLES_DIR / "call_legit_bank_call.wav"
    with open(sample, "rb") as f:
        response = client.post("/call-shield/recording", files={"file": ("call.wav", f, "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["is_scam"] is False
    assert len(body["transcript"]) > 0


def test_payment_pause_full_flow_flag_then_pause_then_reset(client):
    client.post("/payment-pause/reset")

    # Nothing flagged yet -> a payment attempt proceeds normally.
    confirm = client.post("/payment-pause/confirm")
    assert confirm.json()["paused"] is False

    # Flag a real scam through Scam Shield.
    client.post(
        "/scam-shield/text",
        json={"text": "URGENT: verify your account now or it will be suspended. Click immediately."},
    )

    log = client.get("/payment-pause/log")
    assert log.json()["total_flags"] == 1

    # Now a payment attempt should be paused, citing that exact flag.
    confirm = client.post("/payment-pause/confirm")
    body = confirm.json()
    assert body["paused"] is True
    assert body["flags"][0]["source"] == "Scam Shield"

    # Reset clears it.
    reset = client.post("/payment-pause/reset")
    assert reset.json()["ok"] is True
    assert client.get("/payment-pause/log").json()["total_flags"] == 0
