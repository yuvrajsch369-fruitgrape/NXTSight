"""NXTSight demo UI — dark-mode only, themed via src/ui/theme.py.

A minimal local web app so the on-device pipeline can be tried live: drop
a screenshot and see the scam verdict, paste transaction text (or scan a
receipt/bill screenshot) and see it categorized with an insight, or paste
a call transcript / upload a recording and check it for scam-call
patterns. No login, no accounts, nothing that talks to a server outside
this machine — every prediction below runs through the same
NXTSightEngine (src/pipeline/engine.py) used by the CLI and the test suite.

Run it with:
    streamlit run app.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Set before any transformers/huggingface_hub import can happen anywhere
# in the process (including transitively, via any module imported below)
# — see src/pipeline/whisper_qai_hub.py's docstring for why this needs to
# be genuinely first, not just "early": HuggingFace's from_pretrained()
# makes a real network call by default to check for a newer revision,
# even with everything cached locally, and this is the documented way to
# force fully-offline loading everywhere in one shot.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Purely cosmetic, but a real, confirmed warning otherwise: HuggingFace's
# tokenizers library warns on stderr about disabling its internal thread
# parallelism after Streamlit/uvicorn forks a worker process. Harmless
# either way (correctness is unaffected), just noisy — this is the
# documented way to pick one explicitly rather than let it warn about it.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.pipeline import preflight

# Check Python itself before importing anything — a too-old interpreter
# can fail in confusing ways deep inside a dependency rather than here.
_version_problem = preflight.python_version_problem()
if _version_problem:
    print(_version_problem, file=sys.stderr)
    raise SystemExit(1)

# Streamlit itself might not be installed — without it we can't show a
# nice on-screen error, so this one case prints to the terminal instead.
try:
    import streamlit as st
except ImportError:
    print(
        "NXTSight requires the 'streamlit' package, which isn't installed "
        "in this environment.\nRun:\n    pip install -r requirements.txt\n"
        "then try again.",
        file=sys.stderr,
    )
    raise SystemExit(1)

_missing = preflight.missing_dependencies()
if _missing:
    st.set_page_config(page_title="NXTSight", layout="centered")
    st.error(
        "**Missing dependencies:** "
        + ", ".join(_missing)
        + "\n\nRun this in your terminal, then restart the app:\n\n"
        "```\npip install -r requirements.txt\n```"
    )
    st.stop()

# Install the network guard before importing anything that touches a
# model — if OCR/classification ever tried to reach the network, this
# makes that attempt fail immediately and loudly instead of silently
# succeeding over wifi during a demo.
from src.pipeline import network_guard

network_guard.install()

from src.call_shield.classifier import analyze_call, analyze_call_recording
from src.pipeline import ocr_qai_hub, text_encoder, whisper_qai_hub
from src.pipeline.ocr import extract_text_from_image
from src.pipeline.payment_pause import WINDOW_MINUTES, add_flag, format_age, recent_flags
from src.pipeline.runtime import select_execution_providers
from src.scam_detector.classifier import classify_scam
from src.spend_categorizer.categorizer import categorize_transactions
from src.spend_categorizer.receipt_parser import process_receipt_screenshot
from src.ui.theme import badge_row, hero, inject_theme, meter, section, status_dots, vision_cta

SAMPLES_DIR = Path(__file__).resolve().parent / "data" / "samples"

CALL_TRANSCRIPT_SAMPLES = {
    "Digital-arrest scam": """Caller: Namaste, I am Sub-Inspector Verma calling from the Cyber Crime Investigation Unit.
You: Yes, what is this regarding?
Caller: Your PAN card has been used to open a bank account involved in a 2 crore rupee money laundering racket linked to human trafficking. There is a digital arrest warrant against your name.
You: This can't be right, I've never opened any such account.
Caller: The evidence is very clear on our end. You must not disconnect this call or leave your house. Keep your camera on for the remainder of this investigation.
You: Okay, I'm scared, what do I do?
Caller: To prove you are not involved, transfer all funds from your savings account to the RBI secure holding account we provide for verification within the next hour, and read out the OTP the moment it arrives. Failing to comply means a police team arrives at your address immediately.""",
    "Fake courier scam": """Caller: Hello, this is DHL express calling about a parcel booked under your name that has been stopped at Chennai customs.
You: I don't remember ordering anything from abroad.
Caller: The parcel contains an international SIM card and some banned cosmetic items, which is a customs violation. A case file has already been opened against your identity.
You: What happens now?
Caller: To close the case and release the parcel, you need to pay a customs penalty of 3200 rupees right now through the payment link, and confirm the OTP you receive so we can verify the transaction went through before the 20 minute deadline, otherwise this gets escalated to the cyber cell.""",
    "Fake bank scam": """Caller: Good evening, this is calling from the fraud prevention department of your bank.
You: Okay, is something wrong?
Caller: We've blocked a suspicious transaction of 78,000 rupees attempted from your account a few minutes ago from a different city.
You: I didn't make that transaction.
Caller: That's exactly why we're calling, to help you cancel it before it processes. Please read out the one time password sent to your registered number right now so I can reverse the transaction on my end before the 5 minute window closes, or the amount will be deducted permanently.""",
    "Legit bank call": """Caller: Hi, this is calling from your bank regarding the credit card limit increase you requested through the app last week.
You: Oh right, has it been approved?
Caller: Yes, it's been approved and will reflect in your account within 2 business days. Is there anything else I can help you with?
You: No, that's all, thank you.
Caller: You're welcome, have a great day.""",
    "Legit friend call": """Caller: Hey, it's Ankit, are you around this weekend?
You: Yeah, I should be free Saturday, why what's up?
Caller: A few of us are planning to go hiking near Lonavala, thought you might want to join.
You: That sounds great, count me in. What time are we leaving?
Caller: Probably 6 AM from my place, I'll send the details on the group chat.""",
}


def _flag_once(session_key, source, reason, confidence):
    """Log a flag to the Payment Pause log, but only the first time this
    exact (source, reason) result is seen for this widget selection.

    Streamlit reruns the *entire* script on every interaction anywhere in
    the app, not just the widget that changed. Scam Shield's screenshot
    path and the Call Shield audio path both analyze as soon as a file is
    selected, with no separate "Analyze" button — so without this guard,
    clicking something unrelated (e.g. Payment Pause's own button) would
    silently re-run that analysis and add a duplicate flag every time.
    """
    flag_key = f"{source}:{reason}"
    if st.session_state.get(session_key) != flag_key:
        add_flag(st.session_state.flag_log, source=source, reason=reason, confidence=confidence)
        st.session_state[session_key] = flag_key


def _verdict_card(is_scam: bool, confidence: float, reason: str, scam_label: str, legit_label: str) -> None:
    """A styled result card shared by Scam Shield and Call Shield — same
    st.error/st.success/st.write calls as before, just grouped visually,
    plus a small confidence gauge (purely decorative)."""
    with st.container(border=True):
        if is_scam:
            st.error(f"**{scam_label}**")
        else:
            st.success(f"**{legit_label}**")
        meter(confidence, danger=is_scam)
        st.write(reason)


def _render_call_verdict(result, flag_session_key):
    if result["reason"].startswith("Couldn't analyze this"):
        st.warning(result["reason"])
        return
    _verdict_card(result["is_scam"], result["confidence"], result["reason"], "Likely a scam call", "Looks like a legitimate call")
    if result["is_scam"]:
        _flag_once(flag_session_key, "Call Shield", result["reason"], result["confidence"])
        st.caption(
            "Logged to **Payment Pause** — open that tab and a simulated payment attempt will "
            "run on its own in a few seconds and get paused on this flag."
        )

SAMPLE_TRANSACTIONS = """Rs 450.00 debited from A/c XX1234 on 12-Sep-25 at SWIGGY BANGALORE. Avl Bal Rs 12,340.50
Rs 3,499.00 debited from A/c XX1234 at AMAZON on 12-Sep-25. Avl Bal Rs 11,200.
Rs 240.00 debited via UPI to UBER INDIA on 12-Sep-25. UPI Ref No 334455667788.
You have received Rs 45,000.00 in your account XX4521 via NEFT from ABC CORP PVT LTD SALARY on 01-Sep-25. Avl Bal: Rs 63,200.
Rs 649.00 debited from A/c XX1234 towards NETFLIX SUBSCRIPTION on 05-Sep-25. Avl Bal Rs 14,100.
Rs 5,000.00 withdrawn from A/c XX1234 at SBI ATM MG ROAD on 12-Sep-25. Avl Bal Rs 9,200.
Rs 5,000.00 debited from A/c XX1234 towards SIP MUTUAL FUND ZERODHA on 05-Sep-25. Avl Bal Rs 22,300.
Rs 1,240.00 debited from A/c XX1234 towards BESCOM ELECTRICITY BILL on 12-Sep-25. Avl Bal Rs 9,760."""

st.set_page_config(page_title="NXTSight", page_icon="◆", layout="centered", initial_sidebar_state="collapsed")
inject_theme()

hero(
    "NXTSight",
    "Your on-device shield against financial fraud — built for the Snapdragon AI Lab Build &amp; Present Challenge",
    icon="⟡",
)

vision_cta(
    "pages/🔭_Future_Vision.py",
    "Future Vision",
    "where NXTSight goes from here — not just the hackathon build.",
)

_, execution_description = select_execution_providers()
is_on_npu = execution_description.startswith("Snapdragon NPU")

try:
    network_guard.verify_blocked()
    network_isolation_ok = True
except Exception as exc:
    network_isolation_ok = False
    network_isolation_error = str(exc)

if not network_isolation_ok:
    st.error(f"**NETWORK ISOLATION CHECK FAILED:** {network_isolation_error}")
    st.stop()

badge_row(
    ("npu" if is_on_npu else "cpu", f"» {execution_description}"),
    ("safe", "✓ Network blocked &amp; verified — inference needs no network"),
)

ocr_ai_hub_active, ocr_ai_hub_status = ocr_qai_hub.status()
minilm_ai_hub_active, minilm_ai_hub_status = text_encoder.status()
whisper_ai_hub_active, whisper_ai_hub_status = whisper_qai_hub.status()
with st.expander("Which real AI Hub models are active right now?"):
    status_dots(
        ("OCR", ocr_ai_hub_active),
        ("MiniLM-v2", minilm_ai_hub_active),
        ("Whisper encoder", whisper_ai_hub_active),
    )
    st.write(f"**OCR (Scam Shield + Receipt Scanner):** {'✓ AI Hub-compiled model active' if ocr_ai_hub_active else '○ fallback (local EasyOCR/PyTorch)'} — {ocr_ai_hub_status}")
    st.write(f"**MiniLM-v2 text encoder (all 3 classifiers' backbone):** {'✓ AI Hub-compiled model active' if minilm_ai_hub_active else '○ fallback (local PyTorch)'} — {minilm_ai_hub_status}")
    st.write(f"**Whisper encoder (Call Shield, decoder stays local):** {'✓ AI Hub-compiled model active' if whisper_ai_hub_active else '○ fallback (local Whisper tiny.en)'} — {whisper_ai_hub_status}")

st.warning(
    "**◐ This is a live working prototype, not the finished product.** You're pasting text or "
    "uploading a screenshot by hand so you can try it live. In the real product, NXTSight "
    "reads your phone's incoming SMS and notifications **automatically in the background** — "
    "the same way apps like Walnut or Money View already do in India — so you'd never open an "
    "app or type anything yourself.\n\n"
    "Two more simplifications worth knowing about: **Call Shield** listens through this "
    "device's microphone, like a person in the room — it doesn't tap into the phone system "
    "itself. **Payment Pause** simulates a payment attempt to show how the warning would "
    "work — it isn't connected to a real payment app yet."
)

if "screenshot_transactions" not in st.session_state:
    st.session_state.screenshot_transactions = []

if "flag_log" not in st.session_state:
    st.session_state.flag_log = []

if "payment_interrupt" not in st.session_state:
    st.session_state.payment_interrupt = None

scam_tab, spend_tab, receipt_tab, call_tab, payment_tab = st.tabs(
    [
        "◈  Scam Shield",
        "¤  Money Insight",
        "▤  Receipt Scanner",
        "☎  Call Shield",
        "⏸  Payment Pause",
    ]
)

with scam_tab:
    section("Scam Shield", "◈")
    st.write("Upload a screenshot of a message, or try one of the sample scam screenshots below.")

    sample_files = sorted(SAMPLES_DIR.glob("scam_*.png")) if SAMPLES_DIR.exists() else []
    sample_names = ["Upload my own"] + [f.stem for f in sample_files]
    choice = st.radio("Image source", sample_names, horizontal=True, label_visibility="collapsed")

    image_path = None
    if choice == "Upload my own":
        uploaded = st.file_uploader("Screenshot", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                image_path = Path(tmp.name)
    else:
        image_path = next((f for f in sample_files if f.stem == choice), None)
        if image_path is None:
            st.error(f"Sample '{choice}' is no longer available — pick another, or upload your own.")

    if image_path is not None:
        try:
            st.image(str(image_path), width=320)

            with st.spinner("Reading text from the image..."):
                text = extract_text_from_image(str(image_path))

            if text.startswith("Error"):
                st.error(text)
            else:
                with st.expander("Extracted text"):
                    st.text(text)

                with st.spinner("Checking for scam patterns..."):
                    result = classify_scam(text)

                if result["reason"].startswith("Couldn't analyze this"):
                    st.warning(result["reason"])
                else:
                    _verdict_card(result["is_scam"], result["confidence"], result["reason"], "Likely a scam", "Looks legitimate")
                    if result["is_scam"]:
                        _flag_once(
                            "_last_scam_flag", "Scam Shield", result["reason"], result["confidence"]
                        )
                        st.caption(
                            "Logged to **Payment Pause** — open that tab and a simulated payment "
                            "attempt will run on its own in a few seconds and get paused on this flag."
                        )
        except Exception as exc:
            st.error(f"Couldn't process this screenshot: {exc}")

with spend_tab:
    section("Money Insight", "¤")
    st.write("Paste bank/UPI transaction messages below, one per line, or load sample transactions.")

    if "transactions_text" not in st.session_state:
        st.session_state.transactions_text = ""

    if st.button("Load sample transactions"):
        st.session_state.transactions_text = SAMPLE_TRANSACTIONS

    transactions_text = st.text_area(
        "Transactions",
        key="transactions_text",
        height=200,
        label_visibility="collapsed",
        placeholder="Paste one transaction message per line...",
    )

    if st.session_state.screenshot_transactions:
        st.caption(
            f"Plus {len(st.session_state.screenshot_transactions)} transaction(s) added from receipt/bill "
            "screenshots (see the **Receipt / Bill Scanner** tab) — included automatically below."
        )
        if st.button("Clear screenshot-added transactions"):
            st.session_state.screenshot_transactions = []
            st.rerun()

    if st.button("Analyze spending", type="primary"):
        try:
            sms_lines = [line.strip() for line in transactions_text.split("\n") if line.strip()]
            combined = sms_lines + st.session_state.screenshot_transactions

            with st.spinner("Categorizing transactions..."):
                result = categorize_transactions(combined)

            if result["insight"].startswith("Couldn't analyze this"):
                st.warning(result["insight"])
            else:
                st.success(result["insight"])

            if result["categorized"]:
                st.table(
                    [
                        {
                            "Transaction": (item["text"][:60] + "...") if len(item["text"]) > 60 else item["text"],
                            "Category": item["category"],
                            "Confidence": f"{item['confidence'] * 100:.0f}%",
                            "Amount": f"₹{item['amount']:,.0f}" if item["amount"] is not None else "—",
                            "Direction": item["direction"] or "—",
                            "Source": "From screenshot" if item["source"] == "screenshot" else "From SMS",
                        }
                        for item in result["categorized"]
                    ]
                )
        except Exception as exc:
            st.error(f"Couldn't analyze these transactions: {exc}")

with receipt_tab:
    section("Receipt / Bill Scanner", "▤")
    st.write(
        "Upload a screenshot of a payment confirmation, receipt, or bill — or try one of the "
        "sample screenshots below — and add what it finds straight into Money Insight."
    )

    receipt_sample_files = sorted(SAMPLES_DIR.glob("receipt_*.png")) if SAMPLES_DIR.exists() else []
    receipt_sample_names = ["Upload my own"] + [f.stem for f in receipt_sample_files]
    receipt_choice = st.radio(
        "Receipt image source", receipt_sample_names, horizontal=True, label_visibility="collapsed"
    )

    receipt_image_path = None
    if receipt_choice == "Upload my own":
        receipt_uploaded = st.file_uploader(
            "Receipt image", type=["png", "jpg", "jpeg"], label_visibility="collapsed", key="receipt_uploader"
        )
        if receipt_uploaded is not None:
            suffix = Path(receipt_uploaded.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(receipt_uploaded.getvalue())
                receipt_image_path = Path(tmp.name)
    else:
        receipt_image_path = next((f for f in receipt_sample_files if f.stem == receipt_choice), None)
        if receipt_image_path is None:
            st.error(f"Sample '{receipt_choice}' is no longer available — pick another, or upload your own.")

    if receipt_image_path is not None:
        try:
            st.image(str(receipt_image_path), width=320)

            with st.spinner("Reading the image and finding the amount, merchant, and date..."):
                receipt_result = process_receipt_screenshot(str(receipt_image_path))

            if not receipt_result["ok"]:
                st.warning(receipt_result["message"])
                if receipt_result.get("raw_text"):
                    with st.expander("Extracted text (for reference)"):
                        st.text(receipt_result["raw_text"])
            else:
                st.success("Found a transaction in this image:")
                found_col1, found_col2, found_col3 = st.columns(3)
                found_col1.metric("Amount", f"₹{receipt_result['amount']:,.2f}")
                found_col2.metric("Merchant", receipt_result["merchant"] or "Unknown")
                found_col3.metric("Date", receipt_result["date"] or "Not found")

                with st.expander("Extracted text (for reference)"):
                    st.text(receipt_result["raw_text"])

                if st.button("Add to Money Insight", type="primary"):
                    st.session_state.screenshot_transactions.append(
                        {"text": receipt_result["transaction_text"], "source": "screenshot"}
                    )
                    # Tabs' code all runs every rerun regardless of which tab
                    # is visible, but in script order — the Money Insight tab
                    # above already rendered by the time this button's click
                    # is handled, so without forcing a fresh rerun here it
                    # wouldn't show this addition until some *later*
                    # unrelated interaction. st.toast() (unlike st.success)
                    # survives the rerun that follows it, so the confirmation
                    # is still visible on the other side.
                    st.toast("Added to Money Insight — open that tab and click Analyze spending to see it.")
                    st.rerun()
        except Exception as exc:
            st.error(f"Couldn't process this image: {exc}")

with call_tab:
    section("Call Shield", "☎")
    st.write(
        "Looks for patterns specific to India's call-fraud landscape: impersonating a bank, police, or "
        "courier service; manufactured urgency; threats of arrest or legal action; requests for an OTP "
        "or a money transfer."
    )

    call_input_mode = st.radio(
        "Call input",
        ["Record live (microphone)", "Paste transcript", "Upload recording (WAV)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if call_input_mode == "Record live (microphone)":
        st.write(
            "Play the call on speakerphone next to this device (or just speak it), click the "
            "microphone, and stop when you're done — it's transcribed on-device and checked "
            "immediately, the same way as an uploaded recording."
        )
        mic_recording = st.audio_input("Record a call", label_visibility="collapsed")

        if mic_recording is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(mic_recording.getvalue())
                    mic_audio_path = Path(tmp.name)

                with st.spinner("Transcribing on-device (Whisper) and checking for scam-call patterns..."):
                    mic_result = analyze_call_recording(str(mic_audio_path))

                if not mic_result["ok"]:
                    st.warning(mic_result["message"])
                else:
                    with st.expander("Transcript"):
                        st.text(mic_result["transcript"])
                    _render_call_verdict(mic_result, "_last_call_mic_flag")
            except Exception as exc:
                st.error(f"Couldn't process this recording: {exc}")

    elif call_input_mode == "Paste transcript":
        if "call_transcript_text" not in st.session_state:
            st.session_state.call_transcript_text = ""

        sample_pick_col, sample_button_col = st.columns([3, 1])
        sample_pick = sample_pick_col.selectbox(
            "Sample transcript", list(CALL_TRANSCRIPT_SAMPLES.keys()), label_visibility="collapsed"
        )
        if sample_button_col.button("Load sample"):
            st.session_state.call_transcript_text = CALL_TRANSCRIPT_SAMPLES[sample_pick]

        transcript_text = st.text_area(
            "Call transcript",
            key="call_transcript_text",
            height=220,
            label_visibility="collapsed",
            placeholder="Paste a call transcript here...",
        )

        if st.button("Analyze call", type="primary"):
            try:
                with st.spinner("Checking for scam-call patterns..."):
                    call_result = analyze_call(transcript_text)
                _render_call_verdict(call_result, "_last_call_paste_flag")
            except Exception as exc:
                st.error(f"Couldn't analyze this transcript: {exc}")

    else:
        call_sample_files = sorted(SAMPLES_DIR.glob("call_*.wav")) if SAMPLES_DIR.exists() else []
        call_sample_names = ["Upload my own"] + [f.stem for f in call_sample_files]
        call_choice = st.radio(
            "Recording source", call_sample_names, horizontal=True, label_visibility="collapsed"
        )

        call_audio_path = None
        if call_choice == "Upload my own":
            call_uploaded = st.file_uploader(
                "Call recording", type=["wav"], label_visibility="collapsed", key="call_uploader"
            )
            if call_uploaded is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(call_uploaded.getvalue())
                    call_audio_path = Path(tmp.name)
        else:
            call_audio_path = next((f for f in call_sample_files if f.stem == call_choice), None)
            if call_audio_path is None:
                st.error(f"Sample '{call_choice}' is no longer available — pick another, or upload your own.")

        if call_audio_path is not None:
            try:
                st.audio(str(call_audio_path))

                with st.spinner("Transcribing on-device (Whisper) and checking for scam-call patterns..."):
                    call_result = analyze_call_recording(str(call_audio_path))

                if not call_result["ok"]:
                    st.warning(call_result["message"])
                else:
                    with st.expander("Transcript"):
                        st.text(call_result["transcript"])
                    _render_call_verdict(call_result, "_last_call_audio_flag")
            except Exception as exc:
                st.error(f"Couldn't process this recording: {exc}")

with payment_tab:
    section("Payment Pause", "⏸")
    st.write(
        "Ties **Scam Shield** and **Call Shield** together: whenever either one "
        f"flags something as a likely scam, it's logged here, and a simulated payment attempt "
        "runs **on its own a few seconds later, with no click needed** — the same way a "
        "scammer's manufactured urgency tries to rush someone straight from a scam message or "
        "call into paying. If that simulated attempt lands while the flag is still within "
        f"{WINDOW_MINUTES} minutes, NXTSight pauses it and shows you what was flagged, when, "
        "and why — the simple rule is: **any flag raised in the last "
        f"{WINDOW_MINUTES} minutes pauses any payment attempt.** No matching against amount or "
        "contact — that would need signals this prototype doesn't have."
    )

    if "auto_checked_version" not in st.session_state:
        st.session_state.auto_checked_version = 0

    with st.expander(f"Flag log (this session, {len(st.session_state.flag_log)} total)"):
        if not st.session_state.flag_log:
            st.caption(
                "Nothing flagged yet. Go flag a scam screenshot or a scam call in the other "
                "tabs — a simulated payment attempt will run automatically a few seconds "
                "later and land right here."
            )
        else:
            for flag in sorted(st.session_state.flag_log, key=lambda f: f.at, reverse=True):
                st.write(f"- **{flag.source}**, {format_age(flag.at)} — {flag.reason}")
            if st.button("Clear flag log"):
                st.session_state.flag_log = []
                st.session_state.payment_interrupt = None
                st.session_state.auto_checked_version = 0
                st.rerun()

    st.divider()

    current_flag_count = len(st.session_state.flag_log)
    if current_flag_count > st.session_state.auto_checked_version:
        # A new flag arrived since the last automatic check — simulate a
        # payment attempt happening on its own a few seconds later, with no
        # click required, rather than waiting for someone to press a button.
        countdown_placeholder = st.empty()
        for remaining in (3, 2, 1):
            countdown_placeholder.info(
                f"New scam flag detected — auto-simulating a payment attempt in {remaining}..."
            )
            time.sleep(1)
        countdown_placeholder.empty()

        matches = recent_flags(st.session_state.flag_log, WINDOW_MINUTES)
        st.session_state.payment_interrupt = matches or None
        st.session_state.auto_checked_version = current_flag_count
        if not matches:
            st.toast("Auto-simulated payment attempt — no recent flags, it would proceed normally.")

    if st.session_state.payment_interrupt:
        matches = st.session_state.payment_interrupt
        with st.container(border=True):
            st.error(
                f"**⏸ Payment paused.** {len(matches)} scam flag(s) in the last {WINDOW_MINUTES} minutes:"
            )
            for flag in matches:
                st.write(
                    f"- **{flag.source}**, {format_age(flag.at)} "
                    f"({flag.confidence * 100:.0f}% confidence) — {flag.reason}"
                )
            cancel_col, proceed_col = st.columns(2)
            if cancel_col.button("Cancel payment", type="primary"):
                st.session_state.payment_interrupt = None
                st.toast("Payment cancelled.")
                st.rerun()
            if proceed_col.button("Proceed anyway"):
                st.session_state.payment_interrupt = None
                st.toast("Payment confirmed despite the warning (simulated).")
                st.rerun()
    else:
        st.success(
            f"**✓ Current status:** no scam flags in the last {WINDOW_MINUTES} minutes — a "
            "confirmed payment would proceed normally."
        )
