"""NXTSight demo UI.

A minimal local web app so the on-device pipeline can be tried live: drop
a screenshot and see the scam verdict, or paste transaction text and see
it categorized with an insight. No login, no accounts, nothing that talks
to a server outside this machine — every prediction below runs through
the same NXTSightEngine (src/pipeline/engine.py) used by the CLI and the
test suite.

Run it with:
    streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from src.pipeline.ocr import extract_text_from_image
from src.scam_detector.classifier import classify_scam
from src.spend_categorizer.categorizer import categorize_transactions

SAMPLES_DIR = Path(__file__).resolve().parent / "data" / "samples"

SAMPLE_TRANSACTIONS = """Rs 450.00 debited from A/c XX1234 on 12-Sep-25 at SWIGGY BANGALORE. Avl Bal Rs 12,340.50
Rs 3,499.00 debited from A/c XX1234 at AMAZON on 12-Sep-25. Avl Bal Rs 11,200.
Rs 240.00 debited via UPI to UBER INDIA on 12-Sep-25. UPI Ref No 334455667788.
You have received Rs 45,000.00 in your account XX4521 via NEFT from ABC CORP PVT LTD SALARY on 01-Sep-25. Avl Bal: Rs 63,200.
Rs 649.00 debited from A/c XX1234 towards NETFLIX SUBSCRIPTION on 05-Sep-25. Avl Bal Rs 14,100.
Rs 5,000.00 withdrawn from A/c XX1234 at SBI ATM MG ROAD on 12-Sep-25. Avl Bal Rs 9,200.
Rs 5,000.00 debited from A/c XX1234 towards SIP MUTUAL FUND ZERODHA on 05-Sep-25. Avl Bal Rs 22,300.
Rs 1,240.00 debited from A/c XX1234 towards BESCOM ELECTRICITY BILL on 12-Sep-25. Avl Bal Rs 9,760."""

st.set_page_config(page_title="NXTSight", layout="centered")

st.title("NXTSight")
st.caption("On-device scam detection + spend insight — Snapdragon AI Lab Build & Present Challenge")

st.warning(
    "**Demo simplification.** You're pasting text or uploading a screenshot by hand so you can "
    "try this live. In the real product, NXTSight reads your phone's incoming SMS and "
    "notifications **automatically in the background** — the same way apps like Walnut or "
    "Money View already do in India — so you'd never open an app or type anything yourself. "
    "This screen exists only to demo the underlying engine; it is not the intended product experience."
)

scam_tab, spend_tab = st.tabs(["Scam Screenshot Scanner", "Spend Insight"])

with scam_tab:
    st.subheader("Scam Screenshot Scanner")
    st.write("Upload a screenshot of a message, or try one of the sample scam screenshots below.")

    sample_files = sorted(SAMPLES_DIR.glob("*.png")) if SAMPLES_DIR.exists() else []
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
        image_path = next(f for f in sample_files if f.stem == choice)

    if image_path is not None:
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
            elif result["is_scam"]:
                st.error(f"**Likely a scam** — {result['confidence'] * 100:.0f}% confidence")
                st.write(result["reason"])
            else:
                st.success(f"**Looks legitimate** — {result['confidence'] * 100:.0f}% confidence")
                st.write(result["reason"])

with spend_tab:
    st.subheader("Spend Insight")
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

    if st.button("Analyze spending", type="primary"):
        lines = [line.strip() for line in transactions_text.split("\n") if line.strip()]

        with st.spinner("Categorizing transactions..."):
            result = categorize_transactions(lines)

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
                    }
                    for item in result["categorized"]
                ]
            )
