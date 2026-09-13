# NXTSight

Built for the Snapdragon AI Lab Build & Present Challenge.

NXTSight is a small on-device app with two features that share one pipeline:

1. **Scam Screenshot Scanner** — point it at a screenshot of a suspicious text/WhatsApp/email message and it flags whether the message looks like a financial scam.
2. **Spend Insight** — feed it sample bank/UPI transaction text and it categorizes the spending and gives a plain-language insight (e.g. "you spent 30% more on food delivery this month").

Everything runs **fully locally** on a Snapdragon-powered HP PC, using the device's NPU via Qualcomm AI Hub — no cloud calls, no data leaving the machine.

## How it works

Both features are really the same three-step pipeline pointed at different inputs: **read** (OCR pulls text out of a screenshot, or the raw transaction text is used as-is), **understand** (a small local model — compiled and run on-device through Qualcomm AI Hub's NPU tooling — classifies or categorizes that text), and **explain** (the result is turned into one plain-language line a non-technical person can read, like "this looks like a scam because it asks you to click a link and act immediately" or "your top spend this month was food delivery"). Sharing that pipeline between the scam-check and spend-insight features means one on-device model-serving layer does both jobs, instead of building two separate apps.

## Project structure

```
NXTSight/
├── src/
│   ├── pipeline/            # shared: model loading, NPU inference (QAI Hub), pre/post-processing
│   ├── scam_detector/       # feature 1: screenshot OCR + scam classification
│   └── spend_categorizer/   # feature 2: transaction text parsing + spend categorization
├── models/                  # exported/compiled model artifacts for on-device NPU inference
├── data/samples/            # sample screenshots and sample transaction text for demos/tests
├── notebooks/               # exploration / model experimentation
├── tests/                   # unit tests
├── assets/screenshots/      # demo screenshots for the submission write-up
├── requirements.txt
└── README.md
```

## Status

Scaffold only — no feature logic implemented yet.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**First run on macOS:** EasyOCR downloads its model weights the first time it runs. If you installed Python from python.org and see a `CERTIFICATE_VERIFY_FAILED` error, fix it with:

```bash
pip install certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```
