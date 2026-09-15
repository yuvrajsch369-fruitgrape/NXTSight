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
│   ├── pipeline/
│   │   ├── ocr.py           # extract_text_from_image(): screenshot -> raw text
│   │   └── runtime.py       # picks the ONNX Runtime execution provider (NPU vs CPU)
│   ├── scam_detector/       # feature 1: screenshot OCR + scam classification
│   └── spend_categorizer/   # feature 2: transaction text parsing + spend categorization
├── models/                  # exported/compiled model artifacts for on-device NPU inference
├── scripts/
│   └── aihub_profile_easyocr.py  # one-off: profile a compiled model on real Snapdragon hardware
├── data/samples/            # sample screenshots and sample transaction text for demos/tests
├── notebooks/               # exploration / model experimentation
├── tests/                   # unit tests
├── assets/screenshots/      # demo screenshots for the submission write-up
├── requirements.txt
└── README.md
```

## Status

OCR stage (`extract_text_from_image`) and the NPU/CPU execution-provider
auto-detect (`runtime.py`) are working. Scam classification and spend
categorization are not implemented yet.

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

## Snapdragon / Qualcomm AI Hub

There are two separate pieces here — a **build-time step** you run once, by hand, to get a Snapdragon-ready model, and a **runtime check** the app makes every time it starts.

### 1. Build-time: compile + profile a model for Snapdragon via AI Hub

This step turns a model into a `.onnx` file specifically compiled for the Snapdragon X Elite, and confirms it actually runs correctly by profiling it on real, physical Snapdragon hardware in Qualcomm's cloud device farm (not a simulator — useful since most of us don't own a Snapdragon PC to test on directly). It needs your own free AI Hub account and API token from [aihub.qualcomm.com](https://aihub.qualcomm.com); nobody else can run this step for you.

```bash
pip install qai-hub "qai-hub-models[easyocr]"
qai-hub configure --api_token <YOUR_API_TOKEN>

python -m qai_hub_models.models.easyocr.export \
    --device "Snapdragon X Elite CRD" \
    --target-runtime onnx \
    --profile-options="--qairt_version=default"
```

That one command uploads the model, compiles it for the Snapdragon X Elite, profiles the compiled model on real cloud-hosted Snapdragon hardware, validates its numerical output against the original model, and downloads the resulting `.onnx` — it prints the exact output path when it finishes. (The `--profile-options` flag works around a version mismatch in the current `qai-hub-models` release — it pins an older QAIRT than AI Hub now serves; without it the profiling step is rejected. The two compiled models still download and load fine.) Copy the result into `models/`.

If you want to profile a `.onnx` you already have (say, to re-verify after a change) without repeating the full export, use the included script — it talks to the AI Hub API directly and does just the profiling step:

```bash
python scripts/aihub_profile_easyocr.py models/easyocr_detector/model.onnx
```

This submits a real job to Qualcomm's cloud, waits for it to run on physical Snapdragon hardware, and prints back the actual measured latency/memory numbers.

**Real numbers, from this exact export, profiled on physical Snapdragon X Elite CRD hardware in Qualcomm's cloud device farm:**

| Model | Inference time | Peak memory |
|---|---|---|
| EasyOCR detector | 38.1 ms | 71.2 MB |
| EasyOCR recognizer | 20.4 ms | 40.9 MB |

Both models load and report their expected shapes locally through `runtime.py` — the detector takes a `(1, 3, 608, 800)` image tensor, matching EasyOCR's documented input resolution. That's solid evidence for the "Present" part of the challenge that this genuinely runs on Snapdragon, not just in theory.

### 2. Runtime: automatic NPU/CPU detection

`src/pipeline/runtime.py` decides, at startup, how to run a compiled `.onnx` model:

- On a **Snapdragon Windows PC** with `onnxruntime-qnn` installed, ONNX Runtime reports a `QNNExecutionProvider` — `runtime.py` detects it and routes inference to the Hexagon NPU.
- On **any other machine** (this dev Mac included — Intel Macs have no Qualcomm NPU and no QNN provider at all), it falls back to the plain CPU provider automatically.

No manual flag to flip, and it never crashes for lacking the NPU provider. Every session logs which path is active in plain language, so it's obvious mid-demo which one is running:

```
[NXTSight] Execution path: CPU (QNN execution provider not available on this machine)
[NXTSight] ONNX Runtime providers available here: CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider
[NXTSight] Session ready on 'models/easyocr_detector.onnx' — active provider: CPUExecutionProvider
```

On the actual Snapdragon HP PC, once `onnxruntime-qnn` is installed, the same code logs `Execution path: Snapdragon NPU (ONNX Runtime QNN execution provider)` instead — nothing else changes.

`tests/test_runtime.py` proves this end-to-end right now, on this dev machine, using a tiny throwaway ONNX model — no Snapdragon hardware or AI Hub account required to verify the detection logic itself works.
