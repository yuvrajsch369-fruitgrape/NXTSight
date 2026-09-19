# NXTSight

Built for Snapdragon-powered HP PCs, it started as a solo entry for Qualcomm's Snapdragon AI Lab Build & Present Challenge, running fully on-device.

NXTSight is a small on-device engine with two jobs: read a message and tell you, in plain language, whether it looks like a scam and why, and read your transaction messages and tell you, in plain language, where your money is going. Everything runs locally on the machine — screenshot OCR, speech-to-text, and every classifier — with no server call involved in a single prediction. On a Snapdragon PC it runs on the Hexagon NPU through Qualcomm's QNN execution provider; everywhere else it falls back to CPU automatically. Same code, same result, either way.

## The problem

Roughly 1 in 5 UPI users in India has been targeted by a payment fraud, and government data shows over ₹805 crore lost to digital payment scams this financial year alone. These frauds — fake KYC updates, OTP-sharing requests, too-good investment offers, "your account is blocked" threats, fake delivery fees — arrive as plain text messages, and the people most exposed to them are often the least equipped to spot the pattern in the moment. At the same time, most people have no real visibility into where their money actually goes month to month; bank SMS and UPI notifications pile up unread.

NXTSight addresses both. It reads a suspicious message and tells you whether it looks like a scam and why, and it reads your transaction messages and tells you where your money is going. Both run through the same local model-serving pipeline, entirely on the device, with no cloud dependency.

## Where this sits next to Norton

Gen Digital, Norton's parent company, already ships an AI scam detector called Genie inside Norton 360 — it's real, it's live globally, and it's a genuine competitor in this space. But Genie's own India page says nothing about UPI-specific fraud, fake KYC messages, digital-arrest call scams, or courier-customs fee scams, and Gen Digital's own numbers show its strength sitting in North America and Europe, not India. 

That's the gap NXTSight is built for: depth in exactly the fraud patterns and payment rails that dominate the Indian market, running fully on-device rather than phoning a server for every check. Gen Digital has already shipped a chip-specific partnership before — Norton Deepfake Protection, built around Intel's on-device silicon — which is the precedent for what a Snapdragon-specific, India-specific fraud engine could become.

## What it does

Five features, one shared pipeline underneath:

| Feature | What you give it | What you get back |
|---|---|---|
| **Scam Shield** | A screenshot of a text/WhatsApp/email message | A scam verdict, a confidence score, and the specific phrase that triggered it |
| **Money Insight** | Bank/UPI transaction SMS, pasted as text | Each transaction categorized into one of 11 spending categories, plus one plain-language summary sentence |
| **Receipt / Bill Scanner** | A screenshot of a receipt, bill, or payment confirmation | Merchant, amount, and date pulled out and folded into Money Insight, tagged separately from SMS |
| **Call Shield** | A live microphone recording, a pasted call transcript, or an uploaded WAV | A scam-call verdict tuned to Indian call-fraud patterns — bank/police/courier impersonation, digital-arrest threats, OTP demands |
| **Payment Pause** | Nothing directly — it watches the other two | If Scam Shield or Call Shield flagged something in the last 10 minutes, a simulated payment attempt pauses automatically and shows what was flagged, when, and why |

Call Shield's "live" mode genuinely captures audio through your device's mic and transcribes it on-device — it just can't tap into an actual phone call the way telephony-level software would, since that needs OS/carrier integration a local app doesn't have. Payment Pause is a simulated payment screen for the same reason: intercepting a real UPI or banking app would need integration with that app or the OS payments layer, which is outside what this prototype does. Both limitations are stated plainly in the app itself.

## The AI models

**Compiled and profiled on real Snapdragon X Elite hardware through Qualcomm AI Hub:**

| Model | Job | Inference time | Runs on |
|---|---|---|---|
| EasyOCR — detector | build + profile | 38.1 ms | NPU |
| EasyOCR — recognizer | build + profile | 20.4 ms | NPU |
| MiniLM-v2 text encoder | build + profile | 1.9 ms | NPU |
| Whisper-tiny encoder | build + profile | 27.8 ms | NPU |
| Scam classifier head | build + profile | 41 µs | NPU |
| Spend classifier head | build + profile | 35 µs | NPU |
| Call Shield classifier head | build + profile | 41 µs | NPU |

These aren't estimates — each one went through a real compile job and a real profiling job on physical Snapdragon silicon in Qualcomm's cloud device farm, and the numbers above are what came back. Job links and the full story for each are in [Snapdragon / Qualcomm AI Hub](#snapdragon--qualcomm-ai-hub) below.

**From outside AI Hub, underneath the above:**

- **`sentence-transformers/all-MiniLM-L6-v2`** (HuggingFace) — the base weights AI Hub's own MiniLM-v2 listing is built from. AI Hub's packaged version needs `torch>=2.4`, which has no installable wheel on this dev machine (or, it turns out, on Windows ARM64 either), so `src/pipeline/text_encoder.py` rebuilds the same architecture by hand with the already-installed torch and exports that.
- **`openai/whisper-tiny`** decoder — stays in plain PyTorch, deliberately never sent through ONNX. Autoregressive, token-by-token decoding is the part of Whisper most likely to quietly go wrong if hand-converted; only the encoder (a single forward pass) was compiled for the NPU.
- **EasyOCR's own architecture** — open source, the base the AI Hub-compiled detector and recognizer come from.
- **scikit-learn `LogisticRegression`** — the three classification heads (scam / spend / call) are trained locally on hand-labeled data, then hand-exported to ONNX and compiled through AI Hub. These aren't from any model zoo; they're NXTSight's own, sitting on top of MiniLM's embeddings.

## The On-Device Pipeline

Every model in NXTSight — OCR, the MiniLM encoder, the Whisper encoder, all three classifier heads — loads through exactly one function: `runtime.create_inference_session()`. That function checks `onnxruntime.get_available_providers()` once, at startup. If Qualcomm's QNN provider shows up (a Snapdragon PC with `onnxruntime-qnn` installed), every session prefers it and inference runs on the Hexagon NPU. If it doesn't, sessions fall back to plain CPU — no flag to set, no separate build, same code path either way. This is checked live, not assumed: the app shows which path is active on screen every time it starts.

OCR is the "read" step ahead of everything else for screenshot-based features — it turns an image into raw text, and that text then goes through the same pipeline as any typed message. Nothing downstream cares whether text came from OCR or from a paste box.

MiniLM-v2 is the shared understanding layer underneath all three classifiers. Scam Shield, Money Insight, and Call Shield each encode their input text into the same 384-dimension embedding through MiniLM, and a small trained head sitting on top of that embedding — different for each task — makes the actual call. One encoder doing the language understanding, three lightweight heads doing task-specific judgment, instead of three separate NLP pipelines duplicating the same work.

Whisper is split for the same reason MiniLM's decoder-equivalent risk doesn't apply to it: the encoder (audio in, hidden states out, one forward pass) runs through ONNX/QNN, and the decoder stays in `qai_hub_models`' own unmodified PyTorch loop. Before trusting the hybrid, transcripts from the ONNX-encoder + PyTorch-decoder combination were checked byte-for-byte against an all-PyTorch baseline on every bundled sample recording — identical output on all four.

If any model's `.onnx` file is missing or fails to load — a bad deploy, a cleared cache, whatever — the engine and the OCR/Whisper modules fall back to calling the plain PyTorch or scikit-learn model directly instead of crashing. The feature still works, just without the NPU-accelerated path.

## What runs on-device (NPU vs. CPU fallback)

All of it, and the fallback is real, not theoretical:

- **Scam Shield, Money Insight, and Call Shield's classifiers** run through the QNN-aware engine today. `select_execution_providers()` picks QNN if present, CPU otherwise, and this switch is covered by its own test (`tests/test_runtime.py`), not just described.
- **OCR** prefers the AI Hub-compiled detector/recognizer through the same QNN/CPU path, and falls back to EasyOCR's own PyTorch reader (CPU-only) if the compiled models aren't present.
- **Call Shield's speech-to-text** prefers the AI Hub-compiled Whisper encoder through the same path, with the decoder always on local PyTorch, and falls back to OpenAI's own Whisper package entirely if the compiled encoder isn't available.

On this dev machine (an Intel Mac, no Snapdragon NPU) everything genuinely runs on CPU, and the on-screen badge says so honestly rather than claiming NPU. On a real Snapdragon Windows PC with `onnxruntime-qnn` installed, the exact same code takes the NPU path automatically. Worst case, if QNN somehow doesn't activate on a judge's machine, the CPU fallback is the same code already running here — not a separate untested branch.

## Architecture

`classify_scam()`, `categorize_transactions()`, and `analyze_call()` don't each load their own model. All three call through one shared `NXTSightEngine` ([`src/pipeline/engine.py`](src/pipeline/engine.py)) — the only piece of code that ever touches ONNX Runtime, the QNN provider, or the MiniLM encoder directly. What differs per feature is registered as a `Task`: its own trained head, its own confidence rules.

```
                         ┌───────────────────────────────────────┐
                         │            NXTSightEngine              │
                         │        src/pipeline/engine.py          │
                         │                                         │
                         │  text_guard: is this text analyzable?  │
                         │  text_encoder: MiniLM-v2 embedding      │
                         │  runtime: QNN (NPU) or CPU, auto-detect │
                         └────────────────┬────────────────────────┘
                                           │
                            engine.analyze(task_name, text)
                                           │
          ┌────────────────────────────────┼────────────────────────────────┐
          │                                │                                │
   Task "scam_detection"        Task "spend_categorization"        Task "call_shield"
   classify_scam(text)          categorize_transactions(texts)     analyze_call(transcript)
   → is_scam, confidence,       → categorized, insight             → is_scam, confidence,
     reason                                                          reason
```

OCR ([`src/pipeline/ocr.py`](src/pipeline/ocr.py), or the AI Hub path in [`src/pipeline/ocr_qai_hub.py`](src/pipeline/ocr_qai_hub.py)) and speech-to-text ([`src/pipeline/stt.py`](src/pipeline/stt.py), or [`src/pipeline/whisper_qai_hub.py`](src/pipeline/whisper_qai_hub.py)) sit ahead of the engine, turning an image or a recording into text before it reaches `engine.analyze()`. Payment Pause ([`src/pipeline/payment_pause.py`](src/pipeline/payment_pause.py)) sits to the side of all three, watching whatever they flag.

## Workflow of NXTSight

How a request actually moves through the system, feature by feature:

**Scam Shield:** screenshot → OCR extracts text → engine encodes it with MiniLM → scam classifier head scores it → verdict returned with a confidence score and the triggering phrase. If it's flagged, that flag is written to Payment Pause's log.

**Money Insight:** pasted SMS text → engine encodes each message with MiniLM → spend classifier head sorts it into one of 11 categories → amount and debit/credit direction are pulled out separately with a regex → one summary sentence built across the batch.

**Receipt / Bill Scanner:** screenshot → the same OCR step Scam Shield uses → merchant/amount/date pulled out with layout-aware parsing → rebuilt into a normalized sentence → handed to the exact same spend classifier as Money Insight, tagged as coming from a screenshot instead of SMS.

**Call Shield:** live mic audio, a WAV upload, or a pasted transcript → if audio, Whisper transcribes it (encoder on NPU, decoder on CPU) → engine encodes the transcript with MiniLM → call classifier head scores it, quoting back the specific line that triggered the verdict. Flagged calls also get written to Payment Pause's log.

**Payment Pause:** watches that shared, in-memory-only flag log. A few seconds after any new flag from Scam Shield or Call Shield, it automatically checks whether anything landed in the last 10 minutes — no button required — and if so, shows a simulated payment attempt pausing, with what was flagged, when, and why, before you choose to cancel or proceed anyway.

## User flow

Open the app in a browser (`streamlit run app.py`, usually `http://localhost:8501`) and there are five tabs, all pre-loaded with sample data:

1. **Scam Shield** — pick one of the sample screenshots or upload your own, and it flags the message as scam or not, with a reason.
2. **Money Insight** — load the sample transactions (or paste your own) and hit analyze; get a categorized table and one summary sentence.
3. **Receipt / Bill Scanner** — pick a sample receipt, review what it pulled out, and add it into Money Insight with one click.
4. **Call Shield** — record live through the mic, paste a transcript, or upload a WAV, and get a scam-call verdict with the line that triggered it.
5. **Payment Pause** — after flagging anything in tabs 1 or 4, switch here: a simulated payment pauses on its own a few seconds later, showing exactly what was flagged.

No login, no account, nothing to configure beyond installing dependencies. If your environment is missing something, `python scripts/check_setup.py` (or the app itself) says exactly what and how to fix it.

## Project structure

```
NXTSight/
├── app.py                    # streamlit run app.py — the demo UI, five tabs
├── pages/
│   └── Future_Vision.py      # a separate page: where this could go beyond the hackathon build
├── FUTURE_VISION.md           # source text for the Future Vision page
├── backend/                   # FastAPI service exposing all five features over HTTP
│   ├── main.py                     # uvicorn backend.main:app
│   └── README.md                   # endpoint docs + curl examples
├── src/
│   ├── pipeline/
│   │   ├── engine.py               # NXTSightEngine — the one shared object every task calls through
│   │   ├── runtime.py              # picks QNN (NPU) vs CPU, auto-detected
│   │   ├── text_encoder.py         # MiniLM-v2 shared text encoder
│   │   ├── ocr.py / ocr_qai_hub.py # screenshot -> text (local EasyOCR / AI Hub-compiled path)
│   │   ├── stt.py / whisper_qai_hub.py  # audio -> text (local Whisper / AI Hub-compiled encoder)
│   │   ├── onnx_export.py          # hand-built ONNX export for the three classifier heads
│   │   ├── payment_pause.py        # the in-memory flag log + the "recent flag pauses a payment" rule
│   │   ├── network_guard.py        # blocks and proves-blocked any outbound network call
│   │   ├── text_guard.py           # shared "is this text analyzable" check
│   │   └── preflight.py            # Python version + dependency checks
│   ├── scam_detector/         # Scam Shield: data, training, classifier
│   ├── spend_categorizer/     # Money Insight + Receipt/Bill Scanner: data, training, categorizer, receipt parser
│   └── call_shield/           # Call Shield: data, training, classifier
├── models/                    # AI Hub-compiled artifacts (OCR, MiniLM-v2, Whisper encoder)
├── scripts/                   # export_*.py (local ONNX export) and aihub_compile_*.py (real AI Hub jobs)
├── data/samples/               # sample screenshots, call recordings, transaction text used in the demo
├── tests/                      # 148 tests covering every module above
├── requirements.txt
└── README.md
```

## Snapdragon / Qualcomm AI Hub

Getting a model onto the NPU has two parts: a build-time step, run once by hand, that compiles a model for the Snapdragon X Elite and profiles it on real physical hardware in Qualcomm's cloud device farm; and the runtime check described above, which happens every time the app starts.

All seven models listed in [The AI models](#the-ai-models) went through real, successful compile and profile jobs — not simulated, not estimated. The three custom classifier heads needed a hand-built ONNX export ([`src/pipeline/onnx_export.py`](src/pipeline/onnx_export.py)) to get there: `skl2onnx`'s default converter produces a dynamic batch dimension and an `ai.onnx.ml.LinearClassifier` op that AI Hub's compiler rejects outright. The fix was a static batch size of 1 and a plain `Gemm → Softmax → ArgMax` graph using only core ONNX ops — mathematically identical to scikit-learn's own `predict_proba`, checked against it directly rather than assumed.

MiniLM-v2 and the Whisper encoder followed the same hand-export path for a different reason: AI Hub's own packaged versions of both need `torch>=2.4`, which isn't installable on this dev machine. Both were rebuilt by hand using the already-installed torch, verified numerically against the original PyTorch model before being submitted (MiniLM: max absolute difference `1.1e-7` against the original), and compiled through the raw `qai_hub` API instead of the `qai_hub_models` CLI.

To run any of this yourself, you'll need your own free AI Hub account and API token from [aihub.qualcomm.com](https://aihub.qualcomm.com):

```bash
pip install qai-hub qai-hub-models
qai-hub configure --api_token <YOUR_API_TOKEN>

python -m src.scam_detector.train_classifier       # produces the .onnx
python scripts/aihub_compile_scam_classifier.py     # compiles + profiles it on real Snapdragon hardware
```

Same pattern for `aihub_compile_spend_categorizer.py`, `aihub_compile_call_shield.py`, `aihub_compile_minilm.py`, and `aihub_compile_whisper_encoder.py`. For OCR, `scripts/export_easyocr.py` does the local export and `scripts/aihub_profile_easyocr.py` handles profiling.

## Running it

```bash
git clone https://github.com/yuvrajsch369-fruitgrape/NXTSight.git
cd NXTSight

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python scripts/check_setup.py   # confirms your environment is ready before you go further

streamlit run app.py
```

Full test suite: 148 tests, all passing — `pytest` from the project root with the venv active.
