# NXTSight

Built for the Snapdragon AI Lab Build & Present Challenge.

NXTSight is a small on-device app with two features that share one pipeline:

1. **Scam Screenshot Scanner** — point it at a screenshot of a suspicious text/WhatsApp/email message and it flags whether the message looks like a financial scam.
2. **Spend Insight** — feed it sample bank/UPI transaction text and it categorizes the spending and gives a plain-language insight (e.g. "you spent 30% more on food delivery this month").

Everything runs **fully locally** on a Snapdragon-powered HP PC, using the device's NPU via Qualcomm AI Hub — no cloud calls, no data leaving the machine.

## Live demo

```bash
streamlit run app.py
```

Opens a minimal local web page ([app.py](app.py)) with two tabs — drop in a screenshot and see the scam verdict, or paste/load sample transaction text and see it categorized with an insight. No login, no accounts, nothing that talks to a server outside this machine. It calls the exact same `classify_scam` / `categorize_transactions` functions used by the CLI and the test suite — the UI is a thin wrapper, not a separate code path.

**The UI itself carries a persistent banner** (not just this README) making clear that manual upload/paste is a demo simplification: the real, intended product reads incoming SMS/notifications automatically in the background — the same way Walnut or Money View already do in India — so nobody ever opens an app or types anything.

**Two more status badges sit right above that banner, on every screen:**

- **Execution path** — reads `runtime.select_execution_providers()` and shows the exact same string that gets logged server-side: `CPU (QNN execution provider not available on this machine)` on a dev machine, or `Snapdragon NPU (ONNX Runtime QNN execution provider)` on the real Snapdragon hardware. No need to explain which path is active out loud — it's on screen.

- **Network: blocked & verified** — this isn't a claim, it's a live self-test. [`src/pipeline/network_guard.py`](src/pipeline/network_guard.py) patches `socket.socket.connect()` at app startup so any connection to anything other than localhost raises immediately, then the app itself attempts a real outbound connection (to `8.8.8.8:53`) and confirms its own code rejected it — *before* rendering the rest of the page. If that check ever fails, the app shows a red error and refuses to render anything else (`st.stop()`) rather than quietly continuing. This is exactly the "turn off wifi and nothing changes" claim made demonstrable: the badge is proof captured at that instant, not something asserted out loud. Verified in [tests/test_network_guard.py](tests/test_network_guard.py): loopback connections still work (Streamlit needs those), a blocked attempt raises `NetworkBlockedError`, and OCR/scam-classification/spend-categorization all still work correctly with the guard active — proving the pipeline doesn't secretly depend on network once its models are on disk.

One honest caveat: this proves the *inference* path is offline. A brand-new machine that has never run NXTSight before would still need network once, on first launch, to let EasyOCR download its model weights (a few hundred MB, cached to `~/.EasyOCR/` afterward) — the guard would loudly block that too if wifi were off on a truly first-ever run. For a presentation, run it once beforehand so the cache exists; after that, the offline guarantee holds for real.

## How it works

Both features are really the same three-step pipeline pointed at different inputs: **read** (OCR pulls text out of a screenshot, or the raw transaction text is used as-is), **understand** (a small local model — compiled and run on-device through Qualcomm AI Hub's NPU tooling — classifies or categorizes that text), and **explain** (the result is turned into one plain-language line a non-technical person can read, like "this looks like a scam because it asks you to click a link and act immediately" or "your top spend this month was food delivery"). Sharing that pipeline between the scam-check and spend-insight features means one on-device model-serving layer does both jobs, instead of building two separate apps.

## Architecture: one engine, two jobs

`classify_scam()` and `categorize_transactions()` don't each load a model and run inference themselves. They both call through **one shared `NXTSightEngine` object** ([src/pipeline/engine.py](src/pipeline/engine.py)) — the only piece of code in the whole app that ever touches ONNX Runtime, the Snapdragon NPU/QNN provider, or a TF-IDF vectorizer. What's different per feature is registered as a `Task` (its own trained artifacts, its own confidence rules) — not a second copy of the model-serving code.

```
                         ┌───────────────────────────────────────┐
                         │            NXTSightEngine              │   ← ONE shared object
                         │        src/pipeline/engine.py          │     (verified in
                         │                                         │      tests/test_engine.py:
                         │  text_guard.unanalyzable_reason(text)  │      both task modules
                         │  TF-IDF vectorize                      │      hold the same
                         │  runtime.create_inference_session()    │      `engine` instance)
                         │    → QNN (Snapdragon NPU) or CPU,      │
                         │      auto-detected, every call         │
                         └────────────────┬────────────────────────┘
                                           │
                            engine.analyze(task_name, text)
                                           │
                  ┌────────────────────────┴─────────────────────────┐
                  │                                                    │
        Task "scam_detection"                            Task "spend_categorization"
        artifacts/vectorizer.joblib                       artifacts/vectorizer.joblib
        artifacts/classifier.onnx (binary)                artifacts/classifier.onnx (11-class)
                  │                                                    │
      src/scam_detector/classifier.py                src/spend_categorizer/categorizer.py
      classify_scam(text):                            categorize_transactions(texts):
        - interprets label as is_scam                   - interprets label as a category name
        - reason: top scam/legit terms                  - amount/direction via regex (non-ML)
          from the engine's own vocabulary               - rolls results into one insight
        → {is_scam, confidence, reason}                 → {categorized, insight}
```

Each task still trains its **own** model — a binary scam flag and an 11-way spending category are genuinely different problems, so faking one shared set of weights would just be a worse model for both jobs. What's genuinely shared, end to end, is the serving code: text validation, vectorization, session creation, NPU/CPU detection, session caching. That's the part that actually runs on-device, and it's one object doing it for both features — which is the whole pitch.

## Scam classifier

`classify_scam(text) -> {"is_scam": bool, "confidence": float, "reason": str}` ([src/scam_detector/classifier.py](src/scam_detector/classifier.py)) looks for the patterns behind fake bank alerts, OTP-sharing requests, too-good-to-be-true investment offers, urgent account-blocked threats, and fake delivery/KYC/job-offer scams.

**The model:** a TF-IDF + Logistic Regression classifier trained on 60 labeled examples ([src/scam_detector/data.py](src/scam_detector/data.py)), exported to ONNX (~15KB) via `skl2onnx`, registered as a Task on the shared `NXTSightEngine` (see [Architecture](#architecture-one-engine-two-jobs) above) — so once compiled for Snapdragon via AI Hub, it runs on the NPU too, no code change. Retrain it with:

```bash
python -m src.scam_detector.train_classifier
```

Text vectorization deliberately includes legit messages that mention OTPs and banks — the model needs to learn the *pattern* (being asked to hand over an OTP, or an urgent threat with a link) rather than reacting to individual words like "OTP" or "bank" that show up constantly in harmless messages too.

**Edge cases** — each returns a clear result rather than crashing:

| Input | Result |
|---|---|
| Empty / whitespace-only | `is_scam: false`, `reason: "Couldn't analyze this: no text provided."` |
| Extremely long text | Truncated to 4000 characters, then classified normally |
| Non-English text | `reason: "Couldn't analyze this: detected language '<code>' — this model only supports English."` |
| Gibberish / symbols-only | `reason: "Couldn't analyze this: ..."` (caught by a letter-ratio check and a language-detection confidence threshold) |

**AI Hub path:** unlike EasyOCR, this is a custom model, not one from the AI Hub model zoo, so it compiles via the raw `qai_hub` API rather than the `qai_hub_models` CLI:

```bash
python -m src.scam_detector.train_classifier      # produces the .onnx
python scripts/aihub_compile_scam_classifier.py    # compiles + profiles it on real Snapdragon hardware
```

## Spend categorizer

`categorize_transactions(list_of_texts) -> {"categorized": [...], "insight": str}` ([src/spend_categorizer/categorizer.py](src/spend_categorizer/categorizer.py)) takes a batch of bank/UPI transaction SMS text and sorts each into one of 11 categories (Food & Dining, Groceries, Shopping, Transport, Bills & Utilities, Entertainment, Transfers & UPI P2P, Income & Refunds, Healthcare, Investment & Savings, Cash Withdrawal), then produces one plain-language spending insight across the batch.

**Calls through the same shared `NXTSightEngine`** as the scam classifier (see [Architecture](#architecture-one-engine-two-jobs) above) — TF-IDF + Logistic Regression trained on 88 labeled examples ([src/spend_categorizer/data.py](src/spend_categorizer/data.py)), exported to ONNX via `skl2onnx`, registered as its own Task. This module never imports `onnxruntime` or `joblib` directly; every model-serving line (text validation, vectorization, NPU/CPU session creation) lives once, in the engine, not once per feature. Retrain it with:

```bash
python -m src.spend_categorizer.train_classifier
```

Each item in `categorized` is `{"text", "category", "confidence", "amount", "direction"}` — `amount` and `direction` (`"debit"`/`"credit"`) are pulled out with a small regex, independent of the ML classifier, so `insight` can sum actual rupee amounts by category rather than just counting messages.

**Edge cases** — each returns a clear result rather than crashing:

| Input | Result |
|---|---|
| Empty list / `None` / non-list | `{"categorized": [], "insight": "Couldn't analyze this: no transactions provided."}` |
| A malformed item in the batch (`None`, `""`, a number, gibberish) | That item alone becomes `{"category": "Unrecognized", "confidence": 0.0, ...}` — the rest of the batch still processes normally |
| Text that isn't actually a transaction (e.g. a casual message) | Categorized "Unrecognized" — gated on *two* independent signals: low classifier confidence (11-way softmax, so a real floor is ~0.20, not 0.5) **and** the absence of any parseable amount/debit-credit cue. Confidence alone let a false positive through in testing ("Happy birthday!" scored 0.28 on "Food & Dining"); requiring an actual amount or debit/credit keyword too closed that gap. |
| All items unrecognized | `insight: "Couldn't analyze this: none of the provided transactions could be understood."` |

**Honest limitation:** with only ~8 training examples per category, a few genuinely novel merchant names get misclassified into a neighboring category (e.g. a rent-split payment landed in "Shopping" instead of "Transfers & UPI P2P" in testing) rather than being rejected outright — expected behavior for a tiny bag-of-words model with zero semantic generalization. More labeled examples in `data.py` is the direct fix.

**AI Hub path** (same shape as the scam classifier):

```bash
python -m src.spend_categorizer.train_classifier          # produces the .onnx
python scripts/aihub_compile_spend_categorizer.py          # compiles + profiles it on real Snapdragon hardware
```

## Project structure

```
NXTSight/
├── app.py                   # live demo UI (streamlit run app.py) — screenshot -> verdict, or transactions -> insight
├── .streamlit/config.toml   # disables Streamlit's own telemetry (would otherwise call out)
├── src/
│   ├── pipeline/
│   │   ├── ocr.py            # extract_text_from_image(): screenshot -> raw text
│   │   ├── runtime.py        # picks the ONNX Runtime execution provider (NPU vs CPU)
│   │   ├── text_guard.py     # shared "is this text analyzable" check (gibberish/non-English/empty)
│   │   ├── engine.py         # NXTSightEngine: the one shared object both tasks call through
│   │   └── network_guard.py  # blocks + proves-blocked any non-loopback network connection
│   ├── scam_detector/       # feature 1: screenshot OCR + scam classification
│   │   ├── data.py               # labeled training examples
│   │   ├── train_classifier.py   # local build step: trains + exports classifier.onnx
│   │   ├── classifier.py         # classify_scam(text) -> {is_scam, confidence, reason}
│   │   └── artifacts/            # trained vectorizer.joblib, classifier.onnx, top_terms.json
│   └── spend_categorizer/   # feature 2: transaction text parsing + spend categorization
│       ├── data.py               # labeled training examples (11 categories)
│       ├── train_classifier.py   # local build step: trains + exports classifier.onnx
│       ├── categorizer.py        # categorize_transactions(texts) -> {categorized, insight}
│       └── artifacts/            # trained vectorizer.joblib, classifier.onnx, categories.json
├── models/                  # exported/compiled AI-Hub model artifacts (OCR) for on-device NPU inference
├── scripts/
│   ├── aihub_profile_easyocr.py             # one-off: profile the OCR model on real Snapdragon hardware
│   ├── aihub_compile_scam_classifier.py     # one-off: compile + profile the scam classifier for Snapdragon
│   └── aihub_compile_spend_categorizer.py   # one-off: compile + profile the spend categorizer for Snapdragon
├── data/samples/            # sample screenshots and sample transaction text for demos/tests
├── notebooks/               # exploration / model experimentation
├── tests/                   # unit tests, incl. test_engine.py (proves the shared-object architecture)
├── assets/screenshots/      # demo screenshots for the submission write-up
├── requirements.txt
└── README.md
```

## Status

All three pipeline stages are working: OCR (`extract_text_from_image`), the
NPU/CPU execution-provider auto-detect (`runtime.py`), scam classification
(`classify_scam`), and spend categorization (`categorize_transactions`).

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
[NXTSight] Session ready on 'models/easyocr_detector/model.onnx' — active provider: CPUExecutionProvider
```

On the actual Snapdragon HP PC, once `onnxruntime-qnn` is installed, the same code logs `Execution path: Snapdragon NPU (ONNX Runtime QNN execution provider)` instead — nothing else changes.

`tests/test_runtime.py` proves this end-to-end right now, on this dev machine, using a tiny throwaway ONNX model — no Snapdragon hardware or AI Hub account required to verify the detection logic itself works.
