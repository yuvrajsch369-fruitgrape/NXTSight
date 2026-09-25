# NXTSight

NXTSight is a small on-device engine with two jobs: read a message and tell you, in plain language, whether it looks like a scam and why, and read your transaction messages and tell you, in plain language, where your money is going. Everything runs locally on the machine — screenshot OCR, speech-to-text, every classifier — with no server call involved in a single prediction. Nothing about that "no cloud" claim is a design nicety: it's the whole reason a bank could ever plug this into a real payment flow.

It runs on any modern PC. At startup it checks what hardware is actually there and automatically uses the fastest real accelerator it finds — a Snapdragon NPU, an NVIDIA GPU, an Intel NPU/GPU, a Windows GPU, an Apple Neural Engine — falling back to plain CPU if none of those are present. Same code, same result, either way; nothing to configure. Snapdragon's Hexagon NPU is the fastest path NXTSight supports today (see the measured numbers in the [appendix](#appendix-snapdragon-specific-verification)) — the platform it's been tuned and specifically verified against — but it's one path among several, not the only one this runs on.

## The problem

Roughly 1 in 5 UPI users in India has been targeted by a payment fraud, and government data shows over ₹805 crore lost to digital payment scams this financial year alone. These frauds — fake KYC updates, OTP-sharing requests, too-good investment offers, "your account is blocked" threats, fake delivery fees — arrive as plain text messages, and the people most exposed to them are often the least equipped to spot the pattern in the moment. At the same time, most people have no real visibility into where their money actually goes month to month; bank SMS and UPI notifications pile up unread.

NXTSight addresses both. It reads a suspicious message and tells you whether it looks like a scam and why, and it reads your transaction messages and tells you where your money is going. Both run through the same local model-serving pipeline, entirely on the device, with no cloud dependency.

## Where this sits next to Norton

Gen Digital, Norton's parent company, already ships an AI scam detector called Genie inside Norton 360 — it's real, it's live globally, and it's a genuine competitor in this space. But Genie's own India page says nothing about UPI-specific fraud, fake KYC messages, digital-arrest call scams, or courier-customs fee scams, and Gen Digital's own numbers show its strength sitting in North America and Europe, not India. 

That's the gap NXTSight is built for: depth in exactly the fraud patterns and payment rails that dominate the Indian market, running fully on-device rather than phoning a server for every check.

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

Five real models, none of them calling out to a cloud API:

- **`sentence-transformers/all-MiniLM-L6-v2`** (HuggingFace) — the shared text encoder underneath all three classifiers. Rebuilt by hand in [`src/pipeline/text_encoder.py`](src/pipeline/text_encoder.py) around the already-installed torch, exported to a portable `.onnx`.
- **`openai/whisper-tiny`** — Call Shield's speech-to-text, three tiers deep: the Snapdragon-specific ONNX-exported encoder when that hardware is present, whisper.cpp/GGUF (real Metal/CUDA/Vulkan acceleration everywhere else) as the second tier, plain PyTorch as the final fallback. The autoregressive decoder never runs through anything hand-converted — see [The On-Device Pipeline](#the-on-device-pipeline).
- **EasyOCR** — the detector + recognizer behind Scam Shield and the Receipt Scanner's screenshot reading.
- **Three `scikit-learn LogisticRegression` heads** (scam / spend / call) — trained locally on hand-labeled data, sitting on top of MiniLM's embeddings. Not from any model zoo; these are NXTSight's own.
- **Qwen2.5-1.5B-Instruct** (GGUF, Q4_K_M, via `llama-cpp-python`) — a local LLM second opinion for Scam Shield and Money Insight only, called only when the fast classifier's own confidence is genuinely ambiguous. See [The On-Device Pipeline](#the-on-device-pipeline) below for why and when.

Every one of those ships a portable, vendor-neutral `.onnx` file (hand-exported where the default tooling didn't cooperate — see [`onnx_export.py`](src/pipeline/onnx_export.py)) that runs on whatever hardware accelerator is actually present — see [The On-Device Pipeline](#the-on-device-pipeline) below. On top of that, all seven (the five models above, split into 7 artifacts — EasyOCR is two stages, Whisper's encoder is separate from its decoder) have additionally been compiled and profiled on **real Snapdragon X Elite hardware** through Qualcomm AI Hub, landing at 35–41 microseconds for the three classifier heads and single-digit milliseconds for the larger encoders — genuine, measured numbers, not estimates. Full job-by-job detail is in the [appendix](#appendix-snapdragon-specific-verification) at the bottom of this README.

## The On-Device Pipeline

Every model in NXTSight — OCR, the MiniLM encoder, the Whisper encoder, all three classifier heads — loads through exactly one function: `runtime.create_inference_session()`. That function is a small hardware-abstraction layer over ONNX Runtime's own execution providers, not something hand-rolled per vendor: it checks `onnxruntime.get_available_providers()` once at startup and prefers, in order, Qualcomm's QNN provider (Snapdragon Hexagon NPU), then CUDA (NVIDIA GPU), then OpenVINO (Intel NPU/GPU, Core Ultra and newer), then DirectML (any GPU on Windows), then CoreML (Apple Neural Engine/GPU), falling back to plain CPU if none of those are present. No flag to set, no separate build — same code path picks whichever real accelerator this machine actually has. This is checked live, not assumed: the app shows which path is active on screen every time it starts.

OCR is the "read" step ahead of everything else for screenshot-based features — it turns an image into raw text, and that text then goes through the same pipeline as any typed message. Nothing downstream cares whether text came from OCR or from a paste box.

MiniLM-v2 is the shared understanding layer underneath all three classifiers. Scam Shield, Money Insight, and Call Shield each encode their input text into the same 384-dimension embedding through MiniLM, and a small trained head sitting on top of that embedding — different for each task — makes the actual call. One encoder doing the language understanding, three lightweight heads doing task-specific judgment, instead of three separate NLP pipelines duplicating the same work.

Whisper's autoregressive decoder never runs through anything hand-converted, on any tier — only the encoder (audio in, hidden states out, one forward pass) gets accelerated. Three tiers, in priority order ([`src/pipeline/stt.py`](src/pipeline/stt.py)): the Snapdragon-specific ONNX-exported encoder (`whisper_qai_hub.py`, decoder stays in `qai_hub_models`' own unmodified PyTorch loop) when that hardware is present; [`whisper.cpp`](src/pipeline/whisper_cpp.py)/GGUF next, a real binding to the upstream `ggml-org/whisper.cpp` project with its own Metal/CUDA/Vulkan/CPU dispatch — genuine GPU acceleration on machines that have no Snapdragon NPU at all, this dev Mac included (previously pure CPU); plain PyTorch as the final, always-available fallback. Every tier was checked against the same bundled sample recordings before being trusted — the AI Hub tier byte-for-byte against an all-PyTorch baseline (identical on all four), whisper.cpp against the classifier's actual verdict on all four (different model artifact — GGUF-quantized — so not byte-identical, but every sample still lands on the correct scam/legit call).

If any model's `.onnx` file is missing or fails to load — a bad deploy, a cleared cache, whatever — the engine and the OCR/Whisper modules fall back to calling the plain PyTorch or scikit-learn model directly instead of crashing. The feature still works, just without the NPU-accelerated path.

**A hybrid fast-path + LLM tier, for Scam Shield and Money Insight only.** MiniLM + a trained head is fast (single-digit milliseconds) but was trained on a few dozen to a couple hundred examples — genuinely uncertain on inputs unlike anything it's seen. `engine.py`'s `_maybe_escalate()` checks the *margin* between the model's top two guesses (not raw confidence alone — the right measure for an 11-way category task, where a binary confidence cutoff doesn't translate) after every fast-path prediction, and only when that margin is below a per-task threshold does it call [`llm_classifier.py`](src/pipeline/llm_classifier.py) — a local Qwen2.5-1.5B-Instruct model (GGUF, via `llama-cpp-python`), prompted with the task's own category descriptions and a JSON schema that constrains its answer to one of the real labels. Real, measured latency when it does fire: a few seconds, not milliseconds — this is why it's an escalation, not the default path. If the LLM is unavailable for any reason (not installed, model not downloaded, a malformed response), the fast path's own result is kept unchanged — escalation is strictly additive, never a new way for classification to break. Call Shield does not use this — its classifier stays exactly as it was.

**The calling code has zero knowledge of any of this.** `classify_scam()` and `categorize_transactions()` call `engine.analyze(task_name, text)` and interpret a `Prediction` — they never import `onnxruntime`, reference a provider name, or know an LLM exists (checked structurally in [`tests/test_llm_escalation.py`](tests/test_llm_escalation.py)). Whether that `Prediction` came from ONNX Runtime on QNN, CoreML, CPU, or the LLM escalation is entirely engine.py's decision, hidden behind one interface — the same engine, unchanged at the call site, could serve predictions from inside a fintech's own backend instead of this standalone app.

## What runs on-device (NPU / GPU vs. CPU fallback)

All of it, and the fallback is real, not theoretical:

- **Scam Shield, Money Insight, and Call Shield's classifiers** run through the hardware-abstracted engine today. `select_execution_providers()` picks the best real accelerator ONNX Runtime reports — QNN, CUDA, OpenVINO, DirectML, CoreML, in that order — CPU otherwise, and every path is covered by its own test (`tests/test_runtime.py`) that runs the real production classifier model, not a synthetic stand-in — see the table below for exactly what that does and doesn't prove per path.
- **OCR** prefers the AI Hub-compiled detector/recognizer through the same hardware-abstracted path, and falls back to EasyOCR's own PyTorch reader (CPU-only) if the compiled models aren't present.
- **Call Shield's speech-to-text** tries the AI Hub-compiled Whisper encoder first (decoder always on local PyTorch), then whisper.cpp/GGUF for real Metal/CUDA/Vulkan acceleration, then falls back to OpenAI's own Whisper package entirely if neither accelerated tier is available.

**Tested honestly, path by path — here's exactly what's real hardware versus simulated:**

| Path | Decision logic tested | Real model runs on it | Genuine hardware exercised |
|---|---|---|---|
| CoreML (Apple Neural Engine/GPU) | ✅ | ✅ | ✅ — this dev machine has it; the badge on screen genuinely reads "Apple Neural Engine / GPU", confirmed by running the real scam classifier through it (0.861 confidence, same verdict as CPU) |
| CPU (no accelerator) | ✅ | ✅ | ✅ — the universal fallback, exercised on every test run |
| QNN (Snapdragon NPU) | ✅ | ✅ | ❌ — no Snapdragon device to test on directly (see the [appendix](#appendix-snapdragon-specific-verification) for the separate, real Snapdragon hardware verification via Qualcomm AI Hub, which *is* genuine physical-chip evidence, just not through this exact code path) |
| CUDA (NVIDIA GPU) | ✅ | ✅ | ❌ — no NVIDIA GPU available to this project |
| OpenVINO (Intel NPU/GPU) | ✅ | ✅ | ❌ — no Intel-NPU/GPU-plugin hardware available to this project, and `onnxruntime-openvino` (the package that provides this exact EP) has no macOS wheels at all, so it can't even be installed on this dev machine to check — confirmed directly against every release on PyPI. Separately, and for real: `scripts/export_openvino_ir.py` converted all 7 of this project's production models to OpenVINO's native IR format with **zero unsupported-operator issues**, numerically verified against ONNX Runtime CPU output (max abs diff `1.08e-07` on MiniLM) — genuine evidence the models themselves are OpenVINO-compatible, just not proof of the runtime EP or the accelerator silicon |
| DirectML (Windows GPU) | ✅ | ✅ | ❌ — no Windows-GPU machine available to this project |

"Decision logic tested" means `tests/test_runtime.py` mocks `onnxruntime.get_available_providers()` to simulate each provider being present and checks `select_execution_providers()` picks it correctly. "Real model runs on it" is a step further, closing a real gap a decision-only test would miss: with each provider mocked as available, NXTSight's actual `classifier.onnx` (not a toy graph) is loaded and run through `create_inference_session()` for real, and — for QNN, CUDA, OpenVINO, and DirectML specifically — checked to reach the *same verdict* as every other path on the same input. What none of this proves for QNN/CUDA/OpenVINO/DirectML is that the acceleration itself is genuine: without the physical chip, ONNX Runtime silently falls back to CPU under the hood (a real, confirmed `UserWarning`, not a guess) even though the code path — every line `create_inference_session()` runs — executes for real. On a real Snapdragon Windows PC with `onnxruntime-qnn` installed, the exact same code takes the genuine NPU path automatically; worst case, if no accelerator activates on a given machine, the CPU fallback is the same code already running in CI, not a separate untested branch.

To reproduce the OpenVINO model-conversion check yourself (`pip install openvino==2025.4.1` — see `requirements.txt` for why that exact version and why it's optional):

```bash
python scripts/export_openvino_ir.py
```

Converts all 7 production models to `models/openvino_ir/` and prints a per-model pass/fail — that script is also what to reach for first if a *future* model addition ever doesn't convert cleanly: it reports the exact op OpenVINO's converter rejected and the real options (operator substitution, opset change — not a silent workaround), rather than guessing.

## Architecture

`classify_scam()`, `categorize_transactions()`, and `analyze_call()` don't each load their own model. All three call through one shared `NXTSightEngine` ([`src/pipeline/engine.py`](src/pipeline/engine.py)) — the only piece of code that ever touches ONNX Runtime, a provider name, the MiniLM encoder, or the LLM directly. What differs per feature is registered as a `Task`: its own trained head, its own confidence rules, and — for scam/spend only — its own LLM escalation config.

```
                         ┌────────────────────────────────────────────┐
                         │               NXTSightEngine                │
                         │           src/pipeline/engine.py            │
                         │                                              │
                         │  text_guard: is this text analyzable?       │
                         │  text_encoder: MiniLM-v2 embedding           │
                         │  runtime: QNN / CUDA / OpenVINO /          │
                         │           DirectML / CoreML / CPU —        │
                         │           auto-detected                    │
                         │  _maybe_escalate(): fast-path margin too     │
                         │    close? → llm_classifier.py (Qwen2.5-1.5B) │
                         └─────────────────────┬────────────────────────┘
                                                │
                                 engine.analyze(task_name, text)
                                   → one Prediction either way,
                                     caller never knows which path
                                                │
          ┌─────────────────────────────────────┼─────────────────────────────────────┐
          │                                     │                                     │
   Task "scam_detection"             Task "spend_categorization"             Task "call_shield"
   classify_scam(text)               categorize_transactions(texts)          analyze_call(transcript)
   → is_scam, confidence, reason     → categorized, insight                  → is_scam, confidence, reason
   (fast path, or LLM escalation)    (fast path, or LLM escalation)          (fast path only — no LLM tier)
```

OCR ([`src/pipeline/ocr.py`](src/pipeline/ocr.py), or the AI Hub path in [`src/pipeline/ocr_qai_hub.py`](src/pipeline/ocr_qai_hub.py)) and speech-to-text ([`src/pipeline/stt.py`](src/pipeline/stt.py), or [`src/pipeline/whisper_qai_hub.py`](src/pipeline/whisper_qai_hub.py)) sit ahead of the engine, turning an image or a recording into text before it reaches `engine.analyze()`. Payment Pause ([`src/pipeline/payment_pause.py`](src/pipeline/payment_pause.py)) sits to the side of all three, watching whatever they flag.

## Workflow of NXTSight

How a request actually moves through the system, feature by feature:

**Scam Shield:** screenshot → OCR extracts text → engine encodes it with MiniLM → scam classifier head scores it → if that result is genuinely ambiguous, a local LLM gives a second opinion instead → verdict returned with a confidence score and the triggering phrase. If it's flagged, that flag is written to Payment Pause's log.

**Money Insight:** pasted SMS text → engine encodes each message with MiniLM → spend classifier head sorts it into one of 11 categories, escalating to the LLM on genuinely ambiguous ones → amount and debit/credit direction are pulled out separately with a regex → one summary sentence built across the batch.

**Receipt / Bill Scanner:** screenshot → the same OCR step Scam Shield uses → merchant/amount/date pulled out with layout-aware parsing → rebuilt into a normalized sentence → handed to the exact same spend classifier as Money Insight, tagged as coming from a screenshot instead of SMS.

**Call Shield:** live mic audio, a WAV upload, or a pasted transcript → if audio, Whisper transcribes it (encoder accelerated where possible — Snapdragon NPU, then whisper.cpp/Metal/CUDA, then CPU — decoder always on CPU) → engine encodes the transcript with MiniLM → call classifier head scores it, quoting back the specific line that triggered the verdict. Flagged calls also get written to Payment Pause's log.

**Payment Pause:** watches that shared, in-memory-only flag log. A few seconds after any new flag from Scam Shield or Call Shield, it automatically checks whether anything landed in the last 10 minutes — no button required — and if so, shows a simulated payment attempt pausing, with what was flagged, when, and why, before you choose to cancel or proceed anyway.

## User flow

Open the app in a browser (`streamlit run app.py`, usually `http://localhost:8501`) and there are five tabs, all pre-loaded with sample data. A highlighted "Future Vision →" card sits right on the main page — no sidebar digging required to find it. The whole app — this page and Future Vision — is dark mode only, no toggle, no light-mode flash on load.

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
│   └── ✦_Future_Vision.py    # a separate page: where this could go beyond today's build
├── FUTURE_VISION.md           # source text for the Future Vision page
├── PROJECT_DESCRIPTION.md     # a shorter, pitch-oriented overview alongside this README
├── assets/theme.css           # the shared dark theme — fonts, gradients, cards, animations
├── backend/                   # FastAPI service exposing all five features over HTTP
│   ├── main.py                     # uvicorn backend.main:app
│   └── README.md                   # endpoint docs + curl examples
├── src/
│   ├── ui/
│   │   └── theme.py                # injects assets/theme.css + small markup helpers (hero, section, badges)
│   ├── pipeline/
│   │   ├── engine.py               # NXTSightEngine — the one shared object every task calls through,
│   │   │                           #   incl. _maybe_escalate() — the fast-path -> LLM decision
│   │   ├── runtime.py              # picks QNN / CUDA / OpenVINO / DirectML / CoreML / CPU, auto-detected
│   │   ├── text_encoder.py         # MiniLM-v2 shared text encoder
│   │   ├── llm_classifier.py       # local LLM second opinion (Qwen2.5-1.5B, GGUF) — optional
│   │   ├── ocr.py / ocr_qai_hub.py # screenshot -> text (local EasyOCR / AI Hub-compiled path)
│   │   ├── stt.py                  # audio -> text: tries whisper_qai_hub, then whisper_cpp, then local Whisper
│   │   ├── whisper_qai_hub.py      # tier 1: AI Hub-compiled Snapdragon encoder (optional)
│   │   ├── whisper_cpp.py          # tier 2: whisper.cpp/GGUF, Metal/CUDA/Vulkan (optional)
│   │   ├── onnx_export.py          # hand-built ONNX export for the three classifier heads
│   │   ├── payment_pause.py        # the in-memory flag log + the "recent flag pauses a payment" rule
│   │   ├── network_guard.py        # blocks and proves-blocked any outbound network call
│   │   ├── text_guard.py           # shared "is this text analyzable" check
│   │   └── preflight.py            # Python version + dependency checks
│   ├── scam_detector/         # Scam Shield: data, training, classifier
│   ├── spend_categorizer/     # Money Insight + Receipt/Bill Scanner: data, training, categorizer, receipt parser
│   └── call_shield/           # Call Shield: data, training, classifier
├── models/
│   ├── llm/                        # Qwen2.5-1.5B-Instruct GGUF weights (downloaded, not committed)
│   └── ...                         # AI Hub-compiled artifacts (OCR, MiniLM-v2, Whisper encoder)
├── scripts/                   # export_*.py (local ONNX export), aihub_compile_*.py (real AI Hub jobs),
│                               #   fix_pywhispercpp_macos.py (a real macOS packaging fix, see requirements.txt)
├── data/samples/               # sample screenshots, call recordings, transaction text used in the demo
├── tests/                      # 178 tests covering every module above
├── requirements.txt
└── README.md
```

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

Full test suite: 178 tests, all passing — `pytest` from the project root with the venv active. Nothing above touches Qualcomm AI Hub, an account, or a network call — see the appendix below if you want the Snapdragon-specific detail.

## Appendix: Snapdragon-specific verification

Everything above already runs on real hardware acceleration wherever it's available — QNN, CUDA, OpenVINO, DirectML, or CoreML, picked automatically (see [The On-Device Pipeline](#the-on-device-pipeline)). This appendix is additional, optional proof that the same models were also compiled and profiled on **real Snapdragon X Elite hardware** through Qualcomm AI Hub, for anyone who wants the specific numbers:

| Model | Inference time | Compute unit |
|---|---|---|
| EasyOCR detector | 38.1 ms | NPU |
| EasyOCR recognizer | 20.4 ms | NPU |
| MiniLM-v2 text encoder | 1.9 ms | NPU |
| Whisper-tiny encoder | 27.8 ms | NPU |
| Scam classifier head | 41 µs | NPU |
| Spend classifier head | 35 µs | NPU |
| Call Shield classifier head | 41 µs | NPU |

All seven went through real, successful compile and profile jobs on physical Snapdragon silicon in Qualcomm's cloud device farm — not simulated, not estimated. Getting there needed a hand-built ONNX export for the three classifier heads ([`onnx_export.py`](src/pipeline/onnx_export.py) — `skl2onnx`'s default converter emits a dynamic batch dimension and an `ai.onnx.ml` op AI Hub's compiler rejects outright) and the same hand-export approach for MiniLM-v2 and the Whisper encoder (AI Hub's own packaged versions need `torch>=2.4`, unavailable on this dev machine) — both verified numerically against the original PyTorch models before submitting.

To reproduce this yourself, you'll need a free AI Hub account and API token from [aihub.qualcomm.com](https://aihub.qualcomm.com):

```bash
pip install qai-hub qai-hub-models
qai-hub configure --api_token <YOUR_API_TOKEN>

python -m src.scam_detector.train_classifier       # produces the .onnx
python scripts/aihub_compile_scam_classifier.py     # compiles + profiles it on real Snapdragon hardware
```

Same pattern for `aihub_compile_spend_categorizer.py`, `aihub_compile_call_shield.py`, `aihub_compile_minilm.py`, and `aihub_compile_whisper_encoder.py`. For OCR, `scripts/export_easyocr.py` does the local export and `scripts/aihub_profile_easyocr.py` handles profiling.
