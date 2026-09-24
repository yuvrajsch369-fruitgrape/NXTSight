# NXTSight Backend

A FastAPI service exposing all five NXTSight features over HTTP, built on the exact same shared engine the Streamlit demo uses — not a second implementation sitting next to it.

**If you only try one thing, try this:** start the server, then run the Payment Pause scenario near the bottom of this file. It flags a real scam message, then shows a simulated payment attempt getting paused because of it — over real HTTP calls, not a UI mock.

## Run it

```bash
uvicorn backend.main:app --port 8000
```

Then either open **http://localhost:8000/docs** for interactive Swagger docs, or use the curl commands below. If a dependency is missing, the server prints exactly which one and refuses to start, rather than failing mid-request with a buried `ImportError`.

## Endpoints

| Feature | Endpoint(s) | Engine task |
|---|---|---|
| Scam Shield | `POST /scam-shield/text`, `POST /scam-shield/screenshot` | `scam_detection` |
| Money Insight | `POST /money-insight/transactions` | `spend_categorization` |
| Screenshot Spend Scanner | `POST /spend-scanner/screenshot` | `spend_categorization` |
| Call Shield | `POST /call-shield/transcript`, `POST /call-shield/recording` | `call_shield` |
| Payment Pause | `GET /payment-pause/log`, `POST /payment-pause/confirm`, `POST /payment-pause/reset` | *(no model — pure logic)* |
| Status | `GET /status` | — |

Every ML-backed route registers its own `Task` on the same `NXTSightEngine` instance ([`src/pipeline/engine.py`](../src/pipeline/engine.py)) — none of them loads a model file or opens an ONNX Runtime session directly. All three classifiers (scam, spend, call) run on MiniLM-v2 embeddings under the hood, each with its own small trained head on top; see the main [README](../README.md#the-ai-models) for how that's wired.

Call Shield and Payment Pause are both scoped honestly: `/call-shield/recording` transcribes a WAV file on-device and analyzes the transcript, but this backend can't intercept a live phone call — that needs telephony-level OS integration a local server doesn't have. `/payment-pause/confirm` simulates a payment attempt against an in-memory flag log; it doesn't and can't intercept a real UPI or banking app.

## What `/status` actually proves

```bash
curl -s http://localhost:8000/status
```

```json
{
  "execution_path": "Apple Neural Engine / GPU (ONNX Runtime CoreML execution provider)",
  "network_isolated": true,
  "ai_hub_models": {
    "ocr": {"active": true, "status": "..."},
    "minilm_v2_text_encoder": {"active": true, "status": "..."},
    "whisper_encoder": {"active": true, "status": "..."}
  },
  "call_shield_speech_to_text": {
    "tier_1_snapdragon_ai_hub": {"active": true, "status": "..."},
    "tier_2_whisper_cpp": {"active": true, "status": "whisper.cpp / GGUF (ggml hardware backend — Metal, CUDA, Vulkan, or CPU, auto-detected at build time)"}
  }
}
```

Call Shield's speech-to-text is three tiers deep — the Snapdragon-specific encoder above, then [`whisper.cpp`](https://github.com/ggml-org/whisper.cpp)/GGUF (a real binding, `pywhispercpp`, giving genuine Metal/CUDA/Vulkan acceleration on machines with no Snapdragon NPU — this dev machine included), then plain PyTorch as the final fallback. `call_shield_speech_to_text` in `/status` reports both accelerated tiers independently; see [`src/pipeline/stt.py`](../src/pipeline/stt.py) for the full chain and `requirements.txt` for why `pywhispercpp` is optional (it needs building from source for GPU acceleration, which needs `cmake` and, on macOS, a known packaging fix — [`scripts/fix_pywhispercpp_macos.py`](../scripts/fix_pywhispercpp_macos.py)).

Two things behind that response are checked for real at startup, not assumed:

- **Execution path** — `select_execution_providers()` ([`src/pipeline/runtime.py`](../src/pipeline/runtime.py)) calls `onnxruntime.get_available_providers()` once and picks the best real accelerator it reports, in order: QNN (Snapdragon NPU), CUDA (NVIDIA GPU), DirectML (Windows GPU), CoreML (Apple Neural Engine/GPU), CPU otherwise. This dev machine is an Intel Mac with no Snapdragon NPU, but it does have Apple's CoreML provider — the response above is genuinely live, not a stand-in for what CPU would say. On a Snapdragon Windows PC, the same code takes the NPU branch with zero changes; QNN, CUDA, and DirectML are verified at the selection-logic level (`tests/test_runtime.py` mocks each), not on real hardware this project has access to.
- **Network isolation** — [`network_guard.py`](../src/pipeline/network_guard.py) patches `socket.socket.connect` to reject anything non-loopback, then makes a real connection attempt to `8.8.8.8:53` to confirm its own patch actually rejected it. If that check fails, the server refuses to start rather than silently serving requests over an unproven "offline" claim.

## The ONNX-optional fallback, actually exercised

Every classifier prefers its `classifier.onnx` through ONNX Runtime, but falls back to calling the trained `classifier.joblib` directly if the ONNX file is missing — proven, not just described:

```
$ mv src/scam_detector/artifacts/classifier.onnx /tmp/classifier.onnx.bak
$ python3 -c "from src.scam_detector.classifier import classify_scam; \
              print(classify_scam('Your account will be blocked in 2 hours unless you verify now. Click here.'))"
[NXTSight] No classifier.onnx for task 'scam_detection' — falling back to the trained
scikit-learn model directly (no ONNX Runtime acceleration for this task until
it's re-exported).
{'is_scam': True, 'confidence': 0.861, 'reason': "Contains phrases commonly seen in scams: 'hours', 'verify', 'click', 'unless'."}
```

Same confidence score as the ONNX path on the same input — the fallback is the same math, just without ONNX Runtime in the loop. (The exact number shifts by a thousandth or two depending on which execution provider is active — CoreML's floating-point kernels aren't bit-identical to CPU's, 0.861 here vs. 0.863 on plain CPU — never enough to flip a verdict, and re-checked live on both paths before writing this down.) OCR and speech-to-text have their own equivalent fallbacks (AI Hub-compiled model → local PyTorch model), and every failure mode returns a plain string instead of crashing:

```bash
curl -s -X POST http://localhost:8000/scam-shield/screenshot -F "file=@/tmp/garbage.png"
# {"detail":"Error: '/tmp/garbage.png' is not a readable image (corrupted or unsupported format)"}
```

That's a 422, not a 500 or a raw stack trace.

## Try every endpoint yourself

### Scam Shield

```bash
curl -s -X POST http://localhost:8000/scam-shield/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Dear customer, your account will be BLOCKED today due to pending KYC update. Click here to update immediately and avoid suspension."}'

curl -s -X POST http://localhost:8000/scam-shield/screenshot \
  -F "file=@data/samples/scam_bank_kyc_alert.png"
```

### Money Insight

```bash
curl -s -X POST http://localhost:8000/money-insight/transactions \
  -H "Content-Type: application/json" \
  -d '{"transactions": [
    "Rs 450.00 debited from A/c XX1234 at SWIGGY BANGALORE on 12-Sep-25.",
    "Rs 3,499.00 debited from A/c XX1234 at AMAZON on 12-Sep-25.",
    "You have received Rs 45,000.00 via NEFT from ABC CORP SALARY on 01-Sep-25."
  ]}'
```

### Screenshot Spend Scanner

```bash
curl -s -X POST http://localhost:8000/spend-scanner/screenshot \
  -F "file=@data/samples/receipt_upi_confirmation.png"
```

### Call Shield

```bash
curl -s -X POST http://localhost:8000/call-shield/transcript \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Caller: This is the Cyber Crime unit. Transfer all funds from your savings account right now and read out the OTP the moment it arrives."}'

curl -s -X POST http://localhost:8000/call-shield/recording \
  -F "file=@data/samples/call_legit_bank_call.wav"
```

### Payment Pause — the full scenario

```bash
# 1. Clean slate
curl -s -X POST http://localhost:8000/payment-pause/reset

# 2. Confirm a payment with nothing flagged — proceeds normally
curl -s -X POST http://localhost:8000/payment-pause/confirm

# 3. Flag a scam
curl -s -X POST http://localhost:8000/scam-shield/text \
  -H "Content-Type: application/json" \
  -d '{"text": "URGENT: verify your account now or it will be suspended. Click immediately."}'

# 4. See it in the log
curl -s http://localhost:8000/payment-pause/log

# 5. Confirm a payment again — now it's paused, citing exactly what was flagged
curl -s -X POST http://localhost:8000/payment-pause/confirm
```

Step 5 is the actual pitch: a scam message flagged moments ago automatically pauses the next payment attempt, with what/when/why shown back.

## Tests

```bash
python -m pytest tests/test_backend.py -v
```

11 tests hitting every endpoint through FastAPI's `TestClient` — real inference, real OCR, real speech-to-text, no mocks. Part of the full suite (`python -m pytest`, 159 tests total).
