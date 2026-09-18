# NXTSight Backend

One FastAPI process exposing all five NXTSight features over HTTP, built on top of the same shared on-device engine the Streamlit demo already uses — not a second implementation next to it.

**Try this first, if you only try one thing:** start the server, then run the "full scenario" curl sequence near the bottom of this file. It flags a real scam message, then shows a simulated payment attempt getting paused because of it — the actual pitch, end to end, over real HTTP calls.

## Run it

```bash
uvicorn backend.main:app --port 8000
```

Then either:
- Open **http://localhost:8000/docs** for interactive Swagger docs (click "Try it out" on any endpoint), or
- Use the curl commands below.

If a dependency is missing, the server prints exactly which one and refuses to start — it does not fail with a buried `ImportError` mid-request.

---

## What was reused vs. what's new here

Requirements 1–5 below describe infrastructure that **already existed and was already tested and audited** before this backend was built: `src/pipeline/engine.py` (the shared inference engine), `src/pipeline/runtime.py` (QNN/CPU auto-detection), `src/pipeline/network_guard.py` (offline proof), `src/pipeline/ocr.py`, and the three trained classifiers — all already covered by 137 passing tests, and all three classifiers already genuinely compiled and profiled on real Snapdragon X Elite hardware via Qualcomm AI Hub (see the main [README.md](../README.md#snapdragon--qualcomm-ai-hub) for those job URLs and numbers).

**What's actually new in this pass:** the `backend/` folder itself (this FastAPI service — Requirement 6), the ONNX-optional fallback in `engine.py` and the `classifier.joblib` saves in every `train_classifier.py` (Requirement 2's resilience clause), and this README (Requirement 7). Said plainly instead of presented as if built from scratch, because that's the honest version.

---

## Requirement-by-requirement

### Requirement 1 — provably Snapdragon-optimized, not just claimed

`backend/main.py` calls `select_execution_providers()` from [`src/pipeline/runtime.py`](../src/pipeline/runtime.py) once, at process startup — not per-request:

```python
available = ort.get_available_providers()
if QNN_PROVIDER in available:
    providers = [(QNN_PROVIDER, {"backend_path": _QNN_BACKEND_PATH}), "CPUExecutionProvider"]
    description = "Snapdragon NPU (ONNX Runtime QNN execution provider)"
else:
    providers = ["CPUExecutionProvider"]
    description = "CPU (QNN execution provider not available on this machine)"
```

This is a real `onnxruntime.get_available_providers()` call, not a comment. `GET /status` surfaces the result:

```
$ curl -s http://localhost:8000/status
{
  "execution_path": "CPU (QNN execution provider not available on this machine)",
  "network_isolated": true,
  ...
}
```

**Honest state on this dev machine (Intel Mac):** `onnxruntime-qnn` is a Windows-only package (see `requirements.txt`'s platform marker), so this always falls back to CPU here — that's the code correctly detecting reality, not a bug. On a real Snapdragon Windows PC with `onnxruntime-qnn` installed, the exact same code takes the NPU branch with zero changes. Separately — and this is real, not aspirational — all three classifiers plus the OCR detector/recognizer have already been compiled and profiled on **actual Snapdragon X Elite hardware** via Qualcomm AI Hub; see the main README for the job URLs.

### Requirement 2 — a real, locally-trained model, with a graceful ONNX-optional fallback

Both classifiers are TF-IDF + Logistic Regression, trained on labeled examples in `data.py`, via `train_classifier.py`:

```bash
python -m src.scam_detector.train_classifier       # 68 labeled scam/legit examples
python -m src.spend_categorizer.train_classifier    # 88 labeled examples, 11 categories
```

**The export step is isolated and fails gracefully, for real — not just described that way.** Every `train_classifier.py` now does this:

```python
dump(classifier, ARTIFACTS_DIR / "classifier.joblib")   # always saved — this never depends on ONNX
try:
    onnx_model = export_logistic_regression(classifier, features.shape[1])
    (ARTIFACTS_DIR / "classifier.onnx").write_bytes(onnx_model.SerializeToString())
except Exception as e:
    print(f"WARNING: ONNX export failed/unavailable ({e}). classifier.joblib was still saved...")
```

And [`src/pipeline/engine.py`](../src/pipeline/engine.py)'s `_ensure_loaded()` prefers the ONNX/QNN-aware path but falls back to calling the raw scikit-learn model directly if `classifier.onnx` is missing:

```python
if onnx_path.exists():
    self._sessions[task_name] = ("onnx", session)          # Requirement 1's path
elif joblib_path.exists():
    logger.warning("No classifier.onnx for task '%s' — falling back to the trained "
                    "scikit-learn model directly...", task_name)
    self._sessions[task_name] = ("sklearn", load(joblib_path))
```

**Proven live, not just claimed** — `classifier.onnx` was temporarily removed for the scam classifier and a real request run through it:

```
$ mv src/scam_detector/artifacts/classifier.onnx /tmp/classifier.onnx.bak
$ python3 -c "from src.scam_detector.classifier import classify_scam; \
              print(classify_scam('Your account will be blocked in 2 hours unless you verify now. Click here.'))"
[NXTSight] No classifier.onnx for task 'scam_detection' — falling back to the trained
scikit-learn model directly (no ONNX Runtime / QNN acceleration for this task until
it's re-exported).
{'is_scam': True, 'confidence': 0.564, 'reason': "Contains phrases commonly seen in scams: ..."}
```

Confidence **0.564** — identical to the ONNX path's result on the same input, restored right after. The fallback isn't a degraded approximation; it's the same math, just without ONNX Runtime / QNN in the loop.

**Honest state in this environment:** ONNX export tooling (`onnx` package) **is** installed here, so it didn't actually fail on its own — the test above was performed by deliberately removing the already-exported file to exercise the exact code path a genuinely missing `onnx` package would trigger, since that's a faithful, honest way to prove the fallback without breaking this venv's real install.

### Requirement 3 — real OCR, never crashes

[`src/pipeline/ocr.py`](../src/pipeline/ocr.py) wraps EasyOCR (`easyocr.Reader(["en"])`), with every failure mode returning a clear string instead of raising:

```
$ curl -s -X POST http://localhost:8000/scam-shield/screenshot -F "file=@/tmp/garbage.png"
{"detail":"Error: '/tmp/garbage.png' is not a readable image (corrupted or unsupported format)"}
```
HTTP 422, not a 500 or a stack trace.

### Requirement 4 — five thin features, one shared engine

| Feature | Endpoint(s) | Module | Engine task |
|---|---|---|---|
| Scam Shield | `POST /scam-shield/text`, `POST /scam-shield/screenshot` | `src/scam_detector/classifier.py` | `scam_detection` |
| Money Insight | `POST /money-insight/transactions` | `src/spend_categorizer/categorizer.py` | `spend_categorization` |
| Screenshot Spend Scanner | `POST /spend-scanner/screenshot` | `src/spend_categorizer/receipt_parser.py` (reuses `ocr.py`, feeds `categorizer.py`) | `spend_categorization` |
| Call Shield | `POST /call-shield/transcript`, `POST /call-shield/recording` | `src/call_shield/classifier.py` | `call_shield` |
| Payment Pause | `GET /payment-pause/log`, `POST /payment-pause/confirm`, `POST /payment-pause/reset` | `src/pipeline/payment_pause.py` | *(pure logic — no model)* |

Every one of the four ML-backed features registers its own `Task` on the exact same `NXTSightEngine` instance ([`src/pipeline/engine.py`](../src/pipeline/engine.py)) — none of them loads a model file or opens an ONNX Runtime session itself. **Call Shield is honestly scoped**: `POST /call-shield/recording` transcribes a WAV *file* on-device first (real Whisper STT), then runs the same transcript analysis as `/call-shield/transcript` — this backend does not and cannot intercept a live phone call; that would need phone/telephony-level OS integration outside a local server's reach. **Payment Pause is honestly scoped** too: `/payment-pause/confirm` simulates a payment attempt against an in-memory flag log; it does not and cannot intercept a real UPI/banking app.

### Requirement 5 — proven offline, not just asserted

At import time, before a single route is registered:

```python
network_guard.install()
try:
    network_guard.verify_blocked()
except Exception as exc:
    print(f"FATAL: network isolation could not be verified ({exc}). Refusing to start...")
    raise SystemExit(1)
```

[`src/pipeline/network_guard.py`](../src/pipeline/network_guard.py) patches `socket.socket.connect` to reject any non-loopback address, then `verify_blocked()` makes a **real** connection attempt to `8.8.8.8:53` and confirms its own patch rejected it. If that check ever failed — meaning offline isolation genuinely isn't working — **the server would refuse to start at all**, not just log a warning. `GET /status` reports the result of that real startup check.

### Requirement 6 — one clean, documented API

This file, plus **http://localhost:8000/docs** (FastAPI's auto-generated interactive Swagger UI — try any endpoint from the browser, no curl needed) and **http://localhost:8000/openapi.json** (the machine-readable schema). 10 endpoints total, listed above and demonstrated below.

### Requirement 7 — this README

You're reading it.

---

## Try every endpoint yourself

Real output from a real run of this exact server is captured throughout this file and in the main [README.md](../README.md); the commands below reproduce it.

### Status (try this first)

```bash
curl -s http://localhost:8000/status
```

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

Step 5 is the actual pitch: a scam message flagged moments ago automatically pauses the next payment attempt, with what/when/why shown back — over a real HTTP call, not a UI mock.

---

## Automated tests

```bash
python -m pytest tests/test_backend.py -v
```

11 tests, hitting every endpoint through FastAPI's `TestClient` — real inference, real OCR, real STT, no mocks. Part of the full suite (`python -m pytest`, 148 tests total).
