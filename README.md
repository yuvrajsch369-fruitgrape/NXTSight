# NXTSight

Built for the Snapdragon AI Lab Build & Present Challenge.

NXTSight is a small on-device app with five features that share one pipeline:

1. **Scam Screenshot Scanner** — point it at a screenshot of a suspicious text/WhatsApp/email message and it flags whether the message looks like a financial scam.
2. **Spend Insight** — feed it bank/UPI transaction text and it categorizes the spending and gives a plain-language insight (e.g. "your top spending category was Food & Dining, 24% of total spend").
3. **Receipt / Bill Scanner** — scan a payment confirmation, printed receipt, or bill screenshot and add it straight into Spend Insight, tagged separately from SMS-derived entries.
4. **Call Shield** — record a call live through your microphone, paste a call transcript, or upload a WAV recording, and it flags scam-call patterns specific to India's fraud landscape (bank/police/courier impersonation, digital-arrest threats, OTP/money-transfer demands) — genuinely live microphone capture and on-device transcription, but still not telephony-level call interception (see [Call Shield](#call-shield) below for the distinction).
5. **Payment Pause** — ties the two scam-detection features together: if Scam Screenshot Scanner or Call Shield flagged something recently, a simulated payment attempt runs automatically a few seconds later — no click needed — and gets paused with what was flagged, when, and why, before letting you proceed or cancel — explicitly a simulated payment screen, not real payment-app interception (see [Payment Pause](#payment-pause) below for why).

Everything runs **fully locally** — no cloud calls, no account, no data leaving the machine. That claim is demonstrable, not just asserted: the app proves it live, every time it starts (see [What runs on-device](#what-runs-on-device-npu-vs-cpu-fallback) below).

## Quick start

Tested on macOS with Python 3.11.9; should work on Windows/Linux with Python 3.9+ (see [Reliability & hardening](#reliability--hardening) for what's independently verified vs. not).

```bash
git clone https://github.com/yuvrajsch369-fruitgrape/NXTSight.git
cd NXTSight

python3 -m venv venv            # use `python` instead if your system has no `python3` alias (e.g. some Windows setups)
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python scripts/check_setup.py   # confirms Python version + every dependency is OK before you go further
```

If you're on macOS and see `CERTIFICATE_VERIFY_FAILED` on first run (a known python.org-installer issue, not an NXTSight bug):

```bash
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

Then run the app:

```bash
streamlit run app.py
```

This opens a local web page in your browser (usually `http://localhost:8501`). Five tabs, all pre-loaded with sample data so there's nothing to hunt for:

- **Scam Screenshot Scanner** — click one of the three sample screenshots already selectable on screen (`scam_bank_kyc_alert`, `scam_lottery_win`, `scam_parcel_customs_fee`), or upload your own PNG/JPG. It reads the text out of the image, then flags it as a scam or not with a confidence score and a plain-language reason.
- **Spend Insight** — click **"Load sample transactions"** to fill the box with 8 realistic bank/UPI SMS messages, then **"Analyze spending"**. Each transaction gets categorized with an amount and direction, and you get one summary sentence across all of them.
- **Receipt / Bill Scanner** — click a sample receipt screenshot, review the extracted merchant/amount/date, then **"Add to Spend Insight"** to fold it into the same summary, tagged "From screenshot".
- **Call Shield** — click the microphone and play a call on speakerphone (or just speak it) for a genuinely live check, or paste a sample call transcript / upload a sample WAV recording and click **"Analyze call"** — either way you get scam-call patterns flagged, with the specific line that triggered the verdict quoted back to you.
- **Payment Pause** — after flagging something scammy in either of the two tabs above, open this tab: a simulated payment attempt runs on its own a few seconds later (no click needed) and shows the interruption — what was flagged, when, and why, with the option to cancel or proceed.

No login. No account. No setup beyond the commands above. If anything about your environment is incomplete, `check_setup.py` (or the app itself) tells you exactly what's missing and how to fix it — it won't fail with a cryptic error partway through.

**Sanity-check without the UI** — if you'd rather verify the code directly, from the project root with the venv active:

```bash
python -c "
from src.scam_detector.classifier import classify_scam
print(classify_scam('Your account will be blocked in 2 hours unless you verify now. Click here.'))
"
```

Expect something like `{'is_scam': True, 'confidence': 0.58, 'reason': "Contains phrases commonly seen in scams: 'verify', 'account'."}`.

## What runs on-device (NPU vs. CPU fallback)

Described honestly, matched to what's actually wired up right now — not what's theoretically possible:

**Scam detection and spend categorization run through the Snapdragon-aware path today.** Both `classify_scam()` and `categorize_transactions()` call through one shared engine ([`src/pipeline/engine.py`](src/pipeline/engine.py)) that loads its model into ONNX Runtime via [`src/pipeline/runtime.py`](src/pipeline/runtime.py). At startup, that code checks whether the machine has Qualcomm's QNN execution provider available — true only on a Snapdragon Windows PC with `onnxruntime-qnn` installed. If it's there, inference runs on the Hexagon NPU. If not (this dev Mac, or any other machine), it falls back to plain CPU inference automatically — same code, same result, no manual flag, no crash. This switch is real, not a plan: proven in [`tests/test_runtime.py`](tests/test_runtime.py), and the live demo shows exactly which path is active on an on-screen badge every time you run it, so nobody has to take it on faith.

**Screenshot OCR does not run through that path yet.** `extract_text_from_image()` currently uses EasyOCR's own PyTorch-based reader, which always runs on CPU regardless of hardware. Separately, we compiled the underlying EasyOCR detector and recognizer models for the Snapdragon X Elite via Qualcomm AI Hub and profiled them on **real physical Snapdragon hardware** in Qualcomm's cloud device farm — genuine measured numbers, not estimates: **38.1ms** (detector) and **20.4ms** (recognizer), see [Snapdragon / Qualcomm AI Hub](#snapdragon--qualcomm-ai-hub) below. That proves the OCR stage *can* run at NPU speed. Wiring that compiled model into the live `extract_text_from_image()` call — replacing EasyOCR's own PyTorch path — is the next integration step, not something already running in the demo. We're telling you this plainly rather than letting the README imply otherwise.

**Call Shield's classifier is on the Snapdragon-aware path; its speech-to-text step isn't.** `analyze_call()` registers as a Task on the same shared engine as the other two classifiers, so it gets the NPU/CPU auto-detect for free. The transcription step ahead of it (`extract_text_from_audio()`, [src/pipeline/stt.py](src/pipeline/stt.py)) runs Whisper via plain PyTorch on CPU — Qualcomm AI Hub does have Whisper models in its zoo, but compiling and wiring one in is future work, not done here.

## Sample inputs to try

Already bundled in the repo — no need to find your own test data:

| Feature | Where | What's in it |
|---|---|---|
| Scam scanner | [`data/samples/scam_bank_kyc_alert.png`](data/samples/scam_bank_kyc_alert.png) | Fake "update your KYC or your account is blocked" message with a suspicious link |
| Scam scanner | [`data/samples/scam_lottery_win.png`](data/samples/scam_lottery_win.png) | Fake lottery/prize-win message asking for personal details |
| Scam scanner | [`data/samples/scam_parcel_customs_fee.png`](data/samples/scam_parcel_customs_fee.png) | Fake "pay a customs fee to release your parcel" message |
| Spend Insight | built into `app.py` (`SAMPLE_TRANSACTIONS`) | 8 realistic bank/UPI SMS lines — Swiggy, Amazon, Uber, a salary credit, Netflix, an ATM withdrawal, a mutual fund SIP, an electricity bill |
| Receipt / Bill Scanner | [`data/samples/receipt_upi_confirmation.png`](data/samples/receipt_upi_confirmation.png) | A GPay/PhonePe-style "Payment Successful" screen |
| Receipt / Bill Scanner | [`data/samples/receipt_printed_grocery.png`](data/samples/receipt_printed_grocery.png) | A printed grocery-store receipt with 7 line items |
| Receipt / Bill Scanner | [`data/samples/receipt_multi_item_bill.png`](data/samples/receipt_multi_item_bill.png) | A restaurant bill with subtotal/tax/service-charge decoys, to test picking the *real* total |
| Receipt / Bill Scanner | [`data/samples/receipt_blurry_photo.png`](data/samples/receipt_blurry_photo.png) | A heavily blurred photo — the "this should fail cleanly, not guess" case |
| Receipt / Bill Scanner | [`data/samples/receipt_handwritten_note.png`](data/samples/receipt_handwritten_note.png) | A handwritten IOU note — the other "should fail cleanly" case |
| Call Shield | no sample file — use your own voice/mic | "Record live": read one of the sample transcripts below out loud (or play it on speakerphone) and it's captured and analyzed live |
| Call Shield | built into `app.py` (5 sample transcripts) | 3 scam calls (digital-arrest, fake courier, fake bank) + 2 legit calls, pasteable with one click |
| Call Shield | [`data/samples/call_digital_arrest_scam.wav`](data/samples/call_digital_arrest_scam.wav) | Spoken digital-arrest scam, synthesized audio, run through real on-device speech-to-text |
| Call Shield | [`data/samples/call_fake_courier_scam.wav`](data/samples/call_fake_courier_scam.wav) | Spoken fake-courier scam |
| Call Shield | [`data/samples/call_fake_bank_scam.wav`](data/samples/call_fake_bank_scam.wav) | Spoken fake-bank scam |
| Call Shield | [`data/samples/call_legit_bank_call.wav`](data/samples/call_legit_bank_call.wav) | Spoken legitimate bank call |
| Payment Pause | no sample files of its own | Reuses whatever Scam Screenshot Scanner or Call Shield just flagged — flag one of the samples above, then switch to this tab |

Want to try your own? The scam scanner accepts any screenshot with legible text; the spend categorizer accepts any bank/UPI SMS text, one message per line, pasted into the text box; the receipt scanner accepts any payment confirmation, receipt, or bill screenshot; Call Shield accepts a pasted transcript or a WAV recording; Payment Pause has no input of its own — it reacts to whatever the other two just flagged.

## Architecture: one engine, three jobs

`classify_scam()`, `categorize_transactions()`, and `analyze_call()` don't each load a model and run inference themselves. All three call through **one shared `NXTSightEngine` object** ([src/pipeline/engine.py](src/pipeline/engine.py)) — the only piece of code in the whole app that ever touches ONNX Runtime, the Snapdragon NPU/QNN provider, or a TF-IDF vectorizer. What's different per feature is registered as a `Task` (its own trained artifacts, its own confidence rules) — not a second copy of the model-serving code. The diagram below shows two of the three (the pattern for `analyze_call()` — [src/call_shield/classifier.py](src/call_shield/classifier.py) — is identical, just a third `Task` alongside the other two, trained on call-transcript data instead of SMS or receipt-derived text).

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

Each task still trains its **own** model — a binary scam flag, an 11-way spending category, and a binary scam-call flag are genuinely different problems (call transcripts especially: multi-turn dialogue, much longer, different vocabulary entirely from a single SMS), so faking one shared set of weights would just be a worse model for every job. What's genuinely shared, end to end, is the serving code: text validation, vectorization, session creation, NPU/CPU detection, session caching. That's the part that actually runs on-device, and it's one object doing it for all three features.

Screenshot OCR ([`src/pipeline/ocr.py`](src/pipeline/ocr.py)) and call-audio transcription ([`src/pipeline/stt.py`](src/pipeline/stt.py)) are the "read" steps ahead of the classifiers — they turn an image or a recording into text, which then goes through the engine like any other input. As noted above, neither runs through this engine's QNN-aware path yet; the three classifiers do.

## Scam classifier

`classify_scam(text) -> {"is_scam": bool, "confidence": float, "reason": str}` ([src/scam_detector/classifier.py](src/scam_detector/classifier.py)) looks for the patterns behind fake bank alerts, OTP-sharing requests, too-good-to-be-true investment offers, urgent account-blocked threats, and fake delivery/KYC/job-offer scams.

**The model:** a MiniLM-v2 embedding + Logistic Regression classifier trained on 77 labeled examples ([src/scam_detector/data.py](src/scam_detector/data.py)), exported to ONNX (~12KB) by hand ([src/pipeline/onnx_export.py](src/pipeline/onnx_export.py) — see why below), registered as a Task on the shared `NXTSightEngine` (see [Architecture](#architecture-one-engine-three-jobs) above) — so once compiled for Snapdragon via AI Hub, it runs on the NPU too, no code change. Retrain it with:

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

**Accuracy:** checked against 40 held-out examples (none copied from training data) spanning romance scams, fake tech support, fake charity, tax-refund phishing, SIM-swap, crypto/seed-phrase phishing, WhatsApp code-forwarding, and more — **40/40 correct**, zero missed scams. Details in [Reliability & hardening](#reliability--hardening) below.

**AI Hub path:** unlike EasyOCR, this is a custom model, not one from the AI Hub model zoo, so it compiles via the raw `qai_hub` API rather than the `qai_hub_models` CLI:

```bash
python -m src.scam_detector.train_classifier      # produces the .onnx
python scripts/aihub_compile_scam_classifier.py    # compiles + profiles it on real Snapdragon hardware
```

**Actually done, not just described** — [compile job](https://workbench.aihub.qualcomm.com/jobs/jgnz326mg/) and [profile job](https://workbench.aihub.qualcomm.com/jobs/jp2rl8xmg/), both SUCCESS on a real **Snapdragon X Elite CRD**: **41 microseconds** per prediction, every op mapped to the **NPU** compute unit. Getting here took two real, fixed bugs, not a clean first try: `skl2onnx`'s default converter emits an ONNX graph with a dynamic batch dimension (AI Hub's compiler: *"Model input 'input' has dynamic shapes. Please use a static shape"*) and an `ai.onnx.ml.LinearClassifier` op (*"Domain name 'ai.onnx.ml' ... is not supported by Qualcomm AI Hub Workbench"*) — a classical-ML op an NPU compiler has no way to lower. Fixed by exporting by hand instead: a static batch size of 1 (the engine only ever runs one text at a time) and a plain `Gemm → Softmax → ArgMax` graph using only core `ai.onnx` ops, mathematically identical to sklearn's own `predict_proba` (verified bit-for-bit against the pre-fix confidence score). The downloaded compiled artifact lives at `src/scam_detector/artifacts/classifier_snapdragon.onnx.onnx.zip`. The live app still runs the portable `classifier.onnx` through the CPU/QNN auto-detecting `runtime.py` — same honest distinction as OCR below.

## Spend categorizer

`categorize_transactions(list_of_texts) -> {"categorized": [...], "insight": str}` ([src/spend_categorizer/categorizer.py](src/spend_categorizer/categorizer.py)) takes a batch of bank/UPI transaction SMS text and sorts each into one of 11 categories (Food & Dining, Groceries, Shopping, Transport, Bills & Utilities, Entertainment, Transfers & UPI P2P, Income & Refunds, Healthcare, Investment & Savings, Cash Withdrawal), then produces one plain-language spending insight across the batch.

**Calls through the same shared `NXTSightEngine`** as the scam classifier (see [Architecture](#architecture-one-engine-three-jobs) above) — TF-IDF + Logistic Regression trained on 88 labeled examples ([src/spend_categorizer/data.py](src/spend_categorizer/data.py)), exported to ONNX by hand ([src/pipeline/onnx_export.py](src/pipeline/onnx_export.py)), registered as its own Task. This module never imports `onnxruntime` or `joblib` directly; every model-serving line (text validation, vectorization, NPU/CPU session creation) lives once, in the engine, not once per feature. Retrain it with:

```bash
python -m src.spend_categorizer.train_classifier
```

Each item in `categorized` is `{"text", "category", "confidence", "amount", "direction", "source"}` — `amount` and `direction` (`"debit"`/`"credit"`) are pulled out with a small regex, independent of the ML classifier, so `insight` can sum actual rupee amounts by category rather than just counting messages. `list_of_texts` accepts plain strings (source defaults to `"sms"`) or `{"text": ..., "source": ...}` dicts — see [Receipt / Bill Scanner](#receipt--bill-scanner) below, which feeds screenshot-derived transactions through this exact same function tagged `source="screenshot"`.

**Edge cases** — each returns a clear result rather than crashing:

| Input | Result |
|---|---|
| Empty list / `None` / non-list | `{"categorized": [], "insight": "Couldn't analyze this: no transactions provided."}` |
| A malformed item in the batch (`None`, `""`, a number, gibberish) | That item alone becomes `{"category": "Unrecognized", "confidence": 0.0, ...}` — the rest of the batch still processes normally |
| Text that isn't actually a transaction (e.g. a casual message) | Categorized "Unrecognized" — gated on *two* independent signals: low classifier confidence (11-way softmax, so a real floor is ~0.20, not 0.5) **and** the absence of any parseable amount/debit-credit cue |
| All items unrecognized | `insight: "Couldn't analyze this: none of the provided transactions could be understood."` |

**Honest limitation:** with only ~8 training examples per category, a few genuinely novel merchant names get misclassified into a neighboring category (e.g. a rent-split payment landed in "Shopping" instead of "Transfers & UPI P2P" in testing) rather than being rejected outright — expected behavior for a tiny bag-of-words model with zero semantic generalization. More labeled examples in `data.py` is the direct fix.

**AI Hub path** (same shape as the scam classifier):

```bash
python -m src.spend_categorizer.train_classifier          # produces the .onnx
python scripts/aihub_compile_spend_categorizer.py          # compiles + profiles it on real Snapdragon hardware
```

**Actually done** — [compile job](https://workbench.aihub.qualcomm.com/jobs/j568n6e7g/) and [profile job](https://workbench.aihub.qualcomm.com/jobs/jgjrexz8p/), both SUCCESS on real **Snapdragon X Elite CRD** hardware: **35 microseconds** per prediction, every op on the **NPU**. Same hand-built export as the scam classifier, same reason (AI Hub's compiler rejects both a dynamic batch dimension and the `ai.onnx.ml` domain `skl2onnx` emits by default) — see the scam classifier's AI Hub section above for the full story.

## Receipt / Bill Scanner

A third input path into the same spend insight: scan a screenshot of a payment confirmation, printed receipt, or bill — not just SMS text — and add what it finds to the same spending summary.

`process_receipt_screenshot(image_path)` ([src/spend_categorizer/receipt_parser.py](src/spend_categorizer/receipt_parser.py)) reuses `extract_text_from_image()` (the exact same OCR function the scam scanner uses) for the "read" step, then does something the SMS path doesn't need to: receipt and payment-app layouts don't read like bank SMS at all — no "debited"/"credited" boilerplate, the merchant name and amount are scattered across separate lines rather than one sentence. Feeding that raw OCR text straight into the SMS-trained spend classifier would be unreliable. Instead, it pulls out `{merchant, amount, date}` with layout-aware heuristics, builds a normalized sentence in the shape the classifier actually knows — `"Rs 450.00 debited via UPI to SWIGGY on 12 Sep 2025."` — and hands that to the **exact same** `categorize_transactions()` used for SMS, tagged `source="screenshot"`. Reused, not duplicated.

**Amount extraction**, in priority order: (1) a currency-marked number (`Rs`/`₹`/`INR`) on a line that also says "total"/"paid"/"amount" — avoids picking a per-item price off a multi-line bill; (2) otherwise the largest currency-marked number anywhere; (3) last resort, a bare number alone on its own line with nothing else — how a payment app's amount often OCRs when its ₹ glyph isn't captured as text (this is the *only* way GPay/PhonePe-style "Payment Successful" screens show an amount at all). That third tier is guarded two ways, because a stray digit an OCR hallucinates from photo noise must never be silently treated as a real amount: it has to be ≥ ₹10, and the text has to contain some actual payment-related word ("paid", "successful", "transaction", ...) somewhere — not just one isolated number with nothing recognizable around it.

**Merchant extraction** takes the first plausible text line (skipping app-chrome labels like "Payment Successful," "Paid to," date labels, and anything that's itself an amount), and stitches in the next line too if the first one is short — OCR often splits a header like "BIG BAZAAR" across two lines ("BIG" / "BAZAAR").

**Tested against 5 varied sample screenshots** (bundled in `data/samples/receipt_*.png`, synthetic but realistic — a UPI confirmation, a printed grocery receipt, a handwritten IOU note, a deliberately blurred photo, and a multi-item restaurant bill with decoy subtotal/tax amounts). Real, current results, locked in as regression tests in [tests/test_receipt_parser.py](tests/test_receipt_parser.py):

| Sample | Result | Where it struggles |
|---|---|---|
| UPI payment confirmation | ✅ Merchant, amount, date all correct (SWIGGY, ₹450, 12 Sep 2025) | — |
| Printed grocery receipt | ✅ All correct (BIG BAZAAR, ₹845, 15/09/2025) | — |
| Multi-item restaurant bill | ✅ Merchant and total-vs-line-item selection both correct | OCR splits the total's decimal cents onto their own line ("933" / "50"), and the amount picker doesn't rejoin them — parsed as ₹933.00 instead of ₹933.50. Small, but a real, known rounding-like error, not hidden. |
| Blurry photo | ✅ Correctly declines: *"couldn't find a clear amount in this image — it may be too blurry, cropped, or in a layout this can't read yet."* | OCR extracts almost nothing usable from a heavily blurred image, so there's nothing to parse — this is the intended, safe outcome, not a bug. |
| Handwritten note | ✅ Correctly declines, same message | OCR misreads the handwritten digit "5" as the letter "S" ("Rs S00" instead of "Rs 500") — a real, common handwriting-OCR failure mode. Because "S00" isn't a valid number, the parser correctly refuses to guess rather than inventing a wrong amount. |

Two genuine, unresolved limitations, not glossed over: **decimal cents can be silently dropped** when OCR splits a total across lines (the multi-item bill case), and **handwritten amounts are essentially unsupported** — any digit an OCR misreads as a letter simply won't parse, which is the correct failure mode (never invent a number) but means legitimate handwritten receipts will often decline entirely rather than succeed with reduced confidence. Neither is silent or crashes; both are exactly the "clear message, never a wrong number" behavior the rest of the app follows.

**In the demo UI:** the **Receipt / Bill Scanner** tab shows the extracted merchant/amount/date (or the decline message) and an **Add to Spend Insight** button. Added items show up in the **Spend Insight** tab's results table with a **Source** column reading "From SMS" or "From screenshot," and the top-line insight sentence includes them in the same total.

## Call Shield

**This still isn't telephony-level call interception — said plainly, in the UI and here.** NXTSight cannot tap into the phone system, a carrier, or a dialer to listen to an actual phone call. Doing that would require phone/telephony-level OS integration — call-audio access, a dialer or carrier hook — that a local Python app fundamentally cannot do and this prototype does not attempt. What *is* real: **"Record live"** captures actual audio through your device's microphone — the same way a person in the room would hear it, e.g. a call held on speakerphone next to the laptop — transcribes it on-device, and analyzes it, all live, no pre-recorded sample required. It just can't reach into a call NXTSight isn't physically in the room for. Alongside that: paste a transcript directly, or upload a pre-recorded WAV to transcribe on-device first. This same point is made in plain language in the single "live working prototype" notice at the top of the app — one place for every honest caveat, rather than a separate warning banner repeated on each tab.

`analyze_call(transcript) -> {"is_scam": bool, "confidence": float, "reason": str}` ([src/call_shield/classifier.py](src/call_shield/classifier.py)) — same result shape as `classify_scam()`, registered as its own Task on the shared `NXTSightEngine` (see [Architecture](#architecture-one-engine-three-jobs) above). Its own model, not a reuse of the SMS classifier's weights: a call transcript is multi-turn dialogue, much longer than a single message, and carries vocabulary specific to India's call-fraud landscape (digital-arrest threats, "stay on video call," police/CBI/customs impersonation) that isn't represented in SMS training data at all — trained on 30 labeled call-transcript examples ([src/call_shield/data.py](src/call_shield/data.py)).

**The `reason` cites which part of the call triggered it, not just which words.** A transcript is long enough that "contains scam phrases" alone isn't actionable — `analyze_call()` scores every line of the transcript against the model's own learned vocabulary and quotes the single line with the strongest match, e.g. *`Flagged because of this part of the call: "Caller: ...transfer all funds from your savings account to the RBI secure holding account..." — contains phrases commonly seen in scam calls: 'otp', 'account', 'verification'.`*

**Speech-to-text** (`extract_text_from_audio()`, [src/pipeline/stt.py](src/pipeline/stt.py)) prefers Qualcomm AI Hub's compiled Whisper-tiny **encoder** ([src/pipeline/whisper_qai_hub.py](src/pipeline/whisper_qai_hub.py)) — decoding stays on `qai_hub_models`' own unmodified PyTorch decode loop, nothing autoregressive runs through ONNX (see [Snapdragon / Qualcomm AI Hub](#snapdragon--qualcomm-ai-hub) above for why). Falls back to OpenAI's Whisper (`tiny.en`, fully on-device via PyTorch) if that path is unavailable for any reason. Either way, WAV only, deliberately: Whisper's own audio loader (either path) shells out to a system `ffmpeg` binary for other formats, which isn't installed on every machine (this dev Mac included, no Homebrew either); WAV files are decoded with the standard-library `wave` module and resampled to 16kHz with plain numpy instead. Retrain the classifier with `python -m src.call_shield.train_classifier`.

**Verified correct before being trusted as the preferred path — real transcriptions, not just "didn't crash":**

| Sample | AI Hub hybrid transcript matches all-PyTorch baseline? | Verdict |
|---|---|---|
| Legit bank call | ✅ Byte-for-byte identical | Correctly cleared |
| Digital-arrest scam | ✅ Byte-for-byte identical | Correctly flagged, OTP/arrest/transfer triggers all present |
| Fake courier scam | ✅ Byte-for-byte identical | Correctly flagged |
| Fake bank scam | ✅ Byte-for-byte identical | Correctly flagged |

Both the pure-PyTorch baseline and the ONNX-encoder hybrid were run against all 4 bundled recordings and diffed line for line before the hybrid became the preferred path — not assumed correct because the encoder export succeeded. One genuine, confirmed gap while verifying this: setting `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` from inside `whisper_qai_hub.py` itself worked in an isolated test script but *not* reliably once actually run through `uvicorn` — only caught by testing the real running server, not the script alone. Fixed by setting both environment variables at the literal top of `app.py`/`backend/main.py`, before any other import.

**"Record live" (microphone) uses Streamlit's own `st.audio_input` widget** (`app.py`), requested at 16kHz — already the pipeline's target sample rate — and hands the recorded WAV bytes straight to the exact same `analyze_call_recording()` used by the WAV-upload path, through a temp file. No separate code path to trust: whatever's already tested against the 4 bundled recordings runs identically here. The one thing that genuinely can't be covered by `pytest` is the browser's own microphone capture (it needs a real mic and a real click) — that part is verified manually, live, not by an automated test.

**Tested against 5 held-out call transcripts** (3 scam: digital-arrest, fake courier, fake bank; 2 legit — none copied from training data) **and the 4 bundled sample recordings, transcribed for real, not just pasted as text.** Locked in as regression tests in [tests/test_call_shield.py](tests/test_call_shield.py) and [tests/test_stt.py](tests/test_stt.py):

| Test | Result |
|---|---|
| Digital-arrest scam (pasted + as spoken audio) | ✅ Flagged, ~71–78% confidence, correct triggering line quoted |
| Fake courier scam (pasted + as spoken audio) | ✅ Flagged, ~72–77% confidence |
| Fake bank scam (pasted + as spoken audio) | ✅ Flagged, ~65–73% confidence |
| Legit bank call (pasted + as spoken audio) | ✅ Cleared, ~67–76% confidence |
| Legit friend call (pasted) | ✅ Cleared, 72% confidence |

**A real bug this testing surfaced, not just a clean result to report:** the legit bank-call transcript, pasted with `"Caller:"/"You:"` speaker labels, correctly cleared at 67% confidence. The *identical content*, spoken as natural continuous audio and transcribed by Whisper — which produces flat, unlabeled text, since there's no speaker diarization — initially **false-positived as a scam at 51% confidence**, right at the decision boundary. Real transcribed-call text doesn't carry dialogue labels; the training data, originally written entirely in labeled-script style, didn't have examples of that shape on the legit side. Fixed at the data level: added a handful of unlabeled, continuous-text training examples (both scam and legit) matching what real transcribed audio actually looks like, then retrained. All 9 held-out checks (5 pasted + 4 audio) pass afterward, with meaningfully higher, more decisive confidence across the board (legit bank call moved from 51% to 76%) — not just barely over the line.

**A real dependency-install bug this testing surfaced too:** `openai-whisper` declares its `numba` dependency with **no version constraint at all**, so a fresh `pip install` picks whichever numba is newest — which needs a newer `llvmlite` than 0.43.0, the last version with an Intel-Mac wheel (the exact same pattern as the `torch` pin elsewhere in this file). Without an explicit pin, installing on an Intel Mac tries to compile `llvmlite` from source and fails outright unless a full LLVM toolchain happens to be present. Confirmed reproducible on a clean venv twice, not a one-off flake — see the `numba==0.60.0` pin and comment in [requirements.txt](requirements.txt).

**AI Hub path** (same shape as the other two custom classifiers):

```bash
python -m src.call_shield.train_classifier          # produces the .onnx
python scripts/aihub_compile_call_shield.py           # compiles + profiles it on real Snapdragon hardware
```

**Actually done** — [compile job](https://workbench.aihub.qualcomm.com/jobs/j5wlqo3jp/) and [profile job](https://workbench.aihub.qualcomm.com/jobs/jgddowqlg/), both SUCCESS on real **Snapdragon X Elite CRD** hardware: **41 microseconds** per prediction, every op on the **NPU**. Same hand-built export, same fix, as the other two classifiers.

## Payment Pause

**This is a simulated payment attempt, not a real payment-app intercept — said plainly, in the UI and here.** NXTSight cannot see or intercept a payment actually being made in a real UPI or banking app; doing that would require integration with that app, or the OS's own payments layer, which is beyond what a local Python app can do and beyond this prototype's scope. What Payment Pause actually does: watch a short-lived, **in-memory-only** log (nothing written to disk, nothing sent anywhere — a plain Python list that lives only as long as the demo session does) of anything Scam Screenshot Scanner or Call Shield has flagged recently, and **automatically** run a simulated payment attempt a few seconds after any new flag — no button, no click — checking that log at the moment of that attempt.

**The matching rule is deliberately simple, because it has to be explainable, not just effective** ([`src/pipeline/payment_pause.py`](src/pipeline/payment_pause.py)): any flag raised in the last `WINDOW_MINUTES` (**10**, hardcoded) pauses any payment attempt. No correlation against the payment's amount, contact, or app — that would need signals a local demo app doesn't have access to. If one or more flags fall inside that window, `recent_flags()` returns them (newest first) and the UI shows exactly what was flagged, by which feature, how long ago, and why, next to **Cancel payment** / **Proceed anyway**. Outside the window, or with no flags at all, the simulated payment proceeds normally.

**Why automatic, not a button:** the point being demonstrated is a scammer's manufactured urgency trying to rush someone from a scam message or call straight into paying, before they've had time to think. A "click here to simulate a payment" button undercuts that — it turns the intervention into something the user has to go looking for. So the Payment Pause tab itself watches for any flag it hasn't auto-checked yet (`auto_checked_version` in `st.session_state`) and, on the next rerun that touches that tab, shows a live 3-second countdown ("auto-simulating a payment attempt in 3... 2... 1...") before evaluating `recent_flags()` — the interruption happens on its own.

**Demo scenario, start to finish:**
1. Open **Scam Screenshot Scanner**, pick `scam_bank_kyc_alert` — it's flagged as a likely scam (~61% confidence), and a caption confirms it was logged to Payment Pause.
2. Switch to the **Payment Pause** tab. The flag log (collapsed by default) shows 1 entry: *Scam Screenshot Scanner, just now — contains phrases commonly seen in scams: 'update', 'kyc', 'immediately', 'suspension'.* A 3-second countdown appears on its own, then resolves.
3. **Payment paused** — no click needed to get here. That exact flag is shown back: what, when, why.
4. Click **Cancel payment** (or **Proceed anyway**, to show the override path exists too) and the interruption clears.

**A real bug this feature's own testing surfaced:** Streamlit reruns the *entire* script on every interaction anywhere in the app — not just the widget that changed. Scam Screenshot Scanner and Call Shield's audio/live-microphone paths all analyze as soon as input is available, with no separate "Analyze" button, so the very first version of this feature silently re-flagged the same still-selected image or recording on *every unrelated click elsewhere in the app* — switching to the Payment Pause tab, for example, was enough to double the flag log. Fixed with a small dedupe guard (`_flag_once()` in `app.py`): each of those auto-triggering inputs only logs a flag the first time a given result is seen, tracked in `st.session_state`, not on every incidental rerun. The auto-trigger check itself needed the same discipline — it's guarded by `auto_checked_version` so an unrelated click doesn't replay the 3-second countdown or re-pause an already-dismissed flag. Verified live: flagging one screenshot, then clicking around the Call Shield and Payment Pause tabs, leaves the flag log at 1 entry and the countdown firing exactly once.

## Project structure

```
NXTSight/
├── app.py                   # live demo UI (streamlit run app.py) — screenshot -> verdict, or transactions -> insight
├── backend/                 # FastAPI service exposing all five features over HTTP — see backend/README.md
│   ├── main.py                   # uvicorn backend.main:app — one server, one shared engine, 10 endpoints
│   └── README.md                 # requirement-by-requirement mapping + curl examples for every endpoint
├── .streamlit/config.toml   # disables Streamlit's own telemetry (would otherwise call out)
├── src/
│   ├── pipeline/
│   │   ├── ocr.py            # extract_text_from_image(): screenshot -> raw text
│   │   ├── runtime.py        # picks the ONNX Runtime execution provider (NPU vs CPU)
│   │   ├── text_guard.py     # shared "is this text analyzable" check (gibberish/non-English/empty)
│   │   ├── engine.py         # NXTSightEngine: the one shared object all three tasks call through
│   │   ├── network_guard.py  # blocks + proves-blocked any non-loopback network connection
│   │   ├── preflight.py      # Python-version + missing-dependency checks (shared by app.py and check_setup.py)
│   │   ├── stt.py            # extract_text_from_audio(): WAV recording -> raw text (Whisper, ffmpeg-free)
│   │   ├── payment_pause.py  # feature 5's logic: in-memory flag log + the "recent flag pauses a payment" rule
│   │   ├── onnx_export.py    # hand-built ONNX export (Gemm->Softmax->ArgMax) — AI Hub rejects skl2onnx's default output
│   │   ├── ocr_qai_hub.py    # AI Hub-compiled EasyOCR detector+recognizer, via runtime.py's QNN/CPU path
│   │   ├── text_encoder.py   # MiniLM-v2 shared text encoder (AI Hub-compiled), all 3 classification heads sit on top of this
│   │   └── whisper_qai_hub.py # AI Hub-compiled Whisper-tiny encoder; decoder stays local, unmodified PyTorch
│   ├── scam_detector/       # feature 1: screenshot OCR + scam classification
│   │   ├── data.py               # labeled training examples
│   │   ├── train_classifier.py   # local build step: trains + exports classifier.onnx
│   │   ├── classifier.py         # classify_scam(text) -> {is_scam, confidence, reason}
│   │   └── artifacts/            # trained vectorizer.joblib, classifier.onnx, top_terms.json
│   ├── spend_categorizer/   # feature 2 & 3: transaction parsing + spend categorization (SMS and screenshots)
│   │   ├── data.py               # labeled training examples (11 categories)
│   │   ├── train_classifier.py   # local build step: trains + exports classifier.onnx
│   │   ├── categorizer.py        # categorize_transactions(texts) -> {categorized, insight}
│   │   ├── receipt_parser.py     # process_receipt_screenshot(): receipt/bill image -> categorize_transactions() input
│   │   └── artifacts/            # trained vectorizer.joblib, classifier.onnx, categories.json
│   └── call_shield/         # feature 4: scam-call detection from a transcript or recording
│       ├── data.py               # labeled call-transcript training examples
│       ├── train_classifier.py   # local build step: trains + exports classifier.onnx
│       ├── classifier.py         # analyze_call(transcript) / analyze_call_recording(wav_path)
│       └── artifacts/            # trained vectorizer.joblib, classifier.onnx, top_terms.json
├── models/                  # exported/compiled AI-Hub model artifacts (OCR, MiniLM-v2, Whisper encoder) — gitignored, regenerate with the export_*.py scripts below
├── scripts/
│   ├── check_setup.py                       # environment doctor — run before the app if unsure setup is complete
│   ├── export_easyocr.py                    # local-only: export EasyOCR detector+recognizer to ONNX (no AI Hub account needed)
│   ├── export_minilm.py                     # local-only: export MiniLM-v2 to ONNX (no AI Hub account needed)
│   ├── export_whisper_encoder.py            # local-only: export Whisper-tiny's encoder (only) to ONNX (no AI Hub account needed)
│   ├── aihub_profile_easyocr.py             # one-off: profile the OCR model on real Snapdragon hardware
│   ├── aihub_compile_scam_classifier.py     # one-off: compile + profile the scam classifier for Snapdragon
│   ├── aihub_compile_spend_categorizer.py   # one-off: compile + profile the spend categorizer for Snapdragon
│   ├── aihub_compile_call_shield.py         # one-off: compile + profile the Call Shield classifier for Snapdragon
│   ├── aihub_compile_minilm.py              # one-off: compile + profile MiniLM-v2 for Snapdragon
│   └── aihub_compile_whisper_encoder.py     # one-off: compile + profile the Whisper-tiny encoder for Snapdragon
├── data/samples/            # sample scam/receipt screenshots, sample call recordings, sample transaction text
├── notebooks/               # exploration / model experimentation
├── tests/                   # unit tests — incl. test_engine.py (architecture), test_hardening.py (crash-proofing),
│                             # test_network_guard.py, test_preflight.py, test_call_shield.py, test_stt.py,
│                             # test_payment_pause.py, test_backend.py
├── assets/screenshots/      # demo screenshots for the submission write-up
├── requirements.txt
└── README.md
```

## Reliability & hardening

### Stress-tested, not just spot-checked

The scam classifier is checked against 40 held-out examples total, none copied from training data: the original 5 scam + 5 legit ([tests/test_classifier.py](tests/test_classifier.py)) plus a larger, deliberately harder 15+15 batch covering scam sub-types the training data was originally thin on — romance/pig-butchering scams, fake tech support, fake charity, tax-refund phishing, SIM-swap, insurance renewal, fake legal threats, crypto wallet/seed-phrase phishing, WhatsApp code-forwarding, subscription-cancellation scares, fake loans/jobs/health offers — against legit messages covering school notices, appointment reminders, calendar invites, weather alerts, security "no action needed" notices, government status updates, and everyday correspondence. Current result: **40/40 correct** on this dev machine.

Two real false positives turned up during this pass and got fixed at the data level, not by patching around them:
- A legit "new sign-in, no action needed" notification was flagged as a scam — the model had only ever seen the *scam* version of that message ("if this WASN'T you, verify immediately"). Added legit examples of the passive framing.
- A legit "your passport application is under processing" notice was flagged — official/institutional language wasn't represented on the legit side of the training data at all. Added a few.

Fixing those introduced two *new* misses (a WhatsApp code-forwarding scam and a crypto-wallet phishing scam got pulled toward "legit" by vocabulary overlap with the new legit examples) — a real example of the tug-of-war a tiny linear model plays with itself. Both got fixed: tightened the new legit examples' wording to reduce overlap, and added one targeted scam example for the previously-uncovered "seed phrase" pattern. All 40 pass now, with **zero false negatives on any scam example** across both batches — when a tradeoff has to be made, this favors never missing a real scam over never annoying someone with a false alarm. (The MiniLM-v2 backbone swap later in this project re-ran into, and re-fixed, the exact same kind of tug-of-war — see [Status](#status) below.)

### Nothing should crash the app on bad input

Audited every function that runs during actual use (not the one-off build/training scripts, which are run manually offline) and fixed what could actually raise an unhandled exception:

- `extract_text_from_image()` crashed on `None` or any non-path input (an unguarded `Path(path)` call) — now returns `"Error: expected a file path, got ..."`.
- The shared engine's model loading and inference (`src/pipeline/engine.py`) weren't guarded against a missing or corrupted `.onnx`/`.joblib` artifact — a bad deploy would have crashed both `classify_scam` and `categorize_transactions`. Any such failure now returns a plain-language reason both features already know how to handle (it reuses the same "couldn't analyze this" path as gibberish/non-English input).
- `classify_scam`'s reason-building vocabulary (`top_terms.json`) and `categorize_transactions`' category-name list (`categories.json`) are each optional now — missing or corrupted, they degrade gracefully (a shorter `reason`, or "Unrecognized" respectively) instead of crashing.
- `app.py`'s file-upload and transaction-analysis handlers are wrapped so any unexpected exception becomes a plain `st.error(...)` message, never Streamlit's raw traceback.

Proven, not just asserted — [tests/test_hardening.py](tests/test_hardening.py) feeds `None`, wrong types, a corrupted `.onnx` file, and a wildly mixed batch (`None`, numbers, nested lists, control characters, a 10,000-character string) through every entry point and checks each one comes back with a clear result instead of raising.

### Pinned dependencies

Every package in `requirements.txt` is pinned to an exact version captured from a real working install — see the file's own comments for why each pin exists. Two real cross-package/platform bugs got caught and fixed this way, not just theorized about:
- `qai-hub-models` requires plain `opencv-python`, which silently conflicts with the `opencv-python-headless` that `easyocr` needs (both packages install a `cv2` module at the same path; whichever installs second wins, non-deterministically). Fixed by moving `qai-hub-models` out of the base install entirely — it's only needed for one optional AI Hub CLI workflow, which has its own separate install instructions below.
- `openai-whisper` declares its `numba` dependency with **no version constraint at all**, so pip picks whichever is newest — which needs a newer `llvmlite` than 0.43.0, the last version with an Intel-Mac wheel. Without an explicit pin, a clean install on an Intel Mac tries to compile `llvmlite` from source and fails outright. Reproduced twice on a genuinely fresh venv before fixing it with an explicit `numba==0.60.0` pin — see [Call Shield](#call-shield) above for the full story.

### Full-system stress test — fresh clone, all fallbacks, concurrent load

Beyond the per-feature testing described throughout this README, one deliberate end-to-end pass: a genuine `git clone` into a scratch directory with a brand-new venv (not this dev machine's already-configured one), every ONNX artifact for OCR/MiniLM-v2/Whisper temporarily removed at once to force all three fallback chains simultaneously, 10 genuinely concurrent requests across mixed endpoints, and the Streamlit UI driven end to end in a real browser. Real findings, fixed:

- **A test-suite fragility, not a code bug**: two receipt-parser tests were implicitly pinned to AI Hub OCR's specific output, which isn't present on a plain `pip install -r requirements.txt` (the optional `qai_hub_models` extras and exported `.onnx` files aren't part of the base install or the git history). Fixed by making those two tests branch on `ocr_qai_hub.status()`, asserting the behavior that's actually correct for whichever OCR backend is genuinely active — confirmed passing in both configurations, not just the one this dev machine happens to have set up.
- **A confusingly empty error message**: the standard-library `wave` module raises a bare `EOFError` with no message text for a file too short to contain a valid header, producing `"...corrupted or unsupported format) ()"`. Fixed in `src/pipeline/stt.py` to fall back to the exception's type name when its message is empty.
- **A cosmetic but real warning**: HuggingFace's `tokenizers` library warns on stderr about disabling internal parallelism after a fork (Streamlit/uvicorn's worker model triggers this). Silenced with `TOKENIZERS_PARALLELISM=false`, set alongside the offline env vars at the top of `app.py`/`backend/main.py`.
- **A genuinely missing local export path**: OCR's `.onnx` files previously had no equivalent to `scripts/export_minilm.py` / `export_whisper_encoder.py` — only the full AI Hub CLI tool. Added [`scripts/export_easyocr.py`](scripts/export_easyocr.py) for a local-only, no-AI-Hub-account-needed export, verified to produce working (if differently packaged — self-contained rather than external-data-split) files.
- **All three ONNX-optional fallback chains, forced simultaneously**: every feature (Scam Shield, Money Insight, Call Shield — text and audio) re-verified working correctly with OCR, MiniLM-v2, and Whisper's compiled models all absent at once, not just individually. Full 148-test suite passes in that state.
- **10 concurrent requests, mixed endpoint types** (OCR, MiniLM classification, Whisper transcription, categorization, status) run genuinely in parallel against the live backend: no crashes, no corrupted responses, no hang — the shared `NXTSightEngine` singleton handles concurrent access safely.

### What happens on a different machine

| Scenario | What happens |
|---|---|
| **Different OS** (Windows/Linux instead of this dev Mac) | All file paths use `pathlib`; the OS-specific pieces (QNN backend filename, `onnxruntime` vs `onnxruntime-qnn`) are already platform-gated in code and in `requirements.txt`. Should work as-is — not independently tested on Windows/Linux hardware, so run `python scripts/check_setup.py` first to confirm before relying on it. |
| **Missing or incomplete `pip install`** | `app.py` checks the Python version and every required import *before* touching Streamlit's UI machinery, showing exactly which packages are missing and the fix — instead of a raw `ModuleNotFoundError` appearing mid-script. Verified by simulating a partial install (only `streamlit` present): the app showed a clean, itemized error rather than crashing. Run `python scripts/check_setup.py` standalone for the same check before even starting the app. |
| **No internet during `pip install`** | Not something an app can fix after the fact — pip itself gives a normal, clear network error in this case. What *is* fixed: the exact pins above mean that once install succeeds, it's the same install every time, so a "worked yesterday, broke today" failure from an unrelated upstream release doesn't happen later. |
| **Too-old Python** | `engine.py` uses `dict[str, Task]`-style type hints (PEP 585), which need Python 3.9+. Both `app.py` and `scripts/check_setup.py` check this explicitly and name the exact minimum version required, rather than failing with a cryptic `TypeError: 'type' object is not subscriptable` deep inside an import. |
| **Fresh machine, first-ever run, wifi off** | Both EasyOCR and Whisper need network once to download their model weights (documented under [What runs on-device](#what-runs-on-device-npu-vs-cpu-fallback) above) — `network_guard` would (correctly) block that too. Run the app once with internet before a wifi-off presentation. |

## Snapdragon / Qualcomm AI Hub

There are two separate pieces here — a **build-time step** you run once, by hand, to get a Snapdragon-ready model, and a **runtime check** the app makes every time it starts (already covered in [What runs on-device](#what-runs-on-device-npu-vs-cpu-fallback) above).

### Build-time: compile + profile a model for Snapdragon via AI Hub

This step turns a model into a `.onnx` file specifically compiled for the Snapdragon X Elite, and confirms it actually runs correctly by profiling it on real, physical Snapdragon hardware in Qualcomm's cloud device farm (not a simulator — useful since most of us don't own a Snapdragon PC to test on directly). It needs your own free AI Hub account and API token from [aihub.qualcomm.com](https://aihub.qualcomm.com); nobody else can run this step for you.

`qai-hub-models` is intentionally *not* in the base `requirements.txt` — it pulls in plain `opencv-python`, which silently conflicts with the `opencv-python-headless` that `easyocr` needs (both packages install a `cv2` module at the same path, and whichever installs second wins). Install it separately, only when you actually need this step:

```bash
pip install qai-hub "qai-hub-models[easyocr]" sounddevice
qai-hub configure --api_token <YOUR_API_TOKEN>

python -m qai_hub_models.models.easyocr.export \
    --device "Snapdragon X Elite CRD" \
    --target-runtime onnx \
    --profile-options="--qairt_version=default"
```

`sounddevice` there is for Call Shield's Whisper encoder path (`src/pipeline/whisper_qai_hub.py`) — a real, hard import-time dependency of `qai_hub_models`' own Whisper app code (used for live microphone demos in their package; unused here, but it's imported unconditionally, so it still needs to be installed). `transformers` itself, unlike `qai-hub-models`, *is* in the base `requirements.txt` — MiniLM-v2's classification backbone needs it unconditionally, not just for this optional AI Hub build step.

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

Both models load and report their expected shapes locally through `runtime.py` — the detector takes a `(1, 3, 608, 800)` image tensor, matching EasyOCR's documented input resolution. That's solid evidence this genuinely runs on Snapdragon, not just in theory — see [What runs on-device](#what-runs-on-device-npu-vs-cpu-fallback) above for the honest caveat that these compiled models aren't yet the ones the live app calls.

### Build-time: the three custom classifiers, too

The same build-time step, run for real against `scripts/aihub_compile_scam_classifier.py`, `aihub_compile_spend_categorizer.py`, and `aihub_compile_call_shield.py` — not just the OCR models:

| Model | Compile job | Profile job | Inference time | Compute unit |
|---|---|---|---|---|
| Scam classifier | [SUCCESS](https://workbench.aihub.qualcomm.com/jobs/jgnz326mg/) | [SUCCESS](https://workbench.aihub.qualcomm.com/jobs/jp2rl8xmg/) | 41 µs | NPU |
| Spend categorizer | [SUCCESS](https://workbench.aihub.qualcomm.com/jobs/j568n6e7g/) | [SUCCESS](https://workbench.aihub.qualcomm.com/jobs/jgjrexz8p/) | 35 µs | NPU |
| Call Shield classifier | [SUCCESS](https://workbench.aihub.qualcomm.com/jobs/j5wlqo3jp/) | [SUCCESS](https://workbench.aihub.qualcomm.com/jobs/jgddowqlg/) | 41 µs | NPU |

Microseconds, not milliseconds — these are tiny linear models (12–29KB), nothing like the OCR networks above, so the gap is expected, not suspicious. Getting a real SUCCESS here took two genuine, fixed bugs first: `skl2onnx`'s default `LogisticRegression` converter — which every classifier used until this point — produces a graph with a dynamic batch dimension and an `ai.onnx.ml.LinearClassifier` op, and AI Hub's compiler rejects both outright (real error messages: *"Model input 'input' has dynamic shapes"* and *"Domain name 'ai.onnx.ml' ... is not supported"*). Fixed by dropping `skl2onnx` entirely for these three models and hand-building the ONNX graph instead ([src/pipeline/onnx_export.py](src/pipeline/onnx_export.py)): a static batch size of 1, and a plain `Gemm → Softmax → ArgMax` graph using only core `ai.onnx` ops — mathematically identical to sklearn's own `predict_proba` (checked bit-for-bit against pre-fix output: same 0.611 confidence on the same test input). The three downloaded compiled artifacts live at each feature's `artifacts/classifier_snapdragon.onnx.onnx.zip`. Same honest caveat as OCR: the live app still runs the portable `classifier.onnx` through the CPU/QNN auto-detecting `runtime.py`, not this downloaded Snapdragon-specific artifact directly.

### Build-time: MiniLM-v2, the shared text encoder

| Model | Compile job | Profile job | Inference time | Compute unit |
|---|---|---|---|---|
| MiniLM-v2 text encoder | [SUCCESS](https://workbench.aihub.qualcomm.com/jobs/j57ej289p/) | [SUCCESS](https://workbench.aihub.qualcomm.com/jobs/jpvl1q8r5/) | 1.9 ms | NPU |

AI Hub's own "MiniLM-v2" listing ([aihub.qualcomm.com/models/minilm_v2](https://aihub.qualcomm.com/models/minilm_v2)) is `sentence-transformers/all-MiniLM-L6-v2` from HuggingFace with a mean-pooling + L2-normalize head — packaged via `qai_hub_models.models.minilm_v2`, which needs `torch>=2.4`. That has no installable wheel for this Intel Mac (confirmed directly: `pip download torch==2.4.0` finds nothing past 2.2.2 for this platform — the exact same wall `torch==2.2.2` is already pinned in `requirements.txt` to work around everywhere else). Rather than block on that, [`src/pipeline/text_encoder.py`](src/pipeline/text_encoder.py) builds the architecturally identical model by hand — same HuggingFace weights, same forward pass — using the already-installed torch, exports it the same hand-built-ONNX way as the three classifiers, and submits *that* to AI Hub via the same raw `qai_hub` API. Verified the export is numerically correct before submitting anything: ONNX output vs. the original PyTorch model, max absolute difference `1.1e-7` (float32 noise) on a real test sentence.

### Build-time: Whisper-tiny's encoder (Call Shield speech-to-text)

| Model | Compile job | Profile job | Inference time | Compute unit |
|---|---|---|---|---|
| Whisper-tiny encoder | [SUCCESS](https://workbench.aihub.qualcomm.com/jobs/jp0m1r7ng/) | [SUCCESS](https://workbench.aihub.qualcomm.com/jobs/j5m0wwvyg/) | 27.8 ms | NPU |

`qai_hub_models.models.whisper_tiny` wraps `openai/whisper-tiny` (the standard multilingual HuggingFace checkpoint, not the English-only `tiny.en` OpenAI's own package uses — a real, small difference, not a bug) with a monkey-patched encoder that directly precomputes cross-attention key/value tensors for every decoder block, so its decoder can consume them without recomputing cross-attention itself. [`scripts/export_whisper_encoder.py`](scripts/export_whisper_encoder.py) exports *only* that encoder — a single forward pass, `(1, 80, 3000)` mel features in, 8 cross-attention KV tensors out for this 4-decoder-layer model — the same low-risk shape as MiniLM-v2's export, deliberately excluding the autoregressive decoder. [`src/pipeline/whisper_qai_hub.py`](src/pipeline/whisper_qai_hub.py) wires the compiled encoder into `qai_hub_models`' own `HfWhisperApp`, whose decode loop runs completely unmodified, in plain PyTorch, never touching ONNX Runtime. Full correctness verification against the all-PyTorch baseline is in [Call Shield](#call-shield) below.

## Status

All pipeline stages work end to end: OCR (`extract_text_from_image`), scam classification (`classify_scam`), spend categorization (`categorize_transactions`), receipt/bill scanning (`process_receipt_screenshot`, which feeds into the same spend categorizer, tagged by source), scam-call detection (`analyze_call` / `analyze_call_recording`), and Payment Pause (`recent_flags`, tying the two scam-detection features to a simulated payment-confirmation screen). All five are also exposed as a standalone HTTP API — [`backend/`](backend/README.md), `uvicorn backend.main:app` — built on this exact same engine, not a second implementation.

**Which models are genuinely real Qualcomm AI Hub models now, versus what's still pending — the honest breakdown:**

| Model | Real AI Hub model? | Compiled + profiled on real Snapdragon hardware? | Wired into the live inference path? |
|---|---|---|---|
| OCR (EasyOCR detector + recognizer) | ✅ Yes ([`src/pipeline/ocr_qai_hub.py`](src/pipeline/ocr_qai_hub.py)) | ✅ Yes — 38.1ms / 20.4ms, real jobs (see [AI Hub section](#snapdragon--qualcomm-ai-hub)) | ✅ Yes — preferred path, live in Scam Shield + Screenshot Spend Scanner right now |
| MiniLM-v2 (shared text encoder, all 3 classifiers) | ✅ Yes ([`src/pipeline/text_encoder.py`](src/pipeline/text_encoder.py)) | ✅ Yes — 1.9ms, real job (see [AI Hub section](#snapdragon--qualcomm-ai-hub)) | ✅ Yes — preferred path, live in Scam Shield, Money Insight, Call Shield right now |
| Scam / Spend / Call classification heads | N/A — these are *our own* trained heads, sitting on top of MiniLM's embeddings, exactly as intended (MiniLM provides language understanding, these provide task-specific judgment) | ✅ Yes, all three — 41µs / 35µs / 41µs | ✅ Yes |
| Call Shield speech-to-text (encoder) | ✅ Yes, partially — the encoder only ([`src/pipeline/whisper_qai_hub.py`](src/pipeline/whisper_qai_hub.py)) | ✅ Yes — 27.8ms, real job (see [AI Hub section](#snapdragon--qualcomm-ai-hub)) | ✅ Yes — preferred path, decoder stays local (see below for why) |

**Both the QNN/CPU execution-provider path (Requirement 1) and the ONNX-optional fallback (Requirement 2) apply uniformly**: every model above prefers ONNX Runtime through `runtime.py` (QNN when available, CPU otherwise — CPU on this dev Intel Mac, honestly) and falls back to calling the raw PyTorch/scikit-learn model directly if its ONNX export is missing, rather than the whole feature going dark. `GET /status` (backend) and the "Which real AI Hub models are active right now?" expander (Streamlit demo) report this live, not as a claim.

**Whisper: encoder on AI Hub, decoder deliberately kept local — said plainly, here's the reasoning.** AI Hub's Whisper models split cleanly into an encoder (one forward pass, audio → hidden states) and a decoder (autoregressive, token-by-token, threading a growing KV cache through each step) — the decoder is the genuinely risky part to hand-wire through ONNX Runtime, where one wrong cache index or mask value can produce a transcript that reads as plausible but is quietly wrong. So only the encoder was compiled and wired through AI Hub; decoding runs through `qai_hub_models`' own **unmodified PyTorch decode loop** — nothing autoregressive was reimplemented here. **Verified before trusting this, not assumed**: transcription through the ONNX-encoder + PyTorch-decoder hybrid is *byte-for-byte identical* to the all-PyTorch baseline on all 4 bundled sample call recordings — same digital-arrest, courier, and bank-scam phrasing, same OTP/urgency triggers, same legit-call clearance. Two real, confirmed issues surfaced getting here, both fixed: (1) `qai_hub_models`' own Whisper app code imports `sounddevice` unconditionally (needed at `pip install` time even though this app never uses live mic recording through it — see the AI Hub section below); (2) the exact same offline-safety violation MiniLM had — HuggingFace's loaders phone home by default even when fully cached — except here the fix needed to be the `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` environment variables set at the very top of `app.py`/`backend/main.py`, before any other import; setting them from inside `whisper_qai_hub.py` itself worked in an isolated test script but *not* reliably once actually run through uvicorn — caught by testing the real server, not just a script, exactly the kind of gap that would have shipped a silent violation of the offline claim if the isolated test alone had been trusted.

**MiniLM-v2's accuracy gap — found, then genuinely closed, not just tuned around:** swapping the scam classifier's backbone from TF-IDF to MiniLM-v2 embeddings initially, and measurably, underperformed the old TF-IDF classifier on the harder held-out stress examples — even after a real grid search over regularization strength, best case 37/40 (3 wrong: a charity-donation scam, a fake tax-refund notice, an advance-fee loan scam — patterns genuinely missing from the training data, not a tuning problem). Fixed at the data level, the same way the original TF-IDF classifier's own accuracy gaps were fixed earlier in this project: added real training examples for those three scam patterns. That fixed all three but broke two *previously-passing* examples in the usual tug-of-war a small linear model plays with itself (an Instagram account-deletion scam, and a legit Google sign-in notice) — added one more legit sign-in example and one more distinct social-media-scam example to cover those too. **Result: 40/40, matching the original TF-IDF classifier exactly** — see the full account in the comment in [`src/scam_detector/train_classifier.py`](src/scam_detector/train_classifier.py). Money Insight and Call Shield's classification heads showed no regression from the MiniLM swap in the first place.

**A critical bug this pass caught and fixed, not just OCR/accuracy tuning:** MiniLM's HuggingFace tokenizer, by default, makes a real network call on every cold load — even fully cached locally — to check for a newer revision. Since `app.py`/`backend/main.py` install the offline network guard *before* any inference call, this would have broken every classification the moment `network_guard` was active, i.e. on every real run of the app — a genuine, confirmed violation of the "runs fully offline" claim, caught by testing, not assumed away. Fixed with `local_files_only=True` on both the tokenizer and model load (see `src/pipeline/text_encoder.py`) — the same "first run needs internet once, every run after that is fully offline" pattern already documented for EasyOCR/Whisper's own model downloads.

Full test suite: 148 tests total, **all 148 passing** — verified on a from-scratch install.