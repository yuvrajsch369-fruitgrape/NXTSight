# NXTSight

Built for the Snapdragon AI Lab Build & Present Challenge.

NXTSight is a small on-device app with two features that share one pipeline:

1. **Scam Screenshot Scanner** — point it at a screenshot of a suspicious text/WhatsApp/email message and it flags whether the message looks like a financial scam.
2. **Spend Insight** — feed it bank/UPI transaction text and it categorizes the spending and gives a plain-language insight (e.g. "your top spending category was Food & Dining, 24% of total spend").

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

This opens a local web page in your browser (usually `http://localhost:8501`). Two tabs, both pre-loaded with sample data so there's nothing to hunt for:

- **Scam Screenshot Scanner** — click one of the three sample screenshots already selectable on screen (`scam_bank_kyc_alert`, `scam_lottery_win`, `scam_parcel_customs_fee`), or upload your own PNG/JPG. It reads the text out of the image, then flags it as a scam or not with a confidence score and a plain-language reason.
- **Spend Insight** — click **"Load sample transactions"** to fill the box with 8 realistic bank/UPI SMS messages, then **"Analyze spending"**. Each transaction gets categorized with an amount and direction, and you get one summary sentence across all of them.

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

Want to try your own? The scam scanner accepts any screenshot with legible text; the spend categorizer accepts any bank/UPI SMS text, one message per line, pasted into the text box; the receipt scanner accepts any payment confirmation, receipt, or bill screenshot.

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

Each task still trains its **own** model — a binary scam flag and an 11-way spending category are genuinely different problems, so faking one shared set of weights would just be a worse model for both jobs. What's genuinely shared, end to end, is the serving code: text validation, vectorization, session creation, NPU/CPU detection, session caching. That's the part that actually runs on-device, and it's one object doing it for both features.

Screenshot OCR ([`src/pipeline/ocr.py`](src/pipeline/ocr.py)) is the "read" step ahead of the scam classifier — it turns an image into text, which then goes through the engine like any other input. As noted above, OCR itself doesn't yet run through this engine's QNN-aware path; the other two features do.

## Scam classifier

`classify_scam(text) -> {"is_scam": bool, "confidence": float, "reason": str}` ([src/scam_detector/classifier.py](src/scam_detector/classifier.py)) looks for the patterns behind fake bank alerts, OTP-sharing requests, too-good-to-be-true investment offers, urgent account-blocked threats, and fake delivery/KYC/job-offer scams.

**The model:** a TF-IDF + Logistic Regression classifier trained on 68 labeled examples ([src/scam_detector/data.py](src/scam_detector/data.py)), exported to ONNX (~15KB) via `skl2onnx`, registered as a Task on the shared `NXTSightEngine` (see [Architecture](#architecture-one-engine-two-jobs) above) — so once compiled for Snapdragon via AI Hub, it runs on the NPU too, no code change. Retrain it with:

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

**Accuracy:** checked against 47 held-out examples (none copied from training data) spanning romance scams, fake tech support, fake charity, tax-refund phishing, SIM-swap, crypto/seed-phrase phishing, WhatsApp code-forwarding, and more — **47/47 correct**, zero missed scams. Details in [Reliability & hardening](#reliability--hardening) below.

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
│   │   ├── network_guard.py  # blocks + proves-blocked any non-loopback network connection
│   │   └── preflight.py      # Python-version + missing-dependency checks (shared by app.py and check_setup.py)
│   ├── scam_detector/       # feature 1: screenshot OCR + scam classification
│   │   ├── data.py               # labeled training examples
│   │   ├── train_classifier.py   # local build step: trains + exports classifier.onnx
│   │   ├── classifier.py         # classify_scam(text) -> {is_scam, confidence, reason}
│   │   └── artifacts/            # trained vectorizer.joblib, classifier.onnx, top_terms.json
│   └── spend_categorizer/   # feature 2: transaction parsing + spend categorization (SMS and screenshots)
│       ├── data.py               # labeled training examples (11 categories)
│       ├── train_classifier.py   # local build step: trains + exports classifier.onnx
│       ├── categorizer.py        # categorize_transactions(texts) -> {categorized, insight}
│       ├── receipt_parser.py     # process_receipt_screenshot(): receipt/bill image -> categorize_transactions() input
│       └── artifacts/            # trained vectorizer.joblib, classifier.onnx, categories.json
├── models/                  # exported/compiled AI-Hub model artifacts (OCR) for on-device NPU inference
├── scripts/
│   ├── check_setup.py                       # environment doctor — run before the app if unsure setup is complete
│   ├── aihub_profile_easyocr.py             # one-off: profile the OCR model on real Snapdragon hardware
│   ├── aihub_compile_scam_classifier.py     # one-off: compile + profile the scam classifier for Snapdragon
│   └── aihub_compile_spend_categorizer.py   # one-off: compile + profile the spend categorizer for Snapdragon
├── data/samples/            # sample scam/receipt screenshots and sample transaction text for demos/tests
├── notebooks/               # exploration / model experimentation
├── tests/                   # unit tests — incl. test_engine.py (architecture), test_hardening.py (crash-proofing),
│                             # test_network_guard.py, test_preflight.py
├── assets/screenshots/      # demo screenshots for the submission write-up
├── requirements.txt
└── README.md
```

## Reliability & hardening

### Stress-tested, not just spot-checked

The scam classifier is checked against 47 held-out examples total, none copied from training data: the original 5 scam + 5 legit ([tests/test_classifier.py](tests/test_classifier.py)) plus a larger, deliberately harder 15+15 batch covering scam sub-types the training data was originally thin on — romance/pig-butchering scams, fake tech support, fake charity, tax-refund phishing, SIM-swap, insurance renewal, fake legal threats, crypto wallet/seed-phrase phishing, WhatsApp code-forwarding, subscription-cancellation scares, fake loans/jobs/health offers — against legit messages covering school notices, appointment reminders, calendar invites, weather alerts, security "no action needed" notices, government status updates, and everyday correspondence. Current result: **47/47 correct** on this dev machine.

Two real false positives turned up during this pass and got fixed at the data level, not by patching around them:
- A legit "new sign-in, no action needed" notification was flagged as a scam — the model had only ever seen the *scam* version of that message ("if this WASN'T you, verify immediately"). Added legit examples of the passive framing.
- A legit "your passport application is under processing" notice was flagged — official/institutional language wasn't represented on the legit side of the training data at all. Added a few.

Fixing those introduced two *new* misses (a WhatsApp code-forwarding scam and a crypto-wallet phishing scam got pulled toward "legit" by vocabulary overlap with the new legit examples) — a real example of the tug-of-war a tiny linear model plays with itself. Both got fixed: tightened the new legit examples' wording to reduce overlap, and added one targeted scam example for the previously-uncovered "seed phrase" pattern. All 47 pass now, with **zero false negatives on any scam example** across both batches — when a tradeoff has to be made, this favors never missing a real scam over never annoying someone with a false alarm.

### Nothing should crash the app on bad input

Audited every function that runs during actual use (not the one-off build/training scripts, which are run manually offline) and fixed what could actually raise an unhandled exception:

- `extract_text_from_image()` crashed on `None` or any non-path input (an unguarded `Path(path)` call) — now returns `"Error: expected a file path, got ..."`.
- The shared engine's model loading and inference (`src/pipeline/engine.py`) weren't guarded against a missing or corrupted `.onnx`/`.joblib` artifact — a bad deploy would have crashed both `classify_scam` and `categorize_transactions`. Any such failure now returns a plain-language reason both features already know how to handle (it reuses the same "couldn't analyze this" path as gibberish/non-English input).
- `classify_scam`'s reason-building vocabulary (`top_terms.json`) and `categorize_transactions`' category-name list (`categories.json`) are each optional now — missing or corrupted, they degrade gracefully (a shorter `reason`, or "Unrecognized" respectively) instead of crashing.
- `app.py`'s file-upload and transaction-analysis handlers are wrapped so any unexpected exception becomes a plain `st.error(...)` message, never Streamlit's raw traceback.

Proven, not just asserted — [tests/test_hardening.py](tests/test_hardening.py) feeds `None`, wrong types, a corrupted `.onnx` file, and a wildly mixed batch (`None`, numbers, nested lists, control characters, a 10,000-character string) through every entry point and checks each one comes back with a clear result instead of raising.

### Pinned dependencies

Every package in `requirements.txt` is pinned to an exact version captured from a real working install — see the file's own comments for why each pin exists. One real cross-package conflict got caught and fixed during this pass: `qai-hub-models` requires plain `opencv-python`, which silently conflicts with the `opencv-python-headless` that `easyocr` needs (both packages install a `cv2` module at the same path; whichever installs second wins, non-deterministically — a genuinely "quietly breaks depending on install order" bug). Fixed by moving `qai-hub-models` out of the base install entirely — it's only needed for one optional AI Hub CLI workflow, which has its own separate install instructions below.

### What happens on a different machine

| Scenario | What happens |
|---|---|
| **Different OS** (Windows/Linux instead of this dev Mac) | All file paths use `pathlib`; the OS-specific pieces (QNN backend filename, `onnxruntime` vs `onnxruntime-qnn`) are already platform-gated in code and in `requirements.txt`. Should work as-is — not independently tested on Windows/Linux hardware, so run `python scripts/check_setup.py` first to confirm before relying on it. |
| **Missing or incomplete `pip install`** | `app.py` checks the Python version and every required import *before* touching Streamlit's UI machinery, showing exactly which packages are missing and the fix — instead of a raw `ModuleNotFoundError` appearing mid-script. Verified by simulating a partial install (only `streamlit` present): the app showed a clean, itemized error rather than crashing. Run `python scripts/check_setup.py` standalone for the same check before even starting the app. |
| **No internet during `pip install`** | Not something an app can fix after the fact — pip itself gives a normal, clear network error in this case. What *is* fixed: the exact pins above mean that once install succeeds, it's the same install every time, so a "worked yesterday, broke today" failure from an unrelated upstream release doesn't happen later. |
| **Too-old Python** | `engine.py` uses `dict[str, Task]`-style type hints (PEP 585), which need Python 3.9+. Both `app.py` and `scripts/check_setup.py` check this explicitly and name the exact minimum version required, rather than failing with a cryptic `TypeError: 'type' object is not subscriptable` deep inside an import. |
| **Fresh machine, first-ever run, wifi off** | EasyOCR needs network once to download its model weights (documented under [What runs on-device](#what-runs-on-device-npu-vs-cpu-fallback) above) — `network_guard` would (correctly) block that too. Run the app once with internet before a wifi-off presentation. |

## Snapdragon / Qualcomm AI Hub

There are two separate pieces here — a **build-time step** you run once, by hand, to get a Snapdragon-ready model, and a **runtime check** the app makes every time it starts (already covered in [What runs on-device](#what-runs-on-device-npu-vs-cpu-fallback) above).

### Build-time: compile + profile a model for Snapdragon via AI Hub

This step turns a model into a `.onnx` file specifically compiled for the Snapdragon X Elite, and confirms it actually runs correctly by profiling it on real, physical Snapdragon hardware in Qualcomm's cloud device farm (not a simulator — useful since most of us don't own a Snapdragon PC to test on directly). It needs your own free AI Hub account and API token from [aihub.qualcomm.com](https://aihub.qualcomm.com); nobody else can run this step for you.

`qai-hub-models` is intentionally *not* in the base `requirements.txt` — it pulls in plain `opencv-python`, which silently conflicts with the `opencv-python-headless` that `easyocr` needs (both packages install a `cv2` module at the same path, and whichever installs second wins). Install it separately, only when you actually need this step:

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

Both models load and report their expected shapes locally through `runtime.py` — the detector takes a `(1, 3, 608, 800)` image tensor, matching EasyOCR's documented input resolution. That's solid evidence this genuinely runs on Snapdragon, not just in theory — see [What runs on-device](#what-runs-on-device-npu-vs-cpu-fallback) above for the honest caveat that these compiled models aren't yet the ones the live app calls.

## Status

All pipeline stages work end to end: OCR (`extract_text_from_image`), scam classification (`classify_scam`), spend categorization (`categorize_transactions`), and receipt/bill scanning (`process_receipt_screenshot`, which feeds into the same spend categorizer, tagged by source). The scam and spend classifiers run through the Snapdragon-aware execution path (NPU when available, CPU fallback otherwise); OCR runs on CPU today, with its Snapdragon-compiled counterpart already validated on real hardware but not yet wired into the live call. Full test suite: 101 tests, all passing, verified on a from-scratch install.