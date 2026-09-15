# NXTSight — Live Demo Script (under 3 minutes)

Prep beforehand: `streamlit run app.py` already running, browser tab open, wifi still **on** for now (you'll turn it off live). Zoom in on the browser so badges/text are readable from the back of the room.

---

## 0:00–0:20 — The hook (say this before touching the screen)

> "Roughly 1 in 5 UPI users in India has been targeted by a payment scam. Government data shows over ₹805 crore lost to these frauds *this financial year alone*. They almost always arrive the same way — a text message. NXTSight reads that message and tells you, in plain language, whether it's a scam — entirely on-device, no cloud, no connection required. Let me show you."

## 0:20–0:40 — Point at the badges before doing anything else

*(Point at the two status badges at the top of the screen.)*

> "Two things before I show you a single feature. This badge tells you exactly which hardware is running the model right now — CPU here, since I'm on a dev machine, but on the actual Snapdragon PC this reads 'Snapdragon NPU.' Same code, it just detects what's available. And this one — 'Network: blocked and verified' — isn't something I'm telling you. The app just tried to make a real network connection to prove to itself that it's blocked, before it even finished loading this page. I'm about to turn off wifi entirely, and nothing here is going to change."

*(Turn off wifi now, visibly, if presenting live. Leave it off for the rest of the demo.)*

## 0:40–1:30 — Scam Screenshot Scanner

*(Click the **Scam Screenshot Scanner** tab. Click the `scam_lottery_win` sample.)*

> "Here's a screenshot of a fake lottery-win message — the kind that shows up on WhatsApp or SMS every day. I drop it in — no typing — and NXTSight reads the text out of the image and checks it."

*(Point at the result as it appears.)*

> "Flagged as a likely scam, with a confidence score, and — this part matters — it tells you *why*: the specific phrases in this message that match known scam patterns. Not a black box. And this ran in under a second, fully on-device."

## 1:30–2:15 — Spend Insight

*(Click the **Spend Insight** tab. Click "Load sample transactions," then "Analyze spending.")*

> "Same engine, second job. This is a batch of real-looking bank and UPI messages — a Swiggy order, an Amazon purchase, a salary credit, an ATM withdrawal. One click, and it categorizes every single one and gives me a plain-language summary."

*(Point at the insight sentence and table.)*

> "'Your top spending category was Cash Withdrawal, 31% of total spend.' No manual sorting, no spreadsheet — and it separates spending from income automatically, so a salary credit doesn't get mistaken for an expense."

## 2:15–2:50 — Tie it together and close

> "Two different jobs — flagging scams, understanding spending — but under the hood, it's *one* on-device engine doing both, which is why this is small enough to run entirely on a phone's NPU with no cloud round-trip. Today you upload a screenshot or paste text to try it live. In the real product, this reads your incoming SMS automatically in the background — the way apps like Walnut and Money View already do in India — so you'd never open the app at all."

*(Gesture at the still-off wifi indicator, or hold up the phone/laptop.)*

> "Wifi's been off this whole time. That's the point: this genuinely runs on-device, it genuinely works at NPU speed on Snapdragon hardware — we profiled it on real Qualcomm cloud hardware, not a simulator — and it genuinely needs nothing from the network to do its job. That's NXTSight."

## 2:50–3:00 — Buffer / Q&A handoff

If you're under time, stop here. If a beat remains:

> "Happy to go deeper into the architecture, the AI Hub profiling numbers, or the test coverage — whatever's most useful."

---

### If something breaks live

- **Screenshot doesn't upload / OCR looks off**: fall back to a different sample screenshot — all three are pre-loaded and known-good.
- **A category looks wrong in Spend Insight**: say so plainly — "this is a small model trained on ~90 examples; it's honest about uncertainty rather than confidently wrong" — then move on. Don't defend it, name it and continue.
- **App won't start at all**: run `python scripts/check_setup.py` on screen — it will name the exact problem and the exact fix, which itself is a good five-second proof point about the project's robustness.
