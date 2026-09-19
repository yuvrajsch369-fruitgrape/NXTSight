# NXTSight — Future Vision

**Every fraud alert arrives after the money is already gone. NXTSight exists to change that — to become the layer that catches the moment before the money moves, not after.**

This started as a solo build for Qualcomm's Snapdragon AI Lab Build & Present Challenge, running fully on-device on a Snapdragon-powered PC. What follows isn't the hackathon build — it's what NXTSight can become from here. What's real today and what's still ahead are marked honestly throughout, because a vision page that can't tell the two apart isn't worth much.

## The Bet

Every fraud-protection layer that exists right now — bank SMS alerts, cybercrime helplines, spam-number lists — tells you what already happened. NXTSight's bet is that the only version of this worth building sits between the scam message and the transfer, and acts at the one moment that actually matters: the decision itself. Everything else on this page follows from that one idea.

## Six Things NXTSight Can Become That Nothing Else Is

### 1. Something that catches the scam call while it's still happening
Call Shield today reads a transcript after a call is already over — fine for a demo, useless for the person still on the phone. The next version can listen live and quietly put a warning on screen while the call is still going — "this matches the digital-arrest script, real police don't ask for money over a call." Truecaller can tell you who's calling. Nothing tells you, while you're still listening, that what's being said matches a known scam. Of everything on this page, this is the one worth building first.

### 2. A scam database that doesn't need anyone's data
Crowdsourced spam-number lists already exist. A crowdsourced list of scam *scripts* doesn't, because the obvious way to build one means uploading people's actual messages. NXTSight can hash the pattern on-device and share only the fingerprint, never the text. One phone catches a new scam, every other phone gets smarter within hours, and nobody's message ever leaves their device. It's also what stops the product from freezing at whatever it happened to be trained on this month.

### 3. A product that puts money behind its own claim
Every security product advertises a detection rate. None of them will pay you if they're wrong. NXTSight can offer a paid tier backed by an actual insurance partner — if a scam gets through that should've been caught, the user gets reimbursed. "We think this works" and "we'll pay you if it doesn't" are very different sentences, and only one of them is a real product.

### 4. Something that stops the fake bank page before anyone types into it
Safe-browsing lists and bank apps only catch sites that are already known to be bad — which usually means someone already lost money to find that out. NXTSight can watch the screen on-device, OCR plus a check on whether the domain actually matches the real bank, and catch the moment someone's about to enter a card number into a site that launched an hour ago. No history required.

### 5. Practice scams, for actual families
Companies run fake phishing emails on their own staff to train them. Nothing does that for a regular family trying to keep a parent safe. NXTSight can send an opt-in, harmless fake scam message every so often, and if someone nearly falls for it, walk them through why — building the instinct before the real one shows up, not just catching it in the moment.

### 6. The thing other apps run on, not just an app
The bigger version of this, long term, isn't a consumer app at all. It's an SDK. Any fintech, gig platform, or marketplace that wants fraud protection inside its own payment flow can plug into the same engine — the way a huge chunk of the internet quietly runs on Stripe or Twilio without anyone thinking about it. That's a bigger business than any subscription line ever gets to be.

## Why It Has to Stay On-Device

None of those six things work if the data leaves the device. A bank is never going to send customer screenshots to a third-party server to get scanned — on-device is the only architecture where that conversation with a bank even starts. It also means no server cost per user, which matters enormously at scale. It keeps working with no signal, which is most of where the fraud actually happens in India. And it's the direction Indian privacy law is already heading, not something it would be fighting on the way there.

## How This Actually Makes Money

The subscription — free core protection, a paid family tier with the guarantee attached — is the part people notice, but it's not where the real number is. The real number is licensing the engine itself: as an SDK to fintechs and marketplaces, as software preloaded on Snapdragon PCs the way antivirus already ships, and eventually as something a telecom bundles in the same way it bundles spam-call blocking today. Three different buyers, one engine underneath all of them.

## The Norton Scenario

Here's the honest test of whether any of this holds up, because a vision page that skips the competition isn't a serious one.

Norton's parent company, Gen Digital, already has an AI scam detector called Genie. It's real, it's built into Norton 360, it went global in late 2025, and by March 2026 it was even running inside ChatGPT. So this isn't a story about an empty market. But Genie's own India page says nothing about UPI-specific fraud, fake KYC, digital-arrest calls, or courier-customs scams, and Gen Digital's own numbers put its strength in North America and Europe, not India. That's the actual opening — not that Norton has nothing, but that it has breadth without depth in exactly the market growing fastest.

There's already a precedent for the kind of deal that could close that gap, too. Gen Digital shipped "Norton Deepfake Protection on Intel PCs" — a feature built specifically around Intel's on-device chip. If NXTSight gets good enough on Snapdragon, Qualcomm has its own reason to pitch that same kind of deal to Norton, the way Intel already did. NXTSight doesn't need a cold pitch to a company worth billions. It needs to be the proof point Qualcomm wants to show off.

There's a second lever sitting right there too. LifeLock, which Gen Digital already owns, built its entire brand around reimbursing people when their identity gets stolen. A scam-loss guarantee attached to an India-specific fraud engine isn't a new business model Norton would have to invent — it's LifeLock's existing idea, pointed at a problem it hasn't solved yet.

Put those two things together and the result is something nobody currently has all the pieces for at once: on-device trust, India-specific depth, Norton's distribution and brand, and an insurance guarantee behind it. That's not "the biggest scam detector." That's the one nobody else can assemble out of parts they already own.

None of this starts with a pitch deck. It starts with proof — real numbers, a scam Genie missed that this caught, a Snapdragon deployment good enough to be worth showing off. Build first. Everything else follows from that.

## How This Actually Changes Things for People

The reason this matters isn't the business model — it's who gets left out right now. The people who most need protection — first-time UPI users, a parent managing money without a bank branch nearby, someone in a small town where the connection drops constantly — are exactly the people current fraud protection assumes won't need help, because it assumes they're already sharp enough to spot the scam themselves. Safety shouldn't depend on how quick someone is in the two seconds before they panic and comply. It's the same shift seatbelts made, or antivirus software made — not asking people to be more careful, just making it survivable when they're not.

This also isn't only about individual losses. Roughly 1 in 5 UPI users have already been defrauded. Every story about someone's parents losing their savings to a fake police call makes the next person more hesitant to trust digital payments at all, and India's whole push toward a cashless economy depends on that trust holding up. A protection layer that actually works, and can prove it with the wifi turned off, does more for that trust than another awareness campaign ever will.

The crowdsourced network is what turns this from "an app that helps one person" into something bigger — every scam one phone catches makes every other phone smarter within hours, with nobody's data ever leaving their device. And because one protected machine at a Common Service Centre or an e-Mitra kiosk covers dozens of people's transactions in a single day, this reaches the people who need it most without any of them having to go find an app store first.

The most ambitious version of this: the same scams — fake police, fake customs, fake KYC, OTP phishing — aren't unique to India. They show up everywhere digital payments are spreading faster than people's ability to spot fraud. Southeast Asia. Africa. Latin America. What starts as a Snapdragon PC hackathon project has a real shot at becoming the template other countries reach for, because the problem repeats everywhere the pattern of adoption repeats. That's not said to sound big. It's said because it's true.

## Where This Can Be Built, in Order

**Now:** screenshot and pasted-text scam detection, plus spend insight, running on a Snapdragon PC, with Call Shield and Payment Pause working on sample data for the demo.

**Next:** NXTSight reading the screen and clipboard directly on Windows instead of relying on manual paste, plus catching remote-access scams specifically — someone getting talked into installing AnyDesk or TeamViewer, a desktop problem phones don't really have.

**After that:** the three hard ones — live in-call intervention, the crowdsourced pattern network, and catching fake bank-login pages before anyone types into them. Each of these needs real infrastructure, not a weekend.

**Later:** the insurance-backed guarantee, the SDK business, and getting this running on shared machines at Common Service Centres and e-Mitra kiosks, where one install protects far more than one person.

## Where This Actually Stands Right Now

Roughly 1 in 5 UPI users have been defrauded, and government data put in front of Parliament shows over ₹805 crore lost to UPI fraud this year alone — most of it never even reported. What's built so far proves the on-device pipeline works, end to end, on real Snapdragon hardware. Everything above is the case for why it's worth finishing.

This page is written assuming someone will hold NXTSight to it. That's on purpose.
