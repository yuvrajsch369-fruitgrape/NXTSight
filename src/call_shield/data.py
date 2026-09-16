"""Labeled training data for the scam-call classifier.

Call transcripts, not single messages — multi-turn dialogue, "Caller:" /
"You:" style, matching what a real speech-to-text transcript of a phone
call looks like. Covers patterns specific to India's call-fraud
landscape: digital-arrest scams (impersonating police/CBI/customs,
threatening arrest, demanding the victim stay on video call), fake
courier/customs scams, fake bank/KYC calls, and — on the legit side —
real bank/customer-support calls that ask for identity verification
without ever asking the caller to *share* an OTP or move money under
threat, so the model learns the pattern (impersonation + threat + urgent
payment/OTP demand) rather than reacting to "bank" or "OTP" alone.
"""

SCAM_EXAMPLES = [
    # Digital-arrest scams (police/CBI/customs impersonation)
    """Caller: Good morning, I am calling from Mumbai Police Cyber Crime Cell. Is this your Aadhaar number ending 4521?
You: Yes, why, what happened?
Caller: A parcel with your name was seized at Mumbai airport customs, it contains illegal substances and five fake passports. This is a very serious case under the Narcotics Act.
You: I haven't sent any parcel, there must be some mistake.
Caller: This is not a mistake sir, we have a digital arrest warrant issued against you. You cannot disconnect this call or you will be arrested within the hour. Switch on your camera immediately and stay on this video call.
You: I'm scared, what do I do?
Caller: To prove your innocence and avoid arrest, you must transfer your entire bank balance to the RBI verification account we will provide, for a 24 hour audit. Do not tell anyone, this is a confidential investigation.""",
    """Caller: This is Inspector Sharma from CBI headquarters Delhi. Your bank account is linked to a money laundering case involving terror funding.
You: What? That's not possible, I've never done anything like that.
Caller: We have evidence. If you do not cooperate right now, a team will arrive at your home to arrest you within 30 minutes. This is your only chance to resolve this quietly.
You: Please, I don't understand, what should I do?
Caller: Stay on this call, do not hang up, do not contact your family. Share the OTP that will come to your phone so we can verify your account is not compromised, and transfer 2 lakh rupees to the secure government escrow account for verification. It will be refunded after the investigation clears you.""",
    """Caller: Sir this is a digital arrest notice from the Customs Department. A courier addressed to you was found containing drugs and it has been traced to your Aadhaar and PAN card.
You: I haven't ordered or sent anything like that.
Caller: The investigation is already underway. To avoid immediate arrest you must remain on video call with us and follow instructions exactly. Do not switch off the phone under any circumstances.
You: Okay, I won't, please tell me what to do.
Caller: Send us your bank OTP right now and initiate a transfer of your full savings for verification purposes, or a police team will be sent to arrest you today.""",
    # Fake courier / customs fee scams (voice call version)
    """Caller: Hello, this is FedEx customer service. We have a parcel under your name that is on hold at customs due to unpaid duty charges.
You: I wasn't expecting any parcel.
Caller: It contains a laptop and some documents. To release it you need to pay a customs clearance fee of 2500 rupees immediately, otherwise the parcel will be destroyed within 6 hours and you may face legal action for undeclared items.
You: That seems urgent, how do I pay?
Caller: Share the OTP you are about to receive and I will process the payment on your behalf right now, before the deadline passes.""",
    """Caller: Good afternoon, this is regarding your Blue Dart shipment. It has been flagged by customs for containing prohibited items and there is a case being registered against you.
You: There must be some confusion, I haven't sent any shipment.
Caller: To avoid your case being escalated to the cyber cell, you must pay a penalty of 4999 rupees right now via the link I am sending, and confirm the OTP so we can verify the payment went through.
You: This is making me nervous.
Caller: You have only 15 minutes before this is escalated, please act quickly and do not disconnect the call.""",
    # Fake bank / KYC calls
    """Caller: Hello, I am calling from your bank's KYC verification department. Your account will be permanently blocked within 24 hours if your KYC is not updated today.
You: I already updated my KYC last month at the branch.
Caller: Our records show it is still pending. To complete it over the phone right now, please tell me the OTP that has just been sent to your registered mobile number.
You: I'm not comfortable sharing an OTP over the phone.
Caller: If you do not share it in the next two minutes your account and debit card will be deactivated immediately and you will have to visit the branch with all original documents, which can take weeks.""",
    """Caller: This is an automated security call from your bank. We have detected suspicious activity and your card will be blocked unless verified now.
You: What kind of suspicious activity?
Caller: Someone attempted a transaction of 45000 rupees from your account in another city. To cancel this transaction and secure your account, please confirm the 6 digit code sent to your phone right now.
You: Okay, let me check my phone.
Caller: Please hurry, this window to cancel the transaction closes in 3 minutes, after which the amount will be debited permanently.""",
    """Caller: Sir I am calling from the RBI compliance office regarding your savings account. Due to a new regulation your account will be frozen tomorrow unless you re-verify today.
You: I haven't heard about any new regulation.
Caller: It was announced this week. To re-verify, please share the OTP and also confirm your debit card number and CVV so we can update our systems before the freeze takes effect at midnight.""",
    # Fake tech support / SIM-block threats
    """Caller: This is Microsoft technical support, we have detected a serious virus on your computer that is stealing your banking information right now.
You: Really? I hadn't noticed anything wrong.
Caller: It's a silent virus. Please do not turn off your computer. I need you to install a remote access application right now so I can remove the virus before it transfers money out of your account.
You: Okay, which application should I install?
Caller: Also, to verify your identity before I begin, please read out the OTP you will receive on your phone in the next minute.""",
    """Caller: Good evening, your SIM card will be deactivated within two hours due to a pending KYC re-verification mandated by TRAI.
You: Nobody told me about this before.
Caller: This is a new directive. To keep your number active, press 9 now and share the OTP sent to confirm your identity, otherwise your number will be permanently disconnected and reassigned tonight.""",
    # Lottery / investment call scam
    """Caller: Congratulations! Your mobile number has won 25 lakh rupees in the KBC lucky draw held this week.
You: I never participated in any KBC draw.
Caller: The draw includes all active mobile numbers. To claim your prize today you need to pay a processing fee of 5000 rupees and share the OTP for prize verification, otherwise the prize will be forfeited within 24 hours and given to the next winner.""",
    """Caller: Sir, I am calling from a SEBI registered investment firm. We have an exclusive one day opportunity guaranteeing 40 percent returns if you invest right now.
You: That sounds too good to be true.
Caller: This offer closes in one hour and slots are limited. Please transfer 50000 rupees to the account I will share now and read out the OTP to confirm the transaction before the window closes.""",
    # Unlabeled, single-track versions (no "Caller:"/"You:" markers) — this
    # is what a real speech-to-text transcript of a recorded call actually
    # looks like (no diarization), as opposed to a dialogue script someone
    # typed up by hand. The model needs to recognize scam patterns either way.
    "This is Inspector Rao calling from the Cyber Crime Cell. Your Aadhaar number is linked to a parcel seized at customs containing illegal items. You are under digital arrest and must not disconnect this call. Transfer your savings to the verification account immediately and read out the OTP or a team will arrive to arrest you within the hour.",
    "This is your bank's fraud prevention team. We have detected a suspicious withdrawal attempt on your account just now. Please read out the one time password sent to your phone immediately so we can block this transaction, otherwise the funds will be withdrawn permanently in the next few minutes.",
    "This is Blue Dart courier service. Your package has been held at customs for containing prohibited items and a case has been opened under your name. To release it and avoid legal action you must pay the customs penalty right now and confirm the OTP sent to your number before the deadline expires today.",
]

LEGIT_EXAMPLES = [
    # Real bank customer service calls
    """Caller: Hello, this is calling from your bank's customer service regarding the home loan application you submitted last week.
You: Yes, I remember, is everything okay?
Caller: Everything is fine, we just need one more document, your latest salary slip. You can email it to us or upload it in the app whenever convenient, there's no urgency.
You: Sure, I'll upload it tonight.
Caller: Take your time, thank you for banking with us.""",
    """Caller: Good morning, this is your bank calling to inform you that your fixed deposit of 1 lakh rupees is maturing next month. Would you like to renew it or have it credited to your savings account?
You: I'd like to renew it for another year please.
Caller: Sure, I've noted that down, you'll receive a confirmation SMS in a day or two. Have a great day.""",
    # Real courier/delivery confirmation calls
    """Caller: Hi, this is Swiggy delivery calling, I'm outside your building, could you confirm the flat number?
You: Yes, it's flat 402, third floor.
Caller: Got it, I'll be there in two minutes.""",
    """Caller: Hello, this is Amazon delivery, your package requires an OTP that should be on your screen, can you read it out to confirm delivery at your doorstep?
You: Sure, one second, it's 8842.
Caller: Thank you, delivered successfully, have a nice day.""",
    # Appointment/reminder calls
    """Caller: Hi, this is Dr. Mehta's clinic calling to confirm your appointment tomorrow at 4 PM.
You: Yes, that works for me.
Caller: Great, please bring your previous prescription. See you tomorrow.""",
    """Caller: Good afternoon, this is a reminder call from City Dental for your cleaning appointment on Friday at 11 AM.
You: Can we reschedule to Saturday instead?
Caller: Sure, let me check availability. Yes, Saturday 11 AM works, I've updated it. See you then.""",
    # Personal / family calls
    """Caller: Hey, it's mom, are you free to talk for a bit?
You: Yeah, what's up?
Caller: Just wanted to check how the new job is going, and remind you about your cousin's wedding next month.
You: It's going well, and yes I remember, I've already blocked those dates.""",
    """Caller: Hey man, are we still on for the match tonight?
You: Yeah for sure, what time should I come over?
Caller: Around 7, I'll order some food too.""",
    # Workplace calls
    """Caller: Hi, this is Priya from HR, just calling to confirm your joining date is still the 15th.
You: Yes, that's correct.
Caller: Perfect, please bring your original documents and a passport photo on your first day. Welcome to the team.""",
    """Caller: Hello, this is regarding tomorrow's client meeting, can you send over the presentation deck before end of day?
You: Sure, I'll have it ready by 5 PM.
Caller: Thanks, appreciate it.""",
    # Legit OTP context (always confirming an action the user initiated, never demanding transfer under threat)
    """Caller: Hi, this is your bank calling to verify the fund transfer you just initiated on the app for 3000 rupees to your landlord.
You: Yes, that was me.
Caller: Great, just wanted to confirm since it was a new payee. No action needed from your side, the transfer will go through shortly.""",
    # Legit telemarketing / survey (pushy but not scam patterns)
    """Caller: Good evening, this is a quick customer satisfaction survey about your recent visit to our store, do you have two minutes?
You: Sure, go ahead.
Caller: On a scale of 1 to 10, how would you rate your experience? Also would you be interested in our loyalty membership, completely optional?
You: I'd rate it an 8, and no thanks on the membership for now.
Caller: Thank you for your time, have a great day.""",
    # Unlabeled, single-track versions (see the matching note in
    # SCAM_EXAMPLES above) — real transcribed-call text, no dialogue markers.
    "Hi, this is calling from your bank regarding the credit card limit increase you requested through the app last week. It has been approved and will reflect in your account within two business days. Is there anything else I can help you with today?",
    "Hi, this is Swiggy delivery calling, I'm outside your building, could you confirm the flat number for the delivery?",
    "Hi, this is Dr Mehta's clinic calling to confirm your appointment tomorrow at 4 PM. Please bring your previous prescription. See you tomorrow.",
]

TRAINING_DATA = [(text, True) for text in SCAM_EXAMPLES] + [
    (text, False) for text in LEGIT_EXAMPLES
]
