"""Labeled training data for the scam-vs-legit classifier.

Deliberately includes legit messages that mention OTPs, banks, and KYC —
the classifier needs to learn the *pattern* (e.g. being asked to share an
OTP) rather than just reacting to individual words like "OTP" or "bank"
that show up in plenty of real, harmless messages too.

Also includes Hindi-English code-mixed ("Hinglish") examples of both
classes, added after measuring that the classifier — trained on English
only — was confidently wrong on real Hinglish legit messages (a bill
reminder, an OTP-warning, a refund notice) even once text_guard.py
correctly let them through (see src/pipeline/text_guard.py and
tests/test_code_mixed_text.py). These are deliberately worded
differently from tests/test_code_mixed_text.py's held-out examples —
the point is for the classifier to learn the Hinglish *pattern* the way
it already learns the English one, not to memorize specific test
sentences.
"""

SCAM_EXAMPLES = [
    # Fake bank alerts / urgent account-blocked threats
    "Dear customer, your SBI account will be blocked today due to pending KYC. Click http://sbi-verify-now.co to update immediately.",
    "ALERT: Unusual activity detected on your ICICI account. Verify your identity within 24 hours or your account will be permanently frozen. Click here: icici-secure-verify.com",
    "Your Axis Bank debit card has been temporarily suspended due to a security breach. Confirm your card details at axis-cardsafe.net to reactivate.",
    "Final warning: Your bank account will be closed in 6 hours due to KYC non-compliance. Update now at hdfc-kyc-update.info to avoid closure.",
    "We detected a login from a new device on your account. If this wasn't you, secure your account immediately by verifying your password here: secure-bank-login.cc",
    "Your net banking has been locked for security reasons. Unlock it instantly by entering your card number and PIN at the link below.",
    # OTP-phishing (asking the victim to hand over an OTP)
    "We are calling from your bank's fraud department. To cancel this unauthorized transaction, please read out the OTP you just received.",
    "Your account is under review. Share the 6-digit verification code sent to your phone so our support team can restore access.",
    "This is Paytm support. A refund of Rs 5000 is pending. Send us the OTP to process it instantly.",
    "Security alert: Someone tried to access your account. Reply with the OTP to confirm it was you and prevent suspension.",
    "Your SIM card will be deactivated unless you confirm the OTP sent to you within the next 10 minutes.",
    # Too-good-to-be-true investment offers
    "Double your money in 7 days! Join our exclusive trading group and get guaranteed daily profits. Limited seats, DM now.",
    "Invest Rs 5,000 today and earn Rs 50,000 in one month, guaranteed. No risk, 100% return assured by our expert traders.",
    "Crypto giveaway: Send 0.1 BTC and receive 1 BTC back instantly as part of our anniversary promotion. Act fast, offer ends soon.",
    "Your crypto exchange account requires immediate verification. Enter your wallet seed phrase at this link to avoid permanent suspension.",
    "Our AI trading bot guarantees 20% daily returns with zero risk. Thousands have already become millionaires. Start with just Rs 1000.",
    "Exclusive stock tip: This penny stock will 10x by next week. Buy now before the news breaks, guaranteed profit.",
    "Congratulations, you've been chosen for our VIP investment plan offering fixed 15% monthly returns, risk-free and government backed.",
    # Fake delivery / customs-fee scams
    "Your parcel is on hold at customs. A fee of Rs 199 is required to release it. Pay now at track-parcel-payfee.com within 12 hours.",
    "FedEx: We were unable to deliver your package due to an address issue. Pay a redelivery fee of $2.99 here to reschedule.",
    "Your order is stuck at the warehouse due to an unpaid handling charge. Clear the Rs 89 fee immediately to avoid return to sender.",
    "DHL Notice: Your shipment requires an import tax payment of $4.50 before it can be released. Pay online now to avoid delays.",
    "We attempted delivery of your parcel but it is being held due to an outstanding customs duty. Settle Rs 249 within 24 hours or the item will be returned to origin.",
    # Fake social-media account-suspension threats
    "Your Facebook Page has been flagged for a copyright violation and will be permanently removed within 24 hours unless you confirm your identity through the link below.",
    "We detected unusual activity that violates our Community Standards. Your account is scheduled for deletion in 24 hours — appeal now by logging in through this link.",
    # KYC phishing / fake job offers
    "Your mobile number KYC is incomplete. Update your Aadhaar and PAN details now at the link below or your SIM will be blocked.",
    "Earn Rs 3000/day working from home, no experience needed. Just pay a Rs 499 registration fee to get started today.",
    "Part-time job alert: Like and share 50 YouTube videos daily and earn Rs 2000. Register now with a small joining fee.",
    "Update your KYC immediately to continue using your wallet. Failure to comply within 24 hours will result in permanent suspension.",
    "We are hiring data entry workers, Rs 25000/month guaranteed. Send your registration fee of Rs 599 via UPI to confirm your seat.",
    # Lottery / prize scams
    "CONGRATULATIONS! Your number has WON Rs 25,00,000 in the KBC Lucky Draw. Send your details to claim your prize within 24 hours.",
    "You have won a brand new iPhone 15! Claim your free prize now by paying a small shipping fee of Rs 199.",
    "Google Rewards: You are today's lucky winner of $10,000. Click here to claim before the offer expires at midnight.",
    "Your WhatsApp number has been selected in the Coca-Cola anniversary lucky draw. Claim your Rs 1,00,000 prize now.",
    # Fake charity / donation-urgency scams
    "Urgent appeal from Hope India Trust: earthquake survivors need shelter tonight. Transfer any amount via UPI to this number right now, every rupee counts before midnight.",
    "This is a fundraiser for children affected by the recent floods. Donate immediately to this UPI ID before our collection window closes in one hour.",
    # Fake tax-refund phishing (contrast with the legit e-verification
    # notice below: this asks for bank details and a link, that doesn't)
    "GST Department Notice: A pending refund of Rs 8,450 is ready for processing. Submit your bank account number and IFSC code at the link below within 24 hours to receive it.",
    "Tax Refund Alert: You are eligible for Rs 12,300 back. Verify your bank details and debit card number here to avoid forfeiting the refund this week.",
    # Advance-fee loan scams
    "Instant personal loan of Rs 3,00,000 approved, zero paperwork, zero interest for the first year. Pay a refundable processing fee of Rs 999 via UPI to release the amount today.",
    "Pre-approved loan offer: Rs 2,00,000 credited within 1 hour, no credit check required. Transfer a one-time processing charge of Rs 1,200 now to activate disbursal.",
    # Hindi-English code-mixed ("Hinglish") scam examples
    "Aapka bank account 24 ghante mein suspend ho jayega KYC update na karne ki wajah se. Turant is link par click karke apni details verify karein.",
    "Bank fraud department se bol rahe hain, aapke account mein ek suspicious transaction dikha hai use cancel karne ke liye humein OTP bataiye.",
    "Aapne lottery mein Rs 10,00,000 jeete hain! Prize claim karne ke liye apni bank details is number par turant bhejein.",
    "Aapka parcel customs mein rukka hua hai, Rs 299 ki fee turant pay karein warna parcel wapas bhej diya jayega.",
    "Sirf Rs 500 invest karke roz Rs 5000 kamayein, 100% guarantee hai, abhi humare group mein join karein.",
    "Aapka Aadhaar KYC pending hai, is link par apni Aadhaar aur PAN details turant update karein warna SIM band ho jayega.",
    "Ghar baithe kaam karein, Rs 2000 roz kamayein, bas ek chhoti si registration fee humein UPI se bhej dein.",
    "Aapka WhatsApp number ek lucky draw mein select hua hai, Rs 1,00,000 ka prize claim karne ke liye apni details yahan bhejein.",
    "Aapka mobile number 2 ghante mein band ho jayega KYC verification na hone ki wajah se. Jo code aapko mila hai wo turant is number par forward kar dein.",
    "GST department ki taraf se Rs 9,200 ka refund aapke liye ready hai. Apna bank account number aur IFSC code is link par turant submit karein 24 ghante ke andar.",
    "Income Tax refund Rs 14,000 ka pending hai aapke liye. Apni bank details aur debit card number yahan verify karein warna refund cancel ho jayega.",
]

LEGIT_EXAMPLES = [
    # Routine bank/financial notifications
    "Your account balance as of today is Rs 24,560.00. For details, visit your nearest branch or log in to net banking.",
    "A sum of Rs 15,000 has been credited to your account ending 4521 on 12-Sep. Available balance: Rs 39,200.",
    "Your credit card statement for this month is ready. Total amount due: Rs 8,240. Due date: 25th of this month.",
    "Your fixed deposit of Rs 50,000 will mature on 30th October. Visit the app to renew or withdraw.",
    "Your monthly account summary is now available in the app. Log in anytime to review your transactions.",
    "Thank you for banking with us. Your recent transaction of Rs 1,200 at Big Bazaar was successful.",
    # Legit OTPs (always warn against sharing, never ask you to share)
    "Your OTP for the login attempt is 738291. It is valid for 5 minutes. Never share this code with anyone, including bank staff.",
    "Use 482913 as your one-time password to complete your purchase. This code expires in 10 minutes. Do not share it with anyone.",
    "Your verification code is 105774. For your security, do not share this code with anyone, even if they claim to be from our support team.",
    "OTP 990213 to reset your password. If you did not request this, please ignore this message.",
    # Legit security notices (informational, "no action needed" — contrast
    # with the "verify immediately or be locked out" scam framing above)
    "A sign-in attempt was made on your account from an unfamiliar location in Chicago. If this was you, no action is needed.",
    "We noticed a login from an unrecognized browser at 9:42 AM. If this was you, you can safely ignore this alert.",
    "New sign-in to your Google Account from a Windows device. If this was you, you don't need to do anything. If not, we recommend you review your recent activity.",
    "Your password was changed successfully. If you did not make this change, contact support from your account settings.",
    # Legit official/government status updates (no ask, no link, no urgency)
    "Your passport renewal application (file no. PP2024981) has been received and is under processing. Track status on the official portal.",
    "Your visa application reference VA88213 has moved to the document verification stage. No action is required at this time.",
    "Your income tax return for AY 2025-26 has been successfully e-verified. Processing typically takes 20-45 days.",
    # Personal / casual messages
    "Hey, are we still on for lunch at 1pm tomorrow? Let me know if you need to reschedule.",
    "Can you send me the notes from today's class? I missed the last 20 minutes.",
    "Happy birthday! Hope you have an amazing day, let's catch up this weekend.",
    "Running 10 minutes late, traffic is bad. Be there soon!",
    "Don't forget to pick up milk and eggs on your way home.",
    "Great seeing you today, let's plan that trip soon.",
    # Legit order/delivery notifications
    "Your Amazon order #402-1928374 has shipped and is expected to arrive on Thursday. Track your package in the app.",
    "Your Zomato order has been delivered. Enjoy your meal! Rate your experience in the app.",
    "Your Swiggy order is out for delivery and will arrive in approximately 20 minutes.",
    "Your Flipkart package was delivered today at 3:45 PM. Thank you for shopping with us.",
    "Your Uber is arriving in 3 minutes. Driver: Raj, White Swift, plate MH12AB1234.",
    # Bills / reminders
    "Reminder: your electricity bill of Rs 1,240 is due on the 28th. Pay via the MyUtility app to avoid a late fee.",
    "Your monthly Netflix subscription of Rs 649 has been renewed successfully.",
    "Your gas cylinder booking is confirmed. Delivery expected within 2 days.",
    "This is a reminder that your annual health checkup is scheduled for next Monday at 10 AM.",
    "Your broadband bill for this month is Rs 899, due on the 5th. Pay now to avoid service interruption.",
    # Work / professional
    "Thanks for your purchase! Your receipt for $42.50 at Blue Bottle Coffee has been emailed to you.",
    "Team meeting moved to 3 PM today, same conference link as usual.",
    "Please review the attached document and share your feedback by end of day Friday.",
    "Your flight PNR ABC123 is confirmed for departure on the 14th at 6:45 AM. Check in online 24 hours before.",
    # Hindi-English code-mixed ("Hinglish") legit examples
    "Aapka account balance Rs 18,650 hai. Poori jaankari ke liye net banking mein login karein.",
    "Aapke account mein Rs 12,000 credit hue hain 10-Sep ko. Available balance Rs 34,500 hai.",
    "Aapka login OTP 583920 hai. Yeh 5 minute ke liye valid hai. Kisi ke saath bhi share na karein, bank staff ke saath bhi nahi.",
    "Aapke account mein ek naya sign-in Chicago se hua hai. Agar yeh aap the, toh koi action nahi chahiye.",
    "Aapka passport renewal application process ho raha hai. Status official portal par check karein.",
    "Kal 1 baje milte hain lunch ke liye? Agar reschedule karna ho toh bata dena.",
    "Aapka bijli ka bill Rs 980 hai, 28 tarikh tak pay kar dein late fee se bachne ke liye.",
    "Aapka broadband bill is mahine Rs 899 hai, 5 tarikh tak app se pay karein service interruption avoid karne ke liye.",
    "Aapka mobile postpaid bill Rs 599 ka due hai, is hafte pay karein service disconnect hone se bachne ke liye.",
    "Aapka Amazon order deliver ho gaya hai aaj 3 baje. Shopping ke liye dhanyawad.",
    "Aapka refund Rs 1,500 initiate ho gaya hai, 5 din mein aapke account mein aa jayega.",
    "Aapka gas cylinder booking confirm ho gaya hai, 2 din mein deliver ho jayega.",
]

TRAINING_DATA = [(text, True) for text in SCAM_EXAMPLES] + [
    (text, False) for text in LEGIT_EXAMPLES
]
