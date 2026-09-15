"""Labeled training data for the scam-vs-legit classifier.

Deliberately includes legit messages that mention OTPs, banks, and KYC —
the classifier needs to learn the *pattern* (e.g. being asked to share an
OTP) rather than just reacting to individual words like "OTP" or "bank"
that show up in plenty of real, harmless messages too.
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
    "Our AI trading bot guarantees 20% daily returns with zero risk. Thousands have already become millionaires. Start with just Rs 1000.",
    "Exclusive stock tip: This penny stock will 10x by next week. Buy now before the news breaks, guaranteed profit.",
    "Congratulations, you've been chosen for our VIP investment plan offering fixed 15% monthly returns, risk-free and government backed.",
    # Fake delivery / customs-fee scams
    "Your parcel is on hold at customs. A fee of Rs 199 is required to release it. Pay now at track-parcel-payfee.com within 12 hours.",
    "FedEx: We were unable to deliver your package due to an address issue. Pay a redelivery fee of $2.99 here to reschedule.",
    "Your order is stuck at the warehouse due to an unpaid handling charge. Clear the Rs 89 fee immediately to avoid return to sender.",
    "DHL Notice: Your shipment requires an import tax payment of $4.50 before it can be released. Pay online now to avoid delays.",
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
]

TRAINING_DATA = [(text, True) for text in SCAM_EXAMPLES] + [
    (text, False) for text in LEGIT_EXAMPLES
]
