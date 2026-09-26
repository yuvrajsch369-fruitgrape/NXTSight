"""Labeled training data for the spend-category classifier.

Realistic Indian bank/UPI SMS phrasing across banks (SBI, HDFC, ICICI,
Axis, Kotak, PNB, Canara, Yes Bank) and payment apps (UPI, Google Pay,
PhonePe, NEFT/IMPS). Merchant names carry most of the signal here, similar
to how a real bank statement categorizer works.

Also includes Hindi-English code-mixed ("Hinglish") examples across
several categories, added after measuring that a Hinglish salary-credit
message was confidently miscategorized as a P2P transfer (see
tests/test_code_mixed_text.py) — the classifier had only ever seen
"Income & Refunds" and "Transfers & UPI P2P" phrased in English. Extra
Hinglish examples for those two categories specifically emphasize the
distinguishing cue (a company/salary/refund source vs. a person's name),
the same cue the English-only examples already rely on. Worded
differently from tests/test_code_mixed_text.py's held-out examples on
purpose — the goal is the classifier learning the Hinglish pattern, not
memorizing one test sentence.
"""

CATEGORIES = sorted(
    [
        "Food & Dining",
        "Groceries",
        "Shopping",
        "Transport",
        "Bills & Utilities",
        "Entertainment",
        "Transfers & UPI P2P",
        "Income & Refunds",
        "Healthcare",
        "Investment & Savings",
        "Cash Withdrawal",
    ]
)

TRAINING_DATA = [
    # Food & Dining
    ("Rs 450.00 debited from A/c XX1234 on 12-Sep-25 at SWIGGY BANGALORE. Avl Bal Rs 12,340.50", "Food & Dining"),
    ("Rs 620 spent on your HDFC Debit Card at ZOMATO on 10-Sep-25.", "Food & Dining"),
    ("Rs 1,150 debited via UPI to DOMINOS PIZZA on 08-Sep. UPI Ref No 302981736452", "Food & Dining"),
    ("Your card was charged Rs 890 at STARBUCKS COFFEE on 14-Sep-25.", "Food & Dining"),
    ("Rs 350 paid to CAFE COFFEE DAY via Google Pay UPI on 11-Sep.", "Food & Dining"),
    ("A/c XX7788 debited Rs 720.00 towards SWIGGY INSTAMART on 09-Sep. Avl Bal Rs 8,900.", "Food & Dining"),
    ("Rs 480 debited via UPI to MCDONALDS INDIA on 13-Sep. UPI Ref No 512873645210.", "Food & Dining"),
    ("Rs 1,020 spent at BARBEQUE NATION on your Axis Bank Credit Card on 07-Sep-25.", "Food & Dining"),
    # Groceries
    ("Rs 2,340.00 debited from A/c XX1234 at BIGBASKET on 05-Sep-25. Avl Bal Rs 15,200.", "Groceries"),
    ("Rs 1,890 spent on your SBI Debit Card at DMART on 03-Sep-25.", "Groceries"),
    ("Rs 950 paid to RELIANCE FRESH via UPI on 06-Sep. UPI Ref No 887364521098.", "Groceries"),
    ("A/c XX4521 debited Rs 3,100.00 towards BLINKIT GROCERY on 02-Sep. Avl Bal Rs 22,400.", "Groceries"),
    ("Rs 640 debited via UPI to ZEPTO on 09-Sep. UPI Ref No 445621983700.", "Groceries"),
    ("Rs 1,275 spent at MORE SUPERMARKET on your ICICI Debit Card on 04-Sep-25.", "Groceries"),
    ("Rs 2,050 paid to NATURES BASKET via PhonePe UPI on 08-Sep.", "Groceries"),
    ("Rs 780 debited from A/c XX9081 at JIOMART on 11-Sep-25. Avl Bal Rs 9,650.", "Groceries"),
    # Shopping
    ("Rs 3,499.00 debited from A/c XX1234 at AMAZON on 12-Sep-25. Avl Bal Rs 11,200.", "Shopping"),
    ("Rs 2,199 spent on your HDFC Credit Card at FLIPKART on 10-Sep-25.", "Shopping"),
    ("Rs 1,650 paid to MYNTRA via UPI on 09-Sep. UPI Ref No 776543210987.", "Shopping"),
    ("A/c XX5566 debited Rs 4,200.00 towards AJIO FASHION on 06-Sep. Avl Bal Rs 18,900.", "Shopping"),
    ("Rs 899 debited via UPI to NYKAA on 13-Sep. UPI Ref No 998877665544.", "Shopping"),
    ("Rs 5,600 spent at CROMA ELECTRONICS on your Axis Bank Debit Card on 07-Sep-25.", "Shopping"),
    ("Rs 1,299 paid to H&M via Google Pay UPI on 05-Sep.", "Shopping"),
    ("Rs 3,750 debited from A/c XX3344 at DECATHLON on 11-Sep-25. Avl Bal Rs 7,800.", "Shopping"),
    # Transport
    ("Rs 240.00 debited via UPI to UBER INDIA on 12-Sep-25. UPI Ref No 334455667788.", "Transport"),
    ("Rs 180 paid to OLA CABS via PhonePe UPI on 10-Sep.", "Transport"),
    ("A/c XX2211 debited Rs 2,000.00 towards INDIAN OIL PETROL PUMP on 08-Sep. Avl Bal Rs 6,400.", "Transport"),
    ("Rs 500 spent at HP PETROL PUMP on your SBI Debit Card on 09-Sep-25.", "Transport"),
    ("Rs 45 debited via UPI to DELHI METRO RAIL on 13-Sep. UPI Ref No 112233445566.", "Transport"),
    ("Rs 320 paid to RAPIDO BIKE TAXI via UPI on 11-Sep.", "Transport"),
    ("Rs 1,200 debited from A/c XX7766 at IRCTC RAILWAYS on 06-Sep-25 for ticket booking. Avl Bal Rs 5,500.", "Transport"),
    ("Rs 90 debited via UPI to BMTC BUS PASS on 07-Sep. UPI Ref No 665544332211.", "Transport"),
    # Bills & Utilities
    ("Rs 1,240.00 debited from A/c XX1234 towards BESCOM ELECTRICITY BILL on 12-Sep-25. Avl Bal Rs 9,760.", "Bills & Utilities"),
    ("Rs 899 paid to ACT FIBERNET BROADBAND via UPI on 10-Sep.", "Bills & Utilities"),
    ("A/c XX8899 debited Rs 450.00 towards JIO MOBILE RECHARGE on 08-Sep. Avl Bal Rs 3,200.", "Bills & Utilities"),
    ("Rs 620 debited via UPI to INDANE GAS CYLINDER BOOKING on 09-Sep. UPI Ref No 223344556677.", "Bills & Utilities"),
    ("Rs 1,890 paid towards BWSSB WATER BILL via net banking on 11-Sep-25.", "Bills & Utilities"),
    ("Rs 399 debited via UPI to AIRTEL DTH RECHARGE on 13-Sep. UPI Ref No 887766554433.", "Bills & Utilities"),
    ("Rs 750 paid to VODAFONE IDEA POSTPAID BILL via PhonePe UPI on 06-Sep.", "Bills & Utilities"),
    ("A/c XX5544 debited Rs 1,050.00 towards MAHANAGAR GAS BILL on 07-Sep. Avl Bal Rs 4,900.", "Bills & Utilities"),
    # Entertainment
    ("Rs 649.00 debited from A/c XX1234 towards NETFLIX SUBSCRIPTION on 05-Sep-25. Avl Bal Rs 14,100.", "Entertainment"),
    ("Rs 199 paid to SPOTIFY PREMIUM via UPI on 03-Sep.", "Entertainment"),
    ("A/c XX6677 debited Rs 500.00 towards BOOKMYSHOW MOVIE TICKETS on 09-Sep. Avl Bal Rs 7,200.", "Entertainment"),
    ("Rs 299 debited via UPI to AMAZON PRIME VIDEO on 11-Sep. UPI Ref No 334422115566.", "Entertainment"),
    ("Rs 850 spent at PVR CINEMAS on your HDFC Debit Card on 08-Sep-25.", "Entertainment"),
    ("Rs 149 paid to HOTSTAR DISNEY SUBSCRIPTION via Google Pay UPI on 12-Sep.", "Entertainment"),
    ("Rs 1,500 debited from A/c XX4433 for SUNBURN FESTIVAL TICKETS on 06-Sep-25. Avl Bal Rs 6,900.", "Entertainment"),
    ("Rs 399 debited via UPI to YOUTUBE PREMIUM on 13-Sep. UPI Ref No 776655443322.", "Entertainment"),
    # Transfers & UPI P2P
    ("Rs 2,000 sent to RAHUL SHARMA via UPI on 12-Sep. UPI Ref No 998877001122.", "Transfers & UPI P2P"),
    ("Rs 500 debited via UPI to 9876543210@okhdfcbank on 10-Sep. UPI Ref No 445566778899.", "Transfers & UPI P2P"),
    ("Rs 1,500 paid to PRIYA VERMA via Google Pay UPI on 09-Sep.", "Transfers & UPI P2P"),
    ("A/c XX1122 debited Rs 3,000.00 to SUNIL KUMAR via IMPS on 08-Sep. Avl Bal Rs 12,400.", "Transfers & UPI P2P"),
    ("Rs 700 sent to ANJALI GUPTA via PhonePe UPI on 11-Sep.", "Transfers & UPI P2P"),
    ("Rs 1,000 debited via UPI to 8899776655@okaxis on 13-Sep. UPI Ref No 665544778899.", "Transfers & UPI P2P"),
    ("Rs 5,000 transferred to VIKRAM SINGH via NEFT on 07-Sep-25 from A/c XX3322. Avl Bal Rs 18,200.", "Transfers & UPI P2P"),
    ("Rs 250 paid to ROOMMATE for rent split via UPI on 06-Sep.", "Transfers & UPI P2P"),
    # Income & Refunds
    ("You have received Rs 45,000.00 in your account XX4521 via NEFT from ABC CORP PVT LTD SALARY on 01-Sep-25. Avl Bal: Rs 63,200.", "Income & Refunds"),
    ("Rs 850 credited to your account XX7788 as CASHBACK from AMAZON PAY on 10-Sep-25.", "Income & Refunds"),
    ("A/c XX2233 credited with Rs 1,200.00 REFUND from FLIPKART on 09-Sep. Avl Bal Rs 9,400.", "Income & Refunds"),
    ("You have received Rs 3,000 from PRIYA VERMA via UPI on 08-Sep. UPI Ref No 112233998877.", "Income & Refunds"),
    ("Rs 500 credited to your wallet as REFERRAL BONUS from PAYTM on 07-Sep-25.", "Income & Refunds"),
    ("A/c XX5566 credited Rs 60,000.00 SALARY from XYZ TECHNOLOGIES on 01-Sep-25. Avl Bal Rs 72,300.", "Income & Refunds"),
    ("Rs 2,400 credited to your account as INSURANCE CLAIM SETTLEMENT on 11-Sep-25.", "Income & Refunds"),
    ("You have received Rs 1,800 from RAHUL SHARMA via Google Pay UPI on 12-Sep.", "Income & Refunds"),
    # Healthcare
    ("Rs 1,200.00 debited from A/c XX1234 at APOLLO PHARMACY on 12-Sep-25. Avl Bal Rs 8,900.", "Healthcare"),
    ("Rs 850 spent on your HDFC Debit Card at FORTIS HOSPITAL on 10-Sep-25.", "Healthcare"),
    ("Rs 450 paid to MEDPLUS PHARMACY via UPI on 09-Sep. UPI Ref No 334455112233.", "Healthcare"),
    ("A/c XX6655 debited Rs 2,500.00 towards DR LAL PATHLABS on 08-Sep. Avl Bal Rs 11,200.", "Healthcare"),
    ("Rs 600 debited via UPI to PRACTO CONSULTATION FEE on 11-Sep. UPI Ref No 998811223344.", "Healthcare"),
    ("Rs 1,800 spent at MANIPAL HOSPITAL on your Axis Bank Debit Card on 07-Sep-25.", "Healthcare"),
    ("Rs 320 paid to NETMEDS PHARMACY via PhonePe UPI on 06-Sep.", "Healthcare"),
    ("Rs 950 debited from A/c XX7799 at 1MG HEALTHCARE on 13-Sep-25. Avl Bal Rs 5,600.", "Healthcare"),
    # Investment & Savings
    ("Rs 5,000.00 debited from A/c XX1234 towards SIP MUTUAL FUND ZERODHA on 05-Sep-25. Avl Bal Rs 22,300.", "Investment & Savings"),
    ("Rs 10,000 invested in GROWW MUTUAL FUND via net banking on 03-Sep-25.", "Investment & Savings"),
    ("A/c XX8877 debited Rs 2,000.00 towards RECURRING DEPOSIT on 09-Sep. Avl Bal Rs 14,500.", "Investment & Savings"),
    ("Rs 15,000 debited via UPI to UPSTOX STOCK PURCHASE on 11-Sep. UPI Ref No 556677889900.", "Investment & Savings"),
    ("Rs 3,000 paid towards LIC PREMIUM via net banking on 08-Sep-25.", "Investment & Savings"),
    ("Rs 8,000 debited from A/c XX9900 towards PPF DEPOSIT on 06-Sep-25. Avl Bal Rs 31,200.", "Investment & Savings"),
    ("Rs 2,500 invested in PAYTM MONEY MUTUAL FUND SIP on 12-Sep-25.", "Investment & Savings"),
    ("Rs 20,000 debited towards FIXED DEPOSIT at ICICI BANK on 07-Sep-25 from A/c XX1122.", "Investment & Savings"),
    # Cash Withdrawal
    ("Rs 5,000.00 withdrawn from A/c XX1234 at SBI ATM MG ROAD on 12-Sep-25. Avl Bal Rs 9,200.", "Cash Withdrawal"),
    ("Rs 2,000 debited via ATM withdrawal at HDFC BANK ATM on 10-Sep-25.", "Cash Withdrawal"),
    ("A/c XX3344 debited Rs 10,000.00 CASH WITHDRAWAL at AXIS ATM on 08-Sep. Avl Bal Rs 24,600.", "Cash Withdrawal"),
    ("Rs 3,000 withdrawn at ICICI BANK ATM KORAMANGALA on 09-Sep-25.", "Cash Withdrawal"),
    ("Rs 1,500 debited via ATM at CANARA BANK ATM on 11-Sep-25 from A/c XX5566.", "Cash Withdrawal"),
    ("Rs 4,000.00 CASH WITHDRAWAL at PNB ATM on 07-Sep-25. Avl Bal Rs 16,800.", "Cash Withdrawal"),
    ("Rs 2,500 withdrawn from A/c XX7788 at YES BANK ATM on 06-Sep-25.", "Cash Withdrawal"),
    ("Rs 6,000 debited via ATM withdrawal at KOTAK MAHINDRA ATM on 13-Sep-25.", "Cash Withdrawal"),
    # Hindi-English code-mixed ("Hinglish") examples, spread across categories
    ("Rs 420 Swiggy se khana order karne par aapke account se debit hua.", "Food & Dining"),
    ("Rs 1,600 BigBasket se grocery order karne par UPI se kat gaye.", "Groceries"),
    ("Rs 2,800 Myntra par shopping ke liye credit card se pay kiye gaye.", "Shopping"),
    ("Rs 150 Ola cab book karne par UPI se debit hua.", "Transport"),
    ("Aapka mobile recharge Rs 299 ka successfully ho gaya hai Jio par.", "Bills & Utilities"),
    ("Rs 199 Hotstar subscription ke liye aapke account se kat gaye.", "Entertainment"),
    ("Rs 1,000 Sunil Kumar ko UPI ke through bhej diye gaye.", "Transfers & UPI P2P"),
    ("Rs 600 roommate ko rent share ke liye Google Pay se bheje gaye.", "Transfers & UPI P2P"),
    ("Aapki salary Rs 52,000 company ki taraf se aapke account mein credit ho gayi hai.", "Income & Refunds"),
    ("Rs 900 ka refund Myntra se aapke account mein wapas aa gaya hai.", "Income & Refunds"),
    ("Rs 300 cashback Google Pay se aapke wallet mein credit hua hai.", "Income & Refunds"),
    ("Rs 550 Apollo Pharmacy mein dawai ke liye UPI se pay kiye gaye.", "Healthcare"),
    ("Rs 4,000 mutual fund SIP mein invest kiye gaye aapke account se is mahine.", "Investment & Savings"),
    ("Rs 3,000 ATM se nikale gaye HDFC Bank ATM par.", "Cash Withdrawal"),
]
