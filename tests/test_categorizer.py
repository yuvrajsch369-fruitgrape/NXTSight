from src.spend_categorizer.categorizer import categorize_transactions

# A realistic 18-transaction batch. Deliberately NOT copied from
# src/spend_categorizer/data.py (the training set) — different merchants,
# phrasing, and amounts, to check the model generalizes rather than
# memorizes. Mirrors a month's worth of a typical Indian bank/UPI feed.
SAMPLE_BATCH = [
    "Rs 380.00 debited from A/c XX2109 on 02-Oct-25 at BEHROUZ BIRYANI via Swiggy. Avl Bal Rs 18,540.00",
    "Rs 1,540 spent on your Kotak Debit Card at STAR BAZAAR on 03-Oct-25.",
    "Rs 2,999 paid to FLIPKART via UPI on 04-Oct. UPI Ref No 665544332211.",
    "Rs 210 debited via UPI to OLA CABS on 05-Oct. UPI Ref No 998877665511.",
    "Rs 1,450.00 debited from A/c XX2109 towards ADANI ELECTRICITY BILL on 05-Oct-25. Avl Bal Rs 16,900.00",
    "Rs 199 debited via UPI to JIOCINEMA PREMIUM on 06-Oct. UPI Ref No 332211998877.",
    "Rs 3,000 sent to KIRAN MEHTA via UPI on 06-Oct. UPI Ref No 776655998811.",
    "You have received Rs 55,000.00 in your account XX2109 via NEFT from BRIGHTWAVE SOFTWARE SALARY on 01-Oct-25. Avl Bal: Rs 71,200.00",
    "Rs 680 spent on your Kotak Debit Card at WELLNESS FOREVER PHARMACY on 07-Oct-25.",
    "Rs 6,000.00 debited from A/c XX2109 towards SIP MUTUAL FUND KUVERA on 08-Oct-25. Avl Bal Rs 12,300.00",
    "Rs 3,000 withdrawn from A/c XX2109 at BANK OF BARODA ATM on 08-Oct-25.",
    "Rs 540 paid to URBAN GROCER via UPI on 09-Oct. UPI Ref No 112233889900.",
    "Rs 899 debited via UPI to SONY LIV SUBSCRIPTION on 09-Oct. UPI Ref No 998811776655.",
    "Rs 1,150 spent at TATA PETROL PUMP on your HDFC Debit Card on 10-Oct-25.",
    "Rs 420 credited to your account as CASHBACK from FLIPKART on 10-Oct-25.",
    "Rs 750 debited via UPI to APOLLO CLINIC CONSULTATION on 11-Oct. UPI Ref No 665511998822.",
    "Rs 900 paid to LANDLORD RENT SHARE via Google Pay UPI on 11-Oct.",
    "Rs 2,200.00 debited from A/c XX2109 towards ICICI CREDIT CARD BILL PAYMENT on 12-Oct-25. Avl Bal Rs 9,100.00",
]


def test_categorizes_realistic_batch():
    result = categorize_transactions(SAMPLE_BATCH)

    assert len(result["categorized"]) == len(SAMPLE_BATCH)
    recognized = [item for item in result["categorized"] if item["category"] != "Unrecognized"]
    # Realistic bar: most of an 18-message realistic batch should be recognized.
    assert len(recognized) >= 14
    assert isinstance(result["insight"], str) and result["insight"]
    for item in result["categorized"]:
        assert 0.0 <= item["confidence"] <= 1.0


def test_empty_list_is_unanalyzable():
    result = categorize_transactions([])
    assert result["categorized"] == []
    assert "couldn't analyze" in result["insight"].lower()


def test_none_input_is_unanalyzable():
    result = categorize_transactions(None)
    assert result["categorized"] == []
    assert "couldn't analyze" in result["insight"].lower()


def test_non_list_input_is_unanalyzable():
    result = categorize_transactions("Rs 500 debited at SWIGGY")
    assert result["categorized"] == []
    assert "couldn't analyze" in result["insight"].lower()


def test_malformed_items_do_not_crash_the_batch():
    batch = [
        "Rs 450.00 debited from A/c XX1234 at SWIGGY BANGALORE on 12-Sep-25.",
        None,
        "",
        "   ",
        12345,
        "asdkjf qpwoei zxcvbn qqqqq",
        "Bonjour, comment allez-vous aujourd'hui?",
    ]
    result = categorize_transactions(batch)

    assert len(result["categorized"]) == len(batch)
    assert result["categorized"][0]["category"] == "Food & Dining"
    for item in result["categorized"][1:]:
        assert item["category"] == "Unrecognized"
        assert item["confidence"] == 0.0
        assert item["amount"] is None
    assert isinstance(result["insight"], str) and result["insight"]


def test_unrecognized_transaction_format_does_not_crash():
    batch = [
        "Rs 450.00 debited from A/c XX1234 at SWIGGY BANGALORE on 12-Sep-25.",
        "Hey, are we still on for lunch at 1pm tomorrow?",
        "Team meeting moved to 3 PM today, same conference link as usual.",
    ]
    result = categorize_transactions(batch)

    assert len(result["categorized"]) == 3
    assert result["categorized"][0]["category"] == "Food & Dining"
    # Non-transaction English sentences should not be confidently mis-categorized.
    for item in result["categorized"][1:]:
        assert isinstance(item["category"], str)
    assert isinstance(result["insight"], str) and result["insight"]


def test_all_unrecognized_gives_clear_insight():
    result = categorize_transactions(["", None, "   "])
    assert all(item["category"] == "Unrecognized" for item in result["categorized"])
    assert "couldn't" in result["insight"].lower()
