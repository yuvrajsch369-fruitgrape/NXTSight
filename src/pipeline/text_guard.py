"""Shared "is this text analyzable" guard for the pipeline's text stages.

Both classify_scam() and categorize_transactions() need to reject the same
kinds of bad input before it ever reaches a model: empty text, gibberish,
non-English text. Centralized here so the two stages share one definition
instead of drifting apart, and so a Snapdragon-side language model (if one
replaces langdetect later) only needs to change in one place.

Two real problems were found and fixed here after measuring this gate
directly against realistic Hindi-English code-mixed ("Hinglish") text —
what real Indian scam/transaction SMS actually look like, not just clean
English:

1. langdetect ships with NO fixed random seed, so its own README documents
   this: the exact same input string can get a different language guess
   on different calls, purely from internal randomness, nothing else
   changing. `DetectorFactory.seed = 0` below is langdetect's own
   documented fix — it doesn't change how good the detector is, only
   makes a given input's verdict consistent from one call to the next.

2. langdetect's n-gram models have no notion of Hindi-English code-mixing
   at all — it was trained one-language-per-document. Fed a genuinely
   Hinglish sentence ("Aapka SBI account 24 ghante mein block ho jayega"),
   it doesn't detect "mixed" or "hi" — it typically misidentifies the
   whole string as Indonesian, Estonian, or Turkish (short-text n-gram
   collisions with romanized Hindi), then this gate would reject it as
   "not English". No amount of tuning MIN_LANGDETECT_CONFIDENCE or
   allowed_langs fixes this cleanly, because the detector was never given
   a real Hindi-English-mixed class to output. _looks_like_hindi_english_
   code_mixed() below sidesteps the problem instead of tuning around it:
   it checks directly for unambiguous Hindi grammatical markers (common
   postpositions, pronouns, verb forms in their everyday Roman-script
   spelling) and, if there are enough of them, treats the text as
   analyzable without ever asking langdetect at all. Measured directly
   against realistic Hinglish scam and transaction text before and after
   this change — see tests/test_code_mixed_text.py.
"""

import re

from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0  # langdetect's own documented fix for non-deterministic output

MIN_CHARS = 3
MIN_LETTER_RATIO = 0.3  # below this, text is mostly symbols/numbers, not language
MIN_LANGDETECT_CONFIDENCE = 0.70

# Common Hindi postpositions, pronouns, and verb/auxiliary forms in their
# everyday Roman-script ("Hinglish") spelling — the words that make
# "Aapka account block ho jayega" instantly readable as Hindi to a human
# reader even though every character is plain ASCII. Deliberately excludes
# anything that collides with a common standalone English word (e.g. "hi",
# "to", "is", "so", "us", "he" are all real Hindi/Urdu romanizations too,
# but including them would make ordinary English text falsely match) —
# every entry here is unambiguous.
_HINDI_MARKERS = frozenset(
    {
        # postpositions
        "ka", "ki", "ke", "ko", "se", "mein", "par", "tak", "liye",
        # pronouns / possessives
        "aap", "aapka", "aapke", "aapki", "aapne", "aapko", "hum", "humein",
        "unka", "unke", "iska", "iske", "uska", "uske", "yeh", "wala", "waala",
        # verbs / auxiliaries
        "hai", "hain", "tha", "thi", "the", "karein", "kare", "karo", "karna",
        "kiya", "kijiye", "diya", "diye", "dena", "dijiye", "milega", "milegi",
        "mila", "mili", "jayega", "jayegi", "gaya", "gayi", "gaye", "hua",
        "hue", "bhejein", "bhejo", "bhej", "dekhein", "hoga", "hogi", "honge",
        # adverbs / time
        "turant", "abhi", "aaj", "kal", "jaldi", "baje", "ghante", "tarikh",
        # negation / conjunctions
        "nahi", "mat", "aur", "agar", "lekin", "warna", "toh",
    }
)
# Both floors matter: the absolute count guards against a single incidental
# collision (e.g. "hui" inside the French word "aujourd'hui" tokenizing to
# a standalone "hui", which is also a Hindi word for "happened") tipping a
# genuinely different-language string; the ratio guards against a long
# English message that happens to contain two or three of these short
# tokens as an incidental substring match across an otherwise-huge word count.
_MIN_HINDI_MARKER_MATCHES = 2
_MIN_HINDI_MARKER_RATIO = 0.15

_WORD_PATTERN = re.compile(r"[a-zA-Z]+")


def _looks_like_hindi_english_code_mixed(text: str) -> bool:
    """True if `text` contains enough unambiguous Hindi grammatical markers
    to be confidently read as Hindi-English code-mixed text, regardless of
    what a general-purpose statistical language detector (trained on
    single-language documents, not code-mixed SMS) makes of it.
    """
    words = [w.lower() for w in _WORD_PATTERN.findall(text)]
    if not words:
        return False
    matches = sum(1 for w in words if w in _HINDI_MARKERS)
    return matches >= _MIN_HINDI_MARKER_MATCHES and (matches / len(words)) >= _MIN_HINDI_MARKER_RATIO


def unanalyzable_reason(text, allowed_langs=("en",)) -> "str | None":
    """Return a reason string if `text` can't be analyzed, else None.

    `allowed_langs` defaults to English only, matching both models today.
    Hindi-English code-mixed text is accepted independently of
    `allowed_langs` — see _looks_like_hindi_english_code_mixed() above —
    since it's a real pattern in the SMS this pipeline is meant to read,
    not a second "language" to opt into.
    """
    if text is None or not isinstance(text, str):
        return "no text provided"

    cleaned = text.strip()
    if not cleaned:
        return "no text provided"

    if len(cleaned) < MIN_CHARS:
        return "text is too short to judge"

    letter_ratio = sum(c.isalpha() for c in cleaned) / len(cleaned)
    if letter_ratio < MIN_LETTER_RATIO:
        return "text doesn't look like readable language (too few letters)"

    if _looks_like_hindi_english_code_mixed(cleaned):
        return None

    try:
        candidates = detect_langs(cleaned)
    except LangDetectException:
        return "text doesn't look like readable language (gibberish)"
    if not candidates:
        return "text doesn't look like readable language (gibberish)"

    top = candidates[0]
    if top.prob < MIN_LANGDETECT_CONFIDENCE:
        return "couldn't confidently identify the language (possibly gibberish)"
    if top.lang not in allowed_langs:
        return f"detected language '{top.lang}' — this model only supports English"

    return None
