"""Train the spend-category classification head on MiniLM-v2 embeddings,
and export it to ONNX.

Mirrors src/scam_detector/train_classifier.py — same local, no-AI-Hub-
account-needed build step, same MiniLM-v2 embeddings + Logistic Regression
+ hand-built ONNX export (src/pipeline/onnx_export.py) approach, reused
here for a different (multi-class) label space:

    python -m src.spend_categorizer.train_classifier

A TF-IDF vectorizer is still fit and saved (vectorizer.joblib, engine.py
requires one per task) but plays no role in the actual category decision
anymore — categorizer.py never calls vocabulary_terms() for its reason
text (unlike the scam classifier), so this is just kept for interface
consistency with every other task on the shared engine.

Once this .onnx file is compiled for Snapdragon via AI Hub (see
scripts/aihub_compile_spend_categorizer.py), it runs on the NPU through
the same runtime.py used by the scam classifier and OCR — one on-device
model-serving layer, two features.

Also saves the fitted classifier itself (classifier.joblib) and isolates
the ONNX export in its own try/except — see the identical note in
src/scam_detector/train_classifier.py for why.
"""

import json
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.pipeline import text_encoder
from src.pipeline.onnx_export import export_logistic_regression
from src.spend_categorizer.data import CATEGORIES, TRAINING_DATA

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def main():
    category_to_id = {category: i for i, category in enumerate(CATEGORIES)}

    texts = [text for text, _ in TRAINING_DATA]
    labels = [category_to_id[category] for _, category in TRAINING_DATA]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        stop_words="english",
        # Letters only: account numbers, dates, and amounts are unique per
        # transaction and would otherwise dilute the real merchant-name signal.
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
    )
    vectorizer.fit(texts)

    print(f"Encoding {len(texts)} examples with MiniLM-v2...")
    features = np.stack([text_encoder.encode(t) for t in texts]).astype(np.float32)

    # C=10: carried over from the TF-IDF version's empirical tuning (see
    # git history) as the starting point; re-checked against held-out
    # text after switching to MiniLM features — still gives confident-but-
    # not-overconfident scores without inflating confidence on ambiguous text.
    classifier = LogisticRegression(max_iter=2000, C=10.0)
    classifier.fit(features, labels)

    train_accuracy = classifier.score(features, labels)
    print(f"Training accuracy on {len(texts)} examples, {len(CATEGORIES)} categories (MiniLM-v2 features): {train_accuracy:.1%}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dump(vectorizer, ARTIFACTS_DIR / "vectorizer.joblib")
    dump(classifier, ARTIFACTS_DIR / "classifier.joblib")

    try:
        onnx_model = export_logistic_regression(classifier, features.shape[1])
        (ARTIFACTS_DIR / "classifier.onnx").write_bytes(onnx_model.SerializeToString())
        onnx_exported = True
    except Exception as e:
        onnx_exported = False
        print(
            f"WARNING: ONNX export failed/unavailable in this environment ({type(e).__name__}: {e}). "
            "classifier.joblib was still saved — the engine will fall back to calling it directly "
            "(no ONNX Runtime / QNN acceleration until this is re-run somewhere ONNX export works)."
        )

    (ARTIFACTS_DIR / "categories.json").write_text(json.dumps(CATEGORIES, indent=2))

    artifacts = "vectorizer.joblib, classifier.joblib"
    artifacts += ", classifier.onnx" if onnx_exported else " (classifier.onnx NOT written — see warning above)"
    print(f"Saved {artifacts}, categories.json -> {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
