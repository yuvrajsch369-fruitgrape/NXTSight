"""Train the spend-category classifier and export it to ONNX.

Mirrors src/scam_detector/train_classifier.py — same local, no-AI-Hub-
account-needed build step, same TF-IDF + Logistic Regression + hand-built
ONNX export (src/pipeline/onnx_export.py) approach, reused here for a
different (multi-class) label space:

    python -m src.spend_categorizer.train_classifier

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
    features = vectorizer.fit_transform(texts).toarray().astype(np.float32)

    # C=10: with only ~8 examples per category spread across 11 classes,
    # the default regularization spreads probability mass too thin even on
    # clear-cut merchant names (e.g. "swiggy" seen in training still only
    # scored ~0.27 at C=2). Checked C in {2,5,10,20,50} against held-out
    # text; C=10 gives confident-but-not-overconfident scores (~0.6 on
    # clear matches) without inflating confidence on ambiguous/unrelated text.
    classifier = LogisticRegression(max_iter=2000, C=10.0)
    classifier.fit(features, labels)

    train_accuracy = classifier.score(features, labels)
    print(f"Training accuracy on {len(texts)} examples, {len(CATEGORIES)} categories: {train_accuracy:.1%}")

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
