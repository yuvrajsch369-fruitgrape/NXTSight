"""Train the scam-vs-legit classifier and export it to ONNX.

This is a local build step (unlike the OCR model, it needs no AI Hub
account) — run it once:

    python -m src.scam_detector.train_classifier

It fits a TF-IDF + Logistic Regression pipeline on the small labeled
dataset in data.py, then exports the classifier half to ONNX — by hand,
via src/pipeline/onnx_export.py, not skl2onnx's default converter, since
that emits an ai.onnx.ml op Qualcomm AI Hub's compiler rejects — so it can
run through the same QNN-aware runtime.py used for the OCR models. Once
this exact .onnx file is compiled for Snapdragon via AI Hub (see
scripts/aihub_compile_scam_classifier.py), it runs on the NPU instead of
CPU — no code change in classifier.py.

Text vectorization (TF-IDF) stays in Python/scikit-learn rather than
being folded into the ONNX graph, the same tradeoff the OCR stage makes
by keeping pre/post-processing outside the compiled model.
"""

import json
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.pipeline.onnx_export import export_logistic_regression
from src.scam_detector.data import TRAINING_DATA

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TOP_TERMS_PER_CLASS = 40


def main():
    texts = [text for text, _ in TRAINING_DATA]
    labels = [1 if is_scam else 0 for _, is_scam in TRAINING_DATA]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english"
    )
    features = vectorizer.fit_transform(texts).toarray().astype(np.float32)

    classifier = LogisticRegression(max_iter=1000, C=1.0)
    classifier.fit(features, labels)

    train_accuracy = classifier.score(features, labels)
    print(f"Training accuracy on {len(texts)} examples: {train_accuracy:.1%}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dump(vectorizer, ARTIFACTS_DIR / "vectorizer.joblib")

    onnx_model = export_logistic_regression(classifier, features.shape[1])
    (ARTIFACTS_DIR / "classifier.onnx").write_bytes(onnx_model.SerializeToString())

    # Precompute human-readable top terms per class for classifier.py's
    # `reason` field, so it doesn't need scikit-learn at inference time.
    vocabulary = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    order = np.argsort(coefficients)
    top_legit = [vocabulary[i] for i in order[:TOP_TERMS_PER_CLASS]]
    top_scam = [vocabulary[i] for i in order[::-1][:TOP_TERMS_PER_CLASS]]
    (ARTIFACTS_DIR / "top_terms.json").write_text(
        json.dumps({"scam": top_scam, "legit": top_legit}, indent=2)
    )

    print(f"Saved vectorizer.joblib, classifier.onnx, top_terms.json -> {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
