"""Train the scam-call classifier and export it to ONNX.

Same local, no-AI-Hub-account-needed build step as the other two
classifiers — run it once:

    python -m src.call_shield.train_classifier

TF-IDF + Logistic Regression on the labeled call-transcript dataset in
data.py, exported to ONNX via skl2onnx, registered as its own Task on the
shared NXTSightEngine. Call transcripts are much longer and more varied
than a single SMS, so this uses a larger vocabulary cap than the other
two classifiers' training scripts.
"""

import json
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from src.call_shield.data import TRAINING_DATA

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TOP_TERMS_PER_CLASS = 40


def main():
    texts = [text for text, _ in TRAINING_DATA]
    labels = [1 if is_scam else 0 for _, is_scam in TRAINING_DATA]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english"
    )
    features = vectorizer.fit_transform(texts).toarray().astype(np.float32)

    classifier = LogisticRegression(max_iter=2000, C=5.0)
    classifier.fit(features, labels)

    train_accuracy = classifier.score(features, labels)
    print(f"Training accuracy on {len(texts)} examples: {train_accuracy:.1%}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dump(vectorizer, ARTIFACTS_DIR / "vectorizer.joblib")

    onnx_model = convert_sklearn(
        classifier,
        initial_types=[("input", FloatTensorType([None, features.shape[1]]))],
        options={id(classifier): {"zipmap": False}},
        target_opset=13,
    )
    (ARTIFACTS_DIR / "classifier.onnx").write_bytes(onnx_model.SerializeToString())

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
