"""Train the scam-call classification head on MiniLM-v2 embeddings, and
export it to ONNX.

Same local, no-AI-Hub-account-needed build step as the other two
classifiers — run it once:

    python -m src.call_shield.train_classifier

MiniLM-v2 embeddings (src/pipeline/text_encoder.py) + Logistic Regression
on the labeled call-transcript dataset in data.py, exported to ONNX by
hand (src/pipeline/onnx_export.py, not skl2onnx's default converter — see
that module's docstring for why), registered as its own Task on the
shared NXTSightEngine.

Honest limitation worth knowing: MiniLM-v2 (matching AI Hub's own spec)
truncates to 128 tokens. A short SMS-length scam message fits easily; a
multi-turn call transcript often doesn't — the classification decision
only "sees" the first ~100 words of a long call. The `reason` field's
"which line triggered this" quoting is unaffected (it still runs the TF-
IDF explanation vocabulary against the *full* transcript, not the
truncated one) — only the classification decision itself is subject to
this limit.

A TF-IDF vectorizer is still fit here too — not for the decision anymore,
only to generate top_terms.json, the vocabulary classifier.py quotes when
explaining which line/words triggered a verdict.

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

from src.call_shield.data import TRAINING_DATA
from src.pipeline import text_encoder
from src.pipeline.onnx_export import export_logistic_regression

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TOP_TERMS_PER_CLASS = 40


def main():
    texts = [text for text, _ in TRAINING_DATA]
    labels = [1 if is_scam else 0 for _, is_scam in TRAINING_DATA]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english"
    )
    tfidf_features = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    explanation_classifier = LogisticRegression(max_iter=2000, C=5.0)
    explanation_classifier.fit(tfidf_features, labels)

    print(f"Encoding {len(texts)} examples with MiniLM-v2...")
    features = np.stack([text_encoder.encode(t) for t in texts]).astype(np.float32)

    classifier = LogisticRegression(max_iter=2000, C=5.0)
    classifier.fit(features, labels)

    train_accuracy = classifier.score(features, labels)
    print(f"Training accuracy on {len(texts)} examples (MiniLM-v2 features): {train_accuracy:.1%}")

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

    vocabulary = vectorizer.get_feature_names_out()
    coefficients = explanation_classifier.coef_[0]
    order = np.argsort(coefficients)
    top_legit = [vocabulary[i] for i in order[:TOP_TERMS_PER_CLASS]]
    top_scam = [vocabulary[i] for i in order[::-1][:TOP_TERMS_PER_CLASS]]
    (ARTIFACTS_DIR / "top_terms.json").write_text(
        json.dumps({"scam": top_scam, "legit": top_legit}, indent=2)
    )

    artifacts = "vectorizer.joblib, classifier.joblib"
    artifacts += ", classifier.onnx" if onnx_exported else " (classifier.onnx NOT written — see warning above)"
    print(f"Saved {artifacts}, top_terms.json -> {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
