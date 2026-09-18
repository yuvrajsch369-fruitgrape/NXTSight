"""MiniLM-v2 — the shared on-device text encoder every classification head
now runs on top of, instead of each task fitting its own TF-IDF vocabulary.

This is exactly what Qualcomm AI Hub's own "MiniLM-v2" model listing wraps
(https://aihub.qualcomm.com/models/minilm_v2): `sentence-transformers/
all-MiniLM-L6-v2` from HuggingFace, mean-pooled and L2-normalized into a
384-dim sentence embedding. qai_hub_models' own packaged version of this
model (qai_hub_models.models.minilm_v2) needs torch>=2.4, which has no
installable wheel for this Intel Mac (torch dropped Intel Mac wheels
after 2.2.2 — the same reason requirements.txt already pins torch==2.2.2
for everything else). Rather than block on that, this module builds the
identical model directly — same HF weights, same forward pass (mean pool
+ L2 normalize) as AI Hub's own qai_hub_models/models/minilm_v2/model.py
— using the torch version already installed, and exports it to ONNX by
hand so it can still be submitted to Qualcomm AI Hub for a real
compile+profile job via the same raw qai_hub API already used for the
three classifiers (see scripts/aihub_compile_minilm.py).

One shared encoder, not one per task — a scam flag, a spending category,
and a scam-call flag are different classification heads sitting on top of
the same general-purpose sentence embedding, the same way three people
can read the same paragraph and reach three different conclusions about
it. Each task keeps its own TF-IDF vectorizer too (see engine.py) — not
for classification anymore, only to keep generating the "which words
triggered this" explanation text, which needs an interpretable bag-of-
words signal that a dense embedding fundamentally can't give you.
"""

import logging
from pathlib import Path

import numpy as np
import torch

from src.pipeline.runtime import create_inference_session

logger = logging.getLogger("nxtsight")

HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 128

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ONNX_PATH = _PROJECT_ROOT / "models" / "minilm_v2" / "model.onnx"

_tokenizer = None
_onnx_session = None
_onnx_description = None
_torch_model = None


class MiniLMEmbedder(torch.nn.Module):
    """Sentence embedding: transformer -> mean pooling (attention-mask
    weighted) -> L2 normalize. Architecturally identical to AI Hub's own
    qai_hub_models/models/minilm_v2/model.py AllMiniLML6V2 — same HF
    weights, same forward pass — built by hand only because that package
    itself needs a newer torch than this machine can install.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, weights: str = HF_MODEL_ID) -> "MiniLMEmbedder":
        from transformers import AutoModel

        # local_files_only=True: without it, HuggingFace's from_pretrained()
        # makes a real network call (a HEAD/GET to check the latest revision
        # hash) even when the model is fully cached locally already — a real,
        # confirmed violation of the "runs fully offline" guarantee (it fails
        # loudly with our own NetworkBlockedError once network_guard is
        # installed, exactly as it should). Same "first run needs internet,
        # every run after that doesn't" pattern already documented for
        # EasyOCR/Whisper's one-time model downloads.
        return cls(AutoModel.from_pretrained(weights, local_files_only=True))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state

        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        embeddings = sum_embeddings / sum_mask

        return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        # local_files_only=True — see the identical note on
        # MiniLMEmbedder.from_pretrained() above. Confirmed by testing:
        # without it, this makes a real outbound connection attempt even
        # with the tokenizer fully cached, which network_guard (correctly)
        # blocks the moment it's installed — i.e. on every real run of the
        # app, since app.py/backend install it before any inference call.
        _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, local_files_only=True)
    return _tokenizer


def _tokenize(text: str):
    tok = _get_tokenizer()
    encoded = tok(
        [text],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    return encoded["input_ids"], encoded["attention_mask"]


def _get_onnx_session():
    """Prefer the QNN/CPU-aware ONNX path — same shared
    runtime.create_inference_session() every other model in this app
    uses. Raises if the exported model isn't present; encode() below
    catches that and falls back to the raw PyTorch model directly.
    """
    global _onnx_session, _onnx_description
    if _onnx_session is None:
        if not ONNX_PATH.exists():
            raise FileNotFoundError(f"MiniLM-v2 ONNX export not found at {ONNX_PATH}")
        _onnx_session, _onnx_description = create_inference_session(str(ONNX_PATH))
    return _onnx_session


def _get_torch_model():
    global _torch_model
    if _torch_model is None:
        logger.warning(
            "MiniLM-v2 ONNX path unavailable — falling back to the raw PyTorch model "
            "directly (no ONNX Runtime / QNN acceleration until it's exported)."
        )
        _torch_model = MiniLMEmbedder.from_pretrained()
        _torch_model.eval()
    return _torch_model


def encode(text: str) -> np.ndarray:
    """Text -> 384-dim L2-normalized sentence embedding. Prefers the
    ONNX/QNN-aware path, falls back to the raw PyTorch model — same
    two-tier pattern as every classifier's ONNX/scikit-learn fallback in
    engine.py. Never raises for a missing model; only for tokenizer
    failures, which would indicate a genuinely broken environment.
    """
    input_ids, attention_mask = _tokenize(text)

    try:
        session = _get_onnx_session()
        outputs = session.run(
            ["embeddings"],
            {
                "input_ids": input_ids.numpy().astype(np.int64),
                "attention_mask": attention_mask.numpy().astype(np.int64),
            },
        )
        return outputs[0][0]
    except Exception:
        model = _get_torch_model()
        with torch.no_grad():
            embedding = model(input_ids, attention_mask)
        return embedding.numpy()[0]


def status() -> tuple:
    """(is_active, description) for the ONNX/QNN-aware path specifically
    — used for the startup status log. Never raises."""
    try:
        _get_onnx_session()
        return True, _onnx_description
    except Exception as e:
        return False, f"not active, using PyTorch fallback ({type(e).__name__}: {e})"
