"""Export MiniLM-v2 (src/pipeline/text_encoder.py) to ONNX.

A local build step, no AI Hub account needed — run it once:

    python scripts/export_minilm.py

Produces models/minilm_v2/model.onnx: static shapes (batch=1, seq_len=128,
matching AI Hub's own minilm_v2 export spec), pure ai.onnx ops only (a
real transformer graph — MatMul/LayerNorm/Softmax/etc. — no ai.onnx.ml
domain to trip up AI Hub's compiler, unlike our first attempt at the
custom classifiers' export). Once this file exists, text_encoder.py's
encode() uses it automatically through the QNN/CPU-aware runtime.py.

To also compile + profile it on real Snapdragon hardware via Qualcomm AI
Hub, once this export exists:

    python scripts/aihub_compile_minilm.py
"""

import sys
from pathlib import Path

# Run directly (`python scripts/export_minilm.py`), same as the other
# scripts/ files — but this one needs `from src....` imports, so put the
# project root on sys.path the way `python -m` would have done for us.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.pipeline.text_encoder import MAX_SEQ_LENGTH, ONNX_PATH, MiniLMEmbedder, _get_tokenizer


def main():
    print("Loading sentence-transformers/all-MiniLM-L6-v2 from HuggingFace...")
    model = MiniLMEmbedder.from_pretrained()
    model.eval()

    tokenizer = _get_tokenizer()
    sample = tokenizer(
        ["sample sentence for tracing"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {ONNX_PATH} (batch=1, seq_len={MAX_SEQ_LENGTH})...")
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        str(ONNX_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        opset_version=14,
        dynamic_axes=None,  # static shapes on purpose — see onnx_export.py's note on why
    )
    print(f"Saved {ONNX_PATH} ({Path(ONNX_PATH).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
