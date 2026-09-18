"""Export Whisper-tiny's encoder (only) to ONNX.

A local build step, no AI Hub account needed — run it once:

    python scripts/export_whisper_encoder.py

Deliberately encoder-only. Whisper's decoder is the autoregressive,
KV-cache-dependent part — genuinely risky to hand-wire through ONNX
Runtime with any confidence. The encoder is a single forward pass (audio
mel features -> hidden states, here specifically AI Hub's own monkey-
patched encoder that directly precomputes cross-attention K/V for every
decoder block) — the same low-risk shape as MiniLM-v2's export. Decoding
stays exactly as qai_hub_models' own HfWhisperApp already implements it
(src/pipeline/whisper_qai_hub.py reuses that class and its decode loop
unchanged) — nothing autoregressive is reimplemented here.

Verified before submitting anything to AI Hub: transcription through the
ONNX-encoder + PyTorch-decoder hybrid is *identical* to the all-PyTorch
baseline on all 4 bundled sample call recordings — see backend/README.md
or the main README's Call Shield section for that comparison.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qai_hub_models.models.whisper_tiny.model import WhisperTiny

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "models" / "whisper_tiny_encoder" / "model.onnx"


def main():
    print("Loading openai/whisper-tiny (AI Hub's monkey-patched encoder/decoder pair)...")
    whisper = WhisperTiny.from_pretrained()

    sample_input = torch.zeros(1, 80, 3000)  # (batch, num_mel_bins, mel-feature length)
    output_names = whisper.encoder.get_output_names()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting encoder to {OUTPUT_PATH} ({len(output_names)} cross-attention KV outputs)...")
    torch.onnx.export(
        whisper.encoder,
        (sample_input,),
        str(OUTPUT_PATH),
        input_names=["input_features"],
        output_names=output_names,
        opset_version=14,
        dynamic_axes=None,
    )
    print(f"Saved {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
