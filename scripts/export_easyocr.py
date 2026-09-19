"""Export EasyOCR's detector + recognizer (only) to ONNX, locally.

A local build step, no AI Hub account needed — run it once:

    python scripts/export_easyocr.py

This mirrors scripts/export_minilm.py and export_whisper_encoder.py: a
plain local PyTorch -> ONNX export using qai_hub_models' own model
wrappers (EasyOCRDetector/EasyOCRRecognizer — the exact same weights
EasyOCR's own Reader already uses locally), producing the two files
src/pipeline/ocr_qai_hub.py actually loads through the QNN/CPU-aware
runtime.py.

Separate from — and does not require — the AI Hub compile+profile step
(scripts/aihub_profile_easyocr.py) or the full `qai_hub_models` CLI
export documented in the README's Snapdragon/AI Hub section, which
additionally submits real cloud jobs. This script only reproduces the
portable, locally-runnable .onnx files; use the CLI or aihub_profile
scripts separately if you want to re-verify on real Snapdragon hardware.

Produces larger, self-contained .onnx files than the CLI tool's export
does (weights embedded directly rather than split into a companion
model.data file via ONNX's external-data convention) — functionally
identical, just packaged differently; verified numerically by re-running
extract_text_from_image() against the bundled samples after export.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DETECTOR_SHAPE = (1, 3, 608, 800)
RECOGNIZER_SHAPE = (1, 1, 64, 1000)


def main():
    from qai_hub_models.models.easyocr.model import EasyOCRDetector, EasyOCRRecognizer

    print("Loading EasyOCR detector + recognizer (same weights EasyOCR's own Reader uses)...")
    detector = EasyOCRDetector.from_pretrained()
    recognizer = EasyOCRRecognizer.from_pretrained()
    detector.eval()
    recognizer.eval()

    detector_path = MODELS_DIR / "easyocr_detector" / "model.onnx"
    detector_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting detector to {detector_path} {DETECTOR_SHAPE}...")
    torch.onnx.export(
        detector,
        (torch.zeros(*DETECTOR_SHAPE),),
        str(detector_path),
        input_names=["image"],
        output_names=detector.get_output_names(),
        opset_version=14,
        dynamic_axes=None,
    )
    print(f"Saved {detector_path} ({detector_path.stat().st_size / 1024:.0f} KB)")

    recognizer_path = MODELS_DIR / "easyocr_recognizer" / "model.onnx"
    recognizer_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting recognizer to {recognizer_path} {RECOGNIZER_SHAPE}...")
    torch.onnx.export(
        recognizer,
        (torch.zeros(*RECOGNIZER_SHAPE),),
        str(recognizer_path),
        input_names=["image"],
        output_names=["output_preds"],
        opset_version=14,
        dynamic_axes=None,
    )
    print(f"Saved {recognizer_path} ({recognizer_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
