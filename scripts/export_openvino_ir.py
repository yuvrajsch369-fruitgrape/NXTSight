"""Export every NXTSight model to OpenVINO's native IR format (.xml + .bin).

A local, optional verification/optimization step — the same shape as
scripts/aihub_compile_*.py for Snapdragon, except this one runs entirely
locally (no account, no cloud job) via the `openvino` Python package's
own model converter. Run it once:

    python scripts/export_openvino_ir.py

This does NOT change what the live app runs by default: every model
already ships a portable .onnx that the hardware-abstraction layer in
src/pipeline/runtime.py picks up automatically (QNN, CUDA, OpenVINO,
DirectML, or CoreML, whichever ONNX Runtime reports as available, CPU
otherwise — see that file). What this script proves is that NXTSight's
own models genuinely convert to OpenVINO's own graph representation
without hitting an unsupported operator — a real, run-it-yourself check,
not an assumption.

Platform note, confirmed directly against PyPI (not assumed): the
`openvino` package used here (model conversion + local CPU-plugin
verification) publishes macOS wheels only through version 2025.4.1 for
Intel Macs (x86_64) — 2026.4.0 dropped Intel-Mac support entirely, the
same pattern already documented for torch elsewhere in this project.
requirements.txt pins accordingly. `onnxruntime-openvino` — the
separate package that actually plugs the OpenVINO execution provider
into ONNX Runtime for the live app — publishes NO macOS wheels at all,
any version: this script can prove the *models* convert cleanly, but
the runtime EP itself can only be verified on Windows or Linux.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "models" / "openvino_ir"

# (friendly name, source .onnx path)
MODELS = [
    ("scam_classifier", PROJECT_ROOT / "src/scam_detector/artifacts/classifier.onnx"),
    ("spend_classifier", PROJECT_ROOT / "src/spend_categorizer/artifacts/classifier.onnx"),
    ("call_shield_classifier", PROJECT_ROOT / "src/call_shield/artifacts/classifier.onnx"),
    ("minilm_v2", PROJECT_ROOT / "models/minilm_v2/model.onnx"),
    ("easyocr_detector", PROJECT_ROOT / "models/easyocr_detector/model.onnx"),
    ("easyocr_recognizer", PROJECT_ROOT / "models/easyocr_recognizer/model.onnx"),
    ("whisper_tiny_encoder", PROJECT_ROOT / "models/whisper_tiny_encoder/model.onnx"),
]


def _explain_failure(name: str, error: Exception) -> None:
    """Printed only if a conversion genuinely fails — real, actionable
    options, not a silent workaround. Which of these actually applies
    depends on the specific op OpenVINO's converter names in `error`."""
    print(f"\n  {name}: CONVERSION FAILED")
    print(f"  {type(error).__name__}: {error}")
    print(
        "\n  Real options, in the order worth trying:\n"
        "    1. Check the error above for the specific unsupported op name —\n"
        "       OpenVINO's converter names it directly (e.g. 'OpenVINO does not\n"
        "       support the following ONNX operations: <Op>').\n"
        "    2. Operator substitution: if it's a custom or rarely-used op, check\n"
        "       whether an equivalent standard ONNX op exists and re-export from\n"
        "       the source model using it (the same fix this project already\n"
        "       used for AI Hub — see src/pipeline/onnx_export.py's hand-built\n"
        "       Gemm->Softmax->ArgMax graph, built specifically because\n"
        "       skl2onnx's default output used an op AI Hub's compiler rejected).\n"
        "    3. Opset version: try re-exporting the source model at a different\n"
        "       ONNX opset (older ops sometimes have broader converter support\n"
        "       than their newer equivalents, or vice versa).\n"
        "    4. Quantization is NOT the first thing to reach for here — it\n"
        "       changes numerical precision to solve a *performance* problem,\n"
        "       not an unsupported-operator problem. Only relevant once a model\n"
        "       already converts cleanly and you're tuning speed/size.\n"
    )


def main() -> int:
    import openvino as ov

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for name, source_path in MODELS:
        if not source_path.exists():
            print(f"  {name}: SKIPPED (source not found: {source_path})")
            results.append((name, "skipped", None))
            continue

        print(f"  {name}: converting {source_path.relative_to(PROJECT_ROOT)} ...")
        start = time.time()
        try:
            model = ov.convert_model(str(source_path))
            target_dir = OUTPUT_DIR / name
            target_dir.mkdir(parents=True, exist_ok=True)
            xml_path = target_dir / "model.xml"
            ov.save_model(model, str(xml_path))
            elapsed = time.time() - start
            size_kb = (xml_path.stat().st_size + (target_dir / "model.bin").stat().st_size) / 1024
            print(f"    OK — {xml_path.relative_to(PROJECT_ROOT)} + .bin ({size_kb:.0f} KB total, {elapsed:.1f}s)")
            results.append((name, "ok", elapsed))
        except Exception as e:
            _explain_failure(name, e)
            results.append((name, "failed", str(e)))

    print("\n--- Summary ---")
    ok = [r for r in results if r[1] == "ok"]
    failed = [r for r in results if r[1] == "failed"]
    skipped = [r for r in results if r[1] == "skipped"]
    print(f"{len(ok)} converted, {len(failed)} failed, {len(skipped)} skipped (out of {len(MODELS)})")
    if failed:
        print("Failed:", ", ".join(name for name, _, _ in failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
