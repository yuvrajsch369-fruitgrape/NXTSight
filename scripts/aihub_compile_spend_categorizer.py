"""Compile + profile the spend categorizer for Snapdragon via Qualcomm AI Hub.

Mirrors scripts/aihub_compile_scam_classifier.py — same custom-model path
through the raw `qai_hub` API (this model isn't in the AI Hub zoo, so the
`qai_hub_models` CLI doesn't apply): upload the ONNX file, compile it for
the Snapdragon X Elite, then profile the compiled model on real physical
Snapdragon hardware in Qualcomm's cloud device farm.

This is a one-off cloud step, not something the app runs itself. Run it
manually after training the classifier:

    python -m src.spend_categorizer.train_classifier      # produces the .onnx
    python scripts/aihub_compile_spend_categorizer.py

Prerequisites (you do these yourself — they need your own AI Hub account):
    pip install qai-hub
    qai-hub configure --api_token <YOUR_API_TOKEN>   # from aihub.qualcomm.com
"""

from pathlib import Path

import qai_hub as hub

DEVICE_NAME = "Snapdragon X Elite CRD"
SOURCE_ONNX = Path("src/spend_categorizer/artifacts/classifier.onnx")
COMPILED_ONNX = Path("src/spend_categorizer/artifacts/classifier_snapdragon.onnx")


def main():
    if not SOURCE_ONNX.exists():
        raise SystemExit(
            f"{SOURCE_ONNX} not found — run `python -m src.spend_categorizer.train_classifier` first."
        )

    device = hub.Device(DEVICE_NAME)

    print(f"[NXTSight/AI Hub] Submitting compile job for '{SOURCE_ONNX}' targeting {DEVICE_NAME} ...")
    compile_job = hub.submit_compile_job(
        model=str(SOURCE_ONNX),
        device=device,
        name="NXTSight spend categorizer",
        options="--target_runtime onnx",
    )
    print(f"[NXTSight/AI Hub] Compile job: {compile_job.url}")
    compile_job.wait()
    status = compile_job.get_status()
    print(f"[NXTSight/AI Hub] Compile status: {status}")
    if not status.success:
        raise SystemExit("Compile job failed — see the job URL above for details.")

    compile_job.download_target_model(str(COMPILED_ONNX))
    print(f"[NXTSight/AI Hub] Downloaded Snapdragon-compiled model -> {COMPILED_ONNX}")

    print(f"[NXTSight/AI Hub] Submitting profile job on real {DEVICE_NAME} hardware ...")
    profile_job = hub.submit_profile_job(
        model=compile_job.get_target_model(),
        device=device,
        name="NXTSight spend categorizer",
        options="--qairt_version=default",
    )
    print(f"[NXTSight/AI Hub] Profile job: {profile_job.url}")
    profile_job.wait()
    print(f"[NXTSight/AI Hub] Profile status: {profile_job.get_status()}")
    print("[NXTSight/AI Hub] Profiling results (from real hardware, not a simulator):")
    print(profile_job.download_profile())


if __name__ == "__main__":
    main()
