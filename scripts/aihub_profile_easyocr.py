"""Submit a profile job to Qualcomm AI Hub's cloud device farm.

This confirms a compiled model actually runs correctly on real, physical
Snapdragon hardware — useful since most of us don't own a Snapdragon PC to
test on directly. This is a one-off cloud step, not something the app runs
itself: run it manually, from a terminal, after you've exported a model
(see the "Snapdragon / Qualcomm AI Hub" section in README.md).

Prerequisites (you do these yourself — they need your own AI Hub account):
    pip install qai-hub
    qai-hub configure --api_token <YOUR_API_TOKEN>   # from aihub.qualcomm.com

Usage:
    python scripts/aihub_profile_easyocr.py path/to/easyocr_detector.onnx
"""

import sys

import qai_hub as hub

DEVICE_NAME = "Snapdragon X Elite CRD"


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} path/to/model.onnx")
        raise SystemExit(1)

    model_path = sys.argv[1]
    device = hub.Device(DEVICE_NAME)

    print(f"[NXTSight/AI Hub] Submitting profile job for '{model_path}' on {DEVICE_NAME} ...")
    job = hub.submit_profile_job(
        model=model_path,
        device=device,
        name="NXTSight EasyOCR detector",
        # This qai_hub_models release targets an older QAIRT version than AI
        # Hub currently serves; without this override the job is rejected.
        options="--qairt_version=default",
    )

    print(f"[NXTSight/AI Hub] Job submitted: {job.url}")
    print("[NXTSight/AI Hub] Waiting for it to run on real, physical Snapdragon hardware in Qualcomm's cloud device farm...")
    job.wait()

    print(f"[NXTSight/AI Hub] Status: {job.get_status()}")
    profile = job.download_profile()
    print("[NXTSight/AI Hub] Profiling results (from real hardware, not a simulator):")
    print(profile)


if __name__ == "__main__":
    main()
