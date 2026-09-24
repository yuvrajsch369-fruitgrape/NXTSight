"""Execution-provider selection for on-device ONNX inference.

A small hardware-abstraction layer over ONNX Runtime's own execution
providers — not something hand-rolled per vendor. At session-creation
time we ask ONNX Runtime what's actually available on *this* machine
(`onnxruntime.get_available_providers()`) and pick the best real
accelerator it reports, in this priority order:

    1. QNN       — Snapdragon Hexagon NPU (what this app is built for)
    2. CUDA       — NVIDIA GPU
    3. DirectML   — any GPU on Windows (AMD/Intel/NVIDIA, via DirectX 12)
    4. CoreML     — Apple Neural Engine / GPU on macOS
    5. CPU        — always available, the universal fallback

Every one of those five is a first-class execution provider ONNX
Runtime itself ships and maintains (QNN and DirectML from Microsoft/
Qualcomm, CUDA from Microsoft against NVIDIA's stack, CoreML from
Microsoft against Apple's) — this module doesn't talk to any vendor
SDK directly, it just asks ONNX Runtime which of its own providers
loaded successfully and prefers the fastest one that did.

The check happens once, at session-creation time, and is logged in
plain language so it's obvious during a demo which path is actually
running — no manual flag to flip, no crash if a given provider isn't
there. Genuinely verified on this dev machine: only CoreML and CPU are
installed here (`onnxruntime`'s macOS wheel ships both), so those two
paths are exercised for real every time this file's tests run; QNN,
CUDA, and DirectML are exercised with the real selection *logic*
(tests/test_runtime.py mocks `get_available_providers()` to simulate
each), not on real hardware — this codebase doesn't have a Snapdragon,
NVIDIA, or Windows-GPU machine to test on directly.
"""

import logging
import platform

import onnxruntime as ort

logger = logging.getLogger("nxtsight")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[NXTSight] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

QNN_PROVIDER = "QNNExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
DIRECTML_PROVIDER = "DmlExecutionProvider"
COREML_PROVIDER = "CoreMLExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"

# The Hexagon-NPU backend for QNN. ("QnnCpu.dll"/".so" also exists, but that
# would silently run on CPU while claiming the NPU provider — not what we want.)
_QNN_BACKEND_PATH = "QnnHtp.dll" if platform.system() == "Windows" else "libQnnHtp.so"

# Priority order: Qualcomm's own NPU first — a real, verified hardware
# target this project has measured performance on — then whichever other
# real accelerator ONNX Runtime reports as available, CPU last as the
# universal fallback.
# (provider_name, description, badge_variant, extra_provider_options)
_PROVIDER_PRIORITY = (
    (QNN_PROVIDER, "Snapdragon NPU (ONNX Runtime QNN execution provider)", "npu", {"backend_path": _QNN_BACKEND_PATH}),
    (CUDA_PROVIDER, "NVIDIA GPU (ONNX Runtime CUDA execution provider)", "accel", {}),
    (DIRECTML_PROVIDER, "Windows GPU (ONNX Runtime DirectML execution provider)", "accel", {}),
    (COREML_PROVIDER, "Apple Neural Engine / GPU (ONNX Runtime CoreML execution provider)", "accel", {}),
)


def select_execution_providers():
    """Decide which ONNX Runtime providers to use on this machine.

    Returns (providers, description). `providers` is ready to pass straight
    to onnxruntime.InferenceSession(..., providers=providers). CPU is always
    appended as the last entry so a provider that's *reported* available but
    fails to actually initialize (a real, observed ONNX Runtime behavior)
    still has somewhere safe to land instead of crashing the session.
    """
    available = ort.get_available_providers()

    for provider_name, description, _badge_variant, extra_options in _PROVIDER_PRIORITY:
        if provider_name in available:
            providers = [(provider_name, extra_options), CPU_PROVIDER]
            break
    else:
        providers = [CPU_PROVIDER]
        description = "CPU (no hardware accelerator available on this machine)"

    logger.info("Execution path: %s", description)
    logger.info("ONNX Runtime providers available here: %s", ", ".join(available))
    return providers, description


def badge_variant_for(description: str) -> str:
    """Classify an execution-path description into a UI badge style:
    "npu" for the Snapdragon path this app is built for, "accel" for any
    other real hardware accelerator, "cpu" for the plain fallback."""
    for _provider_name, known_description, variant, _extra_options in _PROVIDER_PRIORITY:
        if description == known_description:
            return variant
    return "cpu"


def create_inference_session(model_path: str):
    """Create an ONNX Runtime session using the best provider this machine has.

    Returns (session, description) — description is the same plain-language
    string that gets logged, handy for showing in a UI during a demo.
    """
    providers, description = select_execution_providers()
    session = ort.InferenceSession(model_path, providers=providers)

    active = session.get_providers()[0]
    logger.info("Session ready on '%s' — active provider: %s", model_path, active)
    return session, description
