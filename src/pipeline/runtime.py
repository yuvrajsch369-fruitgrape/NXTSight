"""Execution-provider selection for on-device ONNX inference.

On a Snapdragon Windows PC with `onnxruntime-qnn` installed, ONNX Runtime
exposes a QNNExecutionProvider that runs a compiled model on the Hexagon
NPU. On any other machine (this dev Mac included) that provider simply
isn't there, so we fall back to ONNX Runtime's default CPU provider.

The check happens once, at session-creation time, and is logged in plain
language so it's obvious during a demo which path is actually running —
no manual flag to flip, no crash if the NPU provider is missing.
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

# The Hexagon-NPU backend for QNN. ("QnnCpu.dll"/".so" also exists, but that
# would silently run on CPU while claiming the NPU provider — not what we want.)
_QNN_BACKEND_PATH = "QnnHtp.dll" if platform.system() == "Windows" else "libQnnHtp.so"


def select_execution_providers():
    """Decide which ONNX Runtime providers to use on this machine.

    Returns (providers, description). `providers` is ready to pass straight
    to onnxruntime.InferenceSession(..., providers=providers).
    """
    available = ort.get_available_providers()

    if QNN_PROVIDER in available:
        providers = [
            (QNN_PROVIDER, {"backend_path": _QNN_BACKEND_PATH}),
            "CPUExecutionProvider",
        ]
        description = "Snapdragon NPU (ONNX Runtime QNN execution provider)"
    else:
        providers = ["CPUExecutionProvider"]
        description = "CPU (QNN execution provider not available on this machine)"

    logger.info("Execution path: %s", description)
    logger.info("ONNX Runtime providers available here: %s", ", ".join(available))
    return providers, description


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
