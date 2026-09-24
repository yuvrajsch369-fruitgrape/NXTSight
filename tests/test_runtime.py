import numpy as np
import onnx
from onnx import TensorProto, helper

from src.pipeline import runtime
from src.pipeline.runtime import (
    badge_variant_for,
    create_inference_session,
    select_execution_providers,
)


def _build_dummy_add_one_model(path):
    """A trivial Y = X + 1 ONNX graph — enough to prove a session actually runs."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])
    node = helper.make_node("Add", ["X", "one"], ["Y"])
    graph = helper.make_graph([node], "add_one", [x], [y], initializer=[one])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10  # match the IR version this onnxruntime build supports
    onnx.save(model, path)


def test_select_execution_providers_never_empty():
    providers, description = select_execution_providers()
    assert providers
    assert "CPUExecutionProvider" in str(providers)
    assert description


def test_create_inference_session_runs_end_to_end(tmp_path):
    model_path = tmp_path / "add_one.onnx"
    _build_dummy_add_one_model(str(model_path))

    session, description = create_inference_session(str(model_path))
    result = session.run(["Y"], {"X": np.array([41.0], dtype=np.float32)})

    assert result[0][0] == 42.0
    assert isinstance(description, str) and description


# --------------------------------------------------------------------
# Hardware-abstraction priority: QNN > CUDA > DirectML > CoreML > CPU.
# This dev machine only ever reports CoreML/CPU for real (verified by
# test_select_execution_providers_never_empty above, on real ONNX
# Runtime output) — the other three provider paths are exercised here
# by mocking get_available_providers() rather than on real hardware,
# since this codebase has no Snapdragon/NVIDIA/Windows-GPU machine to
# test on. Honest about that, not pretending it's the same as a real
# hardware check.
# --------------------------------------------------------------------


def test_qnn_wins_over_every_other_accelerator(monkeypatch):
    monkeypatch.setattr(
        runtime.ort,
        "get_available_providers",
        lambda: ["QNNExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    providers, description = select_execution_providers()
    assert description.startswith("Snapdragon NPU")
    assert providers[0][0] == "QNNExecutionProvider"
    assert providers[-1] == "CPUExecutionProvider"


def test_cuda_wins_when_qnn_absent(monkeypatch):
    monkeypatch.setattr(
        runtime.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    providers, description = select_execution_providers()
    assert description.startswith("NVIDIA GPU")
    assert providers[0][0] == "CUDAExecutionProvider"


def test_directml_wins_when_qnn_and_cuda_absent(monkeypatch):
    monkeypatch.setattr(
        runtime.ort,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    providers, description = select_execution_providers()
    assert description.startswith("Windows GPU")
    assert providers[0][0] == "DmlExecutionProvider"


def test_coreml_wins_when_nothing_else_available(monkeypatch):
    monkeypatch.setattr(
        runtime.ort,
        "get_available_providers",
        lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    providers, description = select_execution_providers()
    assert description.startswith("Apple Neural Engine")
    assert providers[0][0] == "CoreMLExecutionProvider"


def test_cpu_only_fallback_when_no_accelerator_present(monkeypatch):
    monkeypatch.setattr(runtime.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    providers, description = select_execution_providers()
    assert providers == ["CPUExecutionProvider"]
    assert description == "CPU (no hardware accelerator available on this machine)"


def test_badge_variant_classification():
    assert badge_variant_for("Snapdragon NPU (ONNX Runtime QNN execution provider)") == "npu"
    assert badge_variant_for("NVIDIA GPU (ONNX Runtime CUDA execution provider)") == "accel"
    assert badge_variant_for("Windows GPU (ONNX Runtime DirectML execution provider)") == "accel"
    assert badge_variant_for("Apple Neural Engine / GPU (ONNX Runtime CoreML execution provider)") == "accel"
    assert badge_variant_for("CPU (no hardware accelerator available on this machine)") == "cpu"
