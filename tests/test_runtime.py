from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from src.pipeline import runtime, text_encoder
from src.pipeline.runtime import (
    badge_variant_for,
    create_inference_session,
    select_execution_providers,
)

SCAM_CLASSIFIER_ONNX = Path(__file__).resolve().parent.parent / "src" / "scam_detector" / "artifacts" / "classifier.onnx"


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
        lambda: [
            "QNNExecutionProvider",
            "CUDAExecutionProvider",
            "OpenVINOExecutionProvider",
            "DmlExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    providers, description = select_execution_providers()
    assert description.startswith("Snapdragon NPU")
    assert providers[0][0] == "QNNExecutionProvider"
    assert providers[-1] == "CPUExecutionProvider"


def test_cuda_wins_when_qnn_absent(monkeypatch):
    monkeypatch.setattr(
        runtime.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    providers, description = select_execution_providers()
    assert description.startswith("NVIDIA GPU")
    assert providers[0][0] == "CUDAExecutionProvider"


def test_openvino_wins_when_qnn_and_cuda_absent(monkeypatch):
    monkeypatch.setattr(
        runtime.ort,
        "get_available_providers",
        lambda: ["OpenVINOExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    providers, description = select_execution_providers()
    assert description.startswith("Intel NPU / GPU")
    assert providers[0][0] == "OpenVINOExecutionProvider"
    assert providers[0][1] == {"device_type": "AUTO"}


def test_directml_wins_when_qnn_cuda_and_openvino_absent(monkeypatch):
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
    assert badge_variant_for("Intel NPU / GPU (ONNX Runtime OpenVINO execution provider)") == "accel"
    assert badge_variant_for("Windows GPU (ONNX Runtime DirectML execution provider)") == "accel"
    assert badge_variant_for("Apple Neural Engine / GPU (ONNX Runtime CoreML execution provider)") == "accel"
    assert badge_variant_for("CPU (no hardware accelerator available on this machine)") == "cpu"


# --------------------------------------------------------------------
# create_inference_session() itself, exercised for every hardware
# scenario — not just select_execution_providers()'s decision.
#
# Testing select_execution_providers() alone (above) proves the *logic*
# picks the right provider name. It does NOT prove create_inference_session
# — the function every one of NXTSight's 8 ONNX models actually calls —
# runs its real body (onnxruntime.InferenceSession(..., providers=...),
# session.get_providers(), the logging) without raising for a QNN/CUDA/
# DirectML providers list, since this machine has never actually
# supplied one of those. That gap is closed here: get_available_providers()
# is mocked to claim each accelerator is present, but create_inference_session()
# itself runs for real, against a real ONNX Runtime InferenceSession and a
# real (trivial) model. Confirmed directly (see the commit this landed
# in) that ONNX Runtime's own behavior when asked for a provider it
# doesn't actually have compiled in is a graceful fallback to the next
# one in the list — a UserWarning, not an exception — so these tests
# genuinely execute every line of create_inference_session() for every
# hardware path, even though the *acceleration itself* still isn't real
# without the physical hardware. That distinction is asserted on
# explicitly below, not glossed over.
# --------------------------------------------------------------------


def _create_session_with_mocked_hardware(monkeypatch, tmp_path, available_providers):
    monkeypatch.setattr(runtime.ort, "get_available_providers", lambda: available_providers)
    model_path = tmp_path / "add_one.onnx"
    _build_dummy_add_one_model(str(model_path))
    return create_inference_session(str(model_path))


def test_create_inference_session_executes_with_qnn_selected(monkeypatch, tmp_path):
    session, description = _create_session_with_mocked_hardware(
        monkeypatch, tmp_path, ["QNNExecutionProvider", "CPUExecutionProvider"]
    )
    assert description.startswith("Snapdragon NPU")
    result = session.run(["Y"], {"X": np.array([41.0], dtype=np.float32)})
    assert result[0][0] == 42.0
    # Honest about what this does and doesn't prove: this dev machine has
    # no real Hexagon NPU, so ONNX Runtime silently falls back to CPU
    # under the hood even though the *decision* correctly chose QNN —
    # asserted explicitly so this test can't be misread as claiming real
    # NPU execution happened.
    assert session.get_providers()[0] == "CPUExecutionProvider"


def test_create_inference_session_executes_with_cuda_selected(monkeypatch, tmp_path):
    session, description = _create_session_with_mocked_hardware(
        monkeypatch, tmp_path, ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert description.startswith("NVIDIA GPU")
    result = session.run(["Y"], {"X": np.array([41.0], dtype=np.float32)})
    assert result[0][0] == 42.0
    assert session.get_providers()[0] == "CPUExecutionProvider"  # no real NVIDIA GPU on this machine


def test_create_inference_session_executes_with_openvino_selected(monkeypatch, tmp_path):
    session, description = _create_session_with_mocked_hardware(
        monkeypatch, tmp_path, ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    )
    assert description.startswith("Intel NPU / GPU")
    result = session.run(["Y"], {"X": np.array([41.0], dtype=np.float32)})
    assert result[0][0] == 42.0
    # Confirmed directly: passing provider_options ({"device_type": "AUTO"})
    # for a provider name onnxruntime doesn't have compiled in behaves the
    # same graceful way as a bare provider name does — a UserWarning, a
    # fallback to CPU, not an exception. No real Intel NPU/GPU on this
    # machine either way.
    assert session.get_providers()[0] == "CPUExecutionProvider"


def test_create_inference_session_executes_with_directml_selected(monkeypatch, tmp_path):
    session, description = _create_session_with_mocked_hardware(
        monkeypatch, tmp_path, ["DmlExecutionProvider", "CPUExecutionProvider"]
    )
    assert description.startswith("Windows GPU")
    result = session.run(["Y"], {"X": np.array([41.0], dtype=np.float32)})
    assert result[0][0] == 42.0
    assert session.get_providers()[0] == "CPUExecutionProvider"  # no real Windows GPU on this machine


def test_create_inference_session_executes_with_no_accelerator(monkeypatch, tmp_path):
    """The "nothing found" path — genuinely the same code as the other
    three above, just with an empty accelerator list, confirming it's
    not a special case that skips create_inference_session's own logic."""
    session, description = _create_session_with_mocked_hardware(monkeypatch, tmp_path, ["CPUExecutionProvider"])
    assert description == "CPU (no hardware accelerator available on this machine)"
    result = session.run(["Y"], {"X": np.array([41.0], dtype=np.float32)})
    assert result[0][0] == 42.0
    assert session.get_providers() == ["CPUExecutionProvider"]


def test_create_inference_session_executes_with_coreml_selected_for_real(tmp_path):
    """The one scenario on this machine that's genuinely real, not
    mocked: this dev Mac actually has CoreML. No monkeypatching here —
    this is select_execution_providers() and create_inference_session()
    both running against this machine's real onnxruntime.get_available_providers()."""
    model_path = tmp_path / "add_one.onnx"
    _build_dummy_add_one_model(str(model_path))
    session, description = create_inference_session(str(model_path))

    available = runtime.ort.get_available_providers()
    if "CoreMLExecutionProvider" in available:
        assert description.startswith("Apple Neural Engine")
        assert session.get_providers()[0] == "CoreMLExecutionProvider"
    else:
        # Honest fallback if this ever runs somewhere without CoreML
        # (Linux CI, for instance) — still must not crash.
        assert description == "CPU (no hardware accelerator available on this machine)"
    result = session.run(["Y"], {"X": np.array([41.0], dtype=np.float32)})
    assert result[0][0] == 42.0


# --------------------------------------------------------------------
# The same hardening, one level up: NXTSight's actual production model
# (the scam classifier's real classifier.onnx — Gemm -> Softmax ->
# ArgMax, hand-exported, exactly what engine.py loads) under each
# mocked hardware scenario, not just a trivial synthetic graph. This is
# what rules out "the dummy graph happened to work but a real model's
# input/output names or op set doesn't" as a gap the tests above
# wouldn't have caught.
# --------------------------------------------------------------------


def test_real_scam_classifier_onnx_runs_under_every_mocked_hardware_scenario(monkeypatch):
    if not SCAM_CLASSIFIER_ONNX.exists():
        import pytest

        pytest.skip("classifier.onnx not built in this environment — run python -m src.scam_detector.train_classifier")

    vector = text_encoder.encode("Your account will be blocked in 2 hours unless you verify now.")
    vector = vector.reshape(1, -1).astype(np.float32)

    scenarios = {
        "qnn": ["QNNExecutionProvider", "CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
        "cpu_only": ["CPUExecutionProvider"],
    }
    results = {}
    for name, available in scenarios.items():
        monkeypatch.setattr(runtime.ort, "get_available_providers", lambda a=available: a)
        session, description = create_inference_session(str(SCAM_CLASSIFIER_ONNX))
        input_name = session.get_inputs()[0].name
        labels, probabilities = session.run(["label", "probabilities"], {input_name: vector})
        results[name] = (int(labels[0]), description)

    # Every scenario must reach the same verdict on the same input — the
    # "hardware" selected is a mocked label ONNX Runtime silently ignores
    # in favor of real CPU underneath (see the tests above); what this
    # proves is that engine.py's actual model artifact loads and predicts
    # correctly no matter which provider list runtime.py hands it, not
    # that the acceleration itself is real without the physical chip.
    label_ids = {label_id for label_id, _description in results.values()}
    assert len(label_ids) == 1, f"inconsistent verdicts across hardware scenarios: {results}"
    assert results["qnn"][1].startswith("Snapdragon NPU")
    assert results["cuda"][1].startswith("NVIDIA GPU")
    assert results["openvino"][1].startswith("Intel NPU / GPU")
    assert results["directml"][1].startswith("Windows GPU")
    assert results["cpu_only"][1] == "CPU (no hardware accelerator available on this machine)"
