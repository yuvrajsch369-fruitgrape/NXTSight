import numpy as np
import onnx
from onnx import TensorProto, helper

from src.pipeline.runtime import create_inference_session, select_execution_providers


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
