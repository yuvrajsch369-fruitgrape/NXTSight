"""Hand-built ONNX export for our TF-IDF + Logistic Regression classifiers.

skl2onnx's default LogisticRegression converter emits an `ai.onnx.ml.LinearClassifier`
node. That's ONNX Runtime's own extension domain for classical ML ops — Qualcomm AI
Hub's compiler rejects it outright ("Domain name 'ai.onnx.ml' for node
'LinearClassifier' in input model is not supported by Qualcomm AI Hub Workbench"),
since an NPU/DSP compiler backend has no idea how to lower an opaque "run sklearn
logistic regression" op, only genuine tensor ops.

So this builds the exact same linear model directly out of core `ai.onnx` ops —
Gemm -> Softmax -> ArgMax — which any ONNX consumer, AI Hub included, understands.

This reproduces sklearn's LogisticRegression.predict_proba() exactly, not
approximately: for binary classification sklearn stores only one coefficient row
(the positive class's weights, implicitly relative to an all-zero baseline for the
negative class) — modeled here as a genuine two-row weight matrix so a plain softmax
over [0, z] reproduces the same sigmoid probabilities [sigmoid(-z), sigmoid(z)]
bit-for-bit (up to float32 rounding).
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def export_logistic_regression(classifier, n_features: int, model_name: str = "classifier") -> onnx.ModelProto:
    """Build a minimal ONNX graph for a fitted sklearn LogisticRegression.

    Inputs: "input", float32 [1, n_features] (the engine always runs one
    text at a time — see engine.py's vectorizer.transform([cleaned])).
    Outputs: "label" (int64 [1]) and "probabilities" (float32 [1, n_classes]),
    the same names and shapes engine.py already expects from the old
    skl2onnx-produced graph.
    """
    coef = classifier.coef_.astype(np.float32)
    intercept = classifier.intercept_.astype(np.float32)

    if coef.shape[0] == 1:
        # Binary case: sklearn stores one row for the positive class only,
        # implicitly relative to an all-zero baseline for the negative
        # class — make that baseline explicit as a real zero row.
        weight = np.vstack([np.zeros_like(coef[0]), coef[0]])
        bias = np.array([0.0, intercept[0]], dtype=np.float32)
    else:
        weight = coef
        bias = intercept

    n_classes = weight.shape[0]

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, n_features])
    label_tensor = helper.make_tensor_value_info("label", TensorProto.INT64, [1])
    proba_tensor = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, n_classes])

    weight_init = numpy_helper.from_array(weight, name="weight")
    bias_init = numpy_helper.from_array(bias, name="bias")

    # Gemm with transB=1 computes input @ weight.T + bias -> logits.
    gemm_node = helper.make_node("Gemm", ["input", "weight", "bias"], ["logits"], transB=1)
    softmax_node = helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1)
    argmax_node = helper.make_node("ArgMax", ["probabilities"], ["label"], axis=1, keepdims=0)

    graph = helper.make_graph(
        [gemm_node, softmax_node, argmax_node],
        model_name,
        [input_tensor],
        [label_tensor, proba_tensor],
        initializer=[weight_init, bias_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    # onnx.helper.make_model() defaults to whatever IR version the installed
    # `onnx` package itself was built with — which can be newer than what's
    # installed of onnxruntime supports (the exact issue requirements.txt's
    # onnx==1.19.1 pin already works around for the *package* version; this
    # is the same problem at the *model file* level). IR version 10 is opset
    # 13-compatible and loads on the onnxruntime version this project pins.
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model
