# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conversion to KernelOps and the ordered inference, on the Chain (``kernels.chain``) as a model.

qonnx's whole-graph passes ask every node at once, so they refuse a converted
graph whose KernelOps' inputs are not inferred yet; ``InferKernelTensors``
visits the nodes in order, and qonnx's passes agree after it.
"""

from __future__ import annotations

import numpy as np
import pytest
from kernels import chain
from onnx import TensorProto, helper, numpy_helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.kernels.base import PLATFORM_KEYS, KernelOpError, read_target
from finn.transformation.general import ApplyConfig
from finn.transformation.kernels import InferKernelTensors, ToKernelOps, kernel_choices_config
from kernel_ops.models import DOMAIN, TARGET, chain_source


def converted(**options: bool) -> ModelWrapper:
    """The Chain, its source shapes inferred (qonnx's passes run on the source), converted."""
    source = chain_source(**options).transform(InferShapes())
    return source.transform(ToKernelOps(TARGET))


def inferred(**options: bool) -> ModelWrapper:
    return converted(**options).transform(InferKernelTensors())


def tensors(model: ModelWrapper) -> dict[str, tuple[object, str]]:
    return {
        name: (model.get_tensor_shape(name), model.get_tensor_datatype(name).name)
        for name in ("hidden", "levels", "y")
    }


def test_conversion_rewrites_the_nodes_and_states_the_target() -> None:
    model = converted()
    assert [(node.op_type, node.domain, node.name) for node in model.graph.node] == [
        ("MatMul", DOMAIN, "first"),
        ("Thresholding", DOMAIN, "activate"),
        ("MatMul", DOMAIN, "second"),
    ]
    assert model.get_customop_wrapper(model.graph.node[1]).get_nodeattr("bias") == 0
    assert model.get_opset_imports()[DOMAIN] == 1
    assert read_target(model) == TARGET
    assert {key.entry for key in PLATFORM_KEYS.values()} <= {
        item.key for item in model.graph.metadata_props
    }
    # A model that imports the domain keeps its version: inserting never raises it.
    again = chain_source().transform(InferShapes())
    again.set_opset_import(DOMAIN, 2)
    assert again.transform(ToKernelOps(TARGET)).get_opset_imports()[DOMAIN] == 2


def test_a_multithreshold_over_another_axis_is_left_alone() -> None:
    source = chain_source()  # hidden's shape is not known: its channel axis neither
    nodes = source.transform(ToKernelOps(TARGET)).graph.node
    assert [node.op_type for node in nodes] == ["MatMul", "MultiThreshold", "MatMul"]


def test_qonnx_inference_refuses_before_the_ordered_pass() -> None:
    with pytest.raises(
        KernelOpError, match="hidden has no datatype annotation.*InferKernelTensors"
    ):
        converted().transform(InferShapes())


def test_the_ordered_pass_states_the_chains_types_and_qonnx_agrees() -> None:
    model = inferred()
    assert tensors(model) == {
        "hidden": ([chain.ROWS, chain.HIDDEN], chain.H.name),
        "levels": ([chain.ROWS, chain.HIDDEN], chain.T.name),
        "y": ([chain.ROWS, chain.OUTPUTS], chain.Y.name),
    }
    assert (chain.H.name, chain.T.name, chain.Y.name) == ("INT8", "UINT2", "INT7")
    after = model.transform(InferShapes()).transform(InferDataTypes())
    assert tensors(after) == tensors(model)


def test_an_unannotated_input_and_a_narrower_annotation_are_refused() -> None:
    with pytest.raises(KernelOpError, match="first: x has no datatype annotation"):
        converted(annotate_input=False).transform(InferKernelTensors())
    narrow = converted()
    narrow.set_tensor_datatype("hidden", chain.T)
    with pytest.raises(KernelOpError, match="hidden is annotated UINT2, narrower than INT8"):
        narrow.transform(InferKernelTensors())
    # A wider statement is replaced by the exact type.
    wide = converted()
    wide.set_tensor_datatype("hidden", DataType["INT16"])
    assert tensors(wide.transform(InferKernelTensors()))["hidden"][1] == "INT8"


@pytest.mark.parametrize("second_weights", (True, False))
def test_the_converted_graph_computes_the_source_graphs_results(second_weights: bool) -> None:
    x = np.array(chain.X, dtype=np.float32)
    feed = {"x": x} if second_weights else {"x": x, "w2": np.array(chain.W2, dtype=np.float32)}
    expected = execute_onnx(
        chain_source(second_weights=second_weights).transform(InferShapes()), feed
    )
    produced = execute_onnx(inferred(second_weights=second_weights), feed)
    assert np.array_equal(produced["y"], expected["y"])


# ApplyConfig warns of a node without an entry: here second, which has no choices.
@pytest.mark.filterwarnings("ignore:\\nNo HW configuration for nodes")
def test_the_choices_round_trip_through_apply_config() -> None:
    model = inferred()
    first = model.get_customop_wrapper(model.graph.node[0])
    first.save({"compute.packed.pe": 2, "w.transport": "direct"})
    activate = model.get_customop_wrapper(model.graph.node[1])
    activate.save({"ram_style": "auto", "ultra_stages": 0})
    config = kernel_choices_config(model)
    assert config == {
        "first": {"compute.packed.pe": 2, "w.transport": "direct"},
        "activate": {"ram_style": "auto", "ultra_stages": 0},
    }
    applied = inferred().transform(ApplyConfig(config))
    assert kernel_choices_config(applied) == config


# -- the TFC path's input: a Reshape, then thresholds as streamlining leaves them ---------


def tfc_input(
    thresholds: object,
    *,
    dtype: str = "UINT8",
    shared: bool = False,
) -> ModelWrapper:
    """TFC's input as finn-dev's streamlining leaves it: x (1, 1, 4, 4) -> Reshape to
    (1, 16) by an INT64 shape initializer -> MultiThreshold with float thresholds,
    FLOAT32 (no statement). ``shared``: a second MultiThreshold reads the same table
    on another input type (INT4)."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)]
    nodes = [
        helper.make_node("Reshape", ["x", "shape"], ["flat"], name="flatten"),
        helper.make_node(
            "MultiThreshold",
            ["flat", "thresholds"],
            ["y"],
            name="quantize",
            domain="qonnx.custom_op.general",
            out_dtype="UINT2",
        ),
    ]
    inputs = [x]
    if shared:
        inputs.append(helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 16]))
        outputs.append(helper.make_tensor_value_info("v", TensorProto.FLOAT, None))
        nodes.append(
            helper.make_node(
                "MultiThreshold",
                ["z", "thresholds"],
                ["v"],
                name="again",
                domain="qonnx.custom_op.general",
                out_dtype="UINT2",
            )
        )
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(nodes, "tfc_input", inputs, outputs),
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("qonnx.custom_op.general", 1),
            ],
        )
    )
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([1, 16], dtype=np.int64), "shape")
    )
    model.set_initializer("thresholds", np.asarray(thresholds, dtype=np.float32))
    model.set_tensor_datatype("x", DataType[dtype])
    if shared:
        model.set_tensor_datatype("z", DataType["INT4"])
    return model.transform(InferShapes())


def through_the_kernel_path(source: ModelWrapper) -> ModelWrapper:
    return source.transform(ToKernelOps(TARGET)).transform(InferKernelTensors())


def test_a_reshape_keeps_the_shape_its_initializer_states() -> None:
    model = through_the_kernel_path(tfc_input([[63.75, 191.25]]))
    assert [node.op_type for node in model.graph.node] == ["Reshape", "Thresholding"]
    assert model.get_tensor_shape("flat") == [1, 16]
    assert (model.get_tensor_shape("y"), model.get_tensor_datatype("y").name) == ([1, 16], "UINT2")


def test_a_shape_the_inference_cannot_know_is_not_overwritten() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    s = helper.make_tensor_value_info("s", TensorProto.INT64, [2])
    flat = helper.make_tensor_value_info("flat", TensorProto.FLOAT, None)
    node = helper.make_node("Reshape", ["x", "s"], ["flat"], name="flatten")
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph([node], "dynamic", [x, s], [flat]),
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )
    model.set_tensor_shape("flat", [1, 16])
    assert model.transform(InferKernelTensors()).get_tensor_shape("flat") == [1, 16]


@pytest.mark.parametrize(
    "thresholds, dtype, table, annotation",
    [
        # Rounded up; a broadcast row (one for every channel) becomes 16 rows.
        ([[63.75, 191.25]], "UINT8", [64, 192], "UINT8"),
        # Clipped to max + 1 of the input: annotated by what it holds, not by the input.
        ([[0.5, 300.0]], "UINT8", [1, 256], "UINT9"),
        # A signed input: clipped to its minimum, a signed annotation.
        ([[-20.5, 3.2, 9.0]], "INT4", [-8, 4, 8], "INT5"),
    ],
)
def test_the_thresholds_become_integers_against_the_input_type(
    thresholds: list[list[float]], dtype: str, table: list[int], annotation: str
) -> None:
    source = tfc_input(thresholds, dtype=dtype)
    model = through_the_kernel_path(source)
    values = model.get_initializer("thresholds")
    assert values.shape == (16, len(table))
    assert (values == np.array(table, dtype=np.float32)).all()
    assert model.get_tensor_datatype("thresholds").name == annotation
    # Exact: every value of the input type, in both graphs.
    low, high = int(DataType[dtype].min()), int(DataType[dtype].max())
    for start in range(low, high + 1, 16):
        image = np.clip(np.arange(start, start + 16), low, high).reshape(1, 1, 4, 4)
        feed = {"x": image.astype(np.float32)}
        assert np.array_equal(execute_onnx(model, feed)["y"], execute_onnx(source, feed)["y"])


def test_a_shared_table_is_normalized_once_for_each_reader() -> None:
    model = through_the_kernel_path(tfc_input([[-20.5, 3.2, 200.0]], shared=True))
    nodes = {node.name: node for node in model.graph.node}
    quantize, again = nodes["quantize"], nodes["again"]
    assert quantize.input[1] != again.input[1]
    assert model.get_initializer(quantize.input[1])[0].tolist() == [0, 4, 200]
    assert model.get_tensor_datatype(quantize.input[1]).name == "UINT8"
    assert model.get_initializer(again.input[1])[0].tolist() == [-8, 4, 8]
    assert model.get_tensor_datatype(again.input[1]).name == "INT5"
