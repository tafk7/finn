# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conversion to KernelOps and the ordered inference, on test_design's Chain as a model.

qonnx's whole-graph passes ask every node at once, so they refuse a converted
graph whose KernelOps' inputs are not inferred yet; ``InferKernelTensors``
visits the nodes in order, and qonnx's passes agree after it.
"""

from __future__ import annotations

import numpy as np
import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

from finn.custom_op.kernels.base import TARGET_DSP, TARGET_PERIOD, KernelOpError, target
from finn.kernels.target import DspBlock
from finn.transformation.general import ApplyConfig
from finn.transformation.kernels import InferKernelTensors, ToKernelOps, kernel_choices_config
from kernel_ops.models import DOMAIN, chain_source
from kernels import test_design as chain


def converted(**options: bool) -> ModelWrapper:
    """The Chain, its source shapes inferred (qonnx's passes run on the source), converted."""
    source = chain_source(**options).transform(InferShapes())
    return source.transform(ToKernelOps(DspBlock.DSP48E2, 5.0))


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
    assert target(model) == (DspBlock.DSP48E2, 5.0)
    assert {TARGET_DSP, TARGET_PERIOD} <= {item.key for item in model.graph.metadata_props}
    # A model that imports the domain keeps its version: inserting never raises it.
    again = chain_source().transform(InferShapes())
    again.set_opset_import(DOMAIN, 2)
    assert again.transform(ToKernelOps(DspBlock.DSP48E2, 5.0)).get_opset_imports()[DOMAIN] == 2


def test_a_multithreshold_over_another_axis_is_left_alone() -> None:
    source = chain_source()  # hidden's shape is not known: its channel axis neither
    nodes = source.transform(ToKernelOps(DspBlock.DSP48E2, 5.0)).graph.node
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
