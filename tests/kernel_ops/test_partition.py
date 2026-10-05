# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The partition root: test_design's Chain built from KernelOp nodes.

Each node's choices are saved on it; the root's adapter memories, open (several
viable), are the flow's to choose and are saved on their consumers (D8); each
edge's adapter is forced. The
rebuilt root is test_design's Chain, configured the same way: the same flat
netlist and pins, and in XSim what ``execute_onnx`` computes on the source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from kernels import test_design as chain
from kernels.helpers import ADAPTER_RAM_STYLES, labels
from kernels.xsim import pack, requires_xsim, stream_through
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes

from finn.custom_op.kernels.base import KernelOpError
from finn.custom_op.kernels.partition import PartitionRoot, partition_root, save_partition_choices
from finn.custom_op.kernels.roots import StreamedMatMulNode
from finn.kernels.configure import commit, undecided
from finn.transformation.kernels import InferKernelTensors, ToKernelOps
from kernel_ops.models import INT3, TARGET, chain_source, lift, matmul_model

MATMUL = {
    "compute": "packed",
    "compute.packed.pe": chain.PE,
    "compute.packed.simd": chain.SIMD,
    "compute.packed.compute_pumping": False,
    "w.source.memstream.ram_style": "auto",
    "w.source.memstream.pumped_memory": False,
    "w.transport": "direct",
}
THRESHOLDING = {
    "pe": chain.PE,
    "use_axilite": False,
    "deep_pipeline": False,
    "ram_style": "auto",
    "ultra_stages": 0,
}


def kernel_model(**options: bool) -> ModelWrapper:
    """The Chain as KernelOps, each node's choices saved as test_design configures them."""
    model = (
        chain_source(**options)
        .transform(InferShapes())
        .transform(ToKernelOps(TARGET))
        .transform(InferKernelTensors())
    )
    for node in model.graph.node:
        choices = MATMUL if node.op_type == "MatMul" else THRESHOLDING
        if node.op_type == "MatMul" and model.get_initializer(node.input[1]) is None:
            # Streamed weights: no value, so no source; the weight edge's transport is
            # the root's.
            choices = {k: v for k, v in choices.items() if not k.startswith("w.source.")}
        model.get_customop_wrapper(node).save(choices)
    return model


def open_memories(root: PartitionRoot) -> tuple[Any, list[str]]:
    """The root's point and its open adapter memories."""
    return root.point, undecided(root.point, ADAPTER_RAM_STYLES)


def configured(model: ModelWrapper) -> tuple[PartitionRoot, Any]:
    """The root, its open adapter memories chosen and saved on their owners, rebuilt."""
    root = partition_root(model, model.graph.node, name="chain")
    _, styles = open_memories(root)
    save_partition_choices(model, root, dict.fromkeys(styles, "auto"))
    root = partition_root(model, model.graph.node, name="chain")
    point, open_styles = open_memories(root)
    assert open_styles == [] and root.dropped == ()
    return root, point


def test_the_root_of_the_chains_nodes_is_test_designs_chain() -> None:
    model = kernel_model()
    root, point = configured(model)
    reference = chain.chain()
    assert labels(point.module) == labels(reference.module)
    assert point.module.fragment == reference.module.fragment
    assert point.module.pins == reference.module.pins
    assert root.boundary == (("x", "s_axis_0"), ("y", "m_axis_0"))


def test_edge_choices_persist_on_their_consumers() -> None:
    model = kernel_model()
    configured(model)
    ops = {node.name: model.get_customop_wrapper(node) for node in model.graph.node}
    assert "x.adapter.input_gen.input_gen.ram_style" in ops["first"].choices()
    assert "x.adapter.input_gen.input_gen.ram_style" in ops["second"].choices()  # levels


def test_a_stale_edge_choice_is_dropped_and_the_forced_adapter_applies() -> None:
    model = kernel_model()
    configured(model)
    second = model.get_customop_wrapper(model.graph.node[2])
    second.save({"compute.packed.simd": 4})
    root = partition_root(model, model.graph.node, name="chain")
    # The levels edge now converts widths: its input_gen memory no longer applies.
    assert root.dropped == ("levels.adapter.input_gen.input_gen.ram_style",)
    point, _ = open_memories(root)
    assert point.levels.query(type(point.levels).adapter).value.startswith("vpc")


def test_a_lifted_initializers_source_choices_are_stale_in_the_partition() -> None:
    """Weights lifted to a graph input leave the node streamed: its weight stream has no
    value, so no source, and the source's choices, now the weight edge's, are the
    partition's to replay: it drops them as stale."""
    model = matmul_model()
    stored = model.get_customop_wrapper(model.graph.node[0])
    stored.save({"compute.packed.pe": 2, "w.source.memstream.ram_style": "block"})
    lift(model, "w")
    model.set_tensor_datatype("w", INT3)
    streamed = model.get_customop_wrapper(model.graph.node[0])
    assert streamed.facts().root is StreamedMatMulNode
    # The node replays its own choices only: the weight edge's are the partition's.
    assert streamed.point().matmul.compute.pe == 2 and streamed.verify_node() == []
    model = model.transform(InferKernelTensors())
    assert partition_root(model, model.graph.node).dropped == ("w.source.memstream.ram_style",)


def test_a_partition_has_ports_for_its_onnx_inputs_and_outputs_only() -> None:
    model = kernel_model()
    front = partition_root(model, model.graph.node[:2], name="front")
    assert front.boundary == (("x", "s_axis_0"), ("levels", "m_axis_0"))
    point, styles = open_memories(front)
    point = commit(point, dict.fromkeys(styles, "auto"))
    assert sorted(port.name for port in point.module.pins.ports) == [
        "ap_clk",
        "ap_rst_n",
        "m_axis_0",
        "s_axis_0",
    ]


def test_streamed_weights_are_a_boundary_of_the_partition() -> None:
    model = kernel_model(second_weights=False)
    root = partition_root(model, model.graph.node, name="chain")
    assert root.boundary == (("x", "s_axis_0"), ("w2", "s_axis_1"), ("y", "m_axis_0"))
    assert root.owners["w2"] == ("second", "w.")


def test_the_owner_map() -> None:
    root = partition_root(kernel_model(), kernel_model().graph.node, name="chain")
    assert dict(root.owners) == {
        "first": ("first", ""),
        "x": ("first", "x."),
        "w1": ("first", "w."),
        "activate": ("activate", ""),
        "hidden": ("activate", "x."),
        "second": ("second", ""),
        "levels": ("second", "x."),
        "w2": ("second", "w."),
    }


def test_a_choice_in_the_root_persists_on_the_weight_streams_owner() -> None:
    model = kernel_model()
    root = partition_root(model, model.graph.node, name="chain")
    written = save_partition_choices(model, root, {"w2.source.memstream.ram_style": "block"})
    assert written == {"second": {"w.source.memstream.ram_style": "block"}}
    second = model.get_customop_wrapper(model.graph.node[2])
    assert second.choices()["w.source.memstream.ram_style"] == "block"
    rebuilt = partition_root(model, model.graph.node, name="chain")
    assert rebuilt.point.w2.source.ram_style == "block" and rebuilt.dropped == ()


def test_a_node_named_like_a_tensor_is_refused() -> None:
    model = kernel_model()
    model.graph.node[1].name = "hidden"
    with pytest.raises(KernelOpError, match="a node and a tensor are both named hidden"):
        partition_root(model, model.graph.node)


@requires_xsim
def test_the_partition_computes_what_onnx_computes(tmp_path: Path) -> None:
    model = kernel_model()
    _, point = configured(model)
    x = np.array(chain.X, dtype=np.float32)
    y = execute_onnx(chain_source().transform(InferShapes()), {"x": x})["y"].astype(int)
    a_bits, y_bits = chain.A.bitwidth(), chain.Y.bitwidth()
    stream_through(
        point.module,
        tmp_path,
        inputs={
            "s_axis_0": (
                [
                    pack(chain.X[r][f : f + chain.SIMD], a_bits)
                    for r in range(chain.ROWS)
                    for f in range(0, chain.INPUTS, chain.SIMD)
                ],
                chain.SIMD * a_bits,
            )
        },
        outputs={
            "m_axis_0": (
                [
                    pack(y[r][f : f + chain.PE].tolist(), y_bits)
                    for r in range(chain.ROWS)
                    for f in range(0, chain.OUTPUTS, chain.PE)
                ],
                chain.PE * y_bits,
            )
        },
    )
