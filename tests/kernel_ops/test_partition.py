# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The partition root: the Chain (``kernels.chain``) built from KernelOp nodes.

Each node's choices are saved on it; the root's adapter memories, open (several
viable), are the flow's to choose and are saved on their consumers; each
edge's adapter is forced. The
rebuilt root is the Chain (``kernels.chain``), configured the same way: the same flat
netlist and pins, and in XSim what ``execute_onnx`` computes on the source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from kernels import chain
from kernels.helpers import labels
from kernels.xsim import pack, requires_xsim, stream_through
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes

from finn.core.space import inspection
from finn.custom_op.kernels.base import KernelOpError, kernel_op
from finn.custom_op.kernels.partition import partition_root, persist
from finn.kernels.configure import chosen, commit
from finn.transformation.kernels import InferKernelTensors
from kernel_ops.models import (
    INT3,
    chain_source,
    configure_partition,
    kernel_model,
    lift,
    matmul_model,
    open_memories,
)


def test_the_root_of_the_chains_nodes_is_the_chain() -> None:
    model = kernel_model()
    root, point = configure_partition(model)
    reference = chain.chain()
    assert labels(point.module) == labels(reference.module)
    assert point.module.fragment == reference.module.fragment
    assert point.module.abi == reference.module.abi
    assert root.boundary == (("x", "s_axis_0"), ("y", "m_axis_0"))


def test_a_partition_root_refuses_a_node_that_is_not_a_kernel_op() -> None:
    model = chain_source()
    with pytest.raises(KernelOpError, match="activate: MultiThreshold is not a KernelOp"):
        partition_root(model, model.graph.node[1:2])


def test_edge_choices_persist_on_their_consumers() -> None:
    model = kernel_model()
    configure_partition(model)
    ops = {node.name: kernel_op(model, node) for node in model.graph.node}
    assert "x.adapter.input_gen.input_gen.ram_style" in ops["first"].choices()
    assert "x.adapter.input_gen.input_gen.ram_style" in ops["second"].choices()  # levels


def test_a_refold_that_converts_widths_keeps_the_input_side_s_memory() -> None:
    model = kernel_model()
    configure_partition(model)
    second = kernel_op(model, model.graph.node[2])
    second.save({"compute.packed.simd": 4})
    root = partition_root(model, model.graph.node, name="chain")
    # The levels edge now converts widths before its transport; its replay after it is
    # the same input_gen, whose memory choice still applies.
    assert not root.dropped
    assert [stage.label for stage in root.point.levels.stages] == [
        "output_adapter.vpc.vpc",
        "adapter.input_gen.input_gen",
    ]


def test_a_stale_edge_choice_is_dropped_and_the_forced_adapter_applies() -> None:
    model = kernel_model()
    configure_partition(model)
    second = kernel_op(model, model.graph.node[2])
    # A memory of a chain the levels edge does not take.
    second.save({"x.adapter.input_gen_vpc.input_gen.ram_style": "auto"})
    root = partition_root(model, model.graph.node, name="chain")
    assert list(root.dropped) == ["levels.adapter.input_gen_vpc.input_gen.ram_style"]
    assert root.dropped["levels.adapter.input_gen_vpc.input_gen.ram_style"] == "inapplicable"
    point, _ = open_memories(root)
    assert point.levels.query(type(point.levels).adapter).value == "input_gen"


def test_a_lifted_initializers_source_choices_are_stale_in_the_partition() -> None:
    """Weights lifted to a graph input leave the node streamed: its weight channel has no
    value, so no source, and the source's choices, now the weight edge's, are the
    partition's to replay: it drops them as stale."""
    model = matmul_model()
    stored = kernel_op(model, model.graph.node[0])
    stored.save({"compute.packed.pe": 2, "w.source.memstream.ram_style": "block"})
    lift(model, "w")
    model.set_tensor_datatype("w", INT3)
    streamed = kernel_op(model, model.graph.node[0])
    assert streamed.facts().owned == ()
    # The output was inferred from the stored weights' columns; unknown, they give the
    # datatypes' wider range, so the graph's y is stale: the node refuses it until
    # inference, run again, states it from the node's new facts.
    (stale,) = streamed.verify_node()
    assert "matmul-tensor" in stale
    model = model.transform(InferKernelTensors())
    streamed = kernel_op(model, model.graph.node[0])
    # The node replays its own choices only: the weight edge's are the partition's.
    point: Any = streamed.point()
    assert point.matmul.compute.pe == 2 and streamed.verify_node() == []
    assert list(partition_root(model, model.graph.node).dropped) == ["w.source.memstream.ram_style"]


def test_a_partition_has_ports_for_its_onnx_inputs_and_outputs_only() -> None:
    model = kernel_model()
    front = partition_root(model, model.graph.node[:2], name="front")
    assert front.boundary == (("x", "s_axis_0"), ("levels", "m_axis_0"))
    point, styles = open_memories(front)
    point = commit(point, dict.fromkeys(styles, "auto"))
    assert sorted(port.name for port in point.module.abi.pins) == [
        "ap_clk",
        "ap_rst_n",
        "m_axis_0",
        "s_axis_0",
    ]


def test_an_edge_between_partitions_has_one_transport_its_consumers() -> None:
    """``levels`` leaves the front partition for a KernelOp of the back one:
    the back partition's input boundary, its transport its consumer's; the front pins it
    ``direct``, owns nothing of it, and drops a producer's transport set while the edge
    left the graph."""
    model = kernel_model()
    front = partition_root(model, model.graph.node[:2], name="front")
    back = partition_root(model, model.graph.node[2:], name="back")
    assert ("levels", "m_axis_0") in front.boundary and ("levels", "s_axis_0") in back.boundary
    point = front.point
    assert point.levels.query(type(point.levels).transport).value == "direct"
    assert "levels" not in front.owners and back.owners["levels"] == ("second", "x.")
    written = persist(model, back, commit(back.point, {"levels.transport": "fifo"}))
    assert written["second"]["x.transport"] == "fifo"

    kernel_op(model, model.graph.node[1]).save({"y.transport": "fifo"})
    assert list(partition_root(model, model.graph.node[:2], name="front").dropped) == [
        "levels.transport"
    ]


def test_streamed_weights_are_a_boundary_of_the_partition() -> None:
    model = kernel_model(second_weights=False)
    root = partition_root(model, model.graph.node, name="chain")
    assert root.boundary == (("x", "s_axis_0"), ("w2", "s_axis_1"), ("y", "m_axis_0"))
    assert root.owners["w2"] == ("second", "w.")
    # Every channel has a transport, the weight edge's as x's: each its consumer's.
    keys = {decision.key for decision in inspection.decisions(root.point)}
    assert {"w2.transport", "x.transport"} <= keys


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
        "y": ("second", "y."),
    }


def test_a_choice_in_the_root_persists_on_the_weight_streams_owner() -> None:
    model = kernel_model()
    root = partition_root(model, model.graph.node, name="chain")
    written = persist(model, root, commit(root.point, {"w2.source.memstream.ram_style": "block"}))
    assert written["second"]["w.source.memstream.ram_style"] == "block"
    second = kernel_op(model, model.graph.node[2])
    assert second.choices()["w.source.memstream.ram_style"] == "block"
    rebuilt = partition_root(model, model.graph.node, name="chain")
    assert rebuilt.point.w2.source.ram_style == "block" and not rebuilt.dropped


def test_a_node_named_like_a_tensor_is_refused() -> None:
    model = kernel_model()
    model.graph.node[1].name = "hidden"
    with pytest.raises(KernelOpError, match="a node and a tensor are both named hidden"):
        partition_root(model, model.graph.node)


def test_two_nodes_of_one_member_name_are_refused() -> None:
    model = kernel_model()
    model.graph.node[2].name = "first"
    with pytest.raises(KernelOpError, match="a node and another node are both named first"):
        partition_root(model, model.graph.node)


def test_a_kernel_choice_the_root_refuses_is_dropped_named_and_cleared_by_persist() -> None:
    """A choice written past ``save`` reaches the root's replay, which drops it as stale
    with why; the rest replays, and persisting the point clears the stale attribute."""
    model = kernel_model()
    first = kernel_op(model, model.graph.node[0])
    first.set_nodeattr("compute.packed.simd", 2)
    first.set_nodeattr("compute.packed.pe", 3)
    root = partition_root(model, model.graph.node, name="chain")
    assert list(root.dropped) == ["first.compute.packed.pe"]
    assert "domain-membership" in root.dropped["first.compute.packed.pe"]
    assert root.point.first.compute.simd == 2
    # Its pe is open again; once chosen, the node holds the new value, nothing stale.
    persist(model, root, commit(root.point, {"first.compute.packed.pe": 2}))
    assert first.choices()["compute.packed.pe"] == 2
    assert not partition_root(model, model.graph.node, name="chain").dropped


def test_persist_clears_what_the_point_does_not_commit() -> None:
    model = kernel_model()
    root = partition_root(model, model.graph.node, name="chain")
    second = kernel_op(model, model.graph.node[2])
    second.save({"x.adapter.input_gen_vpc.input_gen.ram_style": "auto"})
    stale = partition_root(model, model.graph.node, name="chain")
    assert list(stale.dropped) == ["levels.adapter.input_gen_vpc.input_gen.ram_style"]
    persist(model, stale, stale.point)
    assert "x.adapter.input_gen_vpc.input_gen.ram_style" not in second.choices()
    assert second.choices() == {
        key.replace("levels.", "x.", 1).replace("w2.", "w.", 1).replace("second.", "", 1): value
        for key, value in chosen(root.point).items()
        if key.split(".")[0] in {"second", "levels", "w2", "y"}
    }


@requires_xsim
def test_the_partition_computes_what_onnx_computes(tmp_path: Path) -> None:
    model = kernel_model()
    _, point = configure_partition(model)
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
