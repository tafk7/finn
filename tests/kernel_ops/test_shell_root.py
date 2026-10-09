# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shell root of KernelOp nodes: the Chain (``kernels.chain``) built from them.

Each node's choices are saved on it and each channel's on its tensor; the root's
adapter memories, open (several viable), are the flow's to choose and are saved on
their channels' tensors; each edge's adapter is forced. The root's members are its
channels, the boundary's among them, and its kernels, named as the graph. The rebuilt
root's module is the Chain's (``kernels.chain``), configured the same way: the same
flat netlist and pins, and in XSim what ``execute_onnx`` computes on the source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from kernels import chain
from kernels.helpers import Lanes, labels
from kernels.xsim import requires_xsim
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes

from finn.core.executors.xsim.rtl import pack_lanes, stream_through
from finn.core.space import inspection
from finn.custom_op.kernels.base import CHANNEL_KEYS, KernelOpError, channel_choices, kernel_op
from finn.custom_op.kernels.shell import persist, save_channels, shell_root
from finn.custom_op.partition.kernel_partitions import partition_body
from finn.kernels.configure import chosen, commit
from finn.kernels.explore import Ranked, SizeFifos
from finn.transformation.kernels import (
    InferKernelTensors,
    ToKernelOps,
    explore_kernel_choices,
    kernel_choices_config,
)
from finn.transformation.kernels.cut import CutKernelPartition
from kernel_ops.models import (
    INT3,
    TARGET,
    chain_source,
    configure_partition,
    fan_out_source,
    kernel_model,
    lift,
    matmul_model,
    open_memories,
    shared_weights_source,
)


def test_the_root_of_the_chains_nodes_is_the_chain() -> None:
    model = kernel_model()
    root, point = configure_partition(model)
    reference = chain.chain()
    assert labels(point.module) == labels(reference.module)
    assert point.module.fragment == reference.module.fragment
    assert point.module.abi == reference.module.abi
    assert root.boundary == (("x", "s_axis_0"), ("y", "m_axis_0"))


def test_a_shell_root_refuses_a_node_that_is_not_a_kernel_op() -> None:
    model = chain_source()
    with pytest.raises(KernelOpError, match="activate: MultiThreshold is not a KernelOp"):
        shell_root(model, model.graph.node[1:2])


def test_channel_choices_persist_on_their_tensors() -> None:
    """The adapter memories chosen in the root are stated on the tensors of x and levels,
    and no node holds a channel's choice."""
    model = kernel_model()
    configure_partition(model)
    assert "adapter.input_gen.input_gen.ram_style" in channel_choices(model, "x")
    assert "adapter.input_gen.input_gen.ram_style" in channel_choices(model, "levels")
    attributes = {attribute.name for node in model.graph.node for attribute in node.attribute}
    assert not any(name.startswith(("x.", "w.", "y.")) for name in attributes)


def test_a_refold_that_converts_widths_keeps_the_input_side_s_memory() -> None:
    model = kernel_model()
    configure_partition(model)
    second = kernel_op(model, model.graph.node[2])
    second.save({"compute.packed.simd": 4})
    root = shell_root(model, model.graph.node, name="chain")
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
    # A memory of a chain the levels edge does not take: the writer refuses it, so it is
    # written past it.
    other = {"adapter.input_gen_vpc.input_gen.ram_style": "auto"}
    with pytest.raises(KernelOpError, match="levels: refused choices: adapter.input_gen_vpc"):
        save_channels(model, {"levels": other})
    model.set(CHANNEL_KEYS["adapter.input_gen_vpc.input_gen.ram_style"], "auto", tensor="levels")
    root = shell_root(model, model.graph.node, name="chain")
    stale = "levels.adapter.input_gen_vpc.input_gen.ram_style"
    assert list(root.dropped) == [stale]
    assert root.dropped[stale] == "inapplicable"
    point, _ = open_memories(root)
    levels = point.levels
    assert levels.query(type(levels).adapter).value == "input_gen"


def test_a_lifted_initializers_source_choices_are_stale_in_the_partition() -> None:
    """Weights lifted to a graph input leave the node streamed: its weight channel has no
    value, so no source, and the source's choices, stated on the tensor w, now the
    weight edge's, are the partition's to replay: it drops them as stale."""
    model = matmul_model()
    stored = kernel_op(model, model.graph.node[0])
    stored.save({"compute.packed.pe": 2})
    save_channels(model, {"w": {"source.memstream.ram_style": "block"}})
    owned: Any = stored.point()
    assert owned.w.source.ram_style == "block"  # the node owns w's channel
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
    assert channel_choices(model, "w") == {"source.memstream.ram_style": "block"}
    point: Any = streamed.point()
    assert point.matmul.compute.pe == 2 and streamed.verify_node() == []
    assert list(shell_root(model, model.graph.node).dropped) == ["w.source.memstream.ram_style"]


def test_a_partition_has_ports_for_its_onnx_inputs_and_outputs_only() -> None:
    model = kernel_model()
    root = shell_root(model, model.graph.node, name="chain")
    # hidden and levels run between its nodes: channels with no port.
    assert root.boundary == (("x", "s_axis_0"), ("y", "m_axis_0"))
    point, styles = open_memories(root)
    point = commit(point, dict.fromkeys(styles, "auto"))
    assert sorted(port.name for port in point.module.abi.pins) == [
        "ap_clk",
        "ap_rst_n",
        "m_axis_0",
        "s_axis_0",
    ]


def test_a_second_partition_of_kernel_ops_is_refused_by_name() -> None:
    """The cut decides which KernelOps go together: a root of some of a model's
    KernelOps would be one of two partitions, refused, naming the KernelOps left out."""
    model = kernel_model()
    with pytest.raises(KernelOpError, match="second: KernelOps outside the partition 'front'"):
        shell_root(model, model.graph.node[:2], name="front")


def test_streamed_weights_are_a_boundary_of_the_partition() -> None:
    model = kernel_model(second_weights=False)
    root = shell_root(model, model.graph.node, name="chain")
    assert root.boundary == (("x", "s_axis_0"), ("w2", "s_axis_1"), ("y", "m_axis_0"))
    assert root.owners["w2"] == "w2"
    assert root.owner("w2.transport") == ("w2", "transport")
    # Every channel has a transport, the weight edge's as x's: each its tensor's.
    keys = {decision.key for decision in inspection.decisions(root.point)}
    assert {"w2.transport", "x.transport"} <= keys


def test_the_owner_map() -> None:
    model = kernel_model()
    root = shell_root(model, model.graph.node, name="chain")
    # Each member's owner is its graph name: a channel's tensor, a kernel's node.
    assert dict(root.owners) == {
        "x": "x",
        "w1": "w1",
        "hidden": "hidden",
        "levels": "levels",
        "w2": "w2",
        "y": "y",
        "first": "first",
        "activate": "activate",
        "second": "second",
    }
    # A key's owner is its longest member path's, the key there the rest of it.
    assert root.owner("levels.adapter.input_gen.input_gen.ram_style") == (
        "levels",
        "adapter.input_gen.input_gen.ram_style",
    )
    assert root.owner("second.compute.packed.pe") == ("second", "compute.packed.pe")
    assert root.owner("y.transport") == ("y", "transport")
    assert root.owner("levelsx.transport") is None
    assert root.members == (
        "x",
        "w1",
        "hidden",
        "levels",
        "w2",
        "y",
        "first",
        "activate",
        "second",
    )


def test_a_choice_in_the_root_persists_on_the_weights_tensor() -> None:
    model = kernel_model()
    root = shell_root(model, model.graph.node, name="chain")
    written = persist(model, root, commit(root.point, {"w2.source.memstream.ram_style": "block"}))
    assert written["w2"]["source.memstream.ram_style"] == "block"
    assert channel_choices(model, "w2")["source.memstream.ram_style"] == "block"
    replayed: Any = kernel_op(model, model.graph.node[2]).point()
    assert replayed.w.source.ram_style == "block"
    rebuilt = shell_root(model, model.graph.node, name="chain")
    assert rebuilt.point.w2.source.ram_style == "block" and not rebuilt.dropped


def test_an_explored_point_persists_and_replays_as_itself() -> None:
    """Every choice an exploration commits, a boundary channel's and the others',
    goes back to its owner (a node or a tensor) by path and replays onto the same
    point."""
    model = kernel_model()
    explored = explore_kernel_choices(model, [Ranked(Lanes(2)), SizeFifos()])
    made = chosen(explored.point)
    assert {"x.transport", "y.transport", "levels.transport"} <= set(made)
    assert "first.compute.packed.pe" in made
    rebuilt = shell_root(model, model.graph.node)
    assert not rebuilt.dropped and chosen(rebuilt.point) == made
    # Written again from the replayed point, every node and tensor holds what it held.
    held = kernel_choices_config(model)
    assert {"first", "x", "levels", "y"} <= set(held)
    persist(model, rebuilt, rebuilt.point)
    assert kernel_choices_config(model) == held


def test_a_node_named_like_a_tensor_is_refused() -> None:
    model = kernel_model()
    model.graph.node[1].name = "hidden"
    with pytest.raises(KernelOpError, match="a node and a tensor are both named hidden"):
        shell_root(model, model.graph.node)


def test_two_nodes_of_one_member_name_are_refused() -> None:
    model = kernel_model()
    model.graph.node[2].name = "first"
    with pytest.raises(KernelOpError, match="a node and another node are both named first"):
        shell_root(model, model.graph.node)


FAN_OUT = {
    "two KernelOps": (None, "hidden: tensor-fan-out: read by a, b"),
    "a graph output listed last": ("last", "hidden: tensor-fan-out: read by a and leaves"),
    "a graph output listed first": ("first", "hidden: tensor-fan-out: read by a and leaves"),
}


@pytest.mark.parametrize("case", FAN_OUT)
def test_a_tensor_read_more_than_once_is_refused_by_name(case: str) -> None:
    """A channel has one consumer: hidden read by two KernelOps, or by one and as a graph
    output (in either order of the graph's outputs), is refused before any owner is
    chosen, not given to its last reader nor left without a port."""
    output, named = FAN_OUT[case]
    model = fan_out_source(output=output).transform(ToKernelOps(TARGET))
    assert all(node.domain == "finn.custom_op.kernels" for node in model.graph.node)
    with pytest.raises(KernelOpError, match=f"fan: .*{named}"):
        shell_root(model, model.graph.node, name="fan")


@pytest.mark.parametrize("case", FAN_OUT)
def test_a_build_stops_at_its_first_shell_root_naming_the_tensor(case: str, tmp_path: Path) -> None:
    """The cut accepts the KernelOps (their partition's body keeps hidden read more than
    once), and the build's first shell root, exploration's, refuses it by name."""
    output, named = FAN_OUT[case]
    model = fan_out_source(output=output).transform(ToKernelOps(TARGET))
    _, body, _ = partition_body(model.transform(CutKernelPartition(tmp_path)))
    with pytest.raises(KernelOpError, match=f"partition: .*{named}"):
        explore_kernel_choices(body, [])


def test_a_parameter_two_nodes_own_is_refused_by_name() -> None:
    """One initializer, two MatMuls' weights: each would own its channel. Conversion
    gives each node its own copy, so the second is pointed back at the first's."""
    model = shared_weights_source().transform(ToKernelOps(TARGET))
    a, b = model.graph.node
    assert b.input[1] != "w"
    b.input[1] = a.input[1]
    with pytest.raises(KernelOpError, match="w: tensor-fan-out: read by a, b"):
        shell_root(model, model.graph.node)


def test_a_kernel_choice_the_root_refuses_is_dropped_named_and_cleared_by_persist() -> None:
    """A choice written past ``save`` reaches the root's replay, which drops it as stale
    with why; the rest replays, and persisting the point clears the stale attribute."""
    model = kernel_model()
    first = kernel_op(model, model.graph.node[0])
    first.set_nodeattr("compute.packed.simd", 2)
    first.set_nodeattr("compute.packed.pe", 3)
    root = shell_root(model, model.graph.node, name="chain")
    assert list(root.dropped) == ["first.compute.packed.pe"]
    assert "domain-membership" in root.dropped["first.compute.packed.pe"]
    assert root.point.first.compute.simd == 2
    # Its pe is open again; once chosen, the node holds the new value, nothing stale.
    persist(model, root, commit(root.point, {"first.compute.packed.pe": 2}))
    assert first.choices()["compute.packed.pe"] == 2
    assert not shell_root(model, model.graph.node, name="chain").dropped


def test_persist_clears_what_the_point_does_not_commit() -> None:
    model = kernel_model()
    root = shell_root(model, model.graph.node, name="chain")
    model.set(CHANNEL_KEYS["adapter.input_gen_vpc.input_gen.ram_style"], "auto", tensor="levels")
    stale = shell_root(model, model.graph.node, name="chain")
    assert list(stale.dropped) == ["levels.adapter.input_gen_vpc.input_gen.ram_style"]
    persist(model, stale, stale.point)
    assert "adapter.input_gen_vpc.input_gen.ram_style" not in channel_choices(model, "levels")
    # Every owner holds what the point commits, by graph name and the key there.
    expected: dict[str, dict[str, object]] = {}
    for key, value in chosen(root.point).items():
        owner = root.owner(key)
        assert owner is not None
        expected.setdefault(owner[0], {})[owner[1]] = value
    assert kernel_choices_config(model) == expected


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
                    pack_lanes(chain.X[r][f : f + chain.SIMD], a_bits)
                    for r in range(chain.ROWS)
                    for f in range(0, chain.INPUTS, chain.SIMD)
                ],
                chain.SIMD * a_bits,
            )
        },
        outputs={
            "m_axis_0": (
                [
                    pack_lanes(y[r][f : f + chain.PE].tolist(), y_bits)
                    for r in range(chain.ROWS)
                    for f in range(0, chain.OUTPUTS, chain.PE)
                ],
                chain.PE * y_bits,
            )
        },
        cycles=point.cycles,
    )
