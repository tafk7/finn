# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The graph's network of KernelOp nodes: the Chain (``kernels.chain``) bound from them.

Each node's choices are saved on it and each channel's on its tensor; the network's
adapter memories, open (several viable), are the flow's to choose and are saved on
their channels' tensors; each edge's adapter is forced. The network's members are its
channels, the boundary's among them, and its kernels, named as the graph. Its module is
the Chain's (``kernels.chain``), configured the same way: the same flat netlist and
pins, and in XSim what ``execute_onnx`` computes on the source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from kernels import chain
from kernels.helpers import Lanes, built, choose, cycles_of, labels
from kernels.xsim import requires_xsim
from qonnx.transformation.infer_shapes import InferShapes

import finn.custom_op.kernels.network as graph_network
from finn.core.graph.annotations.channel import CHANNEL_KEYS, channel_choices
from finn.core.graph.errors import KernelOpError
from finn.core.onnx_exec import execute_onnx
from finn.core.reporting.findings import FindingKind
from finn.core.rtlsim_bench import pack_lanes, stream_through
from finn.core.space import Available
from finn.custom_op.kernels import shell
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import network, persist
from finn.custom_op.kernels.shell import save_channels
from finn.kernels.explore import Ranked, SizeFifos
from finn.kernels.network import dotted
from finn.toolchain.launcher import host_toolchain
from finn.transformation.kernels import (
    InferKernelTensors,
    ToKernelOps,
    explore_kernel_choices,
    kernel_choices_config,
)
from finn.transformation.kernels.cut import CutKernelPartition
from kernel_ops.models import (
    INT3,
    MATMUL,
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


def test_the_network_of_the_chains_nodes_is_the_chain() -> None:
    model = kernel_model()
    bound = configure_partition(model)
    module = built(bound, "chain")
    reference = built(chain.chain(), chain.STEM)
    assert labels(module) == labels(reference)
    assert module.fragment == reference.fragment
    assert module.abi == reference.abi
    assert bound.boundary == (("x", "s_axis_0"), ("y", "m_axis_0"))


def test_a_network_refuses_a_node_that_is_not_a_kernel_op() -> None:
    model = chain_source()
    with pytest.raises(KernelOpError, match="activate: MultiThreshold is not a KernelOp"):
        network(model, model.graph.node[1:2])


def test_channel_choices_persist_on_their_tensors() -> None:
    """The adapter memories chosen in the network are stated on the tensors of x and
    levels, and no node holds a channel's choice."""
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
    bound = network(model, model.graph.node, name="chain")
    # The levels edge now converts widths before its transport; its replay after it is
    # the same input_gen, whose memory choice still applies.
    assert not bound.stale
    assert [stage.label for stage in bound.member("levels").stages] == [
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
    bound = network(model, model.graph.node, name="chain")
    stale = "levels.adapter.input_gen_vpc.input_gen.ram_style"
    assert list(bound.stale) == [stale]
    assert bound.stale[stale] == "inapplicable"
    levels = bound.member("levels")
    assert levels.query(type(levels).adapter).value == "input_gen"


def test_a_lifted_initializers_source_choices_are_stale_in_the_partition() -> None:
    """Weights lifted to a graph input leave the node streamed: its weight channel has no
    value, so no source, and the source's choices, stated on the tensor w, now the
    weight edge's, are the partition's to replay: it drops them as stale."""
    model = matmul_model()
    stored = kernel_op(model, model.graph.node[0])
    stored.save(MATMUL)  # folded, so its channels' ends are known
    save_channels(model, {"w": {"source.memstream.ram_style": "block"}})
    owned = network(model, model.graph.node)
    assert owned.member("w").source.ram_style == "block"  # the node owns w's channel
    lift(model, "w")
    model.set_tensor_datatype("w", INT3)
    streamed = kernel_op(model, model.graph.node[0])
    assert streamed.facts().owned == ()
    # The output was inferred from the stored weights' columns; unknown, they give the
    # datatypes' wider range. The kernel states its results from its new facts, so its
    # choices stand; the graph's y is stale, and the network refuses it by name until
    # inference, run again, states it.
    assert streamed.verify_node() == []
    with pytest.raises(
        KernelOpError,
        match="stale_outputs: tensor-stale: the graph states y .*run InferKernelTensors",
    ):
        network(model, model.graph.node)
    model = model.transform(InferKernelTensors())
    streamed = kernel_op(model, model.graph.node[0])
    # The node replays its own choices only: the weight edge's are the partition's.
    assert channel_choices(model, "w") == {"source.memstream.ram_style": "block"}
    point: Any = streamed.point()
    assert point.compute.pe == 2 and streamed.verify_node() == []
    assert list(network(model, model.graph.node).stale) == ["w.source.memstream.ram_style"]


def test_a_stale_output_is_refused_before_its_channel_reads_it() -> None:
    """The product is authoritative: an output annotation its producer no longer states
    is the graph's staleness, named with its tensor and producer, not the channel's
    refusal of the producer's end."""
    model = matmul_model()
    model.set_tensor_datatype("y", INT3)
    with pytest.raises(
        KernelOpError, match="stale_outputs: tensor-stale: the graph states y .* its producer first"
    ) as error:
        network(model, model.graph.node)
    assert [finding.code for finding in error.value.findings] == ["tensor-stale"]
    # Another shape of the same elements and last axis is what the kernel requires.
    model = matmul_model().transform(InferKernelTensors())
    assert not network(model, model.graph.node).stale


def test_a_partition_has_ports_for_its_onnx_inputs_and_outputs_only() -> None:
    model = kernel_model()
    bound = network(model, model.graph.node, name="chain")
    # hidden and levels run between its nodes: channels with no port.
    assert bound.boundary == (("x", "s_axis_0"), ("y", "m_axis_0"))
    _, styles = open_memories(bound)
    module = built(choose(bound, dict.fromkeys(styles, "auto")), "chain")
    assert sorted(port.name for port in module.abi.pins) == [
        "ap_clk",
        "ap_rst_n",
        "m_axis_0",
        "s_axis_0",
    ]


def test_any_selection_of_nodes_is_a_network() -> None:
    """A slice of a model's KernelOps binds as the whole does: its boundary its ONNX
    inputs and outputs, an edge it cuts a port (no second-partition refusal)."""
    model = kernel_model()
    front = network(model, model.graph.node[:2], name="front")
    assert front.kernels == ("first", "activate")
    assert front.boundary == (("x", "s_axis_0"), ("levels", "m_axis_0"))
    alone = network(model, model.graph.node[1:2], name="alone")
    assert alone.boundary == (("hidden", "s_axis_0"), ("levels", "m_axis_0"))
    assert not alone.whole and not front.stale


def test_streamed_weights_are_a_boundary_of_the_partition() -> None:
    model = kernel_model(second_weights=False)
    bound = network(model, model.graph.node, name="chain")
    assert bound.boundary == (("x", "s_axis_0"), ("w2", "s_axis_1"), ("y", "m_axis_0"))
    # Every channel has a transport, the weight edge's as x's: each its tensor's.
    keys = set(bound.decisions())
    assert {("w2", "transport"), ("x", "transport")} <= keys


def test_members_are_named_as_the_graph() -> None:
    model = kernel_model()
    bound = network(model, model.graph.node, name="chain")
    # Each member is its graph name: a channel's tensor, a kernel's node; ranked kernels
    # first; declared in the module's order.
    assert bound.members == (
        "first",
        "activate",
        "second",
        "x",
        "w1",
        "hidden",
        "levels",
        "w2",
        "y",
    )
    assert bound.order == (
        "x",
        "w1",
        "hidden",
        "levels",
        "w2",
        "first",
        "activate",
        "second",
        "y",
    )
    assert ("levels", "adapter.input_gen.input_gen.ram_style") in bound.decisions()
    assert ("second", "compute.packed.pe") in bound.decisions()


def test_a_choice_in_the_network_persists_on_the_weights_tensor() -> None:
    model = kernel_model()
    bound = network(model, model.graph.node, name="chain")
    written = persist(model, choose(bound, {"w2.source.memstream.ram_style": "block"}))
    assert written["w2"]["source.memstream.ram_style"] == "block"
    assert channel_choices(model, "w2")["source.memstream.ram_style"] == "block"
    rebuilt = network(model, model.graph.node, name="chain")
    assert rebuilt.member("w2").source.ram_style == "block" and not rebuilt.stale


def test_an_explored_network_persists_and_replays_as_itself() -> None:
    """Every choice an exploration commits, a boundary channel's and the others',
    goes back to its owner (a node or a tensor) and replays onto the same network."""
    model = kernel_model()
    explore_kernel_choices(model, Ranked(Lanes(2)))
    explored = explore_kernel_choices(model, SizeFifos())
    made = {dotted(key): value for key, value in explored.network.chosen().items()}
    assert {"x.transport", "y.transport", "levels.transport"} <= set(made)
    assert "first.compute.packed.pe" in made
    rebuilt = network(model, model.graph.node)
    assert not rebuilt.stale
    assert {dotted(key): value for key, value in rebuilt.chosen().items()} == made
    # Written again from the replayed network, every node and tensor holds what it held.
    held = kernel_choices_config(model)
    assert {"first", "x", "levels", "y"} <= set(held)
    persist(model, rebuilt)
    assert kernel_choices_config(model) == held


def test_a_node_named_like_a_tensor_is_refused() -> None:
    model = kernel_model()
    model.graph.node[1].name = "hidden"
    with pytest.raises(KernelOpError, match="a node and a tensor are both named hidden"):
        network(model, model.graph.node)


def test_two_nodes_of_one_name_are_refused() -> None:
    model = kernel_model()
    model.graph.node[2].name = "first"
    with pytest.raises(KernelOpError, match="a node and another node are both named first"):
        network(model, model.graph.node)


FAN_OUT = {
    "two KernelOps": (None, "fan_out: tensor-fan-out: hidden is read by a, b"),
    "a graph output listed last": (
        "last",
        "fan_out: tensor-fan-out: hidden is read by a and leaves",
    ),
    "a graph output listed first": (
        "first",
        "fan_out: tensor-fan-out: hidden is read by a and leaves",
    ),
}


@pytest.mark.parametrize("case", FAN_OUT)
def test_a_tensor_read_more_than_once_is_refused_by_name(case: str) -> None:
    """A channel has one consumer: hidden read by two KernelOps, or by one and as a graph
    output (in either order of the graph's outputs), is refused before anything is
    bound, not given to its last reader nor left without a port."""
    output, named = FAN_OUT[case]
    model = fan_out_source(output=output).transform(ToKernelOps(TARGET))
    assert all(node.domain == "finn.custom_op.kernels" for node in model.graph.node)
    with pytest.raises(KernelOpError, match=f"fan: .*{named}"):
        network(model, model.graph.node, name="fan")


@pytest.mark.parametrize("case", FAN_OUT)
def test_a_build_stops_at_its_first_network_naming_the_tensor(case: str, tmp_path: Path) -> None:
    """The cut partitions the KernelOps (their partition's body keeps hidden read more than
    once), and the build's first network, the cut's, where the channels' choices are
    created, refuses it by name."""
    output, named = FAN_OUT[case]
    model = fan_out_source(output=output).transform(ToKernelOps(TARGET))
    with pytest.raises(KernelOpError, match=f"partition: .*{named}"):
        model.transform(CutKernelPartition(tmp_path))


def test_a_parameter_two_nodes_own_is_refused_by_name() -> None:
    """One initializer, two MatMuls' weights: each would own its channel. Conversion
    gives each node its own copy, so the second is pointed back at the first's."""
    model = shared_weights_source().transform(ToKernelOps(TARGET))
    a, b = model.graph.node
    assert b.input[1] != "w"
    b.input[1] = a.input[1]
    with pytest.raises(KernelOpError, match="fan_out: tensor-fan-out: w is read by a, b") as fan:
        network(model, model.graph.node)
    # Fan-out refuses the network: a rejection.
    assert [each.kind for each in fan.value.findings] == [FindingKind.REJECTION]


def test_a_kernel_choice_the_network_refuses_is_dropped_named_and_cleared_by_persist() -> None:
    """A choice written past ``save`` reaches the network's replay, which drops it as
    stale with why; the rest replays, and persisting the network clears the stale
    attribute."""
    model = kernel_model()
    first = kernel_op(model, model.graph.node[0])
    first.set_nodeattr("compute.packed.simd", 2)
    first.set_nodeattr("compute.packed.pe", 3)
    bound = network(model, model.graph.node, name="chain")
    assert list(bound.stale) == ["first.compute.packed.pe"]
    assert "domain-membership" in bound.stale["first.compute.packed.pe"]
    assert bound.member("first").compute.simd == 2
    # Its pe is open again; once chosen, the node holds the new value, nothing stale.
    persist(model, choose(bound, {"first.compute.packed.pe": 2}))
    assert first.choices()["compute.packed.pe"] == 2
    assert not network(model, model.graph.node, name="chain").stale


def test_persist_clears_what_the_network_does_not_hold() -> None:
    model = kernel_model()
    bound = network(model, model.graph.node, name="chain")
    model.set(CHANNEL_KEYS["adapter.input_gen_vpc.input_gen.ram_style"], "auto", tensor="levels")
    stale = network(model, model.graph.node, name="chain")
    assert list(stale.stale) == ["levels.adapter.input_gen_vpc.input_gen.ram_style"]
    persist(model, stale)
    assert "adapter.input_gen_vpc.input_gen.ram_style" not in channel_choices(model, "levels")
    # Every owner holds what the network holds, by graph name and the key there.
    expected: dict[str, dict[str, object]] = {}
    for (owner, key), value in bound.chosen().items():
        expected.setdefault(owner, {})[key] = value
    assert kernel_choices_config(model) == expected


def test_persist_checks_every_value_before_it_writes_any(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A value an owner refuses stops the write, by name, before any owner is written:
    nothing is written halfway (the kernels are checked first, then each channel)."""
    model = kernel_model()
    bound = configure_partition(model)
    held = kernel_choices_config(model)
    changed = choose(bound, {"second.compute.packed.pe": 1})
    checked = graph_network._encoded

    def refuse_y(tensor: str, values: Any) -> None:
        if tensor == "y":
            raise KernelOpError(f"{tensor}: refused")
        checked(tensor, values)

    monkeypatch.setattr(graph_network, "_encoded", refuse_y)
    with pytest.raises(KernelOpError, match="y: refused"):
        persist(model, changed)
    assert kernel_choices_config(model) == held


def test_the_shell_s_cycles_and_resources_are_its_members() -> None:
    bound = configure_partition(kernel_model())
    cycles = cycles_of(bound)
    assert isinstance(cycles, Available) and cycles.value > 0
    used = bound.own_resources()
    assert isinstance(used, Available)
    # The ip shell has no ends or static region: its resources are its members' sum.
    split = shell.shell_resources(bound)
    assert isinstance(split, shell.ShellResources)
    assert split.ends == () and split.static_region == ()
    assert split.total == used.value


@requires_xsim
def test_the_partition_computes_what_onnx_computes(tmp_path: Path) -> None:
    model = kernel_model()
    bound = configure_partition(model)
    module = built(bound, "chain")
    cycles = cycles_of(bound)
    assert isinstance(cycles, Available)
    x = np.array(chain.X, dtype=np.float32)
    y = execute_onnx(chain_source().transform(InferShapes()), {"x": x})["y"].astype(int)
    a_bits, y_bits = chain.A.bitwidth(), chain.Y.bitwidth()
    stream_through(
        module,
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
        cycles=cycles.value,
        toolchain=host_toolchain(),
    )
