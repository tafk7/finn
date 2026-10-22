# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The graph's network of KernelOp nodes: the Chain (``support.chain``) bound from them.

Each node's choices are saved on it and each channel's on its tensor; the network's
adapter memories, open (several viable), are the flow's to choose and are saved on
their channels' tensors; each edge's adapter is forced. The network's members are its
channels, the boundary's among them, and its kernels, named as the graph. Its module is
the Chain's (``support.chain``), configured the same way: the same flat netlist and
pins, and in XSim what ``execute_onnx`` computes on the source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qonnx.transformation.infer_shapes import InferShapes
from support import chain
from support.configure import configured, write_raw
from support.helpers import Lanes, built, choose, cycles_of, labels
from support.models import (
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
from support.refusals import refused
from support.xsim import requires_xsim

import finn.custom_op.kernels.network as graph_network
from finn.core.graph.annotations.channel import CHANNEL_KEYS, channel_choices
from finn.core.graph.errors import KernelOpError
from finn.core.onnx_exec import execute_onnx
from finn.core.reporting.findings import FindingKind
from finn.core.rtlsim_bench import pack_lanes, stream_through
from finn.core.space import Available, inspection
from finn.custom_op.kernels import shell
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import network, persist
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
    with refused(
        "not-a-kernel-op",
        error=KernelOpError,
        owner="kernel_op",
        node="activate",
        message="MultiThreshold is not a KernelOp",
    ):
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
    configured(model, {model.graph.node[2].name: {"simd": 4}})
    bound = network(model, model.graph.node, name="chain")
    # The levels edge now converts widths before its transport; its replay after it is
    # the same input_gen, whose memory choice still applies.
    assert not bound.stale
    assert [stage.label for stage in bound.member("levels").stages] == [
        "output_adapter.vpc.vpc",
        "adapter.input_gen.input_gen",
    ]


def test_an_edge_choice_under_a_chain_not_taken_is_kept_out_of_play() -> None:
    """A memory of a chain the levels edge does not take: set or written past the seam,
    it is kept out of play (dormant), nothing stale, and the adapter the edge takes
    applies."""
    model = kernel_model()
    configure_partition(model)
    other = {"adapter.input_gen_vpc.input_gen.ram_style": "auto"}
    configured(model, {"levels": other})
    bound = network(model, model.graph.node, name="chain")
    assert not bound.stale
    key = ("levels", "adapter.input_gen_vpc.input_gen.ram_style")
    assert dict(bound.states(["levels"]))[key] == inspection.Dormant("auto")
    levels = bound.member("levels")
    assert levels.query(type(levels).adapter).value == "input_gen"


def test_a_lifted_initializers_source_choices_are_out_of_play_in_the_partition() -> None:
    """Weights lifted to a graph input leave the node streamed: its weight channel has no
    value, so no source, and the source's choices, stated on the tensor w, now the
    weight edge's, are the partition's: it keeps them out of play (dormant)."""
    model = matmul_model()
    configured(model, {"first": MATMUL})  # folded, so its channels' contracts are known
    configured(model, {"w": {"source.memstream.ram_style": "block"}})
    owned = network(model, model.graph.node)
    assert owned.member("w").source.ram_style == "block"  # the node owns w's channel
    lift(model, "w")
    model.set_tensor_datatype("w", INT3)
    streamed = kernel_op(model, model.graph.node[0])
    assert streamed.owned() == {}
    # The output was inferred from the stored weights' columns; unknown, they give the
    # datatypes' wider range. The kernel states its results from its new facts, so its
    # choices stand; the graph's y is stale, and the network refuses it by name until
    # inference, run again, states it.
    assert streamed.verify_node() == []
    with refused(
        "tensor-stale",
        error=KernelOpError,
        owner="stale_outputs",
        node="first",
        tensor="y",
        message="the graph states y .*run InferKernelTensors",
    ):
        network(model, model.graph.node)
    model = model.transform(InferKernelTensors())
    streamed = kernel_op(model, model.graph.node[0])
    # The node replays its own choices only: the weight edge's are the partition's, the
    # source and its memory's (the interim channel rule's beside the one set).
    source = {
        "source": "memstream",
        "source.memstream.ram_style": "block",
        "source.memstream.pumped_memory": False,
    }
    assert channel_choices(model, "w") == {**source, "transport": "direct"}
    point: Any = streamed.point()
    assert point.compute.pe == 2 and streamed.verify_node() == []
    edge = network(model, model.graph.node)
    assert not edge.stale
    states = dict(edge.states(["w"]))
    assert {key: states[("w", key)] for key in source} == {
        key: inspection.Dormant(value) for key, value in source.items()
    }


def test_a_stale_output_is_refused_before_its_channel_reads_it() -> None:
    """The product is authoritative: an output annotation its producer no longer states
    is the graph's staleness, named with its tensor and producer, not the channel's
    refusal of the producer's contract."""
    model = matmul_model()
    model.set_tensor_datatype("y", INT3)
    with refused(
        "tensor-stale", error=KernelOpError, owner="stale_outputs", node="first", tensor="y"
    ):
        network(model, model.graph.node)
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
    assert ("second", "pe") in bound.decisions()


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
    assert "first.pe" in made
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
    with refused(
        "name-shared",
        error=KernelOpError,
        owner="unique_names",
        node="hidden",
        message="a node and a tensor are both named hidden",
    ):
        network(model, model.graph.node)


def test_two_nodes_of_one_name_are_refused() -> None:
    model = kernel_model()
    model.graph.node[2].name = "first"
    with refused(
        "name-shared",
        error=KernelOpError,
        owner="unique_names",
        node="first",
        message="a node and another node are both named",
    ):
        network(model, model.graph.node)


FAN_OUT = {
    "two KernelOps": (None, "hidden is read by a, b"),
    "a graph output listed last": ("last", "hidden is read by a and leaves"),
    "a graph output listed first": ("first", "hidden is read by a and leaves"),
}


@pytest.mark.parametrize("case", FAN_OUT)
def test_a_tensor_read_more_than_once_is_refused_by_name(case: str) -> None:
    """A channel has one consumer: hidden read by two KernelOps, or by one and as a graph
    output (in either order of the graph's outputs), is refused before anything is
    bound, not given to its last reader nor left without a port."""
    output, named = FAN_OUT[case]
    model = fan_out_source(output=output).transform(ToKernelOps(TARGET))
    assert all(node.domain == "finn.custom_op.kernels" for node in model.graph.node)
    with refused(
        "tensor-fan-out", error=KernelOpError, owner="fan_out", tensor="hidden", message=named
    ):
        network(model, model.graph.node, name="fan")


@pytest.mark.parametrize("case", FAN_OUT)
def test_a_build_stops_at_its_first_network_naming_the_tensor(case: str, tmp_path: Path) -> None:
    """The cut partitions the KernelOps (their partition's body keeps hidden read more than
    once), and the build's first network, the cut's, where the channels' choices are
    created, refuses it by name."""
    output, named = FAN_OUT[case]
    model = fan_out_source(output=output).transform(ToKernelOps(TARGET))
    with refused(
        "tensor-fan-out", error=KernelOpError, owner="fan_out", tensor="hidden", message=named
    ):
        model.transform(CutKernelPartition(tmp_path))


def test_a_parameter_two_nodes_own_is_refused_by_name() -> None:
    """One initializer, two MatMuls' weights: each would own its channel. Conversion
    gives each node its own copy, so the second is pointed back at the first's."""
    model = shared_weights_source().transform(ToKernelOps(TARGET))
    a, b = model.graph.node
    assert b.input[1] != "w"
    b.input[1] = a.input[1]
    with refused(
        "tensor-fan-out",
        error=KernelOpError,
        owner="fan_out",
        tensor="w",
        message="w is read by a, b",
    ) as fan:
        network(model, model.graph.node)
    # Fan-out refuses the network: a rejection.
    assert [each.kind for each in fan.error.findings] == [FindingKind.REJECTION]


def test_a_kernel_choice_the_network_refuses_is_dropped_named_and_cleared_by_persist() -> None:
    """A choice written raw (as a graph FINN did not write holds it) reaches the network's
    bind, which drops it as stale with why; the rest is bound, and persisting the network
    clears the stale attribute."""
    model = kernel_model()
    write_raw(model, "first", "simd", 2)
    write_raw(model, "first", "pe", 3)
    bound = network(model, model.graph.node, name="chain")
    assert list(bound.stale) == ["first.pe"]
    # Its own domain refuses 3: nothing else stands beside it.
    assert {(f.code, f.owner) for f in bound.stale["first.pe"]} == {
        ("domain-membership", "pe"),
    }
    assert bound.member("first").compute.simd == 2
    # Its pe is open again; once chosen, the node holds the new value, nothing stale.
    persist(model, choose(bound, {"first.pe": 2}))
    assert kernel_op(model, model.graph.node[0]).choices()["pe"] == 2
    assert not network(model, model.graph.node, name="chain").stale


def test_a_stale_choice_keeps_the_kernel_s_finding_naming_its_value_and_candidates() -> None:
    """The misfolded MatMul (``pe`` 3, written raw): the network keeps, per key, the
    findings the kernel gave (each code and owner): the key's own, with the value stored and
    the values the decision could take now (its possible values), its message naming
    them."""
    model = kernel_model()
    write_raw(model, "first", "simd", 2)
    write_raw(model, "first", "pe", 3)
    bound = network(model, model.graph.node, name="chain")
    ((key, findings),) = bound.stale.items()
    assert key == "first.pe"
    assert {(f.kind, f.code, f.owner) for f in findings} == {
        (FindingKind.REJECTION, "domain-membership", "pe"),
    }
    (found,) = findings
    ((_, viable),) = bound.viable([("first", "pe")])
    assert 3 not in viable.cases
    assert dict(found.details) == {"value": 3, "candidates": viable.cases}
    assert found.message == "not one of " + ", ".join(map(str, viable.cases))


def test_a_stale_choice_is_refused_naming_its_key_once() -> None:
    """The refusal of the shell's network prints the key once, with the value stored and the
    values it could take, and carries the kernel's findings as its causes."""
    model = kernel_model()
    write_raw(model, "first", "simd", 2)
    write_raw(model, "first", "pe", 3)
    with refused(
        "choices-stale",
        error=KernelOpError,
        owner="configured_network",
        node="label",
        keys=("first.pe",),
    ) as stale:
        shell.configured_network(model, "label")
    assert str(stale.error).count("pe") == 1
    assert "first.pe = 3: " in str(stale.error)
    assert "not one of 1, 2, 4 (domain-membership)" in str(stale.error)
    assert {(cause.code, cause.owner) for cause in stale.finding.causes} == {
        ("domain-membership", "pe"),
    }


def test_persist_clears_what_the_network_does_not_hold() -> None:
    """A stored value the bind drops (outside its domain) is cleared by persisting; every
    owner holds what the network holds, out of play (dormant) included."""
    model = kernel_model()
    model.set(CHANNEL_KEYS["adapter.input_gen_vpc.input_gen.ram_style"], "auto", tensor="levels")
    write_raw(model, "first", "pe", 3)
    bound = network(model, model.graph.node, name="chain")
    assert list(bound.stale) == ["first.pe"]
    persist(model, bound)
    assert "pe" not in kernel_op(model, model.graph.node[0]).choices()
    assert channel_choices(model, "levels")["adapter.input_gen_vpc.input_gen.ram_style"] == "auto"
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
    changed = choose(bound, {"second.pe": 1})
    checked = graph_network._encoded

    def refuse_y(tensor: str, values: Any) -> None:
        if tensor == "y":
            raise KernelOpError.refused(
                "channel-choice-invalid", "finn.channel", f"{tensor}: refused", tensor=tensor
            )
        checked(tensor, values)

    monkeypatch.setattr(graph_network, "_encoded", refuse_y)
    with refused("channel-choice-invalid", error=KernelOpError, owner="finn.channel", tensor="y"):
        persist(model, changed)
    assert kernel_choices_config(model) == held


def test_the_shell_s_cycles_and_resources_are_its_members() -> None:
    bound = configure_partition(kernel_model())
    cycles = cycles_of(bound)
    # the slowest member of the Chain at its baselines
    assert cycles == Available(12)
    used = bound.own_resources()
    assert isinstance(used, Available)
    # The ip shell has no movers or static region: its resources are its members' sum.
    split = shell.shell_resources(bound)
    assert isinstance(split, shell.ShellResources)
    assert split.movers == () and split.static_region == ()
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
