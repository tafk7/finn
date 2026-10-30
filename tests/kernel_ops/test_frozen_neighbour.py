# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A slice's frozen neighbours: an edge of a selection whose node beyond it is a KernelOp.

The edge channel's side there is the neighbour's port's side at its stored choices, given
as a fact (``Side.neighbour``): an upstream neighbour's stream, label and word width, a
downstream one's stream and label. Only an accepted value is a fact, so a neighbour whose
stored choices or kernel are refused, or whose side is not known at them, is refused by
name (``neighbour-unaccepted``). The edge channel is the slice's: its adapters, its hops to
the neighbour, its sizing (the neighbour's pace) and its resources derive as the whole
partition's do; its netlist and module are refused (``slice-not-emitted``): a slice is not
emitted."""

from __future__ import annotations

import copy
import re
from dataclasses import replace
from typing import Any, cast

from qonnx.core.modelwrapper import ModelWrapper
from support.configure import write_raw
from support.models import TARGET, chain_model, fan_out_source, kernel_model
from support.refusals import refused

from finn.core.graph.errors import KernelOpError
from finn.core.space import Available, Rejected, inspection
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import PRODUCER, declaration, network
from finn.kernels import fifo_sizing
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel, OuterHops
from finn.kernels.composer import ChannelDeclaration, PartitionDeclaration
from finn.kernels.partition import Partition, partition
from finn.kernels.port import port_side
from finn.kernels.sides import Side, SideKind
from finn.transformation.kernels import ToKernelOps


def _slice(model: ModelWrapper, *names: str) -> Partition:
    """The partition of the nodes ``names`` of ``model``, in graph order."""
    return network(model, [node for node in model.graph.node if node.name in names])


def _side(point: Kernel, port: str) -> Side:
    """A KernelOp's port's side, its own or served."""
    found = point.query(cast(Any, port_side(type(point), port)))
    assert isinstance(found, Available) and isinstance(found.value, Side)
    return found.value


def _frozen(point: Kernel, port: str, label: str) -> Side:
    """The actual port side as a frozen neighbour's, labelled ``label``."""
    found = _side(point, port)
    assert found.stream is not None
    return Side.neighbour(found.stream, label, found.width)


def test_an_upstream_neighbour_gives_its_width_and_a_downstream_one_none() -> None:
    """``activate`` alone: ``first`` produces hidden, ``second`` reads levels. Each edge's
    side is the neighbour's port's at its stored choices, labelled below it: hidden's
    producer an output port's (a width), levels' consumer an input port's (none). No edge
    is a port of the slice, and both channels read valid."""
    model = kernel_model()
    alone = _slice(model, "activate")
    assert alone.boundary == ()
    nodes = {node.name: node for node in model.graph.node}
    first = kernel_op(model, nodes["first"]).point()
    second = kernel_op(model, nodes["second"]).point()
    hidden, levels = alone.member("hidden"), alone.member("levels")
    assert hidden.producer == _frozen(first, "y", "first.compute.packed.y")
    assert hidden.producer.width is not None
    assert levels.consumer == _frozen(second, "x", "second.compute.packed.x")
    assert levels.consumer.width is None
    assert hidden.consumer.kind is levels.producer.kind is SideKind.KERNEL
    assert inspection.validity(hidden).status == inspection.validity(levels).status == "valid"


def _flipped(model: ModelWrapper, tensor: str, side: str) -> Partition:
    """``activate`` alone, declared with ``tensor``'s ``side`` (a neighbour's) given the
    other direction: a width dropped from an output port's, one added to an input's; its
    stored choices bound."""
    stored = _slice(model, "activate").chosen()
    declared = declaration(model, [node for node in model.graph.node if node.name == "activate"])
    each = declared.channels[tensor]
    given = getattr(each, side)
    assert isinstance(given, Side) and given.kind is SideKind.NEIGHBOUR
    turned = replace(given, width=None if given.width is not None else 8)
    sides = {"producer": each.producer, "consumer": each.consumer, side: turned}
    channels = dict(declared.channels)
    channels[tensor] = ChannelDeclaration(sides["producer"], sides["consumer"], each.facts)
    return partition(PartitionDeclaration(declared.kernels, channels, declared.shell), stored)


def test_a_neighbour_s_side_in_the_other_direction_is_refused_each_way() -> None:
    """A stated side's direction is its width: an upstream neighbour without one is an
    input port's, and a downstream one with one an output port's (``channel-users``)."""
    model = kernel_model()
    for tensor, side, message in (
        (
            "hidden",
            "producer",
            "a channel's producer is an output port, and first.compute.packed.y is an input's",
        ),
        (
            "levels",
            "consumer",
            "a channel's consumer is an input port, and second.compute.packed.x is an output's",
        ),
    ):
        found = _flipped(model, tensor, side).validity(tensor)
        assert found.status == "refused"
        users = {
            (each.owner, each.message)
            for each in getattr(found.result, "findings", ())
            if each.code == "channel-users"
        }
        assert users == {("sides", message)}


def test_a_neighbour_whose_stored_choices_are_refused_is_refused_by_name() -> None:
    """``mm2``'s simd written raw past its domain: its stored choices are refused, so its
    side is no fact. The slice {mm1, mt1} is refused, naming the edge, the neighbour, its
    port and why, the replay's refusal its cause."""
    model = chain_model(4)
    write_raw(model, "mm2", "simd", 3)
    with refused(
        "neighbour-unaccepted",
        error=KernelOpError,
        owner="a1",
        tensor="a1",
        node="mm2",
        message=re.escape(
            "a1: its frozen neighbour mm2 (MatMul), at its port x, gives no accepted value: "
            "its stored choices are refused"
        ),
    ) as found:
        _slice(model, "mm1", "mt1")
    (finding,) = found.error.findings
    assert dict(finding.details)["port"] == "x"
    assert [cause.code for cause in finding.causes] == ["choices-refused"]


def test_a_neighbour_whose_side_waits_is_refused_by_name() -> None:
    """``mt0``'s pe cleared: its output stream waits on it, so it is no fact; and so with
    no stored choice at all (its kernel bound to its facts, nothing set)."""
    model = chain_model(4)
    cleared = copy.deepcopy(model)
    write_raw(cleared, "mt0", "pe", None)
    message = re.escape(
        "a0: its frozen neighbour mt0 (Thresholding), at its port output, gives no accepted "
        "value: its side waits on an open choice"
    )
    with refused("neighbour-unaccepted", error=KernelOpError, node="mt0", message=message):
        _slice(cleared, "mm1", "mt1")
    for key in kernel_op(model, model.graph.node[1]).choices():
        write_raw(model, "mt0", key, None)
    assert kernel_op(model, model.graph.node[1]).choices() == {}
    with refused("neighbour-unaccepted", error=KernelOpError, node="mt0", message=message):
        _slice(model, "mm1", "mt1")


def test_a_neighbour_open_on_a_choice_its_side_does_not_read_is_accepted() -> None:
    """``mt0``'s memory style cleared: its validity is open, and its output's side, which
    does not read it, is known. The side is the fact."""
    model = chain_model(4)
    write_raw(model, "mt0", "ram_style", None)
    mt0 = kernel_op(model, model.graph.node[1]).point()
    assert inspection.validity(mt0).status == "open"
    part = _slice(model, "mm1", "mt1")
    assert part.member("a0").producer == _frozen(mt0, "output", "mt0.output")


def test_a_tensor_read_inside_and_by_a_kernel_op_outside_is_refused_as_the_whole_reads_it() -> None:
    """hidden is read by ``a``, in the slice, and by ``b``, a KernelOp beyond it: two
    consumers, one a frozen neighbour. The slice reads it as the whole partition does, as
    fan-out (``tensor-fan-out``), and so does the slice of its producer alone."""
    model = fan_out_source().transform(ToKernelOps(TARGET))
    for names in (("first", "a", "b"), ("a",), ("first",)):
        with refused(
            "tensor-fan-out",
            error=KernelOpError,
            owner="fan_out",
            tensor="hidden",
            message="hidden is read by a, b",
        ):
            _slice(model, *names)


def _open(model: ModelWrapper, names: tuple[str, ...], tensor: str) -> Partition:
    """The slice of ``names``, its edge ``tensor``'s consumer the boundary, unstated (as an
    edge was before a frozen neighbour), its stored choices bound."""
    stored = _slice(model, *names).chosen()
    declared = declaration(model, [node for node in model.graph.node if node.name in names])
    each = declared.channels[tensor]
    channels = dict(declared.channels)
    channels[tensor] = ChannelDeclaration(each.producer, Side.boundary("m_axis_0"), each.facts)
    return partition(PartitionDeclaration(declared.kernels, channels, declared.shell), stored)


def test_a_downstream_neighbour_brings_the_input_chain_it_needs_into_the_slice() -> None:
    """Stage 7's chain, the slice {mm1, mt1}, ``mm2`` frozen: a1 needs the input generator
    ``mm2`` requires, as in the whole partition, where an open boundary takes whatever
    ``mt1`` sends (no chain). The edge channel's hops reach ``mm2`` from that chain's
    stage; they, its sizing and its resources are the whole partition's."""
    model = chain_model(4)
    whole = network(model, model.graph.node)
    part = _slice(model, "mm1", "mt1")
    assert part.boundary == ()
    assert _open(model, ("mm1", "mt1"), "a1").member("a1").query(Channel.input_chain) == (
        Available("none")
    )
    for name in ("a0", "a1"):
        assert part.member(name).query(Channel.input_chain) == Available("input_gen")
        assert part.neighbours(name) == whole.neighbours(name)
        assert part.hops(name) == whole.hops(name)
        assert part.query(name, Kernel.resources) == whole.query(name, Kernel.resources)
        assert fifo_sizing.size(part.member(name), part.neighbours(name), 1024) == (
            fifo_sizing.size(whole.member(name), whole.neighbours(name), 1024)
        )
    hops = part.hops("a1")
    assert isinstance(hops, OuterHops) and hops.consumer is not None
    stage = hops.consumer.source.instance
    assert stage is not None and stage.startswith("adapter.input_gen")
    assert hops.consumer.sink.instance == "^mm2.compute.packed"
    consumer = part.neighbours("a1").consumer
    assert consumer is not None and consumer.label == "mm2.compute.packed.x"


def test_a_slice_s_netlist_and_module_are_refused_by_their_code() -> None:
    """A frozen edge is a kernel's port outside the module, no AXIS port of it: the slice's
    netlist and module are refused, each edge by name (``slice-not-emitted``), not by a
    bare error."""
    model = chain_model(4)
    part = _slice(model, "mm1", "mt1")
    netlist, module = part.netlist(), part.module_of(PRODUCER)
    assert isinstance(netlist, Rejected) and isinstance(module, Rejected)
    for found in (netlist, module):
        assert [(f.code, f.owner, dict(f.details)["tensor"]) for f in found.findings] == [
            ("slice-not-emitted", "a0", "a0"),
            ("slice-not-emitted", "a1", "a1"),
        ]
    (first, _) = netlist.findings
    assert first.message == (
        "a slice is not emitted: its edge meets mt0.output, a frozen neighbour's port "
        "outside its module, and no AXIS port of it"
    )


def test_another_neighbour_is_another_design_space_of_the_same_class() -> None:
    """``mm2`` at another simd reads another stream: a1's fact changes, its class does not."""
    model = chain_model(4)
    part = _slice(model, "mm1", "mt1")
    write_raw(model, "mm2", "simd", 2)
    other = _slice(model, "mm1", "mt1")
    assert type(other.point) is type(part.point)
    assert other.member("a1").consumer != part.member("a1").consumer
