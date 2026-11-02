# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A slice's frozen neighbours: an edge of a selection whose node beyond it is a KernelOp.

The edge channel's side there is the neighbour's port's side at its stored choices, given
as a fact (``Side.neighbour``): an upstream neighbour's output side, its stream, label and
word width, a downstream one's input side, its stream and label. Only an accepted value is
a fact, so a neighbour whose stored choices or kernel are refused, or whose side is not
known at them, is refused by name (``neighbour-unaccepted``). The edge channel is the
slice's: its adapters, its hops to the neighbour, its sizing (the neighbour's pace) and its
resources derive as the whole partition's do; its netlist and module are refused
(``slice-not-emitted``): a slice is not emitted."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from support.configure import composition, write_raw
from support.helpers import beside, root_facts
from support.models import TARGET, chain_model, fan_out_source, kernel_model
from support.refusals import refused

from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import CUT_NAME
from finn.core.space import (
    Available,
    Decision,
    Param,
    Rejected,
    derived,
    design_space,
    inspection,
    view,
)
from finn.core.space.domains import finite
from finn.custom_op.kernels import network as frozen_network
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import declaration
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels import fifo_sizing
from finn.kernels.artifacts.abi import Direction
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel, Neighbours, OuterHops
from finn.kernels.composer import ChannelDeclaration, DataflowCoreDeclaration, compose
from finn.kernels.dataflow_core import DataflowCoreKernel, bound
from finn.kernels.explore import CoreComposition
from finn.kernels.laws import NATIVE_PROTOCOL, Standing, StatedRate
from finn.kernels.library import ModuleSpec
from finn.kernels.port import stream_port
from finn.kernels.port_model import InPort
from finn.kernels.sides import Side, SideKind
from finn.platform.capabilities import Resources
from finn.transformation.kernels import ToKernelOps


def _slice(model: ModelWrapper, *names: str) -> CoreComposition:
    """The partition of the nodes ``names`` of ``model``, in graph order."""
    return composition(model, [node for node in model.graph.node if node.name in names])


def _side(point: Kernel, port: str) -> Side:
    """A KernelOp's port's side, its own or exposed."""
    found = point.query(cast(Any, stream_port(type(point), port).side))
    assert isinstance(found, Available) and isinstance(found.value, Side)
    return found.value


def _frozen(point: Kernel, port: str, label: str) -> Side:
    """The actual port side as a frozen neighbour's, labelled ``label``."""
    found = _side(point, port)
    assert found.stream is not None and found.direction is not None
    return Side.neighbour(found.direction, found.stream, label, found.width)


def test_an_upstream_neighbour_gives_an_output_side_and_a_downstream_one_an_input_side() -> None:
    """``Thresholding_0`` alone: ``MatMul_0`` produces MatMul_0_out0, ``MatMul_1`` reads
    Thresholding_0_out0. Each edge's side is the neighbour's port's at its stored choices, labelled
    below it: MatMul_0_out0's producer an output port's (with its width), Thresholding_0_out0's
    consumer an input port's (none). No edge is a port of the slice, and both channels read
    valid."""
    model = kernel_model()
    alone = _slice(model, "Thresholding_0")
    assert alone.boundary == ()
    nodes = {node.name: node for node in model.graph.node}
    first = kernel_op(model, nodes["MatMul_0"]).point()
    second = kernel_op(model, nodes["MatMul_1"]).point()
    hidden, levels = alone.member("MatMul_0_out0"), alone.member("Thresholding_0_out0")
    assert hidden.producer == _frozen(first, "y", "MatMul_0.y")
    assert hidden.producer.direction is Direction.OUT and hidden.producer.width is not None
    assert levels.consumer == _frozen(second, "x", "MatMul_1.x")
    assert levels.consumer.direction is Direction.IN and levels.consumer.width is None
    assert hidden.consumer.kind is levels.producer.kind is SideKind.KERNEL
    assert inspection.validity(hidden).status == inspection.validity(levels).status == "valid"


def _flipped(model: ModelWrapper, tensor: str, side: str) -> CoreComposition:
    """``Thresholding_0`` alone, declared with ``tensor``'s ``side`` (a neighbour's) given the
    other direction, its width dropped (an input side states none); its stored choices
    bound."""
    stored = _slice(model, "Thresholding_0").chosen()
    declared = declaration(
        model, [node for node in model.graph.node if node.name == "Thresholding_0"], name=CUT_NAME
    )
    each = declared.channels[tensor]
    given = getattr(each, side)
    assert isinstance(given, Side) and given.kind is SideKind.NEIGHBOUR
    other = Direction.IN if given.direction is Direction.OUT else Direction.OUT
    turned = replace(given, direction=other, width=None)
    sides = {"producer": each.producer, "consumer": each.consumer, side: turned}
    channels = dict(declared.channels)
    channels[tensor] = ChannelDeclaration(sides["producer"], sides["consumer"], each.facts)
    return _redeclared(declared, channels, declared.frozen, stored)


def _redeclared(
    declared: DataflowCoreDeclaration,
    channels: dict[str, ChannelDeclaration],
    frozen: Mapping[str, Neighbours],
    stored: Mapping[tuple[str, str], object],
) -> CoreComposition:
    """``declared`` with ``channels`` and ``frozen`` in place of its own, composed and its
    ``stored`` choices bound."""
    changed = DataflowCoreDeclaration(
        declared.kernels,
        channels,
        name=declared.name,
        order=declared.order,
        owners=declared.owners,
        frozen=frozen,
    )
    return CoreComposition(bound(compose(changed), stored).point)


def test_a_neighbour_s_side_in_the_other_direction_is_refused_each_way() -> None:
    """A stated side's direction is its port's kind, read by the channel: an upstream
    neighbour's input side, and a downstream neighbour's output side, are refused
    (``channel-users``)."""
    model = kernel_model()
    for tensor, side, message in (
        (
            "MatMul_0_out0",
            "producer",
            "a channel's producer is an output port, and MatMul_0.y is an input's",
        ),
        (
            "Thresholding_0_out0",
            "consumer",
            "a channel's consumer is an input port, and MatMul_1.x is an output's",
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
    """``MatMul_2``'s simd written raw past its domain: its stored choices are refused, so its
    side is no fact. The slice {MatMul_1, Thresholding_1} is refused, naming the edge, the
    neighbour, its port and why, the replay's refusal its cause."""
    model = chain_model(4)
    write_raw(model, "MatMul_2", "simd", 3)
    with refused(
        "neighbour-unaccepted",
        error=KernelOpError,
        owner="Thresholding_1_out0",
        tensor="Thresholding_1_out0",
        node="MatMul_2",
        message=re.escape(
            "Thresholding_1_out0: its frozen neighbour MatMul_2 (MatMul), at its port x, gives no "
            "accepted value: its stored choices are refused"
        ),
    ) as found:
        _slice(model, "MatMul_1", "Thresholding_1")
    (finding,) = found.error.findings
    assert dict(finding.details)["port"] == "x"
    assert [cause.code for cause in finding.causes] == ["choices-refused"]


def test_a_neighbour_whose_side_waits_is_refused_by_name() -> None:
    """``Thresholding_0``'s pe cleared: its output stream waits on it, so it is no fact; and so with
    no stored choice at all (its kernel bound to its facts, nothing set)."""
    model = chain_model(4)
    cleared = copy.deepcopy(model)
    write_raw(cleared, "Thresholding_0", "pe", None)
    message = re.escape(
        "Thresholding_0_out0: its frozen neighbour Thresholding_0 (Thresholding), at its port "
        "output, gives no accepted value: its side waits on an open choice"
    )
    with refused(
        "neighbour-unaccepted", error=KernelOpError, node="Thresholding_0", message=message
    ):
        _slice(cleared, "MatMul_1", "Thresholding_1")
    for key in kernel_op(model, model.graph.node[1]).choices():
        write_raw(model, "Thresholding_0", key, None)
    assert kernel_op(model, model.graph.node[1]).choices() == {}
    with refused(
        "neighbour-unaccepted", error=KernelOpError, node="Thresholding_0", message=message
    ):
        _slice(model, "MatMul_1", "Thresholding_1")


def test_a_neighbour_open_on_a_choice_its_side_does_not_read_is_accepted() -> None:
    """``Thresholding_0``'s memory style cleared: its validity is open, and its output's side, which
    does not read it, is known. The side is the fact."""
    model = chain_model(4)
    write_raw(model, "Thresholding_0", "ram_style", None)
    mt0 = kernel_op(model, model.graph.node[1]).point()
    assert inspection.validity(mt0).status == "open"
    part = _slice(model, "MatMul_1", "Thresholding_1")
    assert part.member("Thresholding_0_out0").producer == _frozen(
        mt0, "output", "Thresholding_0.output"
    )


def test_a_tensor_read_inside_and_by_a_kernel_op_outside_is_refused_as_the_whole_reads_it() -> None:
    """MatMul_0_out0 is read by ``Thresholding_0``, in the slice, and by ``Thresholding_1``, a
    KernelOp beyond it: two consumers, one a frozen neighbour. The slice reads it as the whole
    partition does, as fan-out (``tensor-fan-out``), and so does the slice of its producer alone."""
    model = fan_out_source().transform(ToKernelOps(TARGET))
    for names in (
        ("MatMul_0", "Thresholding_0", "Thresholding_1"),
        ("Thresholding_0",),
        ("MatMul_0",),
    ):
        with refused(
            "tensor-fan-out",
            error=KernelOpError,
            owner="fan_out",
            tensor="MatMul_0_out0",
            message="MatMul_0_out0 is read by Thresholding_0, Thresholding_1",
        ):
            _slice(model, *names)


def _open(model: ModelWrapper, names: tuple[str, ...], tensor: str) -> CoreComposition:
    """The slice of ``names``, its edge ``tensor``'s consumer the boundary, unstated (as an
    edge was before a frozen neighbour), its stored choices bound."""
    stored = _slice(model, *names).chosen()
    declared = declaration(
        model, [node for node in model.graph.node if node.name in names], name=CUT_NAME
    )
    each = declared.channels[tensor]
    channels = dict(declared.channels)
    channels[tensor] = ChannelDeclaration(each.producer, Side.boundary("m_axis_0"), each.facts)
    frozen = {name: each for name, each in declared.frozen.items() if name != tensor}
    return _redeclared(declared, channels, frozen, stored)


def test_a_downstream_neighbour_brings_the_input_chain_it_needs_into_the_slice() -> None:
    """Stage 7's chain, the slice {MatMul_1, Thresholding_1}, ``MatMul_2`` frozen:
    Thresholding_1_out0 needs the input generator ``MatMul_2`` requires, as in the whole partition,
    where an open boundary takes whatever ``Thresholding_1`` sends (no chain). The edge channel's
    hops reach ``MatMul_2`` from that chain's stage; they, its sizing and its resources are the
    whole partition's."""
    model = chain_model(4)
    whole = composition(model, model.graph.node)
    part = _slice(model, "MatMul_1", "Thresholding_1")
    assert part.boundary == ()
    assert _open(model, ("MatMul_1", "Thresholding_1"), "Thresholding_1_out0").member(
        "Thresholding_1_out0"
    ).query(Channel.input_chain) == (Available("none"))
    for name in ("Thresholding_0_out0", "Thresholding_1_out0"):
        assert part.member(name).query(Channel.input_chain) == Available("input_gen")
        assert beside(part, name) == beside(whole, name)
        assert part.hops(name) == whole.hops(name)
        assert part.query(name, Kernel.resources) == whole.query(name, Kernel.resources)
        assert fifo_sizing.size(part.member(name), beside(part, name), 1024) == (
            fifo_sizing.size(whole.member(name), beside(whole, name), 1024)
        )
    hops = part.hops("Thresholding_1_out0")
    assert isinstance(hops, OuterHops) and hops.consumer is not None
    stage = hops.consumer.source.instance
    assert stage is not None and stage.startswith("adapter.input_gen")
    assert hops.consumer.sink.instance == "^MatMul_2.compute.packed"
    consumer = beside(part, "Thresholding_1_out0").consumer
    assert consumer is not None and consumer.label == "MatMul_2.x"


def test_a_slice_s_netlist_and_module_are_refused_by_their_code() -> None:
    """A frozen edge is a kernel's port outside the module, no AXIS port of it: the slice's
    netlist and module are refused, each edge by name (``slice-not-emitted``), not by a
    bare error."""
    model = chain_model(4)
    part = _slice(model, "MatMul_1", "Thresholding_1")
    netlist, module = part.netlist(), part.core_answer(DataflowCoreKernel.module)
    assert isinstance(netlist, Rejected) and isinstance(module, Rejected)
    for found in (netlist, module):
        assert [(f.code, f.owner, dict(f.details)["tensor"]) for f in found.findings] == [
            ("slice-not-emitted", "Thresholding_0_out0", "Thresholding_0_out0"),
            ("slice-not-emitted", "Thresholding_1_out0", "Thresholding_1_out0"),
        ]
    (first, _) = netlist.findings
    assert first.message == (
        "a slice is not emitted: its edge meets Thresholding_0.output, a frozen neighbour's port "
        "outside its module, and no AXIS port of it"
    )


def test_another_neighbour_is_another_design_space_of_the_same_class() -> None:
    """``MatMul_2`` at another simd reads another stream: Thresholding_1_out0's fact changes, its
    class does not. The fact is the port beside Thresholding_1_out0 at ``MatMul_2`` and its pace
    (``Thresholding_1_out0-frozen``), and Thresholding_1_out0's neighbours, which FIFO sizing reads,
    follow it."""
    model = chain_model(4)
    part = _slice(model, "MatMul_1", "Thresholding_1")
    write_raw(model, "MatMul_2", "simd", 2)
    other = _slice(model, "MatMul_1", "Thresholding_1")
    assert type(other.point) is type(part.point)
    assert (
        other.member("Thresholding_1_out0").consumer != part.member("Thresholding_1_out0").consumer
    )
    before, after = (
        root_facts(part.point)["Thresholding_1_out0-frozen"],
        root_facts(other.point)["Thresholding_1_out0-frozen"],
    )
    assert isinstance(before, Neighbours) and isinstance(after, Neighbours)
    assert after.consumer != before.consumer
    assert after.consumer_pace != before.consumer_pace
    for network, given in ((part, before), (other, after)):
        found = beside(network, "Thresholding_1_out0")
        assert (found.consumer, found.consumer_pace) == (given.consumer, given.consumer_pace)


#: A rate stated on a four-beat input: its beats as its words arrive, in 11 cycles a frame.
_RATE = StatedRate("input", (1, 3, 5, 7), 11, Standing.MEASURED, "a test's trace", "its words")
_TENSOR = Tensor((4, 1), ScalarEncoding(DataType["INT4"]))


class _Rated(Kernel):
    """A stage stating its input's rate where ``stating`` chooses so, as ``rate`` gives it."""

    id = "test.frozen.rated"
    interface_protocol = NATIVE_PROTOCOL
    module_spec = ModuleSpec(
        "probe", (CopiedSource("finnlib", "rtl/probe.sv", provides=("module:probe",)),)
    )
    input = InPort(tensor=_TENSOR, sequence=BeatSequence(vector_major((4, 1), 1)))
    rate: StatedRate = Param()
    stating: bool = Decision(domain=finite((False, True)))

    @derived
    def stated_rate(self) -> StatedRate | None:
        stated: StatedRate = self.rate
        return stated if self.stating else None

    @view
    def parameters(self) -> dict[str, int | str]:
        return {}

    @view
    def resource_use(self) -> Resources:
        return Resources()


class _Stating(_Rated):
    """``_Rated``, stating its rate wherever."""

    id = "test.frozen.stating"

    @derived
    def stated_rate(self) -> StatedRate | None:
        stated: StatedRate = self.rate
        return stated


def test_a_neighbour_whose_pace_in_cycles_is_not_known_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its kernel's stated rate refused (``kernel-rate-port``), or waiting on an open choice:
    its pace in cycles is no fact, so neither is the neighbour, its causes kept. A stated
    rate is contract (CONTRACT.md L6): refused, it refuses its kernel's design point, so the
    neighbour is refused as its kernel is, before its pace is read. A pace answered None (a
    rate not stated) is no pace, and a stated one the neighbour's. The KernelOp is stood in
    for by its design space (``kernel_op``), the reader is the network's."""
    node = helper.make_node("Rated", ["edge"], ["out"], name="Rated_0")

    def frozen(point: Kernel) -> Any:
        op = SimpleNamespace(
            outputs=(),
            ports=("input",),
            wiring={"input": SimpleNamespace(port="input")},
            op_type="Rated",
            point=lambda: point,
        )
        monkeypatch.setattr(frozen_network, "kernel_op", lambda model, node: op)
        return frozen_network._neighbour(cast(ModelWrapper, None), "edge", "consumer", node)

    space = design_space(_Rated(rate=_RATE))
    for point, why, code in (
        (
            design_space(_Stating(rate=replace(_RATE, times=(1, 3, 5)))),
            "its kernel is refused",
            "kernel-rate-port",
        ),
        (space, "its pace in cycles waits on an open choice", "decision-unassigned"),
    ):
        with refused(
            "neighbour-unaccepted", error=KernelOpError, node="Rated_0", message=re.escape(why)
        ) as found:
            frozen(point)
        (finding,) = found.error.findings
        assert code in {cause.code for cause in finding.causes}
    assert frozen(space.with_choices({_Rated.stating: False}))[2] is None
    assert frozen(space.with_choices({_Rated.stating: True}))[2] == _RATE
