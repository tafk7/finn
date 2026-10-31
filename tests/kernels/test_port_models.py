# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Actual lean port values, local ownership, sparse demand, discovery and exposed ports."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
from qonnx.core.datatype import DataType
from support.conformance import KERNEL, design_member, place, samples
from support.kernel_specs import SPECS
from support.refusals import refused

from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    Group,
    Inapplicable,
    Param,
    Rejected,
    Select,
    Space,
    Unresolved,
    constraint,
    derived,
    design_space,
    inspection,
    reject,
    view,
)
from finn.dataflow.schedule import Index, Pace, Schedule
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd, vector_major
from finn.kernels.artifacts.abi import Direction, Endpoint
from finn.kernels.artifacts.module import Held
from finn.kernels.base import Kernel, PortPins
from finn.kernels.laws import (
    AXIS,
    AXIS_CLOCKING,
    NATIVE,
    NATIVE_CLOCKING,
    NATIVE_PROTOCOL,
    Advance,
    Axis,
    Native,
)
from finn.kernels.port import (
    LoopEnds,
    PhysicalPort,
    data_width,
    exposed_name,
    exposed_pins,
    exposed_side,
    held,
    pins,
    port_direction,
    stream_pins,
    stream_port,
    transport,
)
from finn.kernels.port_model import InPort, OutPort, StreamPort, exposed_ports, stream_ports
from finn.kernels.sides import Side
from finn.kernels.values.domains import Integer
from finn.kernels.vpc import VpcKernel

ELEMENT = ScalarEncoding(DataType["INT3"])
TENSOR = Tensor((2, 4), ELEMENT)
SEQUENCE = BeatSequence(vector_major((2, 4), 4))


class Ports(Space):
    tensor: Tensor | None = Param(default=TENSOR)
    ok: bool = Param(default=True)
    policy: Integer | None = Param(default=None)
    fold: int = Decision(domain=(1, 2, 4))

    @derived
    def sequence(self) -> BeatSequence:
        return BeatSequence(vector_major((2, 4), self.fold))

    input = InPort(
        label="input",
        transport=NATIVE,
        tensor=tensor,
        sequence=sequence,
        clocking=NATIVE_CLOCKING,
    )
    output = OutPort(
        label="output",
        transport=NATIVE,
        tensor=tensor,
        sequence=input.sequence,
        clocking=NATIVE_CLOCKING,
    )
    checked = InPort(
        label="checked",
        transport=Axis(bus="s_axis_x"),
        tensor=tensor,
        sequence=sequence,
        admits=policy,
        clocking=NATIVE_CLOCKING,
    )

    @constraint
    def parent_rule(self) -> bool:
        return self.ok

    @view
    def aggregate(self) -> PortPins:
        return self.input.stream_pins


@pytest.mark.parametrize("permuted,eager", [(False, False), (True, False), (False, True)])
def test_actual_native_values_parent_bindings_and_snapshot_history(
    permuted: bool, eager: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINN_SPACE_PERMUTE", "1" if permuted else "0")
    monkeypatch.setenv("FINN_SPACE_EAGER", "1" if eager else "0")
    point = design_space(Ports())
    assert inspection.statistics(point).owning_decisions == 1
    assert point.input.element == ELEMENT  # no fold/clock/timing required
    assert isinstance(point.query(Ports.input.stream), Unresolved)
    chosen = point.with_choices({Ports.fold: 4})
    for name, endpoint in (("input", Endpoint.TARGET), ("output", Endpoint.INITIATOR)):
        port = chosen.input if name == "input" else chosen.output
        old = PhysicalPort(name=name, endpoint=endpoint, transport=NATIVE)
        expected = Stream(ELEMENT, SEQUENCE)
        assert port.stream == expected
        assert port.width == data_width(old, expected) == 12
        assert port.side == Side.kernel(
            expected, name, 12 if endpoint is Endpoint.INITIATOR else None
        )
        assert port.physical_pins == pins(old, expected, NATIVE_CLOCKING)
        assert port.idle_pins == Held()
        found = stream_pins(old, expected, NATIVE_CLOCKING)
        assert not isinstance(found, Rejected)
        assert port.stream_pins == PortPins(name, found)
        assert transport(port.physical_spec(idle=False), expected, NATIVE_CLOCKING) == transport(
            old, expected, NATIVE_CLOCKING
        )
    refused_parent = chosen.with_facts({Ports.ok: False})
    assert isinstance(refused_parent.query(Ports.aggregate), Rejected)
    assert refused_parent.input.stream_pins == chosen.input.stream_pins
    refused_child = chosen.with_facts({Ports.policy: Integer(max_bits=2)})
    assert isinstance(refused_child.query(Ports.checked.stream_pins), Rejected)
    assert refused_child.checked.stream == chosen.checked.stream
    assert chosen.checked.stream_pins == point.with_choices({Ports.fold: 4}).checked.stream_pins
    assert isinstance(point.query(Ports.input.stream), Unresolved)
    info = next(p for p in inspection.members(chosen) if p.key == "checked.element_taken")
    assert (info.scope, info.owner) == ("checked", "checked.element_taken")


def test_open_owner_does_not_merge_unrelated_waits_into_actual_stream_inputs() -> None:
    class Open(Ports):
        unused: int = Decision(domain=(1, 2))

        @constraint
        def unrelated(self) -> bool:
            return self.unused > 0

    point = design_space(Open()).with_choices({Open.fold: 4})
    assert inspection.validity(point).status == "open"
    assert isinstance(point.query(Open.aggregate), Available)
    assert isinstance(point.query(Open.checked.physical_pins), Available)


def test_optional_idle_skips_refused_sequence_and_keeps_physical_metadata() -> None:
    class Optional(Space):
        @derived
        def bad(self) -> BeatSequence | Rejected:
            return reject("unused-sequence", "must not be demanded")

        port = InPort(
            transport=AXIS,
            label="set",
            dtype=DataType["INT3"],
            sequence=bad,
            clocking=NATIVE_CLOCKING,
        )

    point = design_space(Optional())
    assert point.port.element == ELEMENT
    assert not point.port.placed
    assert isinstance(point.query(Optional.port.stream), Inapplicable)
    assert isinstance(point.query(Optional.port.stream_pins), Inapplicable)
    old = PhysicalPort(name="set", endpoint=Endpoint.TARGET)
    assert point.port.physical_pins == pins(old, ELEMENT, NATIVE_CLOCKING)
    assert point.port.idle_pins == held(old, ELEMENT, NATIVE_CLOCKING)
    visited = inspection.explain(point, Optional.port.physical_pins)
    assert not any(n.declaration.key == "bad" for n in visited.nodes)


def test_an_idle_port_answers_its_stream_inapplicable_and_an_unindexed_one_binds_nothing() -> None:
    class Shape(Space):
        idle = InPort(
            clocking=AXIS_CLOCKING,
            transport=AXIS,
            label="idle",
            dtype=DataType["INT3"],
        )
        live = InPort(
            clocking=AXIS_CLOCKING,
            transport=AXIS,
            label="live",
            tensor=TENSOR,
            sequence=SEQUENCE,
        )

    point = design_space(Shape())
    assert not point.idle.placed
    for member in ("stream", "width", "side", "stream_pins", "access"):
        assert isinstance(point.query(getattr(Shape.idle, member)), Inapplicable), member
    assert point.live.binds is False
    assert isinstance(point.query(Shape.live.access), Inapplicable)
    assert inspection.statistics(point).owning_decisions == 0
    assert point.idle.element == ELEMENT
    ports = stream_ports(Shape)
    assert point.query(ports["live"].stream) == Available(Stream(ELEMENT, SEQUENCE))


@pytest.mark.parametrize("state", [None, False, True])
def test_real_child_presence_is_distinct_from_idle(state: bool | None) -> None:
    class Guarded(Space):
        enabled: bool = Decision(domain=(False, True))
        port = InPort(
            clocking=AXIS_CLOCKING,
            transport=AXIS,
            label="x",
            tensor=TENSOR,
            sequence=SEQUENCE,
            when=enabled,
        )

    point = design_space(Guarded())
    if state is not None:
        point = point.with_choices({Guarded.enabled: state})
    result = point.query(Guarded.port.physical_pins)
    assert isinstance(result, Unresolved if state is None else Available if state else Inapplicable)


def test_native_loop_markers_and_physical_refusals_use_actual_values() -> None:
    sequence = BeatSequence(SEQUENCE.form, markers=(LevelEnd(2),))

    class Marked(Space):
        port = OutPort(
            label="output",
            transport=NATIVE,
            tensor=TENSOR,
            sequence=sequence,
            clocking=NATIVE_CLOCKING,
            loop_ends=LoopEnds((LevelEnd(2),)),
        )

    point = design_space(Marked())
    old = PhysicalPort(
        name="output",
        endpoint=Endpoint.INITIATOR,
        transport=NATIVE,
        loop_ends=LoopEnds((LevelEnd(2),)),
    )
    expected = Stream(ELEMENT, sequence)
    assert point.port.physical_pins == pins(old, expected, NATIVE_CLOCKING)
    found = stream_pins(old, expected, NATIVE_CLOCKING)
    assert not isinstance(found, Rejected)
    assert point.port.stream_pins == PortPins("output", found)

    class Bad(Space):
        port = InPort(
            clocking=AXIS_CLOCKING,
            label="bad",
            tensor=TENSOR,
            sequence=sequence,
            transport=Native(names=("data", "valid", "ready")),
        )

    bad = design_space(Bad())
    assert isinstance(bad.query(Bad.port.physical_pins), Rejected)
    # Its transport cannot carry its stream's marker: the port's own constraint refuses the
    # design point (``presented``), with the carrier's refusal.
    validity = inspection.validity(bad)
    assert validity.status == "refused"
    assert isinstance(validity.result, Rejected)
    assert [finding.code for finding in validity.result.findings] == ["port-markers"]


def test_port_discovery_rebases_overrides_groups_and_stops_at_real_children() -> None:
    class Included(Group):
        nested = InPort(
            clocking=AXIS_CLOCKING,
            transport=AXIS,
            label="nested",
            tensor=TENSOR,
            sequence=SEQUENCE,
        )

    class Base(Space):
        first: StreamPort = InPort(
            clocking=AXIS_CLOCKING,
            transport=AXIS,
            label="first",
            tensor=TENSOR,
            sequence=SEQUENCE,
        )
        second: object = InPort(
            clocking=AXIS_CLOCKING,
            transport=AXIS,
            label="second",
            tensor=TENSOR,
            sequence=SEQUENCE,
        )

    class Child(Base):
        first = InPort(
            clocking=AXIS_CLOCKING,
            label="first",
            tensor=TENSOR,
            sequence=SEQUENCE,
            transport=Axis(bus="replaced"),
        )
        group = Included()
        real = Base()

    ports = stream_ports(Child)
    assert list(ports) == ["first", "second", "group.nested"]
    point = design_space(Child())
    assert point.query(ports["group.nested"].stream) == Available(Stream(ELEMENT, SEQUENCE))
    with pytest.raises(TypeError, match="node declaration"):
        stream_port(point, "first")


def test_port_children_are_not_hardware_kernels_or_member_netlists() -> None:
    class Host(Kernel):
        id = "test.port-host"
        p = InPort(label="p", tensor=TENSOR, sequence=SEQUENCE)

    point = design_space(Host())
    assert isinstance(point.p, Space)
    assert not isinstance(point.p, Kernel)
    assert point.netlists == ()


def test_view_bindings_remain_model_read_in_errors() -> None:
    class Bad(Space):
        @view
        def outward(self) -> BeatSequence:
            return SEQUENCE

        @derived
        def inward(self) -> BeatSequence:
            return self.outward

        port = InPort(transport=AXIS, label="bad", tensor=TENSOR, sequence=inward)

    with pytest.raises(DefinitionError, match="view"):
        design_space(Bad())


# -- the one construct: its advance rule, and what its kernel supplies ---------------------


@pytest.mark.parametrize(
    ("declared", "rule"),
    [
        (
            {
                "schedule": Schedule({Index("r"): 2, Index("c"): 4}),
                "index": (Index("r"), Index("c")),
            },
            Advance.SCHEDULED,
        ),
        ({"sequence": SEQUENCE}, Advance.STEPPED),
        ({"sequence": SEQUENCE, "governs": "input"}, Advance.SELECTED),
        ({"sequence": SEQUENCE, "sourced": True}, Advance.SOURCED),
    ],
)
def test_a_port_follows_the_advance_rule_its_declaration_gives(
    declared: dict[str, object], rule: Advance
) -> None:
    kind = OutPort if declared.get("sourced") else InPort
    port = cast(Any, kind)(
        label="p",
        transport=AXIS,
        clocking=AXIS_CLOCKING,
        tensor=TENSOR,
        **declared,
    )
    point = design_space(port)
    assert point.advance is rule
    assert (
        rule.law.id
        == {"scheduled": "L2", "stepped": "L3", "selected": "L4", "sourced": "L5"}[rule.value]
    )
    # A scheduled port has a pace, its schedule's; a port of another rule has none.
    pace = point.query(StreamPort.pace)
    if rule is Advance.SCHEDULED:
        assert isinstance(pace, Available)
        assert pace.value == Pace(cast(Schedule, declared["schedule"]), (), ())
    else:
        assert isinstance(pace, Inapplicable)


SCHED, BUF, SEL, SRC = Advance.SCHEDULED, Advance.STEPPED, Advance.SELECTED, Advance.SOURCED
LIBRARY_RULES = {
    # CONTRACT.md § 9: each module kernel's ports, by the rule its RTL follows.
    "PackedDotpKernel": {"x": SCHED, "w": SCHED, "y": SCHED},
    "Int8Dsp58DotpKernel": {"x": SCHED, "w": SCHED, "y": SCHED},
    "ThresholdingAxiKernel": {"input": SCHED, "output": SCHED, "set": SEL},
    "EltwiseKernel": {"lhs": SCHED, "rhs": SCHED, "result": SCHED},
    "TransposeKernel": {"input": SCHED, "output": SCHED},
    "MemStreamKernel": {"set": SEL, "output": SRC},
    "FifoKernel": {"input": BUF, "output": BUF},
    "VpcKernel": {"input": BUF, "output": BUF},
    "InputGeneratorKernel": {"input": BUF, "output": BUF},
    "PoolKernel": {"input": SCHED, "output": SCHED},
    "PwPolyKernel": {"input": SCHED, "output": SCHED},
}


@pytest.mark.parametrize("name", sorted(LIBRARY_RULES))
def test_library_ports_follow_their_rtl_s_advance_rules(name: str) -> None:
    """Every port of each module kernel, at its spec's first sampled design point, answers
    the advance rule CONTRACT.md § 9 states: evaluated, not re-derived."""
    (spec,) = [spec for spec in SPECS if spec.name == name]
    case = {key: value for key, value in spec.cases[spec.space]().items() if key != "reference"}
    given = {key: value for key, value in case.items() if key != "tolerance"}
    sample = samples(**given)[0]
    point = place(
        case["space_type"],
        sample,
        case["inputs"],
        case["outputs"],
        choices=case.get("choices", {}),
        facts=case.get("facts", {}),
    )
    kernel = design_member(point, KERNEL)
    assert {port: getattr(kernel, port).advance for port in stream_ports(type(kernel))} == (
        LIBRARY_RULES[name]
    )


def test_every_module_kernel_of_the_specs_has_its_rules_stated() -> None:
    module_kernels = {
        spec.name
        for spec in SPECS
        if spec.kernel.module_spec is not None or stream_ports(spec.kernel)
    }
    assert module_kernels == set(LIBRARY_RULES)


def test_a_kernel_gives_each_port_of_its_body_its_name_and_its_interface_protocol_s_clock() -> None:
    class Native_(Kernel):
        id = "test.native_port_clock"
        interface_protocol = NATIVE_PROTOCOL
        input = InPort(tensor=TENSOR, sequence=SEQUENCE)
        named = InPort(
            label="other",
            tensor=TENSOR,
            sequence=SEQUENCE,
        )

    point = design_space(Native_())
    assert point.input.label == "input" and point.input.clocking == NATIVE_CLOCKING
    assert point.named.label == "other"
    assert point.input.stream_pins.pins.transport.clock == "clk"
    # A port declared outside a kernel's body states both.
    with pytest.raises(DefinitionError, match="clocking is not supplied"):
        design_space(InPort(label="p", transport=AXIS, tensor=TENSOR, sequence=SEQUENCE))


def test_a_port_takes_its_kernel_s_interface_protocol_s_transport_where_it_states_none() -> None:
    """A port of a kernel's body that states no transport takes its kernel's interface
    protocol's (``Kernel.interface_protocol``): AXIS by default, native for a native kernel; a
    port with its own names states its own. A port outside a kernel's body states it: no
    interface protocol supplies it."""

    class Axis_(Kernel):
        id = "test.transport.axis"
        input = InPort(tensor=TENSOR, sequence=SEQUENCE)

    class Native_(Kernel):
        id = "test.transport.native"
        interface_protocol = NATIVE_PROTOCOL
        input = InPort(tensor=TENSOR, sequence=SEQUENCE)
        named = InPort(transport=AXIS, tensor=TENSOR, sequence=SEQUENCE)

    assert design_space(Axis_()).input.transport == AXIS
    native = design_space(Native_())
    assert native.input.transport == NATIVE
    assert native.named.transport == AXIS
    assert [bus.name for bus in native.named.physical_pins] == ["s_axis_named"]
    with pytest.raises(DefinitionError, match="transport is not supplied"):
        design_space(InPort(label="p", clocking=AXIS_CLOCKING, tensor=TENSOR, sequence=SEQUENCE))


class Leaf(Kernel):
    id = "test.exposed.leaf"
    input = InPort(tensor=TENSOR, sequence=SEQUENCE)
    output = OutPort(tensor=TENSOR, sequence=SEQUENCE)


class OtherLeaf(Leaf):
    id = "test.exposed.other_leaf"


class Exposing(Kernel):
    id = "test.exposed"
    core = Select(one=Leaf, other=OtherLeaf)
    x = core.input
    y = core.output
    kept = Leaf()
    z = kept.output
    w = kept.input


def test_a_kernel_exposes_its_member_s_port_under_the_alias_s_name() -> None:
    """An exposed port is its member's (one node, read by two names): its readouts are the
    member's port's, and its kernel presents its side and stream pins under the alias's
    name, its pins on the leaf the alias reaches (the selected case, or the child)."""
    point = design_space(Exposing()).with_choices({Exposing.core: "other"})
    assert list(exposed_ports(Exposing)) == ["x", "y", "z", "w"]
    assert stream_ports(Exposing) == {}  # none of its own
    assert inspection.path(point.x) == "core.other.input"
    assert point.query(stream_port(Exposing, "x").stream) == Available(point.core.input.stream)

    def side(port: str) -> Side:
        found = point.query(stream_port(Exposing, port).side)
        assert isinstance(found, Available), port
        return exposed_side(Exposing, port, found.value)

    def contract_of(port: str) -> PortPins:
        found = point.query(stream_port(Exposing, port).stream_pins)
        assert isinstance(found, Available) and isinstance(found.value, PortPins), port
        return exposed_pins(point, port, found.value)

    assert side("x") == replace(point.core.input.side, label="x")
    assert contract_of("x") == PortPins("x", point.core.input.stream_pins.pins, "core.other")
    assert contract_of("x").under("net") == PortPins(
        "net.x", point.core.input.stream_pins.pins, "net.core.other"
    )
    assert side("y") == replace(point.core.output.side, label="y")
    assert (contract_of("z").label, contract_of("z").instance) == ("z", "kept")
    assert side("w") == replace(point.kept.input.side, label="w")
    assert port_direction(Exposing, "x") is Direction.IN
    assert port_direction(Exposing, "y") is Direction.OUT
    assert port_direction(Exposing, "z") is Direction.OUT
    # Its advance rule is its member's port's, found by the alias's name too.
    assert point.query(stream_port(Exposing, "x").advance) == Available(Advance.STEPPED)
    assert point.query(stream_port(Exposing, "y").advance) == Available(point.core.output.advance)
    # Open, the case is not known, so neither is the port.
    assert isinstance(
        design_space(Exposing()).query(stream_port(Exposing, "x").stream_pins), Unresolved
    )
    # A kernel's own port is presented as it labels it.
    leaf = design_space(Leaf())
    assert exposed_name(Leaf, "input") is None
    assert exposed_pins(leaf, "input", leaf.input.stream_pins) == leaf.input.stream_pins


def test_an_exposed_port_is_a_stream_port_of_each_case_of_its_kernel_s_member() -> None:
    with refused("alias-foreign", error=DefinitionError):

        class Orphan(Kernel):
            id = "test.exposed.orphan"
            x = Leaf().input

    class InputOnly(Kernel):
        id = "test.exposed.input_only"
        input = InPort(tensor=TENSOR, sequence=SEQUENCE)

    with refused("alias-case-absent", error=DefinitionError, message=r"\['other'\]"):

        class Partial(Kernel):
            id = "test.exposed.partial"
            core = Select(one=Leaf, other=InputOnly)
            x = core.output

    # Its kind is one in every selection; its carrier may differ by case.
    class In_(Kernel):
        id = "test.exposed.in"
        port = InPort(tensor=TENSOR, sequence=SEQUENCE)

    class Turned(Kernel):
        id = "test.exposed.turned"
        port = OutPort(tensor=TENSOR, sequence=SEQUENCE)

    class Native_(Kernel):
        id = "test.exposed.native"
        port = InPort(transport=NATIVE, tensor=TENSOR, sequence=SEQUENCE)

    with refused("port-direction", error=DefinitionError, message="not all inputs or all outputs"):

        class Mixed(Kernel):
            id = "test.exposed.mixed"
            core = Select(i=In_, o=Turned)
            x = core.port

    class Carried(Kernel):
        id = "test.exposed.carried"
        core = Select(axis=In_, native=Native_)
        x = core.port

    assert port_direction(Carried, "x") is Direction.IN

    # A kind derived from a kind is its kind (RF2 M1's case: no direction is bound).
    class Tapped(OutPort):
        pass

    class Tapping(Kernel):
        id = "test.exposed.tapping"
        port = Tapped(tensor=TENSOR, sequence=SEQUENCE)

    class Outputs(Kernel):
        id = "test.exposed.outputs"
        core = Select(t=Turned, d=Tapping)
        x = core.port

    assert port_direction(Outputs, "x") is Direction.OUT

    # One port, one name: a second alias of it is refused (RF2 L2).
    with refused("port-exposed-twice", error=DefinitionError, message="exposes the port w"):

        class Twice(Kernel):
            id = "test.exposed.twice"
            kept = Leaf()
            w = kept.input
            w2 = kept.input

    # A node alias of no stream port is no port of its kernel.
    class Holder(Kernel):
        id = "test.exposed.holder"
        exposing = Exposing()
        leaf = exposing.kept
        x = exposing.x

    assert list(exposed_ports(Holder)) == ["x"]
    with pytest.raises(KeyError):
        stream_port(Holder, "leaf")


def test_an_exposed_port_reaches_through_a_wrapper_to_its_leaf() -> None:
    """A wrapper of a kernel exposing its member's port exposes the same port: one node,
    owned where it is declared, its pins on the leaf below both."""

    class Wrapper(Kernel):
        id = "test.exposed.wrapper"
        inner = Exposing()
        x = inner.x
        z = inner.z

    point = design_space(Wrapper()).with_choices({Wrapper.inner.core: "one"})
    assert [d.key for d in inspection.decisions(point)] == ["inner.core"]
    assert inspection.path(point.x) == "inner.core.one.input"
    found = point.query(stream_port(Wrapper, "x").stream_pins)
    assert isinstance(found, Available)
    pins = exposed_pins(point, "x", found.value)
    assert (pins.label, pins.instance) == ("x", "inner.core.one")
    assert exposed_pins(point, "z", point.z.stream_pins).instance == "inner.kept"
    assert port_direction(Wrapper, "x") is Direction.IN


def test_a_port_states_one_thing_its_advance_rule_follows_from() -> None:
    """One of a schedule, the port it governs (an input's), or sourced values (an
    output's); and a port governs another stream port of its kernel."""
    with refused("port-advance-ambiguous", error=DefinitionError, message="schedule and governs"):

        class Governed(Kernel):
            id = "test.advance.governed"
            input = InPort(
                tensor=TENSOR,
                schedule=Schedule({Index("r"): 2, Index("c"): 4}),
                index=(Index("r"), Index("c")),
                governs="other",
            )
            other = InPort(tensor=TENSOR, sequence=SEQUENCE)

    with refused("port-advance-ambiguous", error=DefinitionError, message="schedule and sourced"):

        class Stored(Kernel):
            id = "test.advance.stored"
            output = OutPort(
                tensor=TENSOR,
                schedule=Schedule({Index("r"): 2, Index("c"): 4}),
                index=(Index("r"), Index("c")),
                sourced=True,
            )

    for governs in ("no_such_port", "input"):
        with refused("port-governs-unknown", error=DefinitionError, message=repr(governs)):

            class Governs(Kernel):
                id = "test.advance.governs"
                input = InPort(tensor=TENSOR, sequence=SEQUENCE, governs=governs)

    class Governing(Kernel):
        id = "test.advance.governing"
        input = InPort(tensor=TENSOR, sequence=SEQUENCE)
        set = InPort(tensor=TENSOR, sequence=SEQUENCE, governs="input")

    assert design_space(Governing()).set.advance is Advance.SELECTED


def test_a_selector_s_group_is_the_governed_port_s_beats_over_its_own() -> None:
    """``group`` (R2, L4): the governed port's beats a frame over the selector's, each
    selector beat naming the set of that many; refused where they do not divide, the design
    point with it (``port-group``, its ``presented``); a port that governs none has none."""
    wide = Tensor((4, 4), ELEMENT)
    narrow = Tensor((1, 4), ELEMENT)

    class Grouped(Kernel):
        id = "test.group.grouped"
        input = InPort(tensor=wide, sequence=BeatSequence(vector_major((4, 4), 4)))
        set = InPort(tensor=narrow, sequence=BeatSequence(vector_major((1, 4), 4)), governs="input")

    class Ungrouped(Kernel):
        id = "test.group.ungrouped"
        input = InPort(
            tensor=Tensor((3, 4), ELEMENT), sequence=BeatSequence(vector_major((3, 4), 4))
        )
        set = InPort(tensor=TENSOR, sequence=SEQUENCE, governs="input")

    class Idle(Kernel):
        id = "test.group.idle"
        input = InPort(dtype=DataType["INT4"], sequence=SEQUENCE)
        set = InPort(tensor=TENSOR, sequence=SEQUENCE, governs="input")

    class Governing(Kernel):
        id = "test.group.governing"
        input = InPort(tensor=TENSOR, sequence=SEQUENCE)
        set = InPort(tensor=TENSOR, sequence=SEQUENCE, governs="input")

    point = design_space(Governing())
    assert point.set.group == 1
    assert isinstance(point.query(Governing.input.group), Inapplicable)
    assert design_space(Grouped()).set.group == 4
    for kind, message in ((Ungrouped, "2 beats a frame do not group input's 3"), (Idle, "idle")):
        found = design_space(kind()).query(kind.set.group)
        assert isinstance(found, Rejected) and [f.code for f in found.findings] == ["port-group"]
        assert message in found.findings[0].message
        validity = inspection.validity(design_space(kind()))
        assert validity.status == "refused", kind


def test_a_scheduled_port_presents_its_kernel_s_one_schedule() -> None:
    """A port of a kernel's body presents its kernel's
    ``schedule`` or no schedule (``port-schedule-foreign``), so its kernel's scheduled
    ports present one index point at each iteration and their paces count the same
    iterations; a subclass's override of ``schedule`` is the one its inherited ports read."""
    r, c = Index("r"), Index("c")

    class Scheduled(Kernel):
        id = "test.schedule.own"

        @derived
        def schedule(self) -> Schedule:
            return Schedule({r: 2, c: 4})

        input = InPort(tensor=TENSOR, schedule=schedule, index=(r, c))

    class Rows(Scheduled):
        @derived
        def schedule(self) -> Schedule:
            return Schedule({r: 2, c: 4}, {c: 2})

    for kind, factors in ((Scheduled, {}), (Rows, {c: 2})):
        point = design_space(kind())
        assert point.input.pace == Pace(Schedule({r: 2, c: 4}, factors), (), ())

    for given, named in ((Schedule({r: 2, c: 4}), "a Schedule of its own"), (None, "'other'")):
        with refused("port-schedule-foreign", error=DefinitionError, message=named):

            class Foreign(Kernel):
                id = "test.schedule.foreign"

                @derived
                def other(self) -> Schedule:
                    return Schedule({r: 2, c: 4})

                input = InPort(
                    tensor=TENSOR,
                    schedule=other if given is None else given,
                    index=(r, c),
                )


# -- a valid design point has a well-formed contract ------------------------------------------
#
# Refused when a kernel's class is made, what no constraint sees: a port's direction, its one
# presentation and its relation, and its kernel's schedule. At each design point, a placed
# port's own constraint (``presented``): its stream and its carrier.


def test_a_stream_port_is_an_input_or_an_output() -> None:
    """Its kind, ``InPort`` or ``OutPort``, is its direction, read-only (``direction``); a
    ``StreamPort`` of neither is ``INOUT``, refused when its kernel's class is made and,
    outside one, at its design point (``port-direction``). A stream's bus role follows from
    its kind: an input's AXIS bus is a target's (``s_axis_*``), an output's an initiator's
    (``m_axis_*``)."""
    with refused("port-direction", error=DefinitionError, message="input is a StreamPort"):

        class Neither(Kernel):
            id = "test.direction.neither"
            input = StreamPort(tensor=TENSOR, sequence=SEQUENCE)

    alone = design_space(
        StreamPort(
            label="p", clocking=AXIS_CLOCKING, transport=AXIS, tensor=TENSOR, sequence=SEQUENCE
        )
    )
    validity = inspection.validity(alone)
    assert validity.status == "refused"
    assert isinstance(validity.result, Rejected)
    assert [finding.code for finding in validity.result.findings] == ["port-direction"]

    point = design_space(Leaf())
    assert (point.input.direction, point.output.direction) == (Direction.IN, Direction.OUT)
    assert point.input.physical_spec(idle=False).endpoint is Endpoint.TARGET
    assert point.output.physical_spec(idle=False).endpoint is Endpoint.INITIATOR
    assert [bus.name for bus in point.input.physical_pins] == ["s_axis_input"]
    assert [bus.name for bus in point.output.physical_pins] == ["m_axis_output"]


def test_a_port_class_states_neither_its_direction_nor_its_advance() -> None:
    """Both are its kind's (RF3 M2): a port class overriding ``direction`` (an ``InPort``
    answering ``OUT``, so that its ABI's endpoint and its kernel's checks would disagree)
    is refused when it is made (``port-direction``), as is one overriding ``advance``
    (``port-advance``), or inheriting either from a class of its own; a kind derived from a
    kind keeps both."""
    with refused("port-direction", error=DefinitionError, message=r"Turned\.direction: "):

        class Turned(InPort):
            @derived
            def direction(self) -> Direction:
                return Direction.OUT

    with refused("port-advance", error=DefinitionError, message=r"Sourcing\.advance: "):

        class Sourcing(InPort):
            @derived
            def advance(self) -> Advance:
                return Advance.SOURCED

    with refused("port-advance", error=DefinitionError, message=r"Selecting\.advance: "):

        class Selecting(OutPort):
            @derived
            def advance(self) -> Advance:
                return Advance.SELECTED

    class Facing:
        direction = Direction.OUT

    with refused("port-direction", error=DefinitionError, message=r"Facing\.direction: "):

        class Faced(Facing, InPort):
            pass

    class Kept(OutPort):
        pass

    port = design_space(
        Kept(label="k", clocking=AXIS_CLOCKING, transport=AXIS, tensor=TENSOR, sequence=SEQUENCE)
    )
    assert (port.direction, port.advance) == (Direction.OUT, Advance.STEPPED)


def test_a_port_binds_exactly_one_presentation() -> None:
    """Its kernel's schedule's projection or a given sequence, never neither or both, and a
    projection's parameters only with a schedule (``port-presentation``)."""
    r, c = Index("r"), Index("c")

    class Scheduled(Kernel):
        id = "test.presentation.scheduled"

        @derived
        def schedule(self) -> Schedule:
            return Schedule({r: 2, c: 4})

    with refused("port-presentation", error=DefinitionError, message="input binds none"):

        class Neither(Kernel):
            id = "test.presentation.neither"
            input = InPort(tensor=TENSOR)

    with refused(
        "port-presentation", error=DefinitionError, message="input binds schedule and sequence"
    ):

        class Two(Scheduled):
            id = "test.presentation.both"
            input = InPort(
                transport=AXIS,
                tensor=TENSOR,
                schedule=Scheduled.schedule,
                index=(r, c),
                sequence=SEQUENCE,
            )

    with refused(
        "port-presentation", error=DefinitionError, message="binds lanes, reshaped with a given"
    ):

        class Projected(Kernel):
            id = "test.presentation.projected"
            input = InPort(
                tensor=TENSOR,
                sequence=SEQUENCE,
                lanes=(c,),
                reshaped=False,
            )


def test_a_port_s_relation_is_its_kind_s() -> None:
    """``governs`` is an input's, ``sourced`` an output's: a port of the other kind has no
    such parameter, so neither mypy (strict, ``call-arg``) nor the engine (``unknown
    members``) admits one."""
    with pytest.raises(DefinitionError, match=r"unknown members \['governs'\]"):
        OutPort(tensor=TENSOR, sequence=SEQUENCE, governs="input")  # type: ignore[call-arg]
    with pytest.raises(DefinitionError, match=r"unknown members \['sourced'\]"):
        InPort(tensor=TENSOR, sequence=SEQUENCE, sourced=True)  # type: ignore[call-arg]
    with pytest.raises(DefinitionError, match=r"unknown members \['dtype'\]"):
        OutPort(tensor=TENSOR, sequence=SEQUENCE, dtype=DataType["INT4"])  # type: ignore[call-arg]


def test_a_port_s_relation_is_a_literal() -> None:
    """``sourced`` and ``governs`` are bound to literals (``port-advance-ambiguous``): the
    advance rule is its declaration's."""
    with refused("port-advance-ambiguous", error=DefinitionError, message="binds sourced to"):

        class StoredFact(Kernel):
            id = "test.relation.stored_fact"
            memory: bool = Param(default=True)
            output = OutPort(tensor=TENSOR, sequence=SEQUENCE, sourced=memory)

    with refused("port-advance-ambiguous", error=DefinitionError, message="binds governs to"):

        class GovernsFact(Kernel):
            id = "test.relation.governs_fact"
            governed: str = Param(default="input")
            input = InPort(tensor=TENSOR, sequence=SEQUENCE)
            set = InPort(tensor=TENSOR, sequence=SEQUENCE, governs=governed)


def test_a_scheduled_port_s_kernel_states_a_schedule_never_inapplicable() -> None:
    """A port presenting its kernel's schedule where the kernel states none
    (``port-schedule-unstated``), or one whose declared answer admits ``Inapplicable``, its
    own or a subclass's (``port-schedule-inapplicable``): its stream would be inapplicable,
    which no constraint refuses."""
    r, c = Index("r"), Index("c")

    with refused("port-schedule-unstated", error=DefinitionError, message=r"\['input'\]"):

        class Unstated(Kernel):
            id = "test.schedule.unstated"
            input = InPort(
                tensor=TENSOR,
                schedule=Kernel.schedule,
                index=(r, c),
            )

    with refused("port-schedule-inapplicable", error=DefinitionError, message="Inapplicable"):

        class Declared(Kernel):
            id = "test.schedule.declared_inapplicable"

            @derived
            def schedule(self) -> Schedule | Inapplicable:
                return Inapplicable()

            input = InPort(
                tensor=TENSOR,
                schedule=schedule,
                index=(r, c),
            )

    class Scheduled(Kernel):
        id = "test.schedule.stated"

        @derived
        def schedule(self) -> Schedule | Rejected:
            return Schedule({r: 2, c: 4})

        input = InPort(tensor=TENSOR, schedule=schedule, index=(r, c))

    with refused("port-schedule-inapplicable", error=DefinitionError, message="Inapplicable"):

        class Widened(Scheduled):
            id = "test.schedule.widened"

            @derived
            def schedule(self) -> Schedule | Inapplicable:
                return Inapplicable()


def test_a_placed_port_s_presentation_is_accepted_and_carried_or_its_point_is_refused() -> None:
    """A placed port's presentation (``presented``): a schedule's projection that does not
    project, or a stream its transport cannot carry (a vpc presenting the same marker on
    both sides, which its native pins carry no marker of), refuses the design point, with
    the port's own refusal; an idle port presents nothing."""
    r, c = Index("r"), Index("c")

    class Projection(Kernel):
        id = "test.presented.projection"
        foreign: bool = Decision(domain=(False, True))

        @derived
        def schedule(self) -> Schedule:
            return Schedule({r: 2, c: 4})

        @derived
        def index(self) -> tuple[Index, ...]:
            return (Index("q"), c) if self.foreign else (r, c)

        input = InPort(tensor=TENSOR, schedule=schedule, index=index)

    space = design_space(Projection())
    assert inspection.validity(space.with_choices({Projection.foreign: False})).status == "valid"
    with pytest.raises(Exception, match="would break input.stream"):
        space.with_choices({Projection.foreign: True})

    marked = BeatSequence(vector_major((4, 6), 2), markers=(LevelEnd(3),))
    tensor = Tensor((4, 6), ScalarEncoding(DataType["INT4"]))
    validity = inspection.validity(
        design_space(VpcKernel(input_tensor=tensor, arriving=marked, leaving=marked))
    )
    assert validity.status == "refused"
    assert {piece.owner for piece in validity.pieces if isinstance(piece.result, Rejected)} == {
        "input.presented",
        "output.presented",
    }
    assert isinstance(validity.result, Rejected)
    assert {finding.code for finding in validity.result.findings} == {"port-markers"}

    class Idle(Kernel):
        id = "test.presented.idle"
        input = InPort(dtype=DataType["INT3"], sequence=SEQUENCE)

    point = design_space(Idle())
    assert inspection.validity(point).status == "valid"
    assert point.query(cast(Any, Idle.input.presented)) == Available(True)
