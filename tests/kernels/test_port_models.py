# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Actual lean port values, local ownership, sparse demand, discovery and served ports."""

from __future__ import annotations

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
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.module import Held
from finn.kernels.base import Kernel, PortContract
from finn.kernels.laws import AXIS, AXIS_CLOCKING, NATIVE, NATIVE_CLOCKING, Advance, Axis, Native
from finn.kernels.port import (
    LoopEnds,
    PhysicalPort,
    contract,
    data_width,
    held,
    pins,
    port_contract,
    port_endpoint,
    port_side,
    transport,
)
from finn.kernels.port_model import StreamPort, served, stream_ports
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

    input = StreamPort(
        label="input",
        endpoint=Endpoint.TARGET,
        transport=NATIVE,
        tensor=tensor,
        sequence=sequence,
        clocking=NATIVE_CLOCKING,
    )
    output = StreamPort(
        label="output",
        endpoint=Endpoint.INITIATOR,
        transport=NATIVE,
        tensor=tensor,
        sequence=input.sequence,
        clocking=NATIVE_CLOCKING,
    )
    checked = StreamPort(
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
    def aggregate(self) -> PortContract:
        return self.input.contract


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
        found = contract(old, expected, NATIVE_CLOCKING)
        assert not isinstance(found, Rejected)
        assert port.contract == PortContract(name, found)
        assert transport(port.physical_spec(idle=False), expected, NATIVE_CLOCKING) == transport(
            old, expected, NATIVE_CLOCKING
        )
    refused_parent = chosen.with_facts({Ports.ok: False})
    assert isinstance(refused_parent.query(Ports.aggregate), Rejected)
    assert refused_parent.input.contract == chosen.input.contract
    refused_child = chosen.with_facts({Ports.policy: Integer(max_bits=2)})
    assert isinstance(refused_child.query(Ports.checked.contract), Rejected)
    assert refused_child.checked.stream == chosen.checked.stream
    assert chosen.checked.contract == point.with_choices({Ports.fold: 4}).checked.contract
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

        port = StreamPort(
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
    assert isinstance(point.query(Optional.port.contract), Inapplicable)
    old = PhysicalPort(name="set", endpoint=Endpoint.TARGET)
    assert point.port.physical_pins == pins(old, ELEMENT, NATIVE_CLOCKING)
    assert point.port.idle_pins == held(old, ELEMENT, NATIVE_CLOCKING)
    visited = inspection.explain(point, Optional.port.physical_pins)
    assert not any(n.declaration.key == "bad" for n in visited.nodes)


def test_an_idle_port_answers_its_stream_inapplicable_and_an_unindexed_one_binds_nothing() -> None:
    class Shape(Space):
        idle = StreamPort(
            clocking=AXIS_CLOCKING, transport=AXIS, label="idle", dtype=DataType["INT3"]
        )
        live = StreamPort(
            clocking=AXIS_CLOCKING, transport=AXIS, label="live", tensor=TENSOR, sequence=SEQUENCE
        )

    point = design_space(Shape())
    assert not point.idle.placed
    for member in ("stream", "width", "side", "contract", "access"):
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
        port = StreamPort(
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
        port = StreamPort(
            label="output",
            endpoint=Endpoint.INITIATOR,
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
    found = contract(old, expected, NATIVE_CLOCKING)
    assert not isinstance(found, Rejected)
    assert point.port.contract == PortContract("output", found)

    class Bad(Space):
        port = StreamPort(
            clocking=AXIS_CLOCKING,
            label="bad",
            tensor=TENSOR,
            sequence=sequence,
            transport=Native(names=("data", "valid", "ready")),
        )

    bad = design_space(Bad())
    assert isinstance(bad.query(Bad.port.physical_pins), Rejected)
    assert inspection.validity(bad).status == "valid"  # no invented physical model rule


def test_port_discovery_rebases_overrides_groups_and_stops_at_real_children() -> None:
    class Included(Group):
        nested = StreamPort(
            clocking=AXIS_CLOCKING, transport=AXIS, label="nested", tensor=TENSOR, sequence=SEQUENCE
        )

    class Base(Space):
        first: StreamPort = StreamPort(
            clocking=AXIS_CLOCKING,
            transport=AXIS,
            label="first",
            tensor=TENSOR,
            sequence=SEQUENCE,
        )
        second: object = StreamPort(
            clocking=AXIS_CLOCKING,
            transport=AXIS,
            label="second",
            tensor=TENSOR,
            sequence=SEQUENCE,
        )

    class Child(Base):
        first = StreamPort(
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
        port_side(point, "first")


def test_port_children_are_not_hardware_kernels_or_member_netlists() -> None:
    class Host(Kernel):
        id = "test.port-host"
        p = StreamPort(transport=AXIS, label="p", tensor=TENSOR, sequence=SEQUENCE)

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

        port = StreamPort(transport=AXIS, label="bad", tensor=TENSOR, sequence=inward)

    with pytest.raises(DefinitionError, match="view"):
        design_space(Bad())


# -- the one construct: its advance rule, and what its kernel supplies (W4) ----------------


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
        ({"sequence": SEQUENCE}, Advance.BUFFERED),
        ({"sequence": SEQUENCE, "governs": "input"}, Advance.SELECTED),
        ({"sequence": SEQUENCE, "stored": True}, Advance.SOURCED),
    ],
)
def test_a_port_follows_the_advance_rule_its_declaration_gives(
    declared: dict[str, object], rule: Advance
) -> None:
    port = cast(Any, StreamPort)(
        label="p", transport=AXIS, clocking=AXIS_CLOCKING, tensor=TENSOR, **declared
    )
    point = design_space(port)
    assert point.advance is rule
    assert (
        rule.law.id
        == {"scheduled": "L2", "buffered": "L3", "selected": "L4", "sourced": "L5"}[rule.value]
    )
    # A scheduled port has a pace, its schedule's; a port of another rule has none (W6).
    pace = point.query(StreamPort.pace)
    if rule is Advance.SCHEDULED:
        assert isinstance(pace, Available)
        assert pace.value == Pace(cast(Schedule, declared["schedule"]), (), ())
    else:
        assert isinstance(pace, Inapplicable)


SCHED, BUF, SEL, SRC = Advance.SCHEDULED, Advance.BUFFERED, Advance.SELECTED, Advance.SOURCED
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
    the advance rule CONTRACT.md § 9 states (R2 F4: evaluated, not re-derived)."""
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


def test_a_kernel_gives_each_port_of_its_body_its_name_and_its_clock_domain() -> None:
    class Native_(Kernel):
        id = "test.native_port_clock"
        clock_domain = NATIVE_CLOCKING
        input = StreamPort(transport=NATIVE, tensor=TENSOR, sequence=SEQUENCE)
        named = StreamPort(label="other", transport=NATIVE, tensor=TENSOR, sequence=SEQUENCE)

    point = design_space(Native_())
    assert point.input.label == "input" and point.input.clocking == NATIVE_CLOCKING
    assert point.named.label == "other"
    assert point.input.contract.contract.transport.clock == "clk"
    # A port declared outside a kernel's body states both.
    with pytest.raises(DefinitionError, match="clocking is not supplied"):
        design_space(StreamPort(label="p", transport=AXIS, tensor=TENSOR, sequence=SEQUENCE))


def test_a_port_states_its_transport() -> None:
    """R1 F4: no default transport, so a native port is never taken for an AXIS bus."""

    with pytest.raises(DefinitionError, match="transport is not supplied"):

        class Untransported(Kernel):
            id = "test.untransported"
            input = StreamPort(tensor=TENSOR, sequence=SEQUENCE)

        design_space(Untransported())


class Leaf(Kernel):
    id = "test.served.leaf"
    input = StreamPort(transport=AXIS, tensor=TENSOR, sequence=SEQUENCE)
    output = StreamPort(
        transport=AXIS, tensor=TENSOR, sequence=SEQUENCE, endpoint=Endpoint.INITIATOR
    )


class OtherLeaf(Leaf):
    id = "test.served.other_leaf"


class Serving(Kernel):
    id = "test.served"
    core = Select(one=Leaf, other=OtherLeaf)
    x = served(core, "input")
    y = served(core, "output")
    kept = Leaf()
    z = served(kept, "output", label="kept.leaf")
    w = served(kept, "input")


def test_a_kernel_serves_its_child_s_port_labelled_below_the_child() -> None:
    """A served port's side and contract are its child's port's, labelled below the selected
    case (``core.<case>.input``) or below the child, or the label the kernel states."""
    point = design_space(Serving()).with_choices({Serving.core: "other"})

    def side(port: str) -> object:
        found = point.query(port_side(Serving, port))
        assert isinstance(found, Available), port
        return found.value

    def contract_of(port: str) -> PortContract:
        found = point.query(port_contract(Serving, port))
        assert isinstance(found, Available) and isinstance(found.value, PortContract), port
        return found.value

    child = point.core.input
    assert side("x") == child.side.under("core.other")
    assert contract_of("x") == child.contract.under("core.other")
    assert contract_of("x").label == "core.other.input"
    assert side("y") == point.core.output.side.under("core.other")
    assert contract_of("z").label == "kept.leaf.output"
    assert side("w") == point.kept.input.side.under("kept")
    # Lowered into members of the kernel, which the engine reads like any other.
    assert port_side(Serving, "x") is getattr(Serving, "x_side")
    assert port_contract(Serving, "x") is getattr(Serving, "x_contract")
    assert getattr(point, "core_case") == "other"
    assert port_endpoint(Serving, "x") is Endpoint.TARGET
    assert port_endpoint(Serving, "y") is Endpoint.INITIATOR
    assert port_endpoint(Serving, "z") is Endpoint.INITIATOR
    # Served ports are not the kernel's own stream ports: a module's ports are.
    assert stream_ports(Serving) == {}
    # Open, the case is not known, so neither is the label.
    assert isinstance(design_space(Serving()).query(port_contract(Serving, "x")), Unresolved)


def test_a_served_port_names_a_child_and_a_stream_port_of_it() -> None:
    with pytest.raises(DefinitionError, match="no child declared"):

        class Orphan(Kernel):
            id = "test.served.orphan"
            x = served(Leaf(), "input")

    with pytest.raises(DefinitionError, match="no stream port"):

        class Misnamed(Kernel):
            id = "test.served.misnamed"
            kept = Leaf()
            x = served(kept, "missing")

    with pytest.raises(DefinitionError, match="each case"):

        class Partial(Kernel):
            id = "test.served.partial"
            core = Select(one=Leaf, other=VpcKernel)
            x = served(core, "x")

    with pytest.raises(DefinitionError, match="served port x"):

        class Repeated(Kernel):
            id = "test.served.repeated"
            kept = Leaf()
            x = served(kept, "input")

            @derived
            def x_side(self) -> Side:
                return self.kept.input.side


def test_a_served_port_is_served_in_its_kernel_s_own_body() -> None:
    """R2 F2, F6: a port served in a Group (which reads nothing of its kernel) is refused,
    and so is a body declaring the case a selector's served ports are lowered with."""

    class InGroup(Group):
        x = served(Leaf(), "input")

    with refused("served-port-nested", error=DefinitionError, message=r"\['g\.x'\]"):

        class Holder(Kernel):
            id = "test.served.holder"
            kept = Leaf()
            g = InGroup()

    with refused("served-port-repeated", error=DefinitionError, message="core_case"):

        class OwnCase(Kernel):
            id = "test.served.own_case"
            core = Select(one=Leaf, other=OtherLeaf)
            x = served(core, "input")

            @derived
            def core_case(self) -> str:
                return "one"


def test_a_port_states_one_thing_its_advance_rule_follows_from() -> None:
    """R2 F3: one of a schedule, the port it governs, or stored values; and a port governs
    another stream port of its kernel."""
    with refused(
        "port-advance-ambiguous", error=DefinitionError, message="schedule and governs and stored"
    ):

        class Both(Kernel):
            id = "test.advance.both"
            input = StreamPort(
                transport=AXIS,
                tensor=TENSOR,
                schedule=Schedule({Index("r"): 2, Index("c"): 4}),
                index=(Index("r"), Index("c")),
                governs="other",
                stored=True,
            )
            other = StreamPort(transport=AXIS, tensor=TENSOR, sequence=SEQUENCE)

    for governs in ("no_such_port", "input"):
        with refused("port-governs-unknown", error=DefinitionError, message=repr(governs)):

            class Governs(Kernel):
                id = "test.advance.governs"
                input = StreamPort(
                    transport=AXIS, tensor=TENSOR, sequence=SEQUENCE, governs=governs
                )

    class Governing(Kernel):
        id = "test.advance.governing"
        input = StreamPort(transport=AXIS, tensor=TENSOR, sequence=SEQUENCE)
        set = StreamPort(transport=AXIS, tensor=TENSOR, sequence=SEQUENCE, governs="input")

    assert design_space(Governing()).set.advance is Advance.SELECTED


def test_a_scheduled_port_presents_its_kernel_s_one_schedule() -> None:
    """W5's open decision 6, taken in W6: a port of a kernel's body presents its kernel's
    ``schedule`` or no schedule (``port-schedule-foreign``), so its kernel's scheduled
    ports present one index point at each iteration and their paces count the same
    iterations; a subclass's override of ``schedule`` is the one its inherited ports read."""
    r, c = Index("r"), Index("c")

    class Scheduled(Kernel):
        id = "test.schedule.own"

        @derived
        def schedule(self) -> Schedule:
            return Schedule({r: 2, c: 4})

        input = StreamPort(transport=AXIS, tensor=TENSOR, schedule=schedule, index=(r, c))

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

                input = StreamPort(
                    transport=AXIS,
                    tensor=TENSOR,
                    schedule=other if given is None else given,
                    index=(r, c),
                )
