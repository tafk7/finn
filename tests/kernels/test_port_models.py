# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Actual lean port values, local ownership, sparse demand and authored discovery."""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    Group,
    Inapplicable,
    Param,
    Rejected,
    Space,
    Unresolved,
    constraint,
    derived,
    design_space,
    inspection,
    reject,
    view,
)
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.module import Held
from finn.kernels.base import NATIVE_CLOCKING, KernelSpace, PortContract, UntimedKernel
from finn.kernels.port import LoopEnds, PhysicalPort, contract, data_width, held, pins, transport
from finn.kernels.port_model import PortModel, PortSurface, SequencePort, native_port, port_surfaces
from finn.kernels.sides import Side
from finn.kernels.values.domains import Integer

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

    input = native_port(
        "input", Endpoint.TARGET, tensor, sequence, clocking=NATIVE_CLOCKING, label="input"
    )
    output = native_port(
        "output",
        Endpoint.INITIATOR,
        tensor,
        input.sequence,
        clocking=NATIVE_CLOCKING,
        label="output",
    )
    checked = SequencePort(
        name="s_axis_x",
        label="checked",
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
        old = PhysicalPort(
            name=name,
            endpoint=endpoint,
            signals=("idat", "ivld", "irdy") if name == "input" else ("odat", "ovld", "ordy"),
        )
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

        port = SequencePort(
            name="set", label="set", dtype=DataType["INT3"], sequence=bad, clocking=NATIVE_CLOCKING
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


def test_static_idle_and_nonbinding_ports_omit_members() -> None:
    class Shape(Space):
        idle = PortModel(name="idle", label="idle", dtype=DataType["INT3"])
        live = SequencePort(name="live", label="live", tensor=TENSOR, sequence=SEQUENCE)

    point = design_space(Shape())
    keys = {n.key for n in inspection.members(point)}
    assert not keys & {
        "idle.stream",
        "idle.width",
        "idle.side",
        "idle.contract",
        "idle.access",
        "idle.placed",
        "live.access",
        "live.binds",
    }
    assert inspection.statistics(point).owning_decisions == 0
    assert point.idle.element == ELEMENT
    surfaces = port_surfaces(Shape)
    assert surfaces["idle"].stream is None
    assert surfaces["live"].stream is not None
    assert point.query(surfaces["live"].stream) == Available(Stream(ELEMENT, SEQUENCE))


@pytest.mark.parametrize("state", [None, False, True])
def test_real_child_presence_is_distinct_from_idle(state: bool | None) -> None:
    class Guarded(Space):
        enabled: bool = Decision(domain=(False, True))
        port = SequencePort(name="x", label="x", tensor=TENSOR, sequence=SEQUENCE, when=enabled)

    point = design_space(Guarded())
    if state is not None:
        point = point.with_choices({Guarded.enabled: state})
    result = point.query(Guarded.port.physical_pins)
    assert isinstance(result, Unresolved if state is None else Available if state else Inapplicable)


def test_native_loop_markers_and_physical_refusals_use_actual_values() -> None:
    sequence = BeatSequence(SEQUENCE.form, markers=(LevelEnd(2),))

    class Marked(Space):
        port = native_port(
            "output",
            Endpoint.INITIATOR,
            TENSOR,
            sequence,
            clocking=NATIVE_CLOCKING,
            label="output",
            loop_ends=LoopEnds("olst", (LevelEnd(2),)),
        )

    point = design_space(Marked())
    old = PhysicalPort(
        name="output",
        endpoint=Endpoint.INITIATOR,
        signals=("odat", "ovld", "ordy"),
        loop_ends=LoopEnds("olst", (LevelEnd(2),)),
    )
    expected = Stream(ELEMENT, sequence)
    assert point.port.physical_pins == pins(old, expected, NATIVE_CLOCKING)
    found = contract(old, expected, NATIVE_CLOCKING)
    assert not isinstance(found, Rejected)
    assert point.port.contract == PortContract("output", found)

    class Bad(Space):
        port = SequencePort(
            name="bad", label="bad", tensor=TENSOR, sequence=sequence, signals=("data", "valid")
        )

    bad = design_space(Bad())
    assert isinstance(bad.query(Bad.port.physical_pins), Rejected)
    assert inspection.validity(bad).status == "valid"  # no invented physical model rule


def test_surface_discovery_rebases_overrides_groups_and_stops_at_real_children() -> None:
    class Included(Group):
        nested = SequencePort(name="nested", label="nested", tensor=TENSOR, sequence=SEQUENCE)

    class Base(Space):
        first: PortModel = SequencePort(
            name="first", label="first", tensor=TENSOR, sequence=SEQUENCE
        )
        second: object = SequencePort(
            name="second", label="second", tensor=TENSOR, sequence=SEQUENCE
        )

    class Child(Base):
        first = SequencePort(name="replaced", label="first", tensor=TENSOR, sequence=SEQUENCE)
        group = Included()
        real = Base()

    surfaces = port_surfaces(Child)
    assert list(surfaces) == ["first", "second", "group.nested"]
    assert surfaces["first"].stream is not None
    point = design_space(Child())
    assert point.query(surfaces["group.nested"].stream) == Available(Stream(ELEMENT, SEQUENCE))
    with pytest.raises(TypeError, match="node declaration"):
        PortSurface.of(point.first)


def test_port_children_are_not_hardware_kernels_or_member_netlists() -> None:
    class Host(KernelSpace):
        id = "test.port-host"
        p = SequencePort(name="p", label="p", tensor=TENSOR, sequence=SEQUENCE)

    point = design_space(Host())
    assert isinstance(point.p, Space)
    assert not isinstance(point.p, UntimedKernel)
    assert point.netlists == ()


def test_view_bindings_remain_model_read_in_errors() -> None:
    class Bad(Space):
        @view
        def outward(self) -> BeatSequence:
            return SEQUENCE

        @derived
        def inward(self) -> BeatSequence:
            return self.outward

        port = SequencePort(name="bad", label="bad", tensor=TENSOR, sequence=inward)

    with pytest.raises(DefinitionError, match="view"):
        design_space(Bad())
