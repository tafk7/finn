# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Authored forwarding surfaces retain their actual lexical Group occurrence."""

from __future__ import annotations

from typing import ClassVar

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2

from finn.core.space import Available, Decision, Group, Param, Space, derived, design_space, view
from finn.core.space.declarations import Declaration, MemberRef, declared_path
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import KernelSpace, PortContract
from finn.kernels.composer import Facts, PartitionDeclaration, Port, channel
from finn.kernels.composer import _checked as checked
from finn.kernels.port import port_endpoint, port_surface
from finn.kernels.port_model import PortSurface, SequencePort, port_surfaces
from finn.kernels.sides import Side

SMALL = Tensor((2, 4), ScalarEncoding(DataType["INT3"]))
OTHER = Tensor((3, 2), ScalarEncoding(DataType["INT5"]))
SMALL_SEQUENCE = BeatSequence(vector_major(SMALL.shape, 4))
OTHER_SEQUENCE = BeatSequence(vector_major(OTHER.shape, 2))


class Receiving(Group):
    tensor: Tensor = Param()
    sequence: BeatSequence = Param()
    role: Endpoint = Param(default=Endpoint.TARGET)
    native = SequencePort(
        name="source", label="source", tensor=tensor, sequence=sequence, endpoint=role
    )

    @derived
    def stream(self) -> Stream:
        return self.native.stream

    @derived
    def width(self) -> int:
        return self.native.width

    @derived
    def side(self) -> Side:
        return self.native.side

    @view
    def contract(self) -> PortContract:
        return self.native.contract

    forwarded: ClassVar[PortSurface] = PortSurface(
        None, Endpoint.TARGET, None, None, stream, width, side, contract
    )


class Sending(Receiving):
    role: Endpoint = Param(default=Endpoint.INITIATOR)
    forwarded: ClassVar[PortSurface] = PortSurface(
        None,
        Endpoint.INITIATOR,
        None,
        None,
        Receiving.stream,
        Receiving.width,
        Receiving.side,
        Receiving.contract,
    )


class Frame(Group):
    receive = Receiving(tensor=SMALL, sequence=SMALL_SEQUENCE)
    send = Sending(tensor=OTHER, sequence=OTHER_SEQUENCE)


class InheritedFrame(Frame):
    """Inherit the actual Group placements as well as the forwarding declarations."""


class ReplacedFrame(InheritedFrame):
    receive = Receiving(tensor=OTHER, sequence=OTHER_SEQUENCE)


class Host(KernelSpace):
    id = "test.group-parent-surfaces"
    group = InheritedFrame()

    @derived
    def stream(self) -> Stream:
        # Same spelling as the Group's real-source readout, different value.
        return Stream(OTHER.element, OTHER_SEQUENCE)

    forwarded: ClassVar[PortSurface] = PortSurface(None, Endpoint.INITIATOR, None, None, stream)


class InheritedHost(Host):
    id = "test.group-parent-surfaces.inherited"


class ReplacedHost(Host):
    id = "test.group-parent-surfaces.replaced"
    group = ReplacedFrame()


def reference_path(value: object) -> tuple[Declaration, ...]:
    """A captured member reference has a path; declared_path describes node placements."""
    assert isinstance(value, MemberRef)
    return value.path


@pytest.mark.parametrize("host", [Host, InheritedHost, ReplacedHost])
@pytest.mark.parametrize("key", ["group.receive.forwarded", "group.send.forwarded"])
def test_discovery_lookup_and_real_source_agree_for_inherited_nested_groups(
    host: type[Host], key: str
) -> None:
    discovered = port_surfaces(host)
    assert list(discovered) == [
        "group.receive.native",
        "group.receive.forwarded",
        "group.send.native",
        "group.send.forwarded",
        "forwarded",
    ]
    surface = port_surface(host, key)
    assert surface == discovered[key]
    sending = key == "group.send.forwarded"
    group = host.group.send if sending else host.group.receive
    assert reference_path(surface.stream) == declared_path(group)
    assert reference_path(surface.side) == declared_path(group)
    assert reference_path(surface.contract_reference()) == declared_path(group)
    assert surface.stream == group.stream
    assert surface.contract_reference() == group.contract
    expected = Endpoint.INITIATOR if sending else Endpoint.TARGET
    assert port_endpoint(host, key) is expected
    tensor = OTHER if sending or host is ReplacedHost else SMALL
    sequence = OTHER_SEQUENCE if tensor is OTHER else SMALL_SEQUENCE
    point = design_space(host())
    assert point.query(surface.stream) == Available(Stream(tensor.element, sequence))
    bits = tensor.element.dtype.bitwidth() * sequence.form.lanes
    assert point.query(surface.width) == Available(((bits + 7) // 8) * 8)
    assert point.query(surface.contract_reference()) == Available(
        point.group.send.native.contract if sending else point.group.receive.native.contract
    )
    # Lookup must not borrow the root's same-spelling stream reference.
    assert surface.stream != port_surface(host, "forwarded").stream
    if not sending and host is not ReplacedHost:
        assert point.query(surface.stream) != point.query(host.stream)


@pytest.mark.parametrize("placement", ["left", "right"])
@pytest.mark.parametrize("key", ["group.receive.forwarded", "group.send.forwarded"])
def test_placed_node_surfaces_keep_the_exact_group_and_kernel_occurrence(
    placement: str, key: str
) -> None:
    class Holder(Space):
        left = Host()
        right = ReplacedHost()

    namespace = Holder.left if placement == "left" else Holder.right
    surface = port_surface(namespace, key)
    sending = key == "group.send.forwarded"
    group = namespace.group.send if sending else namespace.group.receive
    assert reference_path(surface.stream) == declared_path(group)
    assert reference_path(surface.contract_reference()) == declared_path(group)
    assert surface.stream == group.stream
    assert surface.contract_reference() == group.contract
    assert reference_path(surface.stream) != reference_path(
        port_surface(Holder.right if placement == "left" else Holder.left, key).stream
    )
    tensor = OTHER if sending or placement == "right" else SMALL
    sequence = OTHER_SEQUENCE if tensor is OTHER else SMALL_SEQUENCE
    point = design_space(Holder())
    assert point.query(surface.stream) == Available(Stream(tensor.element, sequence))


@pytest.mark.parametrize("key", ["group.receive.forwarded", "group.send.forwarded"])
@pytest.mark.parametrize("role", ["producer", "consumer"])
def test_actual_composer_accepts_valid_and_refuses_wrong_group_forwarding_roles(
    key: str, role: str
) -> None:
    endpoint = Port("kernel", key)
    outside = Side.boundary("out0" if role == "producer" else "in0")
    producer, consumer = (endpoint, outside) if role == "producer" else (outside, endpoint)
    declaration = PartitionDeclaration(
        {"kernel": Facts(InheritedHost, {})},
        {"group-link": channel(producer, consumer, tensor=SMALL, platform=FULL_DSP48E2)},
    )
    findings = checked(declaration)
    valid = (key == "group.send.forwarded") == (role == "producer")
    if valid:
        assert findings == []
    else:
        description = (
            "a channel's producer is an output port, and kernel." + key + " is an input's"
            if role == "producer"
            else "a channel's consumer is an input port, and kernel." + key + " is an output's"
        )
        assert [finding.code for finding in findings] == ["channel-users"]
        assert findings[0].message == description
        assert findings[0].owner == "group-link"


class UnknownRoles(Group):
    fact: Endpoint = Param(default=Endpoint.TARGET)
    choice: Endpoint = Decision(domain=(Endpoint.TARGET, Endpoint.INITIATOR))

    @derived
    def inferred(self) -> Endpoint:
        return Endpoint.INITIATOR

    omitted: ClassVar[PortSurface] = PortSurface(None, None, None, None)
    fact_bound: ClassVar[PortSurface] = PortSurface(None, fact, None, None)
    derived_bound: ClassVar[PortSurface] = PortSurface(None, inferred, None, None)
    decision_bound: ClassVar[PortSurface] = PortSurface(None, choice, None, None)


@pytest.mark.parametrize("key", ["omitted", "fact_bound", "derived_bound", "decision_bound"])
def test_endpoint_metadata_stays_unknown_even_when_configuration_resolves_it(key: str) -> None:
    class UnknownHost(KernelSpace):
        id = "test.group-forwarded-unknown-role"
        group = UnknownRoles()

    dotted = "group." + key
    assert port_endpoint(UnknownHost, dotted) is None
    surface = port_surface(UnknownHost, dotted)
    assert surface == port_surfaces(UnknownHost)[dotted]
    point = design_space(UnknownHost()).with_choices({UnknownHost.group.choice: Endpoint.INITIATOR})
    if surface.endpoint is not None:
        expected = Endpoint.TARGET if key == "fact_bound" else Endpoint.INITIATOR
        assert point.query(surface.endpoint) == Available(expected)
    # Static metadata inspection accepts a class, not a bound runtime instance.
    assert port_endpoint(type(point), dotted) is None
    assert port_endpoint(UnknownHost, "group.unknown") is None


@pytest.mark.parametrize(
    "key", ["missing", "group.missing", "group.receive.missing", "child.group.receive.forwarded"]
)
def test_unknown_keys_and_real_space_crossing_remain_rejected(key: str) -> None:
    class Parent(Host):
        id = "test.group-forwarded-no-crossing"
        child = Host()

    assert key not in port_surfaces(Parent)
    assert port_endpoint(Parent, key) is None
    with pytest.raises(KeyError, match=key):
        port_surface(Parent, key)
