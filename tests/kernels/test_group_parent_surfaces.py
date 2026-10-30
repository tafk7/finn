# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A stream port declared in a Group keeps its actual lexical Group occurrence: discovery
(``stream_ports``), lookup (``port_side``, ``port_contract``, ``port_endpoint``) and the
composer find it there, and nothing forwards it by hand (W4: ``PortSurface`` is gone)."""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2

from finn.core.space import Available, Decision, Group, Param, Space, derived, design_space
from finn.core.space.declarations import Declaration, MemberRef, declared_path
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import Kernel
from finn.kernels.composer import Facts, PartitionDeclaration, Port, channel
from finn.kernels.composer import _checked as checked
from finn.kernels.laws import AXIS, AXIS_CLOCKING
from finn.kernels.port import port_contract, port_endpoint, port_side
from finn.kernels.port_model import StreamPort, stream_ports
from finn.kernels.sides import Side

SMALL = Tensor((2, 4), ScalarEncoding(DataType["INT3"]))
OTHER = Tensor((3, 2), ScalarEncoding(DataType["INT5"]))
SMALL_SEQUENCE = BeatSequence(vector_major(SMALL.shape, 4))
OTHER_SEQUENCE = BeatSequence(vector_major(OTHER.shape, 2))


class Receiving(Group):
    tensor: Tensor = Param()
    sequence: BeatSequence = Param()
    native = StreamPort(
        clocking=AXIS_CLOCKING, transport=AXIS, label="source", tensor=tensor, sequence=sequence
    )


class Sending(Group):
    tensor: Tensor = Param()
    sequence: BeatSequence = Param()
    native = StreamPort(
        clocking=AXIS_CLOCKING,
        transport=AXIS,
        label="source",
        tensor=tensor,
        sequence=sequence,
        endpoint=Endpoint.INITIATOR,
    )


class Frame(Group):
    receive = Receiving(tensor=SMALL, sequence=SMALL_SEQUENCE)
    send = Sending(tensor=OTHER, sequence=OTHER_SEQUENCE)


class InheritedFrame(Frame):
    """Inherit the actual Group placements as well as the ports."""


class ReplacedFrame(InheritedFrame):
    receive = Receiving(tensor=OTHER, sequence=OTHER_SEQUENCE)


class Host(Kernel):
    id = "test.group-parent-surfaces"
    group = InheritedFrame()

    # A port of its own, with the same name as the Groups' ports.
    native = StreamPort(tensor=OTHER, sequence=OTHER_SEQUENCE, transport=AXIS)


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
@pytest.mark.parametrize("key", ["group.receive.native", "group.send.native"])
def test_discovery_lookup_and_real_source_agree_for_inherited_nested_groups(
    host: type[Host], key: str
) -> None:
    discovered = stream_ports(host)
    assert list(discovered) == ["group.receive.native", "group.send.native", "native"]
    sending = key == "group.send.native"
    group = host.group.send if sending else host.group.receive
    assert discovered[key] is not None
    assert declared_path(discovered[key]) == declared_path(group.native)
    side, contract = port_side(host, key), port_contract(host, key)
    assert reference_path(side) == reference_path(group.native.side)
    assert reference_path(contract) == reference_path(group.native.contract)
    assert side == group.native.side
    assert contract == group.native.contract
    expected = Endpoint.INITIATOR if sending else Endpoint.TARGET
    assert port_endpoint(host, key) is expected
    tensor = OTHER if sending or host is ReplacedHost else SMALL
    sequence = OTHER_SEQUENCE if tensor is OTHER else SMALL_SEQUENCE
    point = design_space(host())
    found = point.query(side)
    assert isinstance(found, Available)
    assert found.value.stream == Stream(tensor.element, sequence)
    assert point.query(contract) == Available(
        point.group.send.native.contract if sending else point.group.receive.native.contract
    )
    # Lookup must not borrow the root's same-spelling port.
    assert side != port_side(host, "native")


@pytest.mark.parametrize("placement", ["left", "right"])
@pytest.mark.parametrize("key", ["group.receive.native", "group.send.native"])
def test_placed_node_ports_keep_the_exact_group_and_kernel_occurrence(
    placement: str, key: str
) -> None:
    class Holder(Space):
        left = Host()
        right = ReplacedHost()

    namespace = Holder.left if placement == "left" else Holder.right
    side = port_side(namespace, key)
    sending = key == "group.send.native"
    group = namespace.group.send if sending else namespace.group.receive
    assert side == group.native.side
    assert port_contract(namespace, key) == group.native.contract
    assert reference_path(side) != reference_path(
        port_side(Holder.right if placement == "left" else Holder.left, key)
    )
    tensor = OTHER if sending or placement == "right" else SMALL
    sequence = OTHER_SEQUENCE if tensor is OTHER else SMALL_SEQUENCE
    found = design_space(Holder()).query(side)
    assert isinstance(found, Available)
    assert found.value.stream == Stream(tensor.element, sequence)


@pytest.mark.parametrize("key", ["group.receive.native", "group.send.native"])
@pytest.mark.parametrize("role", ["producer", "consumer"])
def test_actual_composer_accepts_valid_and_refuses_wrong_group_port_roles(
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
    valid = (key == "group.send.native") == (role == "producer")
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

    fact_bound = StreamPort(
        tensor=SMALL,
        sequence=SMALL_SEQUENCE,
        endpoint=fact,
        transport=AXIS,
        label="fact_bound",
        clocking=AXIS_CLOCKING,
    )
    derived_bound = StreamPort(
        tensor=SMALL,
        sequence=SMALL_SEQUENCE,
        endpoint=inferred,
        transport=AXIS,
        label="derived_bound",
        clocking=AXIS_CLOCKING,
    )
    decision_bound = StreamPort(
        tensor=SMALL,
        sequence=SMALL_SEQUENCE,
        endpoint=choice,
        transport=AXIS,
        label="decision_bound",
        clocking=AXIS_CLOCKING,
    )


@pytest.mark.parametrize("key", ["fact_bound", "derived_bound", "decision_bound"])
def test_an_endpoint_bound_to_a_member_stays_unknown_even_when_configuration_resolves_it(
    key: str,
) -> None:
    class UnknownHost(Kernel):
        id = "test.group-port-unknown-role"
        group = UnknownRoles()

    dotted = "group." + key
    assert port_endpoint(UnknownHost, dotted) is None
    assert dotted in stream_ports(UnknownHost)
    port_side(UnknownHost, dotted)
    # Static metadata inspection accepts a class, not a bound runtime instance.
    point = design_space(UnknownHost()).with_choices({UnknownHost.group.choice: Endpoint.INITIATOR})
    assert port_endpoint(type(point), dotted) is None
    assert port_endpoint(UnknownHost, "group.unknown") is None


@pytest.mark.parametrize(
    "key", ["missing", "group.missing", "group.receive.missing", "child.group.receive.native"]
)
def test_unknown_keys_and_real_space_crossing_remain_rejected(key: str) -> None:
    class Parent(Host):
        id = "test.group-port-no-crossing"
        child = Host()

    assert key not in stream_ports(Parent)
    assert port_endpoint(Parent, key) is None
    with pytest.raises(KeyError, match=key):
        port_side(Parent, key)
    with pytest.raises(KeyError, match=key):
        port_contract(Parent, key)
