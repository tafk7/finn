# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A stream port declared in a Group keeps its actual lexical Group occurrence: discovery
(``stream_ports``), lookup (``stream_port``, ``port_direction``) and the
composer find it there, and nothing forwards it by hand; its side is its kind's, which a
channel takes on its role."""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, TEST_CORE

from finn.core.space import Available, Group, Param, Rejected, Space, design_space
from finn.core.space.declarations import Declaration, MemberRef, declared_path
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Direction
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.composer import DataflowCoreDeclaration, Facts, Port, channel
from finn.kernels.composer import _checked as checked
from finn.kernels.laws import AXIS, AXIS_CLOCKING
from finn.kernels.port import port_direction, stream_port
from finn.kernels.port_model import InPort, OutPort, stream_ports
from finn.kernels.sides import Side

SMALL = Tensor((2, 4), ScalarEncoding(DataType["INT3"]))
OTHER = Tensor((3, 2), ScalarEncoding(DataType["INT5"]))
SMALL_SEQUENCE = BeatSequence(vector_major(SMALL.shape, 4))
OTHER_SEQUENCE = BeatSequence(vector_major(OTHER.shape, 2))


class Receiving(Group):
    tensor: Tensor = Param()
    sequence: BeatSequence = Param()
    native = InPort(
        clocking=AXIS_CLOCKING,
        transport=AXIS,
        label="source",
        tensor=tensor,
        sequence=sequence,
    )


class Sending(Group):
    tensor: Tensor = Param()
    sequence: BeatSequence = Param()
    native = OutPort(
        clocking=AXIS_CLOCKING,
        transport=AXIS,
        label="source",
        tensor=tensor,
        sequence=sequence,
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
    native = InPort(tensor=OTHER, sequence=OTHER_SEQUENCE)


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
    side, contract = stream_port(host, key).side, stream_port(host, key).stream_pins
    assert reference_path(side) == reference_path(group.native.side)
    assert reference_path(contract) == reference_path(group.native.stream_pins)
    assert side == group.native.side
    assert contract == group.native.stream_pins
    expected = Direction.OUT if sending else Direction.IN
    assert port_direction(host, key) is expected
    tensor = OTHER if sending or host is ReplacedHost else SMALL
    sequence = OTHER_SEQUENCE if tensor is OTHER else SMALL_SEQUENCE
    point = design_space(host())
    found = point.query(side)
    assert isinstance(found, Available)
    assert found.value.stream == Stream(tensor.element, sequence)
    assert point.query(contract) == Available(
        point.group.send.native.stream_pins if sending else point.group.receive.native.stream_pins
    )
    # Lookup must not borrow the root's same-spelling port.
    assert side != stream_port(host, "native").side


@pytest.mark.parametrize("placement", ["left", "right"])
@pytest.mark.parametrize("key", ["group.receive.native", "group.send.native"])
def test_placed_node_ports_keep_the_exact_group_and_kernel_occurrence(
    placement: str, key: str
) -> None:
    class Holder(Space):
        left = Host()
        right = ReplacedHost()

    namespace = Holder.left if placement == "left" else Holder.right
    side = stream_port(namespace, key).side
    sending = key == "group.send.native"
    group = namespace.group.send if sending else namespace.group.receive
    assert side == group.native.side
    assert stream_port(namespace, key).stream_pins == group.native.stream_pins
    assert reference_path(side) != reference_path(
        stream_port(Holder.right if placement == "left" else Holder.left, key).side
    )
    tensor = OTHER if sending or placement == "right" else SMALL
    sequence = OTHER_SEQUENCE if tensor is OTHER else SMALL_SEQUENCE
    found = design_space(Holder()).query(side)
    assert isinstance(found, Available)
    assert found.value.stream == Stream(tensor.element, sequence)


@pytest.mark.parametrize("key", ["group.receive.native", "group.send.native"])
@pytest.mark.parametrize("role", ["producer", "consumer"])
def test_actual_composer_reads_and_channel_takes_or_refuses_group_port_roles(
    key: str, role: str
) -> None:
    """The composer reads a port by its Group key on either role and checks no role; the
    port's side is its kind's, which the channel takes on its role and refuses on the other
    (``channel-users``)."""
    endpoint = Port("kernel", key)
    outside = Side.boundary("out0" if role == "producer" else "in0")
    producer, consumer = (endpoint, outside) if role == "producer" else (outside, endpoint)
    declaration = DataflowCoreDeclaration(
        {"kernel": Facts(InheritedHost, {})},
        {"group-link": channel(producer, consumer, tensor=SMALL, platform=FULL_DSP48E2)},
        name=TEST_CORE,
    )
    assert checked(declaration) == []
    found = design_space(InheritedHost()).query(stream_port(InheritedHost, key).side)
    assert isinstance(found, Available) and isinstance(found.value, Side)
    side = found.value.under("kernel")
    sides = (side, outside) if role == "producer" else (outside, side)
    tensor = OTHER if key == "group.send.native" else SMALL
    point = design_space(
        Channel(tensor=tensor, producer=sides[0], consumer=sides[1], platform=FULL_DSP48E2)
    )
    read = point.query(Channel.sides)
    valid = (key == "group.send.native") == (role == "producer")
    if valid:
        assert isinstance(read, Available)
    else:
        description = (
            "a channel's producer is an output port, and kernel.source is an input's"
            if role == "producer"
            else "a channel's consumer is an input port, and kernel.source is an output's"
        )
        assert isinstance(read, Rejected)
        assert [(each.code, each.message) for each in read.findings] == [
            ("channel-users", description)
        ]


@pytest.mark.parametrize(
    "key", ["missing", "group.missing", "group.receive.missing", "child.group.receive.native"]
)
def test_unknown_keys_and_real_space_crossing_remain_rejected(key: str) -> None:
    class Parent(Host):
        id = "test.group-port-no-crossing"
        child = Host()
        # The child's ports, exposed under names of the parent's own.
        child_receive = child.group.receive.native
        child_send = child.group.send.native
        child_native = child.native

    assert key not in stream_ports(Parent)
    assert port_direction(Parent, key) is None
    with pytest.raises(KeyError, match=key):
        stream_port(Parent, key)
