# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Port kinds and Composer rejection for real ports under lexical Group frames."""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2
from support.refusals import refused

from finn.core.space import Available, DefinitionError, Group, design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Direction
from finn.kernels.base import Kernel
from finn.kernels.composer import Facts, PartitionDeclaration, Port, channel
from finn.kernels.composer import _checked as checked
from finn.kernels.laws import AXIS, AXIS_CLOCKING
from finn.kernels.port import port_direction, stream_port
from finn.kernels.port_model import InPort, OutPort, StreamPort, stream_ports
from finn.kernels.sides import Side

TENSOR = Tensor((2, 4), ScalarEncoding(DataType["INT3"]))
SEQUENCE = BeatSequence(vector_major(TENSOR.shape, 4))


class TappedOutput(OutPort):
    """A kind of output: its direction its kind's."""


class Ports(Group):
    target = InPort(
        clocking=AXIS_CLOCKING,
        transport=AXIS,
        label="t",
        tensor=TENSOR,
        sequence=SEQUENCE,
    )
    explicit = OutPort(
        clocking=AXIS_CLOCKING,
        transport=AXIS,
        label="o",
        tensor=TENSOR,
        sequence=SEQUENCE,
    )
    tapped = TappedOutput(
        clocking=AXIS_CLOCKING,
        transport=AXIS,
        label="d",
        tensor=TENSOR,
        sequence=SEQUENCE,
    )


class Nested(Group):
    ports = Ports()


class Host(Kernel):
    id = "test.group-port-roles"
    group = Nested()


@pytest.mark.parametrize(
    "port,expected",
    [
        ("group.ports.target", Direction.IN),
        ("group.ports.explicit", Direction.OUT),
        ("group.ports.tapped", Direction.OUT),
    ],
)
def test_a_group_port_s_direction_is_its_kind_s(port: str, expected: Direction) -> None:
    assert port_direction(Host, port) is expected
    point = design_space(Host())
    assert point.query(stream_ports(Host)[port].direction) == Available(expected)


def test_a_port_s_direction_is_never_bound_or_chosen() -> None:
    """Its kind is its direction: no port states one, by a literal, a fact or a choice."""
    for value in (Direction.IN, Direction.OUT):
        with pytest.raises(DefinitionError, match="InPort.direction is a Derived"):
            InPort(
                clocking=AXIS_CLOCKING,
                transport=AXIS,
                label="p",
                tensor=TENSOR,
                sequence=SEQUENCE,
                direction=value,  # type: ignore[call-arg]
            )


def test_static_group_lookup_stops_at_real_spaces_and_unknown_names() -> None:
    class Parent(Kernel):
        id = "test.group-port-stop"
        child = Host()

    assert port_direction(Parent, "child.group.ports.target") is None
    assert port_direction(Host, "group.ports.missing") is None
    with pytest.raises(KeyError):
        stream_port(Parent, "child.group.ports.target")


@pytest.mark.parametrize(
    "port,role",
    [
        ("group.ports.target", "producer"),
        ("group.ports.explicit", "consumer"),
        ("group.ports.tapped", "consumer"),
    ],
)
def test_composer_refuses_known_wrong_roles_under_group_keys(port: str, role: str) -> None:
    endpoint = Port("kernel", port)
    outside = Side.boundary("out0" if role == "producer" else "in0")
    producer, consumer = (endpoint, outside) if role == "producer" else (outside, endpoint)
    declared = PartitionDeclaration(
        {"kernel": Facts(Host, {})},
        {"link": channel(producer, consumer, tensor=TENSOR, platform=FULL_DSP48E2)},
    )
    findings = checked(declared)
    assert [finding.code for finding in findings] == ["channel-users"]
    assert f"kernel.{port}" in findings[0].message


def test_inherited_host_retains_group_placement_and_port_directions() -> None:
    class InheritedHost(Host):
        id = "test.inherited-group-port-roles"

    # The actual nested Group placement comes from Host.
    assert InheritedHost.group is Host.group
    point = design_space(InheritedHost())
    for port, expected in (
        ("group.ports.target", Direction.IN),
        ("group.ports.explicit", Direction.OUT),
        ("group.ports.tapped", Direction.OUT),
    ):
        assert port_direction(InheritedHost, port) is expected
        assert point.query(stream_ports(InheritedHost)[port].direction) == Available(expected)


@pytest.mark.parametrize(
    "port,role",
    [("group.ports.target", "consumer"), ("group.ports.explicit", "producer")],
)
def test_composer_accepts_known_valid_group_producer_and_consumer_roles(
    port: str, role: str
) -> None:
    expected = Direction.OUT if role == "producer" else Direction.IN
    assert port_direction(Host, port) is expected
    endpoint = Port("kernel", port)
    outside = Side.boundary("out0" if role == "producer" else "in0")
    producer, consumer = (endpoint, outside) if role == "producer" else (outside, endpoint)
    declared = PartitionDeclaration(
        {"kernel": Facts(Host, {})},
        {"link": channel(producer, consumer, tensor=TENSOR, platform=FULL_DSP48E2)},
    )
    assert checked(declared) == []


def test_a_port_of_neither_kind_is_refused() -> None:
    """A stream port is an input or an output: a ``StreamPort`` of neither kind, in a
    Group as in a kernel's body, is refused (``port-direction``)."""

    class Unstated(Group):
        port = StreamPort(
            clocking=AXIS_CLOCKING, transport=AXIS, label="u", tensor=TENSOR, sequence=SEQUENCE
        )

    with refused("port-direction", error=DefinitionError, message="group.port is a StreamPort"):

        class UnstatedHost(Kernel):
            id = "test.group-port-unstated"
            group = Unstated()
