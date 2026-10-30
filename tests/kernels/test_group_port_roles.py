# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Static directions and Composer rejection for real ports under lexical Group frames."""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2
from support.refusals import refused

from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    Group,
    Param,
    Unresolved,
    design_space,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Direction
from finn.kernels.base import Kernel
from finn.kernels.composer import Facts, PartitionDeclaration, Port, channel
from finn.kernels.composer import _checked as checked
from finn.kernels.laws import AXIS, AXIS_CLOCKING
from finn.kernels.port import port_direction, port_side
from finn.kernels.port_model import StreamPort, stream_ports
from finn.kernels.sides import Side

TENSOR = Tensor((2, 4), ScalarEncoding(DataType["INT3"]))
SEQUENCE = BeatSequence(vector_major(TENSOR.shape, 4))


class DefaultOutput(StreamPort):
    direction: Direction = Param(default=Direction.OUT)


class Ports(Group):
    target = StreamPort(
        direction=Direction.IN,
        clocking=AXIS_CLOCKING,
        transport=AXIS,
        label="t",
        tensor=TENSOR,
        sequence=SEQUENCE,
    )
    explicit = StreamPort(
        clocking=AXIS_CLOCKING,
        transport=AXIS,
        label="o",
        tensor=TENSOR,
        sequence=SEQUENCE,
        direction=Direction.OUT,
    )
    default_output = DefaultOutput(
        direction=Direction.OUT,
        clocking=AXIS_CLOCKING,
        transport=AXIS,
        label="d",
        tensor=TENSOR,
        sequence=SEQUENCE,
    )
    rebound = DefaultOutput(
        clocking=AXIS_CLOCKING,
        transport=AXIS,
        label="b",
        tensor=TENSOR,
        sequence=SEQUENCE,
        direction=Direction.IN,
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
        ("group.ports.default_output", Direction.OUT),
        ("group.ports.rebound", Direction.IN),
    ],
)
def test_stated_group_directions(port: str, expected: Direction) -> None:
    assert port_direction(Host, port) is expected
    point = design_space(Host())
    assert point.query(stream_ports(Host)[port].direction) == Available(expected)


def test_open_model_direction_remains_unknown_without_reading_or_repairing_configuration() -> None:
    class OpenPorts(Group):
        role: Direction = Decision(domain=(Direction.IN, Direction.OUT))
        port = StreamPort(
            clocking=AXIS_CLOCKING,
            transport=AXIS,
            label="p",
            tensor=TENSOR,
            sequence=SEQUENCE,
            direction=role,
        )

    class OpenHost(Kernel):
        id = "test.open-group-port-role"
        group = OpenPorts()

    assert port_direction(OpenHost, "group.port") is None
    point = design_space(OpenHost())
    assert isinstance(point.query(OpenHost.group.port.direction), Unresolved)
    bound = point.with_choices({OpenHost.group.role: Direction.OUT})
    assert bound.query(OpenHost.group.port.direction) == Available(Direction.OUT)
    assert port_direction(type(bound), "group.port") is None


def test_fact_bound_direction_is_not_mistaken_for_a_literal_static_binding() -> None:
    class BoundPorts(Group):
        role: Direction = Param()
        port = StreamPort(
            clocking=AXIS_CLOCKING,
            transport=AXIS,
            label="p",
            tensor=TENSOR,
            sequence=SEQUENCE,
            direction=role,
        )

    class BoundHost(Kernel):
        id = "test.bound-group-port-role"
        role: Direction = Param()
        group = BoundPorts(role=role)

    point = design_space(BoundHost(role=Direction.IN))
    assert point.group.port.direction is Direction.IN
    assert port_direction(BoundHost, "group.port") is None


def test_static_group_lookup_stops_at_real_spaces_and_unknown_names() -> None:
    class Parent(Kernel):
        id = "test.group-port-stop"
        child = Host()

    assert port_direction(Parent, "child.group.ports.target") is None
    assert port_direction(Host, "group.ports.missing") is None
    with pytest.raises(KeyError):
        port_side(Parent, "child.group.ports.target")


@pytest.mark.parametrize(
    "port,role",
    [
        ("group.ports.target", "producer"),
        ("group.ports.rebound", "producer"),
        ("group.ports.explicit", "consumer"),
        ("group.ports.default_output", "consumer"),
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

    # The actual nested Group placement comes from Host, rather than a new
    # placement containing only a Port subclass with an inherited formal default.
    assert InheritedHost.group is Host.group
    point = design_space(InheritedHost())
    for port, expected in (
        ("group.ports.target", Direction.IN),
        ("group.ports.explicit", Direction.OUT),
        ("group.ports.default_output", Direction.OUT),
        ("group.ports.rebound", Direction.IN),
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


def test_a_port_class_default_is_no_stated_direction() -> None:
    """A stream port states its direction in its declaration: a port class's default does
    not state it (``port-direction``)."""

    class Unstated(Group):
        port = DefaultOutput(
            clocking=AXIS_CLOCKING, transport=AXIS, label="u", tensor=TENSOR, sequence=SEQUENCE
        )

    with refused("port-direction", error=DefinitionError, message="group.port states no direction"):

        class UnstatedHost(Kernel):
            id = "test.group-port-unstated"
            group = Unstated()
