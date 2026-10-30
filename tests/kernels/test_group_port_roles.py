# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Static roles and Composer rejection for real ports under lexical Group frames."""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2

from finn.core.space import Available, Decision, Group, Param, Unresolved, design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import Kernel
from finn.kernels.composer import Facts, PartitionDeclaration, Port, channel
from finn.kernels.composer import _checked as checked
from finn.kernels.port import port_endpoint, port_surface
from finn.kernels.port_model import SequencePort
from finn.kernels.sides import Side

TENSOR = Tensor((2, 4), ScalarEncoding(DataType["INT3"]))
SEQUENCE = BeatSequence(vector_major(TENSOR.shape, 4))


class DefaultOutput(SequencePort):
    endpoint: Endpoint = Param(default=Endpoint.INITIATOR)


class Ports(Group):
    target = SequencePort(label="t", tensor=TENSOR, sequence=SEQUENCE)
    explicit = SequencePort(
        label="o",
        tensor=TENSOR,
        sequence=SEQUENCE,
        endpoint=Endpoint.INITIATOR,
    )
    default_output = DefaultOutput(label="d", tensor=TENSOR, sequence=SEQUENCE)
    rebound = DefaultOutput(label="b", tensor=TENSOR, sequence=SEQUENCE, endpoint=Endpoint.TARGET)


class Nested(Group):
    ports = Ports()


class Host(Kernel):
    id = "test.group-port-roles"
    group = Nested()


@pytest.mark.parametrize(
    "port,expected",
    [
        ("group.ports.target", Endpoint.TARGET),
        ("group.ports.explicit", Endpoint.INITIATOR),
        ("group.ports.default_output", Endpoint.INITIATOR),
        ("group.ports.rebound", Endpoint.TARGET),
    ],
)
def test_constant_default_and_explicit_bound_group_endpoints(port: str, expected: Endpoint) -> None:
    assert port_endpoint(Host, port) is expected
    surface = port_surface(Host, port)
    point = design_space(Host())
    assert point.query(surface.endpoint) == Available(expected)


def test_open_model_endpoint_remains_unknown_without_reading_or_repairing_configuration() -> None:
    class OpenPorts(Group):
        role: Endpoint = Decision(domain=(Endpoint.TARGET, Endpoint.INITIATOR))
        port = SequencePort(label="p", tensor=TENSOR, sequence=SEQUENCE, endpoint=role)

    class OpenHost(Kernel):
        id = "test.open-group-port-role"
        group = OpenPorts()

    assert port_endpoint(OpenHost, "group.port") is None
    point = design_space(OpenHost())
    assert isinstance(point.query(OpenHost.group.port.endpoint), Unresolved)
    bound = point.with_choices({OpenHost.group.role: Endpoint.INITIATOR})
    assert bound.query(OpenHost.group.port.endpoint) == Available(Endpoint.INITIATOR)
    assert port_endpoint(type(bound), "group.port") is None


def test_fact_bound_endpoint_is_not_mistaken_for_a_literal_static_binding() -> None:
    class BoundPorts(Group):
        role: Endpoint = Param()
        port = SequencePort(label="p", tensor=TENSOR, sequence=SEQUENCE, endpoint=role)

    class BoundHost(Kernel):
        id = "test.bound-group-port-role"
        role: Endpoint = Param()
        group = BoundPorts(role=role)

    point = design_space(BoundHost(role=Endpoint.TARGET))
    assert point.group.port.endpoint is Endpoint.TARGET
    assert port_endpoint(BoundHost, "group.port") is None


def test_static_group_lookup_stops_at_real_spaces_and_unknown_names() -> None:
    class Parent(Kernel):
        id = "test.group-port-stop"
        child = Host()

    assert port_endpoint(Parent, "child.group.ports.target") is None
    assert port_endpoint(Host, "group.ports.missing") is None
    with pytest.raises(KeyError):
        port_surface(Parent, "child.group.ports.target")


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


def test_inherited_host_retains_group_placement_and_effective_port_roles() -> None:
    class InheritedHost(Host):
        id = "test.inherited-group-port-roles"

    # The actual nested Group placement comes from Host, rather than a new
    # placement containing only a Port subclass with an inherited formal default.
    assert InheritedHost.group is Host.group
    point = design_space(InheritedHost())
    for port, expected in (
        ("group.ports.target", Endpoint.TARGET),
        ("group.ports.explicit", Endpoint.INITIATOR),
        ("group.ports.default_output", Endpoint.INITIATOR),
        ("group.ports.rebound", Endpoint.TARGET),
    ):
        assert port_endpoint(InheritedHost, port) is expected
        surface = port_surface(InheritedHost, port)
        assert point.query(surface.endpoint) == Available(expected)


@pytest.mark.parametrize(
    "port,role",
    [("group.ports.target", "consumer"), ("group.ports.explicit", "producer")],
)
def test_composer_accepts_known_valid_group_producer_and_consumer_roles(
    port: str, role: str
) -> None:
    expected = Endpoint.INITIATOR if role == "producer" else Endpoint.TARGET
    assert port_endpoint(Host, port) is expected
    endpoint = Port("kernel", port)
    outside = Side.boundary("out0" if role == "producer" else "in0")
    producer, consumer = (endpoint, outside) if role == "producer" else (outside, endpoint)
    declared = PartitionDeclaration(
        {"kernel": Facts(Host, {})},
        {"link": channel(producer, consumer, tensor=TENSOR, platform=FULL_DSP48E2)},
    )
    assert checked(declared) == []
