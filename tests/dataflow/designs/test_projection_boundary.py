# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Only a semantically accepted Design projection discharges presentation's precondition."""

from dataclasses import replace

import pytest

import finn.dataflow.designs.design as design_module
from finn.dataflow._engine import Absent, Decided
from finn.dataflow.designs import (
    DataflowDesign,
    EdgeSink,
    KernelChoice,
    NetworkBoundary,
    NetworkEdge,
)
from finn.dataflow.kernels import RegionDeclaration
from finn.dataflow.model import (
    DataflowNetwork,
    DataflowRegion,
    InputInterface,
    NetworkValidationReport,
    Port,
    RegionInputRef,
    boundary_presented_positions,
    edge_presented_positions,
    exposing_ports,
    unpresented_positions,
    validate_network,
)
from finn.dataflow.space import ConstraintGroup, Input, Problem, Space, Subspace, constraint

from dataflow.kernels.test_module_build_spec import ModuleKernel, UnavailableModule, _region


def _ported_region(width: int) -> DataflowRegion:
    region = _region(width)
    requirement = region.internal_inputs[0]
    return replace(
        region,
        inputs=(
            InputInterface(
                Port("in", requirement.operand, region.outputs[0].port.beat_sequence),
                requirement.requirements,
            ),
        ),
    )


class PortedModule(ModuleKernel):
    id = "ported_module"
    region = RegionDeclaration(
        family="test.ported", version="1", construct=_ported_region, width=ModuleKernel.width
    )


class SuppliedDesign(DataflowDesign):
    id = "supplied_test"
    width = Input(int)
    memory = KernelChoice(Subspace(UnavailableModule, width=width), node_id="state")
    compute = KernelChoice(Subspace(PortedModule, width=width))
    link = NetworkEdge(memory.output("out"), EdgeSink(compute.input("in")))
    result = NetworkBoundary(compute.output("out"))


def _design(design_type: type[DataflowDesign] = SuppliedDesign, width: int = 2) -> DataflowDesign:
    class Root(Space):
        width = Problem(int)
        design = Subspace(design_type, width=width)

    return Root.start({Root.width: width}).design


def test_accepted_network_is_validated_once_before_qualified_presentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked: list[DataflowNetwork] = []

    def validate(network: DataflowNetwork) -> NetworkValidationReport:
        checked.append(network)
        return validate_network(network)

    monkeypatch.setattr(design_module, "validate_network", validate)
    design = _design()
    accepted = design.dataflow.accepted_answer
    assert isinstance(accepted, Decided)
    network = accepted.value
    assert checked == [network]

    # Equal bare operand ids belong to different Regions. Topology presents the
    # compute input, while memory's internal requirement has no stream endpoint.
    local = RegionInputRef("state", "value")
    streamed = RegionInputRef("compute", "value")
    positions = frozenset({(0,), (1,)})
    assert exposing_ports(network, local) == ()
    assert unpresented_positions(network, local) == positions
    assert edge_presented_positions(network, local) == frozenset()
    assert boundary_presented_positions(network, local) == frozenset()
    assert edge_presented_positions(network, streamed) == positions
    assert boundary_presented_positions(network, streamed) == frozenset()
    assert unpresented_positions(network, streamed) == frozenset()
    assert checked == [network]

    child = design.kernel("memory")
    assert isinstance(child, Decided)
    assert isinstance(child.value.physical.accepted_answer, Absent)
    # A containing realization may consume this witness without a child build spec.
    assert design.region("memory") == Decided(network.node("state").region)
    assert design.dataflow.accepted_answer == accepted
    assert checked == [network]


class MissingSource(SuppliedDesign):
    link = NetworkEdge(
        SuppliedDesign.memory.output("missing"), EdgeSink(SuppliedDesign.compute.input("in"))
    )


def _wider_region(width: int) -> DataflowRegion:
    return _ported_region(width * 2)


class WiderModule(PortedModule):
    id = "wider_module"
    region = RegionDeclaration(
        family="test.ported", version="1", construct=_wider_region, width=ModuleKernel.width
    )


class MismatchedSequence(DataflowDesign):
    id = "mismatched_test"
    width = Input(int)
    memory = KernelChoice(Subspace(UnavailableModule, width=width))
    compute = KernelChoice(Subspace(WiderModule, width=width))
    link = NetworkEdge(memory.output("out"), EdgeSink(compute.input("in")))
    result = NetworkBoundary(compute.output("out"))


class InternalBoundary(SuppliedDesign):
    internal = NetworkBoundary(SuppliedDesign.memory.input("value"))


@pytest.mark.parametrize("design_type", [MissingSource, MismatchedSequence, InternalBoundary])
def test_a_constructed_network_is_never_an_accepted_projection_before_validation(
    design_type: type[DataflowDesign],
) -> None:
    projection = _design(design_type).dataflow
    assert isinstance(projection.accepted_answer, Absent)
    assert isinstance(projection.output, Decided)
    report = validate_network(projection.output.value)
    assert report.issues
    assert {f"design-network-{issue.code}" for issue in report.issues} <= {
        finding.code for finding in projection.accepted_answer.findings
    }


class RefusingDesign(SuppliedDesign):
    @constraint()
    def supported() -> bool:
        return False

    dataflow_support = ConstraintGroup(supported)


@pytest.mark.parametrize("design_type,width", [(SuppliedDesign, 1), (RefusingDesign, 2)])
def test_canonical_validation_alone_does_not_accept_a_semantically_refused_network(
    design_type: type[DataflowDesign], width: int
) -> None:
    projection = _design(design_type, width).dataflow
    assert isinstance(projection.output, Decided)
    assert not validate_network(projection.output.value).issues
    assert isinstance(projection.accepted_answer, Absent)
    assert any(assessment.verdict is False for assessment in projection.constraints)
