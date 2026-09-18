# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Only a semantically accepted Kernel projection discharges presentation's precondition."""

from dataclasses import replace
from typing import cast

import pytest

import finn.dataflow.kernels.kernel as design_module
from finn.dataflow._engine import Absent, Decided
from finn.dataflow.kernels import (
    Kernel,
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
    LogicalResult,
    NetworkResult,
    NetworkValidationReport,
    Port,
    RegionInputRef,
    RegionResult,
    boundary_presented_positions,
    edge_presented_positions,
    exposing_ports,
    unpresented_positions,
    validate_network,
)
from finn.dataflow.space import (
    ConstraintGroup,
    Input,
    Problem,
    Projection,
    Readiness,
    Space,
    Subspace,
    constraint,
)
from finn.dataflow.space.dataflow_value_semantics import DATAFLOW_LOGICAL_RESULT_SEMANTICS
from finn.dataflow.space.declarations import Derived, semantics_for

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


class SuppliedKernel(Kernel):
    id = "supplied_test"
    width = Input(int)
    memory = KernelChoice(Subspace(UnavailableModule, width=width), node_id="state")
    compute = KernelChoice(Subspace(PortedModule, width=width))
    link = NetworkEdge(memory.output("out"), EdgeSink(compute.input("in")))
    result = NetworkBoundary(compute.output("out"))


def _design(design_type: type[Kernel] = SuppliedKernel, width: int = 2) -> Kernel:
    class Root(Space):
        width = Problem(int)
        kernel = Subspace(design_type, width=width)

    return Root.start({Root.width: width}).kernel


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
    network = cast(DataflowNetwork, accepted.value)
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

    child = design.child("memory")
    assert isinstance(child, Decided)
    assert isinstance(cast(Kernel, child.value).physical.accepted_answer, Absent)
    # A containing realization may consume this witness without a child build spec.
    assert design.child_region("memory") == Decided(network.node("state").region)
    assert design.dataflow.accepted_answer == accepted
    assert checked == [network]


class MissingSource(SuppliedKernel):
    link = NetworkEdge(
        SuppliedKernel.memory.output("missing"), EdgeSink(SuppliedKernel.compute.input("in"))
    )


def _wider_region(width: int) -> DataflowRegion:
    return _ported_region(width * 2)


class WiderModule(PortedModule):
    id = "wider_module"
    region = RegionDeclaration(
        family="test.ported", version="1", construct=_wider_region, width=ModuleKernel.width
    )


class MismatchedSequence(Kernel):
    id = "mismatched_test"
    width = Input(int)
    memory = KernelChoice(Subspace(UnavailableModule, width=width))
    compute = KernelChoice(Subspace(WiderModule, width=width))
    link = NetworkEdge(memory.output("out"), EdgeSink(compute.input("in")))
    result = NetworkBoundary(compute.output("out"))


class InternalBoundary(SuppliedKernel):
    internal = NetworkBoundary(SuppliedKernel.memory.input("value"))


@pytest.mark.parametrize("design_type", [MissingSource, MismatchedSequence, InternalBoundary])
def test_an_intrinsically_malformed_composition_is_rejected_at_construction(
    design_type: type[Kernel],
) -> None:
    projection = _design(design_type).dataflow
    assert isinstance(projection.accepted_answer, Absent)
    assert isinstance(projection.output, Absent)
    assert {finding.code for finding in projection.output.findings} == {
        "kernel-composition-refused"
    }


class RefusingKernel(SuppliedKernel):
    @constraint()
    def supported() -> bool:
        return False

    logical_support = ConstraintGroup(supported)


@pytest.mark.parametrize("design_type,width", [(SuppliedKernel, 1), (RefusingKernel, 2)])
def test_canonical_validation_alone_does_not_accept_a_semantically_refused_network(
    design_type: type[Kernel], width: int
) -> None:
    projection = _design(design_type, width).dataflow
    assert isinstance(projection.output, Decided)
    assert not validate_network(cast(DataflowNetwork, projection.output.value)).issues
    assert isinstance(projection.accepted_answer, Absent)
    assert any(assessment.verdict is False for assessment in projection.constraints)


class NestedCompositeKernel(Kernel):
    id = "nested_composite"
    width = Input(int)
    inner = KernelChoice(Subspace(SuppliedKernel, width=width))
    result = NetworkBoundary(inner.output("result"))


def test_a_composite_can_nest_a_composite_without_collapsing_regions() -> None:
    nested = _design(NestedCompositeKernel)
    logical = nested.logical.accepted_answer
    assert isinstance(logical, Decided)
    result = cast(NetworkResult, logical.value)
    assert {node.id for node in result.network.nodes} == {
        "inner/state",
        "inner/compute",
    }
    assert tuple(child.use_path.value for child in result.children) == (
        "inner",
        "inner/state",
        "inner/compute",
    )
    boundary = next(item for item in result.network.boundaries if item.id == "result")
    assert boundary.endpoint.node_id == "inner/compute"


def _plain_logical(*, width: int) -> RegionResult:
    return RegionResult(_ported_region(width))


class PlainLogicalLeaf(Space):
    id = "plain_logical_leaf"
    version = "1"
    width = Input(int)

    logical_result: Derived[LogicalResult] = Derived(
        semantics_for(DATAFLOW_LOGICAL_RESULT_SEMANTICS),
        None,
        (("width", width),),
        _plain_logical,
    )
    logical_ready = Readiness(properties=(logical_result,))
    logical = Projection(logical_result, readiness=logical_ready)
    exports = (logical_result,)


class PlainChildComposite(Kernel):
    id = "plain_child_composite"
    width = Input(int)
    child_space = KernelChoice(Subspace(PlainLogicalLeaf, width=width))
    source = NetworkBoundary(child_space.input("in"))
    result = NetworkBoundary(child_space.output("out"))


def test_kernel_choice_consumes_an_equivalent_plain_space_capability() -> None:
    composite = _design(PlainChildComposite)
    logical = composite.logical.accepted_answer
    assert isinstance(logical, Decided)
    result = cast(NetworkResult, logical.value)
    assert tuple(node.id for node in result.network.nodes) == ("child_space",)
