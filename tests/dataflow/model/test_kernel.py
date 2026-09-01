# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""K3: Kernel is one constrained specialization of declarative Space."""

from __future__ import annotations

from typing import cast

from typing_extensions import Self

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow._engine import Decided, DependencyKind, Engine, QualifiedPath, Unresolved
from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.computation import ComputationContract
from finn.dataflow.design.region import DATAFLOW_REGION_SEMANTICS
from finn.dataflow.model.compiler import _Ref, _compile_space
from finn.dataflow.model.declarations import (
    AuthoringError,
    Decision,
    Input,
    Problem,
    Space,
    Use,
    constraint,
    derived,
    divisors_of,
)
from finn.dataflow.model.kernel import Kernel, Parameter, configure_kernel
from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)
from finn.dataflow.spec_algebra import assemble_specs

COMPUTATION = ComputationContract("test.copy")


def _region(extent: int, lanes: int) -> DataflowRegion:
    schedule = LogicalSchedule((ScheduleLevel("fold", extent // lanes),))
    operand = Operand("value", DataType["INT8"], (extent,))
    beats = tuple(
        tuple((fold * lanes + lane,) for lane in range(lanes)) for fold in range(extent // lanes)
    )
    sequence = BeatSequence(lanes, beats)
    requirements = ScheduledInputRequirements(
        {
            ((fold,), (fold * lanes + lane,)): 1
            for fold in range(extent // lanes)
            for lane in range(lanes)
        }
    )
    availability = ScheduledOutputAvailability(
        {
            (fold * lanes + lane,): (fold,)
            for fold in range(extent // lanes)
            for lane in range(lanes)
        }
    )
    return DataflowRegion(
        schedule,
        (InputInterface(Port("input", operand, sequence), requirements),),
        (OutputInterface(Port("output", operand, sequence), availability),),
    )


class ToyKernel(Kernel):
    id = "toy"
    version = "1"
    computation = COMPUTATION

    extent = Input(int)
    lanes = Decision(int, domain=divisors_of(extent))
    pumped = Decision(bool, values=(False, True))

    @derived(DATAFLOW_REGION_SEMANTICS, extent=extent, lanes=lanes)
    def region(*, extent: int, lanes: int) -> DataflowRegion:
        return _region(extent, lanes)

    @derived(int, lanes=lanes)
    def width(*, lanes: int) -> int:
        return lanes * 8

    @constraint(width=width)
    def width_supported(*, width: int) -> bool:
        return width <= 32

    LANES = Parameter(lanes)
    WIDTH = Parameter(width)
    FLAG = Parameter.constant(1, why="the test RTL fixes this mode")
    exports = (lanes, region)

    @classmethod
    def component_abi(cls, configured: Self) -> ComponentABI:
        return ComponentABI(
            "toy", (), tuple((name, str(value)) for name, value in configured.parameters.items())
        )


class Harness(Space):
    extent = Problem(int)


def _compiled(extent_path: str = "problem.test.extent"):
    harness = _compile_space(Harness, "test", problem_namespace="problem.test")
    return harness, _compile_space(
        ToyKernel,
        "test.toy",
        {"extent": cast("_Ref[object]", harness.member("extent"))},
        _allow_problem=False,
    )


def _configured(*, extent: int = 8, lanes: int = 2, pumped: bool = False) -> ToyKernel:
    harness, kernel = _compiled()
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, kernel.spec))),
        {"problem.test.extent": extent},
    )
    point = engine.commit_assignments(
        point,
        {"test.toy.lanes": lanes, "test.toy.pumped": pumped},
    ).point
    answer = configure_kernel(engine, kernel, point)
    assert isinstance(answer, Decided)
    assert isinstance(answer.value, ToyKernel)
    return answer.value


def test_kernel_configures_its_own_region_and_parameters() -> None:
    configured = _configured()
    assert configured.resolved_region == _region(8, 2)
    assert configured.region == configured.resolved_region
    assert configured.computation == COMPUTATION
    assert configured.lanes == 2
    assert configured.pumped is False
    assert configured.LANES == 2
    assert configured.WIDTH == 16
    assert configured.FLAG == 1
    assert dict(configured.parameters) == {"LANES": 2, "WIDTH": 16, "FLAG": 1}
    assert configured.abi.entry_point == "toy"


def test_region_dependency_closure_distinguishes_semantic_and_physical_decisions() -> None:
    _harness, compiled = _compiled()
    region = next(item for item in compiled.spec.properties if item.path.value.endswith(".region"))
    paths = {dependency.path.value for dependency in region.evaluator.dependencies}
    assert "test.toy.lanes" in paths
    assert "test.toy.pumped" not in paths


def test_two_configurations_may_resolve_different_regions() -> None:
    assert _configured(lanes=1).resolved_region != _configured(lanes=2).resolved_region


def test_kernel_feasibility_is_automatic() -> None:
    harness, kernel = _compiled()
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, kernel.spec))),
        {"problem.test.extent": 8},
    )
    point = engine.commit_assignments(
        point,
        {"test.toy.lanes": 8, "test.toy.pumped": False},
    ).point
    answer = configure_kernel(engine, kernel, point)
    assert isinstance(answer, Unresolved)
    assert any(finding.code == "kernel-infeasible" for finding in answer.findings)


def test_configured_kernel_retains_no_engine_point_or_network() -> None:
    configured = _configured()
    values = vars(configured)
    assert not any(isinstance(value, Engine) for value in values.values())
    assert not any(hasattr(value, "design_space") for value in values.values())
    assert "network" not in values


def test_kernel_requires_exactly_one_region_and_computation() -> None:
    class NoRegion(Kernel):
        id = "none"
        computation = COMPUTATION

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("none", ())

    with pytest.raises(AuthoringError, match="derived member named 'region'"):
        _compile_space(NoRegion, "none", {}, _allow_problem=False)

    class WrongRegion(Kernel):
        id = "wrong"
        computation = COMPUTATION

        @derived(int)
        def region() -> int:
            return 1

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("wrong", ())

    with pytest.raises(AuthoringError, match="not a DataflowRegion"):
        _compile_space(WrongRegion, "wrong", {}, _allow_problem=False)

    class NoComputation(Kernel):
        id = "no_computation"

        @derived(DATAFLOW_REGION_SEMANTICS)
        def region() -> DataflowRegion:
            return _region(1, 1)

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("no_computation", ())

    with pytest.raises(AuthoringError, match="ComputationContract"):
        _compile_space(NoComputation, "no_computation", {}, _allow_problem=False)


def test_kernel_input_may_be_an_upstream_decision_and_is_recorded_as_provenance() -> None:
    class DecisionHarness(Space):
        extent = Problem(int)
        supplied = Decision(int, domain=divisors_of(extent))

    harness = _compile_space(
        DecisionHarness,
        "test",
        problem_namespace="problem.test",
    )
    kernel = _compile_space(
        ToyKernel,
        "test.toy",
        {"extent": cast("_Ref[object]", harness.member("supplied"))},
        _allow_problem=False,
    )
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, kernel.spec))),
        {"problem.test.extent": 8},
    )
    point = engine.commit_assignments(
        point,
        {
            "test.supplied": 8,
            "test.toy.lanes": 2,
            "test.toy.pumped": False,
        },
    ).point
    configured = configure_kernel(engine, kernel, point)
    assert isinstance(configured, Decided)
    assert configured.value.imported_decisions == (QualifiedPath("test.supplied"),)


def test_parameter_source_must_belong_to_the_kernel_class() -> None:
    outside = Input(int)

    class Broken(ToyKernel):
        id = "broken"
        OTHER = Parameter(outside)

    harness, _kernel = _compiled()
    with pytest.raises(AuthoringError, match="references a value outside the class"):
        _compile_space(
            Broken,
            "test.broken",
            {"extent": cast("_Ref[object]", harness.member("extent"))},
            _allow_problem=False,
        )


def test_kernel_region_is_implicitly_exported() -> None:
    class Root(Space):
        extent = Problem(int)
        child = Use(ToyKernel, extent=extent)

        @derived(DATAFLOW_REGION_SEMANTICS, child=child.region)
        def observed(*, child: DataflowRegion) -> DataflowRegion:
            return child

    compiled = _compile_space(Root, "root", problem_namespace="problem.root")
    assert compiled.child("child").exported("region").kind is DependencyKind.PROPERTY
