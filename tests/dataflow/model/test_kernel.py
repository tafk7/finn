# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""K3: Kernel is one constrained specialization of declarative Space."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from typing_extensions import Self

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow._engine import (
    Decided,
    DependencyKind,
    Engine,
    EvaluationError,
    QualifiedPath,
    Unresolved,
)
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
    exported_members,
)
from finn.dataflow.model.kernel import Kernel, Parameter, Region, configure_kernel
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
    lanes = Input(int)
    pumped = Decision(bool, values=(False, True))

    region = Region(
        family="test.copy",
        version="1",
        construct=_region,
        extent=extent,
        lanes=lanes,
    )

    @derived(int, lanes=lanes)
    def width(*, lanes: int) -> int:
        return lanes * 8

    @constraint(width=width)
    def width_supported(*, width: int) -> bool:
        return width <= 32

    LANES = Parameter(lanes)
    WIDTH = Parameter(width)
    FLAG = Parameter.constant(1, why="the test RTL fixes this mode")

    @classmethod
    def component_abi(cls, configured: Self) -> ComponentABI:
        return ComponentABI(
            "toy", (), tuple((name, str(value)) for name, value in configured.parameters.items())
        )


class Harness(Space):
    """Standing in for the Design: it owns every Region-visible choice."""

    extent = Problem(int)
    lanes = Decision(int, domain=divisors_of(extent))


def _bindings(harness) -> dict[str, _Ref[object]]:
    return {name: cast("_Ref[object]", harness.member(name)) for name in ("extent", "lanes")}


def _compiled():
    harness = _compile_space(Harness, "test", problem_namespace="problem.test")
    return harness, _compile_space(
        ToyKernel,
        "test.toy",
        _bindings(harness),
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
        {"test.lanes": lanes, "test.toy.pumped": pumped},
    ).point
    answer = configure_kernel(engine, kernel, point)
    assert isinstance(answer, Decided)
    assert isinstance(answer.value, ToyKernel)
    return answer.value


def test_kernel_configures_its_own_region_and_parameters() -> None:
    configured = _configured()
    assert configured.resolved_region == _region(8, 2)
    assert configured.region == configured.resolved_region
    assert configured.region_family == "test.copy"
    assert configured.region_version == "1"
    assert configured.computation == COMPUTATION
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
    assert "test.lanes" in paths
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
        {"test.lanes": 8, "test.toy.pumped": False},
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


def _degenerate() -> DataflowRegion:
    return _region(1, 1)


def test_kernel_requires_exactly_one_region_and_computation() -> None:
    class NoRegion(Kernel):
        id = "none"
        computation = COMPUTATION

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("none", ())

    with pytest.raises(AuthoringError, match="Region member named 'region'"):
        _compile_space(NoRegion, "none", {}, _allow_problem=False)

    class GenericRegion(Kernel):
        id = "generic"
        computation = COMPUTATION

        @derived(DATAFLOW_REGION_SEMANTICS)
        def region() -> DataflowRegion:
            return _degenerate()

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("generic", ())

    with pytest.raises(AuthoringError, match="Region member named 'region'"):
        _compile_space(GenericRegion, "generic", {}, _allow_problem=False)

    class TwoRegions(Kernel):
        id = "two"
        computation = COMPUTATION
        region = Region(family="test.copy", version="1", construct=_degenerate)
        another = Region(family="test.other", version="1", construct=_degenerate)

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("two", ())

    with pytest.raises(AuthoringError, match="exactly one DataflowRegion"):
        _compile_space(TwoRegions, "two", {}, _allow_problem=False)

    class RegionFragment(Space):
        @derived(DATAFLOW_REGION_SEMANTICS)
        def nested_region() -> DataflowRegion:
            return _degenerate()

    class NestedRegion(Kernel):
        id = "nested"
        computation = COMPUTATION
        fragment = Use(RegionFragment)
        region = Region(family="test.copy", version="1", construct=_degenerate)

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("nested", ())

    with pytest.raises(AuthoringError, match="exactly one DataflowRegion"):
        _compile_space(NestedRegion, "nested", {}, _allow_problem=False)

    class NoComputation(Kernel):
        id = "no_computation"
        region = Region(family="test.copy", version="1", construct=_degenerate)

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("no_computation", ())

    with pytest.raises(AuthoringError, match="ComputationContract"):
        _compile_space(NoComputation, "no_computation", {}, _allow_problem=False)

    class NoAbi(Kernel):
        id = "no_abi"
        computation = COMPUTATION
        region = Region(family="test.copy", version="1", construct=_degenerate)

    with pytest.raises(AuthoringError, match=r"declare a component_abi\(\)"):
        _compile_space(NoAbi, "no_abi", {}, _allow_problem=False)

    class OwnsProblem(Kernel):
        id = "owns_problem"
        computation = COMPUTATION
        extent = Problem(int)
        region = Region(
            family="test.copy",
            version="1",
            construct=_region,
            extent=extent,
            lanes=extent,
        )

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("owns_problem", ())

    with pytest.raises(AuthoringError, match="external facts through Input"):
        _compile_space(
            OwnsProblem,
            "owns_problem",
            {},
            problem_namespace="problem.owns_problem",
        )


# -- the Region declaration ---------------------------------------------------


def test_region_needs_a_non_empty_family_and_version() -> None:
    with pytest.raises(AuthoringError, match="non-empty family"):
        Region(family="", version="1", construct=_degenerate)
    with pytest.raises(AuthoringError, match="non-empty version"):
        Region(family="test.copy", version="", construct=_degenerate)


def test_region_constructor_signature_must_match_its_dependencies() -> None:
    extent = Input(int)
    with pytest.raises(AuthoringError, match="unbound parameters \\['lanes'\\]"):
        Region(family="test.copy", version="1", construct=_region, extent=extent)
    with pytest.raises(AuthoringError, match="unused dependencies \\['depth'\\]"):
        Region(
            family="test.copy",
            version="1",
            construct=_region,
            extent=extent,
            lanes=extent,
            depth=extent,
        )
    with pytest.raises(AuthoringError, match="only named parameters"):
        Region(family="test.copy", version="1", construct=lambda **kwargs: _degenerate())
    with pytest.raises(AuthoringError, match="callable constructor"):
        Region(family="test.copy", version="1", construct=None)  # type: ignore[arg-type]


def test_region_family_and_version_survive_onto_the_configured_kernel() -> None:
    configured = _configured()
    assert (configured.region_family, configured.region_version) == ("test.copy", "1")
    assert "toy" not in configured.region_family


def test_a_refusing_region_constructor_is_a_refusal_not_a_crash() -> None:
    harness, kernel = _compiled()
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, kernel.spec))),
        {"problem.test.extent": 8},
    )
    point = engine.commit_assignments(point, {"test.lanes": 8, "test.toy.pumped": False}).point
    answer = configure_kernel(engine, kernel, point)
    assert isinstance(answer, Unresolved)


def test_a_region_constructor_returning_the_wrong_type_is_rejected() -> None:
    class WrongResult(Kernel):
        id = "wrong_result"
        computation = COMPUTATION
        region = Region(
            family="test.copy",
            version="1",
            construct=cast("Callable[[], DataflowRegion]", lambda: 1),
        )

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("wrong_result", ())

    compiled = _compile_space(WrongResult, "wrong_result", {}, _allow_problem=False)
    engine = Engine()
    point = engine.start(engine.validate(compiled.spec), {})
    with pytest.raises(EvaluationError):
        engine.query_property(point, "semantic.wrong_result.region")


# -- Design-owned semantics ---------------------------------------------------


def test_a_kernel_local_decision_may_not_reach_its_region() -> None:
    class DirectDependence(Kernel):
        id = "direct"
        computation = COMPUTATION
        extent = Input(int)
        lanes = Decision(int, values=(1, 2))
        region = Region(
            family="test.copy",
            version="1",
            construct=_region,
            extent=extent,
            lanes=lanes,
        )

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("direct", ())

    harness, _kernel = _compiled()
    with pytest.raises(AuthoringError, match="lets its own Decision"):
        _compile_space(
            DirectDependence,
            "test.direct",
            {"extent": cast("_Ref[object]", harness.member("extent"))},
            _allow_problem=False,
        )


def test_a_transitive_kernel_local_decision_may_not_reach_its_region() -> None:
    class TransitiveDependence(Kernel):
        id = "transitive"
        computation = COMPUTATION
        extent = Input(int)
        lanes = Decision(int, values=(1, 2))

        @derived(int, lanes=lanes)
        def widened(*, lanes: int) -> int:
            return lanes

        region = Region(
            family="test.copy",
            version="1",
            construct=_region,
            extent=extent,
            lanes=widened,
        )

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("transitive", ())

    harness, _kernel = _compiled()
    with pytest.raises(AuthoringError, match="lets its own Decision"):
        _compile_space(
            TransitiveDependence,
            "test.transitive",
            {"extent": cast("_Ref[object]", harness.member("extent"))},
            _allow_problem=False,
        )


def test_a_nested_helper_decision_may_not_reach_its_region() -> None:
    class Folding(Space):
        lanes = Decision(int, values=(1, 2))
        exports = (lanes,)

    class NestedDependence(Kernel):
        id = "nested_dependence"
        computation = COMPUTATION
        extent = Input(int)
        folding = Use(Folding)
        region = Region(
            family="test.copy",
            version="1",
            construct=_region,
            extent=extent,
            lanes=folding.lanes,
        )

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("nested_dependence", ())

    harness, _kernel = _compiled()
    with pytest.raises(AuthoringError, match="lets its own Decision"):
        _compile_space(
            NestedDependence,
            "test.nested_dependence",
            {"extent": cast("_Ref[object]", harness.member("extent"))},
            _allow_problem=False,
        )


def test_a_supplied_decision_reaching_the_region_stays_valid() -> None:
    configured = _configured(lanes=2)
    assert configured.resolved_region == _region(8, 2)
    assert configured.imported_decisions == (QualifiedPath("test.lanes"),)


def test_a_kernel_may_not_publish_exports_besides_its_region() -> None:
    class Publishes(Kernel):
        id = "publishes"
        computation = COMPUTATION
        extent = Input(int)
        pumped = Decision(bool, values=(False, True))
        region = Region(
            family="test.copy",
            version="1",
            construct=_region,
            extent=extent,
            lanes=extent,
        )
        exports = (pumped,)

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("publishes", ())

    harness, _kernel = _compiled()
    with pytest.raises(AuthoringError, match="may not publish exports besides its Region"):
        _compile_space(
            Publishes,
            "test.publishes",
            {"extent": cast("_Ref[object]", harness.member("extent"))},
            _allow_problem=False,
        )


# -- unchanged Kernel behavior ------------------------------------------------


def test_kernel_abi_must_expose_every_resolved_physical_parameter() -> None:
    class IncompleteAbi(ToyKernel):
        id = "incomplete_abi"

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("toy", ())

    harness, _kernel = _compiled()
    compiled = _compile_space(
        IncompleteAbi,
        "test.incomplete",
        _bindings(harness),
        _allow_problem=False,
    )
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, compiled.spec))),
        {"problem.test.extent": 8},
    )
    point = engine.commit_assignments(
        point,
        {"test.lanes": 2, "test.incomplete.pumped": False},
    ).point
    with pytest.raises(AuthoringError, match="exact resolved physical parameter table"):
        configure_kernel(engine, compiled, point)


def test_parameter_source_must_belong_to_the_kernel_class() -> None:
    outside = Input(int)

    class Broken(ToyKernel):
        id = "broken"
        OTHER = Parameter(outside)

    harness, _kernel = _compiled()
    with pytest.raises(AuthoringError, match="references a value outside the class"):
        _compile_space(Broken, "test.broken", _bindings(harness), _allow_problem=False)


def test_kernel_region_is_implicitly_exported() -> None:
    class Root(Space):
        extent = Problem(int)
        lanes = Decision(int, domain=divisors_of(extent))
        child = Use(ToyKernel, extent=extent, lanes=lanes)

        @derived(DATAFLOW_REGION_SEMANTICS, child=child.region)
        def observed(*, child: DataflowRegion) -> DataflowRegion:
            return child

    compiled = _compile_space(Root, "root", problem_namespace="problem.root")
    assert compiled.child("child").exported("region").kind is DependencyKind.PROPERTY
    assert tuple(exported_members(ToyKernel)) == ("region",)


def test_kernel_owns_nested_space_decisions_that_do_not_reach_its_region() -> None:
    class Pipeline(Space):
        stages = Decision(int, values=(1, 2))
        exports = (stages,)

    class CompositeKernel(Kernel):
        id = "composite"
        computation = COMPUTATION
        pipeline = Use(Pipeline)
        region = Region(family="test.copy", version="1", construct=_degenerate)

        STAGES = Parameter(pipeline.stages)

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI(
                "composite",
                (),
                (("STAGES", str(configured.STAGES)),),
            )

    compiled = _compile_space(CompositeKernel, "composite", {}, _allow_problem=False)
    engine = Engine()
    point = engine.start(engine.validate(compiled.spec), {})
    pending = configure_kernel(engine, compiled, point)
    assert isinstance(pending, Unresolved)
    point = engine.commit_assignments(point, {"composite.pipeline.stages": 2}).point
    configured = configure_kernel(engine, compiled, point)
    assert isinstance(configured, Decided)
    assert configured.value.STAGES == 2
    assert dict(configured.value.assignments) == {QualifiedPath("composite.pipeline.stages"): 2}
    assert configured.value.imported_decisions == ()
