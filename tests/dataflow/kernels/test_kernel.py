# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""K3: Kernel is one constrained specialization of declarative Space."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast


import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow._engine import (
    Absent,
    Decided,
    DependencyKind,
    Engine,
    EvaluationError,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.model.occurrence import is_attached_occurrence
from finn.dataflow.computation import ComputationContract
from finn.dataflow.model.semantics import DATAFLOW_REGION_SEMANTICS
from finn.dataflow.model.compiler import _Ref, _compile_space
from finn.dataflow.model.declarations import (
    AuthoringError,
    ConstraintGroup,
    Decision,
    Input,
    Problem,
    Space,
    Subspace,
    constraint,
    derived,
    divisors_of,
    exported_members,
)
from finn.dataflow.kernels.kernel import (
    Kernel,
    KernelPhysicalResult,
    Parameter,
    PhysicallyUnsupported,
    Region,
    RegionRefused,
    kernel_dataflow,
    kernel_physical,
)
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
from finn.dataflow.model.spec_algebra import assemble_specs

#: What ``component_abi`` is handed: the resolved physical parameter table.
Scalars = Mapping[str, bool | int | float | str]

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

    physical_support = ConstraintGroup(width_supported, name="realizable")

    @classmethod
    def component_abi(cls, parameters: Mapping[str, bool | int | float | str]) -> ComponentABI:
        return ComponentABI(
            "toy", (), tuple((name, str(value)) for name, value in parameters.items())
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


def _configured(*, extent: int = 8, lanes: int = 2, pumped: bool = False) -> KernelPhysicalResult:
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
    answer = kernel_physical(engine, kernel, point).accepted_answer
    assert isinstance(answer, Decided), answer
    return answer.value


def test_kernel_configures_its_own_region_and_parameters() -> None:
    configured = _configured()
    assert configured.region == _region(8, 2)
    assert configured.region_family == "test.copy"
    assert configured.region_version == "1"
    assert configured.computation == COMPUTATION
    assert dict(configured.assignments) == {"pumped": False}
    assert dict(configured.parameters) == {"LANES": 2, "WIDTH": 16, "FLAG": 1}
    assert configured.abi.entry_point == "toy"
    assert configured.build_unit == "toy"
    assert configured.kernel_id == "toy"
    assert configured.kernel_version == "1"


def test_region_dependency_closure_distinguishes_semantic_and_physical_decisions() -> None:
    _harness, compiled = _compiled()
    region = next(item for item in compiled.spec.properties if item.path.value.endswith(".region"))
    paths = {dependency.path.value for dependency in region.evaluator.dependencies}
    assert "test.lanes" in paths
    assert "test.toy.pumped" not in paths


def test_two_configurations_may_resolve_different_regions() -> None:
    assert _configured(lanes=1).region != _configured(lanes=2).region


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
    assessment = kernel_physical(engine, kernel, point)
    # Refused, not incomplete: everything is decided and a constraint said no.
    assert assessment.readiness.ready is True
    assert isinstance(assessment.accepted_answer, Absent)
    assert any(
        finding.code == "projection-constraint-refused"
        for finding in assessment.accepted_answer.findings
    )


def test_the_detached_physical_result_retains_no_engine_point_or_network() -> None:
    configured = _configured()
    values = {name: getattr(configured, name) for name in KernelPhysicalResult.__slots__}
    assert not any(isinstance(value, Engine) for value in values.values())
    assert not any(hasattr(value, "design_space") for value in values.values())
    assert not any(isinstance(value, Space) for value in values.values())
    assert "network" not in values


def _degenerate() -> DataflowRegion:
    return _region(1, 1)


def test_kernel_requires_exactly_one_region_and_computation() -> None:
    class NoRegion(Kernel):
        id = "none"
        computation = COMPUTATION

        @classmethod
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
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
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
            return ComponentABI("generic", ())

    with pytest.raises(AuthoringError, match="Region member named 'region'"):
        _compile_space(GenericRegion, "generic", {}, _allow_problem=False)

    class TwoRegions(Kernel):
        id = "two"
        computation = COMPUTATION
        region = Region(family="test.copy", version="1", construct=_degenerate)
        another = Region(family="test.other", version="1", construct=_degenerate)

        @classmethod
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
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
        fragment = Subspace(RegionFragment)
        region = Region(family="test.copy", version="1", construct=_degenerate)

        @classmethod
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
            return ComponentABI("nested", ())

    with pytest.raises(AuthoringError, match="exactly one DataflowRegion"):
        _compile_space(NestedRegion, "nested", {}, _allow_problem=False)

    class NoComputation(Kernel):
        id = "no_computation"
        region = Region(family="test.copy", version="1", construct=_degenerate)

        @classmethod
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
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
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
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
    answer = kernel_physical(engine, kernel, point).accepted_answer
    assert isinstance(answer, Absent)


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
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
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
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
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
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
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
        folding = Subspace(Folding)
        region = Region(
            family="test.copy",
            version="1",
            construct=_region,
            extent=extent,
            lanes=folding.lanes,
        )

        @classmethod
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
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
    assert configured.region == _region(8, 2)
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
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
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
        def component_abi(cls, parameters: Mapping[str, bool | int | float | str]) -> ComponentABI:
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
    # A malformed ABI is a defect in contributor code, not an infeasible point,
    # so it stays an EvaluationError rather than becoming a rejecting absence.
    with pytest.raises(EvaluationError, match="physical_result"):
        kernel_physical(engine, compiled, point)


def test_parameter_source_must_belong_to_the_kernel_class() -> None:
    outside = Input(int)

    class Broken(ToyKernel):
        id = "broken"
        OTHER = Parameter(outside)

    harness, _kernel = _compiled()
    with pytest.raises(AuthoringError, match="references a value outside its declarations"):
        _compile_space(Broken, "test.broken", _bindings(harness), _allow_problem=False)


def test_kernel_region_is_implicitly_exported() -> None:
    class Root(Space):
        extent = Problem(int)
        lanes = Decision(int, domain=divisors_of(extent))
        nested = Subspace(ToyKernel, extent=extent, lanes=lanes, name="child")

        @derived(DATAFLOW_REGION_SEMANTICS, child=nested.region)
        def observed(*, child: DataflowRegion) -> DataflowRegion:
            return child

    compiled = _compile_space(Root, "root", problem_namespace="problem.root")
    assert compiled.child("nested").exported("region").kind is DependencyKind.PROPERTY
    assert tuple(exported_members(ToyKernel)) == ("region",)


def test_kernel_owns_nested_space_decisions_that_do_not_reach_its_region() -> None:
    class Pipeline(Space):
        stages = Decision(int, values=(1, 2))
        exports = (stages,)

    class CompositeKernel(Kernel):
        id = "composite"
        computation = COMPUTATION
        pipeline = Subspace(Pipeline)
        region = Region(family="test.copy", version="1", construct=_degenerate)

        STAGES = Parameter(pipeline.stages)

        @classmethod
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
            return ComponentABI(
                "composite",
                (),
                (("STAGES", str(parameters["STAGES"])),),
            )

    compiled = _compile_space(CompositeKernel, "composite", {}, _allow_problem=False)
    engine = Engine()
    point = engine.start(engine.validate(compiled.spec), {})
    pending = kernel_physical(engine, compiled, point).accepted_answer
    assert isinstance(pending, Unresolved)
    point = engine.commit_assignments(point, {"composite.pipeline.stages": 2}).point
    configured = kernel_physical(engine, compiled, point).accepted_answer
    assert isinstance(configured, Decided)
    assert configured.value.parameters["STAGES"] == 2
    # A Decision inside a helper Space the Kernel owns is the Kernel's, so it is
    # an assignment it carries and not an import it depends on.
    assert dict(configured.value.assignments) == {}
    assert configured.value.imported_decisions == ()


def test_a_local_decision_may_not_gate_what_the_region_depends_on() -> None:
    """Applicability counts as reaching: presence is part of the contract."""

    class Folding(Space):
        supplied = Input(int)

        @derived(int, supplied=supplied)
        def lanes(*, supplied: int) -> int:
            return supplied

        exports = (lanes,)

    class GatedHelper(Kernel):
        id = "gated_helper"
        computation = COMPUTATION
        extent = Input(int)
        lanes = Input(int)
        enabled = Decision(bool, values=(False, True))
        folding = Subspace(Folding, supplied=lanes, when=enabled)
        region = Region(
            family="test.copy",
            version="1",
            construct=_region,
            extent=extent,
            lanes=folding.lanes,
        )

        @classmethod
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
            return ComponentABI("gated_helper", ())

    harness, _kernel = _compiled()
    with pytest.raises(AuthoringError, match="including whether it applies at all"):
        _compile_space(GatedHelper, "test.gated", _bindings(harness), _allow_problem=False)


def test_a_local_decision_may_not_gate_the_region_property_itself() -> None:
    class GatedRegion(Kernel):
        id = "gated_region"
        computation = COMPUTATION
        extent = Input(int)
        lanes = Input(int)
        enabled = Decision(bool, values=(False, True))

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
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
            return ComponentABI("gated_region", ())

    harness, _kernel = _compiled()
    compiled = _compile_space(GatedRegion, "test.plain", _bindings(harness), _allow_problem=False)
    assert compiled.member("region") is not None


def test_an_outer_gate_over_the_region_stays_legal() -> None:
    """A Design's segment condition is not Kernel-owned, so it is not a leak."""

    class Conditional(Space):
        extent = Problem(int)
        lanes = Decision(int, domain=divisors_of(extent))
        present = Decision(bool, values=(False, True))
        nested = Subspace(ToyKernel, extent=extent, lanes=lanes, when=present, name="child")

    compiled = _compile_space(Conditional, "outer", problem_namespace="problem.outer")
    assert compiled.child("nested").exported("region").kind is DependencyKind.PROPERTY


def test_only_a_deliberate_refusal_becomes_a_rejecting_absence() -> None:
    """A constructor defect must not read as an ordinary infeasible point."""

    def refusing(extent: int) -> DataflowRegion:
        raise RegionRefused(f"{extent} is not a supported extent")

    def defective(extent: int) -> DataflowRegion:
        return cast("DataflowRegion", (1, 2, 3)[extent])

    class Refusing(Kernel):
        id = "refusing"
        computation = COMPUTATION
        extent = Input(int)
        region = Region(family="test.copy", version="1", construct=refusing, extent=extent)

        @classmethod
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
            return ComponentABI("refusing", ())

    class Defective(Kernel):
        id = "defective"
        computation = COMPUTATION
        extent = Input(int)
        region = Region(family="test.copy", version="1", construct=defective, extent=extent)

        @classmethod
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
            return ComponentABI("defective", ())

    harness = _compile_space(Harness, "test", problem_namespace="problem.test")
    binding = {"extent": cast("_Ref[object]", harness.member("extent"))}
    engine = Engine()

    refusing_kernel = _compile_space(Refusing, "test.refusing", binding, _allow_problem=False)
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, refusing_kernel.spec))),
        {"problem.test.extent": 8},
    )
    answer = engine.query_property(point, "semantic.test.refusing.region")
    assert isinstance(answer, Absent)
    assert {finding.code for finding in answer.findings} == {"kernel-region-refused"}

    defective_kernel = _compile_space(Defective, "test.defective", binding, _allow_problem=False)
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, defective_kernel.spec))),
        {"problem.test.extent": 8},
    )
    with pytest.raises(EvaluationError):
        engine.query_property(point, "semantic.test.defective.region")


def test_the_two_projections_ask_two_different_questions() -> None:
    """A Region resolves before any physical choice, and the two say so apart."""

    harness, kernel = _compiled()
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, kernel.spec))),
        {"problem.test.extent": 8},
    )
    point = engine.commit_assignments(point, {"test.lanes": 2}).point

    # `pumped` is uncommitted, so there is no build unit yet ...
    physical = kernel_physical(engine, kernel, point)
    assert isinstance(physical.accepted_answer, Unresolved)

    # ... and the Region is nevertheless entirely decided.
    dataflow = kernel_dataflow(engine, kernel, point)
    assert dataflow.readiness.ready is True
    assert dataflow.accepted_answer == Decided(_region(8, 2))


def test_a_valid_region_does_not_oblige_a_realizable_kernel() -> None:
    """An implementation with no wiring for this point still has its Region."""

    class Unbuildable(ToyKernel):
        id = "unbuildable"

        @classmethod
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
            raise PhysicallyUnsupported("no wiring for this parameter table")

    harness, _kernel = _compiled()
    compiled = _compile_space(
        Unbuildable, "test.unbuildable", _bindings(harness), _allow_problem=False
    )
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, compiled.spec))),
        {"problem.test.extent": 8},
    )
    point = engine.commit_assignments(
        point, {"test.lanes": 2, "test.unbuildable.pumped": False}
    ).point

    assert kernel_dataflow(engine, compiled, point).accepted_answer == Decided(_region(8, 2))
    physical = kernel_physical(engine, compiled, point).accepted_answer
    assert isinstance(physical, Absent)
    assert any(finding.code == "kernel-physically-unsupported" for finding in physical.findings)


def test_the_physical_result_carries_its_import_provenance() -> None:
    configured = _configured()
    assert QualifiedPath("test.lanes") in configured.imported_decisions
    assert QualifiedPath("test.toy.pumped") not in configured.imported_decisions


def test_an_attached_kernel_occurrence_answers_its_own_declarations() -> None:
    """The Kernel is an ordinary Space: started, it resolves through the runtime."""

    class Placed(Space):
        extent = Problem(int)
        lanes = Decision(int, domain=divisors_of(extent))
        toy = Subspace(ToyKernel, extent=extent, lanes=lanes)

    root = Placed.start({Placed.extent: 8}).assign(Placed.lanes, 2)
    occurrence = root.toy
    assert is_attached_occurrence(occurrence)
    assert type(occurrence) is ToyKernel
    assert occurrence.LANES == 2
    assert occurrence.FLAG == 1
    assert occurrence.region == _region(8, 2)
    assert occurrence.dataflow.accepted_answer == Decided(_region(8, 2))
    assert isinstance(occurrence.physical.accepted_answer, Unresolved)
    built = occurrence.assign(ToyKernel.pumped, True).physical.accepted_answer
    assert isinstance(built, Decided)
    assert built.value.kernel_id == "toy"


# -- which projection each constraint gates -----------------------------------
#
# A Kernel classifies every constraint it declares, and a constraint may be
# classified *twice*: a folding rule that is both a semantic requirement and a
# build feasibility one is ordinary.  Declaring it in both groups must add a
# projection, never remove one -- and the way it removed one was that
# "physical-only" was read as membership in ``physical_support`` rather than as
# a difference, so a shared constraint was excluded from the Design's Network
# question that its author had explicitly said it gates.


def _classified(*, shared: bool, physical: bool) -> frozenset[QualifiedPath]:
    """One Kernel's physical-only set, for one classification of one constraint."""

    class Classified(Kernel):
        id = f"classified_{int(shared)}{int(physical)}"
        computation = COMPUTATION
        extent = Input(int)
        lanes = Input(int)
        region = Region(
            family="test.copy", version="1", construct=_region, extent=extent, lanes=lanes
        )

        @constraint(lanes=lanes)
        def lanes_supported(*, lanes: int) -> bool:
            return lanes <= 2

        if shared:
            dataflow_support = ConstraintGroup(lanes_supported)
        if physical:
            physical_support = ConstraintGroup(lanes_supported, name="realizable")

        @classmethod
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
            return ComponentABI("classified", ())

    harness, _kernel = _compiled()
    compiled = _compile_space(
        Classified, f"test.{Classified.id}", _bindings(harness), _allow_problem=False
    )
    return compiled.extension.physical_only_constraints


def test_a_constraint_in_physical_support_alone_is_physical_only() -> None:
    paths = _classified(shared=False, physical=True)
    assert [path.value for path in paths] == ["constraint.test.classified_01.lanes_supported"]


def test_a_constraint_in_both_groups_is_not_physical_only() -> None:
    """The whole correction: two classifications add a projection, not subtract."""

    assert _classified(shared=True, physical=True) == frozenset()


def test_a_kernel_whose_groups_coincide_has_no_physical_only_constraints() -> None:
    assert _classified(shared=True, physical=False) == frozenset()
    assert _classified(shared=True, physical=True) == frozenset()


def test_a_shared_constraint_still_gates_both_of_the_kernels_own_projections() -> None:
    """Not physical-only does not mean not physical: it still refuses the build."""

    class Shared(Kernel):
        id = "shared_gate"
        computation = COMPUTATION
        extent = Input(int)
        lanes = Input(int)
        region = Region(
            family="test.copy", version="1", construct=_region, extent=extent, lanes=lanes
        )

        @constraint(lanes=lanes)
        def lanes_supported(*, lanes: int) -> bool:
            return lanes <= 2

        dataflow_support = ConstraintGroup(lanes_supported)
        physical_support = ConstraintGroup(lanes_supported, name="realizable")

        @classmethod
        def component_abi(cls, parameters: Scalars) -> ComponentABI:
            return ComponentABI("shared", ())

    harness, _kernel = _compiled()
    compiled = _compile_space(Shared, "test.shared_gate", _bindings(harness), _allow_problem=False)
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, compiled.spec))),
        {"problem.test.extent": 8},
    )
    refusing = engine.commit_assignments(point, {"test.lanes": 4}).point

    dataflow = kernel_dataflow(engine, compiled, refusing)
    assert isinstance(dataflow.accepted_answer, Absent)
    physical = kernel_physical(engine, compiled, refusing)
    assert isinstance(physical.accepted_answer, Absent)

    accepting = engine.commit_assignments(point, {"test.lanes": 2}).point
    assert kernel_dataflow(engine, compiled, accepting).accepted_answer == Decided(_region(8, 2))
