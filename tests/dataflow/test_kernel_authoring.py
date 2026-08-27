# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from finn.dataflow.design import (
    Answer,
    Decided,
    Decision,
    DecisionDomain,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignSpaceSpec,
    DesignSpace,
    Engine,
    EvaluatorSpec,
    ProblemSchema,
    QualifiedPath,
    ReadinessProfile,
    Unresolved,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.kernel import (
    BindingDefinition,
    KernelAuthoringError,
    KernelDefinition,
    RegionDeclaration,
    assemble_kernel_specs,
    instantiate_kernel,
)
from finn.dataflow.mvau.computation import MVAUComputationProfile
from finn.dataflow.mvau.definition import (
    MVAU_COMPUTE_KERNEL,
    MVAU_COMPUTE_KERNEL_SPEC,
    MVAUComputeBinding,
    MVAUComputeKernelPaths,
    MVAUDspBlock,
    build_mvau_compute_kernel_spec,
)
from finn.dataflow.mvau.regions import (
    MVAURegionDeclaration,
    construct_standard_streamed_mvau_region,
)
from finn.dataflow.mvau.source import (
    make_mvau_selection_envelope,
    parse_mvau_selection_envelope,
    reconstitute_mvau_point,
)
from finn.dataflow.parameters.cyclic.definition import (
    CYCLIC_PARAMETER_KERNEL,
    CYCLIC_PARAMETER_KERNEL_SPEC,
    CyclicParameterBinding,
    CyclicParameterKernelPaths,
    CyclicRamStyle,
    build_cyclic_parameter_kernel_spec,
)
from finn.dataflow.region import DataflowRegion, LogicalSchedule, NumericElementType

INT8 = NumericElementType("int", 8)
INT16 = NumericElementType("int", 16)


def _mvau_problem() -> dict[str, object]:
    return {
        str(MVAUComputeKernelPaths.REPETITIONS): 2,
        str(MVAUComputeKernelPaths.MATRIX_WIDTH): 4,
        str(MVAUComputeKernelPaths.MATRIX_HEIGHT): 4,
        str(MVAUComputeKernelPaths.ACTIVATION_ELEMENT_TYPE): INT8,
        str(MVAUComputeKernelPaths.WEIGHT_ELEMENT_TYPE): INT8,
        str(MVAUComputeKernelPaths.ACCUMULATOR_ELEMENT_TYPE): INT16,
        str(MVAUComputeKernelPaths.OUTPUT_ELEMENT_TYPE): INT16,
        str(MVAUComputeKernelPaths.COMPUTATION_PROFILE): (
            MVAUComputationProfile.ACCUMULATOR_INTEGER
        ),
        str(MVAUComputeKernelPaths.WEIGHT_INITIALIZER_AVAILABLE): True,
        str(MVAUComputeKernelPaths.THRESHOLD_INITIALIZER_AVAILABLE): True,
        str(MVAUComputeKernelPaths.TARGET_DSP_BLOCK): MVAUDspBlock.DSP58,
        str(MVAUComputeKernelPaths.WEIGHTS_NARROW): True,
    }


def _mvau_assignments() -> dict[QualifiedPath, object]:
    return {
        MVAUComputeKernelPaths.PE: 2,
        MVAUComputeKernelPaths.SIMD: 2,
        MVAUComputeKernelPaths.REGION_DECLARATION: MVAURegionDeclaration.STANDARD_STREAMED,
        MVAUComputeKernelPaths.BINDING: MVAUComputeBinding.RTL_SOFTVEC,
        MVAUComputeKernelPaths.COMPUTE_PUMPING: False,
    }


def _cyclic_problem() -> dict[str, object]:
    compute = construct_standard_streamed_mvau_region(2, 4, 4, INT8, INT8, INT16, 2, 2)
    return {
        str(CyclicParameterKernelPaths.OUTPUT_PORT): compute.input_interface("weight").port,
        str(CyclicParameterKernelPaths.INITIALIZER_AVAILABLE): True,
        str(CyclicParameterKernelPaths.RUNTIME_WRITABLE): False,
    }


def _cyclic_assignments() -> dict[QualifiedPath, object]:
    return {
        CyclicParameterKernelPaths.BINDING: CyclicParameterBinding.FINN_RTL_MEMSTREAM,
        CyclicParameterKernelPaths.RAM_STYLE: CyclicRamStyle.BRAM,
        CyclicParameterKernelPaths.PUMPED_MEMORY: False,
    }


@pytest.mark.parametrize(
    "definition,rebuilt,problem,assignments",
    [
        (
            MVAU_COMPUTE_KERNEL,
            build_mvau_compute_kernel_spec(),
            _mvau_problem(),
            _mvau_assignments(),
        ),
        (
            CYCLIC_PARAMETER_KERNEL,
            build_cyclic_parameter_kernel_spec(),
            _cyclic_problem(),
            _cyclic_assignments(),
        ),
    ],
)
def test_kernel_definitions_preserve_rebuilt_spec_behavior(
    definition: KernelDefinition,
    rebuilt: DesignSpaceSpec,
    problem: dict[str, object],
    assignments: dict[QualifiedPath, object],
) -> None:
    engine = Engine()
    declared_space = engine.validate(definition.spec)
    rebuilt_space = engine.validate(rebuilt)
    assert engine.inspect(declared_space) == engine.inspect(rebuilt_space)

    declared_point = engine.commit_assignments(
        engine.start(declared_space, problem), assignments
    ).point
    rebuilt_point = engine.commit_assignments(
        engine.start(rebuilt_space, problem), assignments
    ).point
    assert engine.query_property(
        declared_point, definition.selected_region_path
    ) == engine.query_property(rebuilt_point, definition.selected_region_path)
    assert (
        engine.evaluate_constraints(declared_point).answers
        == engine.evaluate_constraints(rebuilt_point).answers
    )

    instance = instantiate_kernel(engine, definition, declared_point)
    assert isinstance(instance, Decided)
    assert instance.value.definition_id == definition.id
    region_answer = engine.query_property(declared_point, definition.selected_region_path)
    assert isinstance(region_answer, Decided)
    assert instance.value.region == region_answer.value


def _synthetic_kernel(prefix: str) -> KernelDefinition:
    region_path = QualifiedPath(f"semantic.{prefix}.region")
    binding_path = QualifiedPath(f"{prefix}.binding")
    selection_path = QualifiedPath(f"binding.{prefix}.selection")
    decision_semantics = as_object_semantics(
        ValueSemantics.immutable_nominal(str, name=f"{prefix} binding")
    )
    region_semantics = as_object_semantics(
        ValueSemantics.immutable_nominal(DataflowRegion, name="DataflowRegion")
    )

    def accepts(value: object, _dependencies: DependencyView) -> Answer[bool]:
        return Decided(value == "only")

    def candidates(_dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        return Decided(("only",))

    def region(_dependencies: DependencyView) -> Answer[object]:
        return Decided(DataflowRegion(LogicalSchedule(()), (), ()))

    binding_ref = DependencyRef.decision("binding", binding_path, decision_semantics)
    spec = DesignSpaceSpec(
        problem_schema=ProblemSchema(),
        decisions=(
            Decision(
                binding_path,
                decision_semantics,
                DecisionDomain((), accepts, EvaluatorSpec((), candidates)),
            ),
        ),
        properties=(
            DerivedProperty(region_path, region_semantics, EvaluatorSpec((), region)),
            DerivedProperty(
                selection_path,
                decision_semantics,
                EvaluatorSpec(
                    (binding_ref,), lambda dependencies: Decided(dependencies["binding"])
                ),
            ),
        ),
        readiness_profiles=(
            ReadinessProfile(f"{prefix}_structural", properties=(region_path,)),
            ReadinessProfile(
                f"{prefix}_binding_ready",
                decisions=(binding_path,),
                properties=(region_path, selection_path),
            ),
        ),
    )
    return KernelDefinition(
        prefix,
        spec,
        (RegionDeclaration("fixed", region_path),),
        (BindingDefinition("only"),),
        region_path,
        binding_path,
        selection_path,
        f"{prefix}_structural",
        f"{prefix}_binding_ready",
    )


def test_synthetic_third_kernel_uses_no_mvau_or_parameter_types() -> None:
    definition = _synthetic_kernel("synthetic")
    engine = Engine()
    space = engine.validate(definition.spec)
    point = engine.start(space, {})
    assert definition.binding_decision_path is not None
    point = engine.commit_assignments(point, {definition.binding_decision_path: "only"}).point
    answer = instantiate_kernel(engine, definition, point)
    assert isinstance(answer, Decided)
    assert answer.value.region == DataflowRegion(LogicalSchedule(()), (), ())


def test_kernel_instance_is_a_selection_not_a_feasibility_verdict() -> None:
    engine = Engine()
    space = engine.validate(MVAU_COMPUTE_KERNEL_SPEC)
    problem = _mvau_problem()
    problem[str(MVAUComputeKernelPaths.ACCUMULATOR_ELEMENT_TYPE)] = INT8
    point = engine.start(space, problem)
    assignments = _mvau_assignments()
    assignments[MVAUComputeKernelPaths.BINDING] = MVAUComputeBinding.LEGACY_HLS_LUT
    del assignments[MVAUComputeKernelPaths.COMPUTE_PUMPING]
    point = engine.commit_assignments(point, assignments).point

    instance = instantiate_kernel(engine, MVAU_COMPUTE_KERNEL, point)
    assert isinstance(instance, Decided)
    assert engine.evaluate_constraint_set(point, "binding_feasibility").verdict is False


def test_duplicate_kernel_and_declaration_paths_are_rejected_deterministically() -> None:
    first = _synthetic_kernel("duplicate")
    second = _synthetic_kernel("duplicate")
    with pytest.raises(KernelAuthoringError) as caught:
        assemble_kernel_specs((first, second))
    codes = [issue.code for issue in caught.value.issues]
    assert codes == [
        issue.code
        for issue in sorted(caught.value.issues, key=lambda issue: (issue.path, issue.code))
    ]
    assert codes.count("declaration-path-duplicate") == 3
    assert codes.count("readiness-profile-name-duplicate") == 2
    assert "kernel-id-duplicate" in codes


@pytest.mark.parametrize(
    "regions,bindings,expected_code",
    [
        (
            (
                RegionDeclaration("same", QualifiedPath("semantic.x.one")),
                RegionDeclaration("same", QualifiedPath("semantic.x.two")),
            ),
            (BindingDefinition("only"),),
            "region-declaration-id-duplicate",
        ),
        (
            (RegionDeclaration("fixed", QualifiedPath("semantic.x.one")),),
            (BindingDefinition("same"), BindingDefinition("same")),
            "binding-definition-id-duplicate",
        ),
    ],
)
def test_duplicate_region_and_binding_ids_are_rejected(
    regions: tuple[RegionDeclaration, ...],
    bindings: tuple[BindingDefinition, ...],
    expected_code: str,
) -> None:
    base = _synthetic_kernel("x")
    if expected_code == "region-declaration-id-duplicate":
        regions = (
            RegionDeclaration("same", base.selected_region_path),
            RegionDeclaration("same", base.selected_region_path),
        )
    with pytest.raises(KernelAuthoringError) as caught:
        KernelDefinition(
            base.id,
            base.spec,
            regions,
            bindings,
            base.selected_region_path,
            base.binding_decision_path,
            base.binding_selection_path,
            base.structural_readiness_profile,
            base.binding_readiness_profile,
        )
    assert expected_code in {issue.code for issue in caught.value.issues}


def test_two_kernels_using_the_same_helper_do_not_share_decisions() -> None:
    left = _synthetic_kernel("left")
    right = _synthetic_kernel("right")
    spec = assemble_kernel_specs((left, right))
    engine = Engine()
    point = engine.start(engine.validate(spec), {})
    assert left.binding_decision_path is not None
    assert right.binding_decision_path is not None
    point = engine.commit_assignments(point, {left.binding_decision_path: "only"}).point
    assert left.binding_decision_path in point.assignments
    assert right.binding_decision_path not in point.assignments


def test_two_mvau_placements_are_independent_and_share_only_the_target_field() -> None:
    shared_target = QualifiedPath("problem.target.dsp_block")
    shared = {MVAUComputeKernelPaths.TARGET_DSP_BLOCK: shared_target}
    left = MVAU_COMPUTE_KERNEL.place("op0", "op0", shared_problem_paths=shared)
    right = MVAU_COMPUTE_KERNEL.place("op1", "op1", shared_problem_paths=shared)
    target_field = next(
        field
        for field in MVAU_COMPUTE_KERNEL.spec.problem_schema.fields
        if field.path == MVAUComputeKernelPaths.TARGET_DSP_BLOCK
    )
    spec = assemble_kernel_specs(
        (left, right),
        additions=DesignSpaceSpec(
            problem_schema=ProblemSchema(
                (
                    type(target_field)(
                        shared_target,
                        target_field.value_semantics,
                        target_field.required,
                        target_field.constraint,
                        target_field.constraint_description,
                    ),
                )
            )
        ),
    )
    engine = Engine()
    space = engine.validate(spec)
    problem: dict[str, object] = {str(shared_target): MVAUDspBlock.DSP58}
    base_problem = _mvau_problem()
    for placement in (left, right):
        for local_path_text, value in base_problem.items():
            local_path = QualifiedPath(local_path_text)
            if local_path == MVAUComputeKernelPaths.TARGET_DSP_BLOCK:
                continue
            problem[str(placement.path(local_path))] = value
    point = engine.start(space, problem)
    assignments: dict[QualifiedPath, object] = {}
    for placement, pe in ((left, 1), (right, 2)):
        for local_path, value in _mvau_assignments().items():
            assignments[placement.path(local_path)] = (
                pe if local_path == MVAUComputeKernelPaths.PE else value
            )
    point = engine.commit_assignments(point, assignments).point

    left_region = engine.query_property(point, left.selected_region_path)
    right_region = engine.query_property(point, right.selected_region_path)
    assert isinstance(left_region, Decided)
    assert isinstance(right_region, Decided)
    assert left_region.value != right_region.value
    assert left.path(MVAUComputeKernelPaths.PE) != right.path(MVAUComputeKernelPaths.PE)
    assert left.path(MVAUComputeKernelPaths.TARGET_DSP_BLOCK) == shared_target
    assert right.path(MVAUComputeKernelPaths.TARGET_DSP_BLOCK) == shared_target
    assert left.binding_readiness_profile is not None
    assert right.binding_readiness_profile is not None
    assert engine.check_readiness(point, left.binding_readiness_profile).ready is True
    assert engine.check_readiness(point, right.binding_readiness_profile).ready is True
    left_instance = instantiate_kernel(engine, left, point)
    assert isinstance(left_instance, Decided)
    assert left_instance.value.definition_id == MVAU_COMPUTE_KERNEL.id
    assert left_instance.value.instance_id == "op0"

    missing_target = dict(problem)
    del missing_target[str(shared_target)]
    unresolved = engine.start(space, missing_target)
    left_assignments = {left.path(path): value for path, value in _mvau_assignments().items()}
    unresolved = engine.commit_assignments(unresolved, left_assignments).point
    target_constraint = left.path(MVAUComputeKernelPaths.BINDING_TARGET_SUPPORTED)
    target_answer = engine.evaluate_constraints(unresolved, (target_constraint,)).answers[
        target_constraint
    ]
    assert isinstance(target_answer, Unresolved)
    assert target_answer.findings[0].path == target_constraint
    assert target_answer.findings[0].trace == (shared_target,)


def _two_placed_mvau_design() -> tuple[
    DesignSpace,
    dict[QualifiedPath, object],
    dict[QualifiedPath, object],
]:
    shared_target = QualifiedPath("problem.target.dsp_block")
    shared = {MVAUComputeKernelPaths.TARGET_DSP_BLOCK: shared_target}
    left = MVAU_COMPUTE_KERNEL.place("op0", "graph.op0", shared_problem_paths=shared)
    right = MVAU_COMPUTE_KERNEL.place("op1", "graph.op1", shared_problem_paths=shared)
    target_field = next(
        field
        for field in MVAU_COMPUTE_KERNEL.spec.problem_schema.fields
        if field.path == MVAUComputeKernelPaths.TARGET_DSP_BLOCK
    )
    spec = assemble_kernel_specs(
        (left, right),
        additions=DesignSpaceSpec(
            problem_schema=ProblemSchema(
                (
                    type(target_field)(
                        shared_target,
                        target_field.value_semantics,
                        target_field.required,
                        target_field.constraint,
                        target_field.constraint_description,
                    ),
                )
            )
        ),
    )
    problem: dict[QualifiedPath, object] = {shared_target: MVAUDspBlock.DSP58}
    for placement in (left, right):
        for local_path_text, value in _mvau_problem().items():
            local_path = QualifiedPath(local_path_text)
            if local_path != MVAUComputeKernelPaths.TARGET_DSP_BLOCK:
                problem[placement.path(local_path)] = value
    assignments: dict[QualifiedPath, object] = {}
    for placement, pe in ((left, 1), (right, 2)):
        for local_path, value in _mvau_assignments().items():
            assignments[placement.path(local_path)] = (
                pe if local_path == MVAUComputeKernelPaths.PE else value
            )
    engine = Engine()
    return engine.validate(spec), problem, assignments


def test_qualified_placed_mvau_paths_round_trip_across_processes(tmp_path: Path) -> None:
    space, problem, assignments = _two_placed_mvau_design()
    engine = Engine()
    point = engine.commit_assignments(engine.start(space, problem), assignments).point
    envelope = make_mvau_selection_envelope("graph.two_mvau", point)
    envelope_path = tmp_path / "placed-selection.json"
    envelope_path.write_text(envelope.to_json())

    restored = reconstitute_mvau_point(
        engine,
        space,
        problem,
        parse_mvau_selection_envelope(envelope.to_json()),
        source_scope_id="graph.two_mvau",
    )
    assert restored.assignments == point.assignments
    assert all(path.value.startswith("graph.op") for path in restored.assignments)

    code = """
import sys
from pathlib import Path
from finn.dataflow.design import Engine
from finn.dataflow.mvau.source import (
    make_mvau_selection_envelope,
    parse_mvau_selection_envelope,
    reconstitute_mvau_point,
)
from dataflow.test_kernel_authoring import _two_placed_mvau_design
space, problem, _assignments = _two_placed_mvau_design()
engine = Engine()
envelope = parse_mvau_selection_envelope(Path(sys.argv[1]).read_text())
point = reconstitute_mvau_point(engine, space, problem, envelope, source_scope_id='graph.two_mvau')
print(make_mvau_selection_envelope('graph.two_mvau', point).to_json())
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path.cwd() / "src"), str(Path.cwd() / "tests"), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(envelope_path)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.stdout.strip() == envelope.to_json()


def test_public_kernel_specs_are_the_definition_specs() -> None:
    assert MVAU_COMPUTE_KERNEL_SPEC is MVAU_COMPUTE_KERNEL.spec
    assert CYCLIC_PARAMETER_KERNEL_SPEC is CYCLIC_PARAMETER_KERNEL.spec
