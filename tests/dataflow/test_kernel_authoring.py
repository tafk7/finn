# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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
    Engine,
    EvaluatorSpec,
    ProblemSchema,
    QualifiedPath,
    ReadinessProfile,
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
from finn.dataflow.parameters.cyclic.definition import (
    CYCLIC_PARAMETER_KERNEL,
    CYCLIC_PARAMETER_KERNEL_SPEC,
    CyclicParameterBinding,
    CyclicParameterKernelPaths,
    CyclicParameterRegionDeclaration,
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
        CyclicParameterKernelPaths.REGION_DECLARATION: (CyclicParameterRegionDeclaration.FULL_TILE),
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
    witness_path = QualifiedPath(f"binding.{prefix}.witness")
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
                witness_path,
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
                properties=(region_path, witness_path),
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
        witness_path,
        f"{prefix}_structural",
        f"{prefix}_binding_ready",
    )


def test_synthetic_third_kernel_uses_no_mvau_or_parameter_types() -> None:
    definition = _synthetic_kernel("synthetic")
    engine = Engine()
    space = engine.validate(definition.spec)
    point = engine.start(space, {})
    point = engine.commit_assignments(point, {definition.binding_decision_path: "only"}).point
    answer = instantiate_kernel(engine, definition, point)
    assert isinstance(answer, Decided)
    assert answer.value.region == DataflowRegion(LogicalSchedule(()), (), ())


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
            base.binding_witness_path,
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
    point = engine.commit_assignments(point, {left.binding_decision_path: "only"}).point
    assert left.binding_decision_path in point.assignments
    assert right.binding_decision_path not in point.assignments


def test_public_kernel_specs_are_the_definition_specs() -> None:
    assert MVAU_COMPUTE_KERNEL_SPEC is MVAU_COMPUTE_KERNEL.spec
    assert CYCLIC_PARAMETER_KERNEL_SPEC is CYCLIC_PARAMETER_KERNEL.spec
