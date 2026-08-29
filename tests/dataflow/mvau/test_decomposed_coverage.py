# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What ``DotProductKernel`` will and will not claim it can build.

A Kernel that declares no conditions claims everything, which is worse than
claiming nothing: the pool reports a design point feasible and the provider
inventory overclaims what it can realize.  These pin the two halves of the
split -- what the *source* must be for this Kernel to serve it at all, and what
the *provider* can build on a given board.
"""

from __future__ import annotations

import pytest

from dataflow.mvau_op_facts import compute_pool_context
from finn.dataflow.authoring import assemble_specs
from finn.dataflow.design import Decided, DesignPoint, Engine, QualifiedPath
from finn.dataflow.mvau.decomposed import (
    ActivationReplayKernel,
    DecomposedMVAUPools,
    DotProductKernel,
    build_decomposed_mvau_pools,
)
from finn.dataflow.mvau_problem import (
    MVAUComputationProfile,
    MVAUDspBlock,
    MVAUProblemPaths,
)
from finn.dataflow.region import NumericElementType

INT8 = NumericElementType("int", 8)
INT16 = NumericElementType("int", 16)
INT20 = NumericElementType("int", 20)
INT32 = NumericElementType("int", 32)
INT64 = NumericElementType("int", 64)
UINT8 = NumericElementType("uint", 8)
FLOAT16 = NumericElementType("float", 16)
BIPOLAR = NumericElementType("bipolar", 1)


def _point(
    *,
    activation: NumericElementType = INT8,
    weight: NumericElementType = INT8,
    accumulator: NumericElementType = INT16,
    output: NumericElementType = INT16,
    profile: MVAUComputationProfile = MVAUComputationProfile.ACCUMULATOR_INTEGER,
    target: MVAUDspBlock = MVAUDspBlock.DSP58,
    narrow: bool = False,
    pe: int = 2,
    simd: int = 2,
    pumping: bool = False,
) -> tuple[Engine, DesignPoint, DecomposedMVAUPools]:
    pools = build_decomposed_mvau_pools()
    engine = Engine()
    space = engine.validate(
        assemble_specs(
            (
                compute_pool_context(),
                pools.dot_product.build_spec(),
                pools.activation_replay.build_spec(),
            )
        )
    )
    problem: dict[QualifiedPath, object] = {
        MVAUProblemPaths.REPETITIONS: 2,
        MVAUProblemPaths.MATRIX_WIDTH: 8,
        MVAUProblemPaths.MATRIX_HEIGHT: 4,
        MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE: activation,
        MVAUProblemPaths.WEIGHT_ELEMENT_TYPE: weight,
        MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE: accumulator,
        MVAUProblemPaths.OUTPUT_ELEMENT_TYPE: output,
        MVAUProblemPaths.COMPUTATION_PROFILE: profile,
        MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE: True,
        MVAUProblemPaths.RUNTIME_WRITABLE: False,
        MVAUProblemPaths.INITIALIZER_EXCLUDES_MINIMUM: narrow,
        MVAUProblemPaths.TARGET_DSP_BLOCK: target,
        MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS: 4.0,
    }
    point = engine.start(space, problem)
    point = engine.commit_assignments(
        point,
        {
            pools.dot_product.paths.kernel: DotProductKernel.id,
            pools.activation_replay.paths.kernel: ActivationReplayKernel.id,
            pools.pe.path: pe,
            pools.simd.path: simd,
            pools.compute_pumping.path: pumping,
        },
    ).point
    return engine, point, pools


def _rejected(engine: Engine, point: DesignPoint, pools: DecomposedMVAUPools) -> set[str]:
    assessment = engine.evaluate_constraint_set(point, pools.dot_product.feasibility_constraint_set)
    return {
        str(path).rsplit(".", 1)[-1]
        for path, answer in assessment.answers.items()
        if isinstance(answer, Decided) and answer.value is False
    }


def _feasible(**overrides: object) -> bool:
    engine, point, pools = _point(**overrides)  # type: ignore[arg-type]
    return (
        engine.evaluate_constraint_set(point, pools.dot_product.feasibility_constraint_set).verdict
        is True
    )


# -- the baseline the rest is measured against -------------------------------


def test_a_supported_point_is_feasible() -> None:
    assert _feasible() is True


# -- source admission: what this Kernel will serve ---------------------------


@pytest.mark.parametrize(
    "profile",
    [MVAUComputationProfile.FUSED_THRESHOLD, MVAUComputationProfile.BIPOLAR_XNOR_ACCUMULATOR],
)
def test_only_the_accumulator_profile_is_served(profile: MVAUComputationProfile) -> None:
    """This slice covers accumulator output; the others are not implemented."""

    engine, point, pools = _point(profile=profile)
    assert "computation_supported" in _rejected(engine, point, pools)


@pytest.mark.parametrize(
    ("activation", "weight"),
    [(FLOAT16, FLOAT16), (INT8, FLOAT16), (BIPOLAR, BIPOLAR), (INT8, BIPOLAR)],
)
def test_non_integer_operands_are_refused(
    activation: NumericElementType, weight: NumericElementType
) -> None:
    engine, point, pools = _point(activation=activation, weight=weight)
    assert "numeric_supported" in _rejected(engine, point, pools)


def test_unsigned_activations_are_served() -> None:
    """Unsigned is integer arithmetic; ``SIGNED_ACTIVATIONS`` carries the sign."""

    assert _feasible(activation=UINT8) is True


def test_an_output_that_is_not_the_accumulator_is_refused() -> None:
    """With no threshold to requantize through, the output *is* the accumulator."""

    engine, point, pools = _point(accumulator=INT16, output=INT8)
    assert "accumulator_output_type_supported" in _rejected(engine, point, pools)


def test_source_admission_reads_no_target_or_decision() -> None:
    """Admission is asked before a board or a folding is known."""

    pools = build_decomposed_mvau_pools()
    declaration = pools.dot_product.kernels[0]
    admission = set(declaration.source_admission_constraints)
    by_path = {item.path: item for item in declaration.spec.constraints}
    decisions = {item.path for item in declaration.spec.decisions}

    for path in admission:
        for dependency in by_path[path].evaluator.dependencies:
            assert dependency.path not in decisions
            assert dependency.path != MVAUProblemPaths.TARGET_DSP_BLOCK
            assert dependency.path != MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS


# -- provider coverage: what the board can build -----------------------------


def test_an_accumulator_wider_than_the_target_is_refused() -> None:
    """DSP48E2 carries 48 accumulator bits; the same point fits on DSP58."""

    engine, point, pools = _point(accumulator=INT64, output=INT64, target=MVAUDspBlock.DSP48E2)
    assert "width_supported" in _rejected(engine, point, pools)
    assert _feasible(accumulator=INT32, output=INT32, target=MVAUDspBlock.DSP48E2) is True


def test_an_operand_wider_than_the_target_multiplier_is_refused() -> None:
    """DSP48E2 takes 18 activation bits; DSP58 takes 24."""

    engine, point, pools = _point(activation=INT20, target=MVAUDspBlock.DSP48E2)
    assert "width_supported" in _rejected(engine, point, pools)
    assert _feasible(activation=INT20, target=MVAUDspBlock.DSP58) is True


def test_dsp48e1_needs_the_narrow_weight_promise() -> None:
    """Coverage, not admission: the source is fine, this board is not."""

    engine, point, pools = _point(target=MVAUDspBlock.DSP48E1, narrow=False)
    assert "narrow_weights_supported" in _rejected(engine, point, pools)
    assert _feasible(target=MVAUDspBlock.DSP48E1, narrow=True) is True


def test_the_narrow_weight_rule_is_not_a_source_question() -> None:
    """It must never remove a graph from inference; only a board from coverage."""

    pools = build_decomposed_mvau_pools()
    declaration = pools.dot_product.kernels[0]
    names = {str(path).rsplit(".", 1)[-1] for path in declaration.source_admission_constraints}
    assert "narrow_weights_supported" not in names


def test_pumping_needs_at_least_two_lanes() -> None:
    engine, point, pools = _point(simd=1, pumping=True)
    assert "pumping_supported" in _rejected(engine, point, pools)
    assert _feasible(simd=1, pumping=False) is True


def test_the_kernel_declares_both_kinds_of_condition() -> None:
    """The regression this file exists for: a Kernel with no conditions at all."""

    pools = build_decomposed_mvau_pools()
    declaration = pools.dot_product.kernels[0]
    assert declaration.source_admission_constraints
    assert len(declaration.feasibility_constraints) > len(declaration.source_admission_constraints)
