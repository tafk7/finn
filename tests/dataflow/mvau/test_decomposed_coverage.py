# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What the decomposed slice will and will not claim, and on which side.

A declaration with no conditions claims everything, which is worse than claiming
nothing: the design space reports a point feasible and the hardware overclaims
what it can realize.  These pin the two halves of the split, which now fall on
opposite sides of the semantic/physical line:

- **the Region** says what the source must be for this to represent it at all:
  the accumulator profile, integer operands, an output that *is* the
  accumulator.  These gate inference, so they must be answerable from graph
  facts alone.
- **the Kernel** says what a given board can build: the DSP family, operand and
  accumulator widths, the narrow-weight promise, pumping, and a minimum operand
  width.  These read the target or a physical choice, so they are coverage
  questions and must never remove a graph from inference.

The line moved during Phase 1+3 and one case moved with it in two pieces:
operand *integrality* is the Region's -- a float MatMul is not a dot-product
Region badly implemented, it is not one at all -- while the two-bit *minimum*
is this multiplier's.
"""

from __future__ import annotations

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

import pytest

from dataflow.mvau_op_facts import compute_pool_context
from finn.dataflow.authoring import assemble_specs
from finn.dataflow.design import Decided, DesignPoint, Engine, QualifiedPath, Unresolved
from finn.dataflow.mvau.compute_kernels import (
    DECOMPOSED_MVAU_KERNELS,
    MVAU_COMPUTE_SELECTION,
    MVAU_REPLAY_SELECTION,
)
from finn.dataflow.mvau.decomposed import (
    HARDWARE_NUMERIC_TYPE_COVERAGE,
    ActivationReplayKernel,
    DecomposedMVAUKernels,
    DotProductKernel,
)
from finn.dataflow.mvau.hardware.dotp_axi import (
    covers_numeric_types,
    covers_operand_types,
    covers_operand_types as dotp_axi_covers_operand_types,
)
from finn.dataflow.mvau.numeric import MVAUNumericTypes
from finn.dataflow.ops.mvau import (
    MVAU_DATAFLOW_OP_SPEC,
)
from finn.dataflow.mvau_problem import (
    MVAUComputationProfile,
    MVAUDspBlock,
    MVAUProblemPaths,
)
from finn.dataflow.region import NumericElementType

INT8 = DataType["INT8"]
INT16 = DataType["INT16"]
INT20 = DataType["INT20"]
INT32 = DataType["INT32"]
INT64 = DataType["INT64"]
UINT8 = DataType["UINT8"]
FLOAT16 = DataType["FLOAT16"]
BIPOLAR = DataType["BIPOLAR"]


def _problem(
    activation: NumericElementType = INT8,
    weight: NumericElementType = INT8,
    accumulator: NumericElementType = INT16,
    output: NumericElementType = INT16,
    profile: MVAUComputationProfile = MVAUComputationProfile.ACCUMULATOR_INTEGER,
    target: MVAUDspBlock = MVAUDspBlock.DSP58,
    narrow: bool = False,
) -> dict[QualifiedPath, object]:
    return {
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
) -> tuple[Engine, DesignPoint, DecomposedMVAUKernels]:
    pools = DECOMPOSED_MVAU_KERNELS
    engine = Engine()
    space = engine.validate(
        assemble_specs(
            (
                compute_pool_context(),
                MVAU_COMPUTE_SELECTION.build_spec(),
                MVAU_REPLAY_SELECTION.build_spec(),
                *(item.spec for item in DECOMPOSED_MVAU_KERNELS.hardware),
            )
        )
    )
    point = engine.start(
        space, _problem(activation, weight, accumulator, output, profile, target, narrow)
    )
    point = engine.commit_assignments(
        point,
        {
            MVAU_COMPUTE_SELECTION.paths.kernel: DotProductKernel.id,
            MVAU_REPLAY_SELECTION.paths.kernel: ActivationReplayKernel.id,
            pools.pe.path: pe,
            pools.simd.path: simd,
            pools.compute_pumping.path: pumping,
        },
    ).point
    return engine, point, pools


def _rejected(engine: Engine, point: DesignPoint, pools: DecomposedMVAUKernels) -> set[str]:
    """Everything refusing this point, from both halves of the split.

    Asking only the pool would silently stop testing everything that moved to
    the hardware, and the file would keep passing while checking less.

    The two halves need different rules, because ``Absent`` means different
    things in them.  The pool's set spans *every* member, so a constraint
    belonging to a Kernel this point did not select is legitimately absent --
    only a flat ``False`` is a refusal there.  The hardware constraints are all
    the selected Kernel's own, so an ``Absent`` among them can only have come
    from a ``reject(...)`` inside the evaluator, which is a refusal carrying a
    reason.
    """

    pool = engine.evaluate_constraint_set(point, MVAU_COMPUTE_SELECTION.feasibility_constraint_set)
    hardware = engine.evaluate_constraints(point, DECOMPOSED_MVAU_KERNELS.coverage_constraints)
    refused = {
        str(path)
        for path, answer in pool.answers.items()
        if isinstance(answer, Decided) and answer.value is False
    } | {
        str(path)
        for path, answer in hardware.answers.items()
        if not isinstance(answer, Unresolved)
        and not (isinstance(answer, Decided) and answer.value is True)
    }
    return {path.rsplit(".", 1)[-1] for path in refused}


def _feasible(**overrides: object) -> bool:
    engine, point, pools = _point(**overrides)  # type: ignore[arg-type]
    return not _rejected(engine, point, pools)


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
def test_operands_no_hardware_can_multiply_are_not_admitted(
    activation: NumericElementType, weight: NumericElementType
) -> None:
    """Inference must not lower a MatMul nothing can build.

    The condition is existential over the hardware inventory, not a property of
    the Region: a float dot product is a Region FINN has no Kernel for, and it
    becomes admissible the day one is added.
    """

    engine, point, pools = _point(activation=activation, weight=weight)
    assert "some_hardware_covers_the_numeric_types" in _rejected(engine, point, pools)


def test_operand_types_and_widths_are_both_the_hardwares() -> None:
    """Neither is a Region restriction; both are coverage on ``dotp_axi``.

    A ``DataflowRegion`` admits floating-point element types and arithmetic is
    binding-owned, so putting integrality on the declaration would have said
    something false about what a Region can be -- and would have made a float
    Kernel unaddable without editing the Region.
    """

    coverage = {
        str(path).rsplit(".", 1)[-1]
        for path in DECOMPOSED_MVAU_KERNELS.dot_product_hardware.coverage_constraints
    }
    assert {"operand_types_supported", "operand_widths_supported"} <= coverage

    semantic = MVAU_COMPUTE_SELECTION.kernel(DotProductKernel.id)
    names = {str(path).rsplit(".", 1)[-1] for path in semantic.source_admission_constraints}
    assert "operand_types_supported" not in names
    assert "operand_widths_supported" not in names


def test_the_hardware_does_not_claim_types_it_cannot_multiply() -> None:
    """The bug the split exposed in the other direction.

    With integrality on the Region, ``dotp_axi`` had no type coverage at all and
    answered True for FLOAT16 -- the declaration refused float while the Kernel
    claimed it.
    """

    engine, point, pools = _point(activation=FLOAT16, weight=FLOAT16)
    assert "operand_types_supported" in _rejected(engine, point, pools)
    assert covers_operand_types(MVAUNumericTypes(INT8, INT8, INT16, INT16)) is True
    assert covers_operand_types(MVAUNumericTypes(FLOAT16, FLOAT16, INT16, INT16)) is False


def test_admission_quantifies_over_the_declared_hardware_inventory() -> None:
    """Adding a Kernel widens admission; it does not edit the Region.

    Pinning the mechanism, not just the outcome: the bridge asks each entry in
    the inventory, and each entry is the same predicate that Kernel's own
    coverage constraint uses, so the two cannot drift.
    """

    supported = MVAUNumericTypes(INT8, INT8, INT16, INT16)
    unsupported = MVAUNumericTypes(FLOAT16, FLOAT16, FLOAT16, FLOAT16)
    assert dotp_axi_covers_operand_types in HARDWARE_NUMERIC_TYPE_COVERAGE
    assert any(covers(supported) for covers in HARDWARE_NUMERIC_TYPE_COVERAGE)
    assert not any(covers(unsupported) for covers in HARDWARE_NUMERIC_TYPE_COVERAGE)


def test_an_integer_product_with_a_floating_accumulator_is_refused() -> None:
    """The hole the complete signature was introduced to close.

    Before it, coverage inspected activation and weight only, and the sole
    condition mentioning the other two roles checked that they equalled *each
    other*.  So this exact configuration was admitted with **no refusals at
    all**, lowered, and elaborated onto a multiplier that cannot produce it.

    Named per role, so the diagnostic says which operand was wrong rather than
    that something was.
    """

    engine, point, pools = _point(accumulator=FLOAT16, output=FLOAT16)
    assert "operand_types_supported" in _rejected(engine, point, pools)
    assert covers_operand_types(MVAUNumericTypes(INT8, INT8, FLOAT16, FLOAT16)) is False

    refused = {
        verdict.role
        for verdict in covers_numeric_types(MVAUNumericTypes(INT8, INT8, FLOAT16, FLOAT16))
        if not verdict.supported
    }
    assert refused == {"accumulator", "output"}


def test_ternary_is_refused_by_identity_not_by_family_and_width() -> None:
    """The §1.1 defect, pinned so it cannot return.

    ``TERNARY`` is two bits wide and QONNX reports ``is_integer()`` true for it,
    so every family-and-width test admitted it -- and the old reduction then
    spelled it ``INT2`` in the artifact, a different value domain reported as
    nothing.  It has to be refused by canonical identity, and ``INT2`` -- a
    genuine two-bit two's-complement integer -- has to keep working, or the fix
    would just be a narrower version of the same mistake.

    **Scope.**  This says ``DotpAxiKernel`` refuses ``TERNARY``, and no artifact
    path relabels it.  It does *not* say MVAU inference rejects ternary
    outright: the legacy HLS and RTL pool members still classify through
    ``element_family``, which maps ``TERNARY`` to ``"int"``, so a ternary MatMul
    is still recognized and lowered through one of them.  Whether that is
    correct is a question about those legacy paths' actual ternary support, and
    it is deliberately not settled here -- see ``element_family``'s own note.
    """

    ternary = DataType["TERNARY"]
    assert ternary.is_integer() is True
    assert ternary.bitwidth() == 2

    assert covers_operand_types(MVAUNumericTypes(ternary, INT8, INT16, INT16)) is False
    assert covers_operand_types(MVAUNumericTypes(DataType["INT2"], INT8, INT16, INT16)) is True

    engine, point, pools = _point(activation=ternary)
    assert "operand_types_supported" in _rejected(engine, point, pools)


def test_special_one_bit_encodings_are_refused_by_identity() -> None:
    """``BINARY`` and ``BIPOLAR`` are integer-valued, not two's-complement.

    ``BIPOLAR`` spans -1..1 in one bit; a datapath told it has a one-bit signed
    integer would compute over -1..0.  ``UINT1`` is the same value as
    ``BINARY``, so it goes with them -- adopting QONNX means adopting that
    merge.
    """

    for name in ("BINARY", "BIPOLAR", "UINT1"):
        assert (
            covers_operand_types(MVAUNumericTypes(DataType[name], INT8, INT16, INT16)) is False
        ), name


def test_an_unsigned_weight_is_refused_with_its_own_reason() -> None:
    """Separately diagnosable: the type is fine, the signedness is not."""

    refused = {
        verdict.role
        for verdict in covers_numeric_types(MVAUNumericTypes(INT8, UINT8, INT16, INT16))
        if not verdict.supported
    }
    assert refused == {"weight"}


@pytest.mark.parametrize(
    ("accumulator", "output", "expected"),
    [
        (DataType["UINT16"], DataType["UINT16"], {"accumulator", "output"}),
        (DataType["UINT16"], INT16, {"accumulator"}),
        (INT16, DataType["UINT16"], {"output"}),
    ],
)
def test_an_unsigned_accumulator_or_output_is_refused(
    accumulator: NumericElementType, output: NumericElementType, expected: set[str]
) -> None:
    """The core declares these roles signed, so it cannot take unsigned ones.

    ``dotp.sv`` and ``dotp_top.sv`` declare the result port as
    ``output logic signed [PE-1:0][ACCU_WIDTH-1:0] p``, so accumulation and the
    value leaving the core are two's-complement signed.  A ``UINT16``
    accumulator would be reinterpreted at that boundary and everything above
    the signed maximum would come back negative.

    Applying one uniform integer rule to all four roles admitted this with no
    refusal at all -- the same shape of omission as the floating accumulator,
    one level finer.
    """

    refused = {
        verdict.role
        for verdict in covers_numeric_types(MVAUNumericTypes(INT8, INT8, accumulator, output))
        if not verdict.supported
    }
    assert refused == expected


def test_an_unsigned_activation_is_still_accepted() -> None:
    """The role contract is deliberately non-uniform.

    ``SIGNED_ACTIVATIONS`` exists exactly so the core can be told which of the
    two it is being given, so refusing unsigned activations along with the
    other roles would be the opposite error.
    """

    assert covers_operand_types(MVAUNumericTypes(UINT8, INT8, INT16, INT16)) is True
    assert _feasible(activation=UINT8) is True


def test_a_one_bit_operand_is_a_hardware_limit_not_a_source_one() -> None:
    """The Region is expressible; this multiplier is what cannot take it."""

    engine, point, pools = _point(activation=DataType["INT1"])
    assert "operand_widths_supported" in _rejected(engine, point, pools)


def test_unsigned_activations_are_served() -> None:
    """Unsigned is integer arithmetic; ``SIGNED_ACTIVATIONS`` carries the sign."""

    assert _feasible(activation=UINT8) is True


def test_an_output_that_is_not_the_accumulator_is_refused() -> None:
    """With no threshold to requantize through, the output *is* the accumulator."""

    engine, point, pools = _point(accumulator=INT16, output=INT8)
    assert "accumulator_output_type_supported" in _rejected(engine, point, pools)


def test_source_admission_reads_no_target_or_decision() -> None:
    """Admission is asked before a board or a folding is known."""

    declaration = MVAU_COMPUTE_SELECTION.kernel(DotProductKernel.id)
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

    declaration = MVAU_COMPUTE_SELECTION.kernel(DotProductKernel.id)
    names = {str(path).rsplit(".", 1)[-1] for path in declaration.source_admission_constraints}
    assert "narrow_weights_supported" not in names


def test_pumping_needs_at_least_two_lanes() -> None:
    engine, point, pools = _point(simd=1, pumping=True)
    assert "pumping_supported" in _rejected(engine, point, pools)
    assert _feasible(simd=1, pumping=False) is True


def test_no_physical_path_appears_in_the_structural_profile() -> None:
    """Stated over the declaration, so it cannot regress by accident."""

    profile = next(
        item
        for item in MVAU_DATAFLOW_OP_SPEC.readiness_profiles
        if item.name == "mvau_op_structural"
    )
    physical = {kernel.namespace for kernel in DECOMPOSED_MVAU_KERNELS.hardware}
    for path in (*profile.decisions, *profile.properties):
        assert not any(str(path).startswith(f"{owner}.") for owner in physical), path
        assert not any(str(path).startswith(f"semantic.{owner}.") for owner in physical), path


def test_both_sides_declare_conditions_and_neither_declares_the_others() -> None:
    """The regression this file exists for, restated across the new line.

    Before Phase 1+3 one declaration carried both kinds and the test compared
    their counts.  Now they are two objects, and what matters is that each has
    conditions of its own kind and none of the other's -- a Region that read the
    target, or a Kernel that gated inference, would be the split undone.
    """

    semantic = MVAU_COMPUTE_SELECTION.kernel(DotProductKernel.id)
    assert semantic.source_admission_constraints
    assert DECOMPOSED_MVAU_KERNELS.dot_product_hardware.coverage_constraints

    # The Region reads no target and no decision -- checked above for admission,
    # asserted here for every condition it has, because it now has only those.
    physical_paths = {
        MVAUProblemPaths.TARGET_DSP_BLOCK,
        MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS,
    }
    for constraint in semantic.spec.constraints:
        read = {item.path for item in constraint.evaluator.dependencies}
        assert not read & physical_paths, constraint.path
