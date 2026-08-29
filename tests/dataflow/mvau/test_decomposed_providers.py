# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Design note section 9.1: every provider parameter has a declared owner.

Four categories are allowed -- projected problem data, a committed decision, a
derived property, or a documented provider constant.  An elaborator computing a
value that is none of these is the disallowed fifth case, and ``SEGMENTLEN``
was exactly that: a timing derivation that reached the artifact without ever
appearing in the design point.
"""

from __future__ import annotations

from typing import cast

import pytest

from dataflow.mvau_op_facts import compute_pool_context
from finn.dataflow.authoring import assemble_specs
from finn.dataflow.design import Decided, DesignPoint, Engine, QualifiedPath, Unresolved
from finn.dataflow.mvau.decomposed import (
    DOT_PRODUCT_PROVIDER,
    REPLAY_PROVIDER,
    ActivationReplayKernel,
    DecomposedMVAUPools,
    DotProductKernel,
    ParameterOwnership,
    build_decomposed_mvau_pools,
)
from finn.dataflow.mvau_problem import (
    MVAU_PROBLEM,
    MVAU_PROBLEM_PROVENANCE,
    MVAUComputationProfile,
    MVAUDspBlock,
    MVAUProblemPaths,
)
from finn.dataflow.region import NumericElementType

INT8 = NumericElementType("int", 8)
INT16 = NumericElementType("int", 16)
UINT8 = NumericElementType("uint", 8)


def _started(
    *,
    clock_period_ns: float = 4.0,
    target: MVAUDspBlock = MVAUDspBlock.DSP58,
    activation: NumericElementType = INT8,
    matrix_width: int = 8,
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
    point = engine.start(
        space,
        {
            MVAUProblemPaths.REPETITIONS: 2,
            MVAUProblemPaths.MATRIX_WIDTH: matrix_width,
            MVAUProblemPaths.MATRIX_HEIGHT: 4,
            MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE: activation,
            MVAUProblemPaths.WEIGHT_ELEMENT_TYPE: INT8,
            MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE: INT16,
            MVAUProblemPaths.OUTPUT_ELEMENT_TYPE: INT16,
            MVAUProblemPaths.COMPUTATION_PROFILE: MVAUComputationProfile.ACCUMULATOR_INTEGER,
            MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE: True,
            MVAUProblemPaths.RUNTIME_WRITABLE: False,
            MVAUProblemPaths.TARGET_DSP_BLOCK: target,
            MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS: clock_period_ns,
        },
    )
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


def _value(engine: Engine, point: DesignPoint, path: QualifiedPath) -> object:
    answer = engine.query_property(point, path)
    assert isinstance(answer, Decided), answer
    return answer.value


# -- the audit ---------------------------------------------------------------


def test_every_provider_parameter_names_a_declared_owner() -> None:
    _engine, point, pools = _started()
    space = point.design_space

    for parameter in pools.provider_parameters():
        if parameter.ownership is ParameterOwnership.CONSTANT:
            assert parameter.value is not None
            assert parameter.why, f"{parameter.name} is a constant with no stated reason"
            continue
        assert parameter.source is not None
        declared = {
            ParameterOwnership.PROBLEM: space.problem_fields,
            ParameterOwnership.DECISION: space.decisions,
            ParameterOwnership.DERIVED: space.properties,
        }[parameter.ownership]
        assert parameter.source in declared, f"{parameter.name} -> {parameter.source}"


def test_every_derived_parameter_actually_resolves() -> None:
    """Naming a path is not enough; the value has to be there at elaboration."""

    engine, point, pools = _started()
    for parameter in pools.provider_parameters():
        if parameter.ownership is not ParameterOwnership.DERIVED:
            continue
        assert parameter.source is not None
        assert isinstance(engine.query_property(point, parameter.source), Decided), parameter.name


#: The module headers of ``finnlib/rtl/dotp_axi.sv`` and
#: ``finn-rtllib/mvu/replay_buffer.sv``, excluding their ``localparam``
#: deductions.  Exact, not a subset: a parameter the audit omits is one whose
#: value the elaborator would have to invent, which is the failure section 9.1
#: exists to prevent.
DOTP_AXI_PARAMETERS = {
    "VERSION",
    "ACTIVATION_BROADCASTING",
    "PE",
    "SIMD",
    "SEGMENTLEN",
    "ACTIVATION_WIDTH",
    "WEIGHT_WIDTH",
    "ACCU_WIDTH",
    "NARROW_WEIGHTS",
    "SIGNED_ACTIVATIONS",
    "PUMPED_COMPUTE",
    "FORCE_BEHAVIORAL",
}
REPLAY_BUFFER_PARAMETERS = {"LEN", "REP", "W"}


def test_the_audit_covers_exactly_the_parameters_the_rtl_takes() -> None:
    pools = build_decomposed_mvau_pools()
    problem = MVAU_PROBLEM

    dot_product = {
        item.name for item in DotProductKernel.provider_parameters(pools.dot_product_paths, problem)
    }
    replay = {item.name for item in ActivationReplayKernel.provider_parameters(pools.replay_paths)}

    assert dot_product == DOTP_AXI_PARAMETERS
    assert replay == REPLAY_BUFFER_PARAMETERS


def test_the_dot_product_takes_no_matrix_geometry() -> None:
    """MW and MH were the fused wrapper's, only to size the replay it contained.

    Their absence from ``dotp_axi`` is the decomposition visible in the
    parameter list: that geometry is now the replay Kernel's ``LEN`` and ``REP``.
    """

    assert not {"MW", "MH"} & DOTP_AXI_PARAMETERS
    assert REPLAY_BUFFER_PARAMETERS <= {"LEN", "REP", "W"}


def test_the_providers_are_declared_on_the_kernels_they_realize() -> None:
    pools = build_decomposed_mvau_pools()
    dot_product = pools.dot_product.kernels[0]
    replay = pools.activation_replay.kernels[0]

    assert [item.id for item in dot_product.providers] == [DOT_PRODUCT_PROVIDER]
    assert [item.id for item in replay.providers] == [REPLAY_PROVIDER]
    assert dot_product.providers[0].kernel_id == DotProductKernel.id
    assert replay.providers[0].kernel_id == ActivationReplayKernel.id


# -- the values themselves ---------------------------------------------------


@pytest.mark.parametrize(
    ("target", "expected"),
    [(MVAUDspBlock.DSP48E1, 1), (MVAUDspBlock.DSP48E2, 2), (MVAUDspBlock.DSP58, 3)],
)
def test_version_maps_the_target_dsp_family(target: MVAUDspBlock, expected: int) -> None:
    engine, point, pools = _started(target=target)
    assert _value(engine, point, pools.dot_product_paths.dsp_version) == expected


@pytest.mark.parametrize(("activation", "expected"), [(INT8, True), (UINT8, False)])
def test_signed_activations_comes_from_the_activation_type(
    activation: NumericElementType, expected: bool
) -> None:
    engine, point, pools = _started(activation=activation)
    assert _value(engine, point, pools.dot_product_paths.signed_activations) is expected


@pytest.mark.parametrize(
    ("clock_period_ns", "simd", "pumping", "expected"),
    [
        (4.0, 2, False, 1),  # the chain, not the clock, is the limit
        (4.0, 12, False, 4),
        (10.0, 12, False, 4),
        (2.0, 12, False, 3),  # a fast clock shortens the cascade
        (4.0, 12, True, 2),  # pumping halves the reference clock and the chain
    ],
)
def test_segment_length_preserves_the_elaborator_derivation(
    clock_period_ns: float, simd: int, pumping: bool, expected: int
) -> None:
    """Declaring it fixed its ownership, not its value."""

    engine, point, pools = _started(
        clock_period_ns=clock_period_ns, matrix_width=24, simd=simd, pumping=pumping
    )
    assert _value(engine, point, pools.dot_product_paths.segment_length) == expected


def test_an_infeasible_clock_is_unresolved_rather_than_silently_wrong() -> None:
    """Below the segment-delay bound there is no honest answer, so say so."""

    engine, point, pools = _started(clock_period_ns=0.5)
    answer = engine.query_property(point, pools.dot_product_paths.segment_length)
    assert isinstance(answer, Unresolved)
    assert answer.findings[0].code == "mvau-segment-length-clock-infeasible"


def test_segment_length_is_never_the_longest_cascade_by_default() -> None:
    """``SEGMENTLEN = 0`` would select ``CHAINLEN``; the derivation must not."""

    engine, point, pools = _started(clock_period_ns=1.5, matrix_width=24, simd=24)
    value = cast(int, _value(engine, point, pools.dot_product_paths.segment_length))
    assert 0 < value < -(-24 // 3)


def test_the_replay_parameters_restate_the_folding() -> None:
    engine, point, pools = _started(pe=2, simd=2)
    assert _value(engine, point, pools.replay_paths.buffer_length) == 4  # MW / SIMD
    assert _value(engine, point, pools.replay_paths.buffer_repetitions) == 2  # MH / PE
    assert _value(engine, point, pools.replay_paths.buffer_width) == 16  # SIMD * width


def test_narrow_weights_is_the_operation_property_not_a_kernel_one() -> None:
    """Ownership: the value authority is an operation question, not a Kernel's."""

    pools = build_decomposed_mvau_pools()
    narrow = next(item for item in pools.provider_parameters() if item.name == "NARROW_WEIGHTS")
    assert narrow.source == MVAUProblemPaths.EFFECTIVE_NARROW_WEIGHTS
    assert MVAU_PROBLEM_PROVENANCE.kind_of(narrow.source) is None
