# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every physical parameter has a declared owner, and the owner is a handle.

Four routes are allowed -- projected problem data, a committed decision, a
derived property, or a documented Kernel constant.  Hardware computing a value
that is none of these is the disallowed fifth case, and ``SEGMENTLEN`` was
exactly that: a timing derivation that reached the artifact without ever
appearing in the design point.

The audit is now structural rather than asserted.  A ``KernelParameter`` holds
a ``Ref``, and the ``Ref`` carries the kind, so a parameter cannot claim one
ownership and name another.  What is left to test is that the table covers
exactly the parameters the RTL takes, and that each value is the one the old
elaborator computed -- because declaring these fixed their ownership, not their
arithmetic.
"""

from __future__ import annotations

from typing import cast

import pytest

from dataflow.mvau_op_facts import compute_pool_context
from finn.dataflow.authoring import assemble_specs
from finn.dataflow.design import (
    Decided,
    DependencyKind,
    DesignPoint,
    Engine,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow.hardware.kernel import HardwareKernelDeclaration, KernelParameter
from finn.dataflow.mvau.compute_kernels import (
    DECOMPOSED_MVAU_KERNELS,
    MVAU_COMPUTE_SELECTION,
    MVAU_REPLAY_SELECTION,
)
from finn.dataflow.mvau.decomposed import ActivationReplayKernel, DotProductKernel
from finn.dataflow.mvau.hardware.dotp_axi import DotpAxiKernel
from finn.dataflow.mvau.hardware.replay_buffer import ReplayBufferKernel
from finn.dataflow.mvau_problem import (
    MVAU_PROBLEM_PROVENANCE,
    MVAUComputationProfile,
    MVAUDspBlock,
    MVAUProblemPaths,
)
from finn.dataflow.region import NumericElementType

INT8 = NumericElementType("int", 8)
INT16 = NumericElementType("int", 16)
UINT8 = NumericElementType("uint", 8)

DOTP_AXI = DECOMPOSED_MVAU_KERNELS.dot_product_hardware
REPLAY_BUFFER = DECOMPOSED_MVAU_KERNELS.replay_hardware


def _started(
    *,
    clock_period_ns: float = 4.0,
    target: MVAUDspBlock = MVAUDspBlock.DSP58,
    activation: NumericElementType = INT8,
    matrix_width: int = 8,
    pe: int = 2,
    simd: int = 2,
    pumping: bool = False,
) -> tuple[Engine, DesignPoint]:
    engine = Engine()
    space = engine.validate(
        assemble_specs(
            (
                compute_pool_context(),
                MVAU_COMPUTE_SELECTION.build_spec(),
                MVAU_REPLAY_SELECTION.build_spec(),
                DOTP_AXI.spec,
                REPLAY_BUFFER.spec,
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
            MVAU_COMPUTE_SELECTION.paths.kernel: DotProductKernel.id,
            MVAU_REPLAY_SELECTION.paths.kernel: ActivationReplayKernel.id,
            DECOMPOSED_MVAU_KERNELS.pe.path: pe,
            DECOMPOSED_MVAU_KERNELS.simd.path: simd,
            DECOMPOSED_MVAU_KERNELS.compute_pumping.path: pumping,
        },
    ).point
    return engine, point


def _parameter(declaration: HardwareKernelDeclaration, name: str) -> KernelParameter:
    found = declaration.parameter(name)
    assert found is not None, f"{declaration.id} declares no {name}"
    return found


def _value(engine: Engine, point: DesignPoint, path: QualifiedPath) -> object:
    answer = engine.query_property(point, path)
    assert isinstance(answer, Decided), answer
    return answer.value


def _derived(engine: Engine, point: DesignPoint, name: str) -> object:
    parameter = _parameter(DOTP_AXI, name)
    assert parameter.source is not None
    return _value(engine, point, parameter.source.path)


# -- the audit ---------------------------------------------------------------

#: The module headers of ``finnlib/rtl/dotp_axi.sv`` and
#: ``finn-rtllib/mvu/replay_buffer.sv``, excluding their ``localparam``
#: deductions.  Exact, not a subset: a parameter the audit omits is one whose
#: value elaboration would have to invent.
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
    assert set(DOTP_AXI.parameter_names) == DOTP_AXI_PARAMETERS
    assert set(REPLAY_BUFFER.parameter_names) == REPLAY_BUFFER_PARAMETERS


def test_every_parameter_names_a_declaration_or_argues_for_itself() -> None:
    _engine, point = _started()
    space = point.design_space
    declared = {
        DependencyKind.PROBLEM: space.problem_fields,
        DependencyKind.DECISION: space.decisions,
        DependencyKind.PROPERTY: space.properties,
    }

    for declaration in (DOTP_AXI, REPLAY_BUFFER):
        for parameter in declaration.parameters:
            if parameter.is_constant:
                assert parameter.value is not None
                assert parameter.why, f"{parameter.name} is a constant with no stated reason"
                continue
            assert parameter.source is not None
            assert parameter.source.path in declared[parameter.source.kind], (
                f"{parameter.name} -> {parameter.source.path}"
            )


def test_every_derived_parameter_actually_resolves() -> None:
    """Naming a path is not enough; the value has to be there at elaboration."""

    engine, point = _started()
    for declaration in (DOTP_AXI, REPLAY_BUFFER):
        for parameter in declaration.parameters:
            if parameter.source is None or parameter.source.kind is not DependencyKind.PROPERTY:
                continue
            answer = engine.query_property(point, parameter.source.path)
            assert isinstance(answer, Decided), parameter.name


def test_the_dot_product_takes_no_matrix_geometry() -> None:
    """MW and MH were the fused wrapper's, only to size the replay it contained.

    Their absence from ``dotp_axi`` is the decomposition visible in the
    parameter list: that geometry is now the replay Kernel's ``LEN`` and ``REP``.
    """

    assert not {"MW", "MH"} & set(DOTP_AXI.parameter_names)


def test_the_widths_are_declared_rather_than_projected_on_the_way_out() -> None:
    """A bit count in the RTL that is in no property is a value nobody chose."""

    for name in ("ACTIVATION_WIDTH", "WEIGHT_WIDTH", "ACCU_WIDTH"):
        source = _parameter(DOTP_AXI, name).source
        assert source is not None
        assert source.kind is DependencyKind.PROPERTY


def test_the_hardware_declares_no_region_and_the_regions_declare_no_hardware() -> None:
    """The split this phase exists to make, asserted at both ends."""

    semantic = MVAU_COMPUTE_SELECTION.kernel(DotProductKernel.id)
    assert semantic.providers == ()
    assert MVAU_REPLAY_SELECTION.kernels[0].providers == ()
    # Nothing physical left in the semantic declarations.
    semantic_paths = {str(item.path) for item in semantic.spec.decisions}
    assert not any("pumping" in path for path in semantic_paths)
    # ...and the Kernels cover Regions they did not declare.
    assert DOTP_AXI.coverage.region_roles == ("compute",)
    assert REPLAY_BUFFER.coverage.region_roles == ("replay",)


def test_each_kernel_declares_the_sources_it_compiles() -> None:
    assert {item.root for item in DOTP_AXI.sources} == {"finnlib"}
    assert {item.root for item in REPLAY_BUFFER.sources} == {"finn"}
    # Compile order, not alphabetical: dotp_axi instantiates dotp.
    assert [item.path for item in DOTP_AXI.sources][-1] == "rtl/linalg/dotp_axi.sv"


def test_the_kernel_ids_are_the_hardware_not_the_semantics() -> None:
    assert DotpAxiKernel.id == "dotp_axi"
    assert ReplayBufferKernel.id == "replay_buffer"
    assert DotProductKernel.id == "dot_product"
    assert ActivationReplayKernel.id == "activation_replay"


# -- the values themselves ---------------------------------------------------


@pytest.mark.parametrize(
    ("target", "expected"),
    [(MVAUDspBlock.DSP48E1, 1), (MVAUDspBlock.DSP48E2, 2), (MVAUDspBlock.DSP58, 3)],
)
def test_version_maps_the_target_dsp_family(target: MVAUDspBlock, expected: int) -> None:
    engine, point = _started(target=target)
    assert _derived(engine, point, "VERSION") == expected


@pytest.mark.parametrize(("activation", "expected"), [(INT8, True), (UINT8, False)])
def test_signed_activations_comes_from_the_activation_type(
    activation: NumericElementType, expected: bool
) -> None:
    engine, point = _started(activation=activation)
    assert _derived(engine, point, "SIGNED_ACTIVATIONS") is expected


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
    """Moving it fixed its ownership, not its value."""

    engine, point = _started(
        clock_period_ns=clock_period_ns, matrix_width=24, simd=simd, pumping=pumping
    )
    assert _derived(engine, point, "SEGMENTLEN") == expected


def test_an_infeasible_clock_is_unresolved_rather_than_silently_wrong() -> None:
    """Below the segment-delay bound there is no honest answer, so say so."""

    engine, point = _started(clock_period_ns=0.5)
    source = _parameter(DOTP_AXI, "SEGMENTLEN").source
    assert source is not None
    answer = engine.query_property(point, source.path)
    assert isinstance(answer, Unresolved)
    assert answer.findings[0].code == "mvau-segment-length-clock-infeasible"


def test_segment_length_is_never_the_longest_cascade_by_default() -> None:
    """``SEGMENTLEN = 0`` would select ``CHAINLEN``; the derivation must not."""

    engine, point = _started(clock_period_ns=1.5, matrix_width=24, simd=24)
    value = cast(int, _derived(engine, point, "SEGMENTLEN"))
    assert 0 < value < -(-24 // 3)


def test_the_replay_parameters_restate_the_folding() -> None:
    engine, point = _started(pe=2, simd=2)
    values = {}
    for name in REPLAY_BUFFER_PARAMETERS:
        source = _parameter(REPLAY_BUFFER, name).source
        assert source is not None
        values[name] = _value(engine, point, source.path)

    assert values["LEN"] == 4  # MW / SIMD
    assert values["REP"] == 2  # MH / PE
    assert values["W"] == 16  # SIMD * activation width


def test_narrow_weights_is_the_operation_property_not_a_kernel_one() -> None:
    """Ownership: the value authority is an operation question, not a Kernel's."""

    narrow = _parameter(DOTP_AXI, "NARROW_WEIGHTS").source
    assert narrow is not None
    assert narrow.path == MVAUProblemPaths.EFFECTIVE_NARROW_WEIGHTS
    assert MVAU_PROBLEM_PROVENANCE.kind_of(narrow.path) is None
