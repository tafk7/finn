# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 4 gate: suppliers consume compute demand and keep distinct identity."""

from __future__ import annotations

from dataclasses import replace

import pytest

from finn.dataflow.design import (
    Absent,
    Decided,
    DependencyRef,
    DesignPoint,
    DesignSpaceSpec,
    Engine,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    Unresolved,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.kernels import NO_KERNEL, SelectedKernel
from finn.dataflow.mvau.regions import (
    construct_batch_interleaved_mvau_weight_port,
    construct_standard_streamed_mvau_region,
)
from finn.dataflow.parameters.cyclic.definition import (
    CyclicParameterKernelPaths,
    CyclicRamStyle,
    CyclicTargetMemoryCapabilities,
)
from finn.dataflow.parameters.supply_kernels import (
    FINNLIB_MEMSTREAM_PATHS,
    FINN_RTL_MEMSTREAM_PATHS,
    OUTPUT_PORT_EXPORT,
    MVAUWeightSupplyKernelId,
    MVAUWeightSupplyProblemPaths,
    WeightOrganization,
    build_mvau_weight_supply_selection,
)
from finn.dataflow.region import DataflowRegion, NumericElementType, Port
from finn.dataflow.region_validation import RegionValidationReport
from finn.dataflow.spec_algebra import assemble_specs

INT8 = NumericElementType("int", 8)
INT16 = NumericElementType("int", 16)
_MISSING = object()

_PORT = as_object_semantics(ValueSemantics.immutable_nominal(Port, name="Port"))
_BOOL = as_object_semantics(ValueSemantics.immutable_nominal(bool, name="boolean"))
_CAPABILITIES = as_object_semantics(
    ValueSemantics.immutable_nominal(
        CyclicTargetMemoryCapabilities, name="CyclicTargetMemoryCapabilities"
    )
)

#: The compute demand and the compute Kernel's natural full tile enter the pool
#: as ordinary problem facts here, standing in for the operation's exports.
DEMAND = QualifiedPath("problem.test.compute_demand")
FULL_TILE = QualifiedPath("problem.test.compute_full_tile")

SELECTION = build_mvau_weight_supply_selection(
    DependencyRef.problem("demand", DEMAND, _PORT),
    DependencyRef.problem("full_tile", FULL_TILE, _PORT),
)
PATHS = SELECTION.paths


def _weight_port(*, elements_per_beat: int = 4) -> Port:
    region = construct_standard_streamed_mvau_region(2, 4, 4, INT8, INT8, INT16, 2, 2)
    port = region.input_interface("weight").port
    if elements_per_beat == port.beat_sequence.elements_per_beat:
        return port
    return replace(
        port,
        beat_sequence=replace(
            port.beat_sequence,
            elements_per_beat=elements_per_beat,
            beats=tuple((position,) for beat in port.beat_sequence.beats for position in beat),
        ),
    )


def _chunked_port() -> Port:
    return construct_batch_interleaved_mvau_weight_port(2, 4, 4, INT8, 2, 2, 2)


def _spec() -> DesignSpaceSpec:
    return assemble_specs(
        (
            SELECTION.build_spec(),
            DesignSpaceSpec(
                ProblemSchema(
                    (
                        ProblemField(DEMAND, _PORT),
                        ProblemField(FULL_TILE, _PORT),
                        ProblemField(MVAUWeightSupplyProblemPaths.INITIALIZER_AVAILABLE, _BOOL),
                        ProblemField(MVAUWeightSupplyProblemPaths.RUNTIME_WRITABLE, _BOOL),
                        ProblemField(
                            MVAUWeightSupplyProblemPaths.TARGET_MEMORY_CAPABILITIES,
                            _CAPABILITIES,
                            required=False,
                        ),
                    )
                ),
            ),
        )
    )


def _started(
    *,
    demand: Port | None = None,
    full_tile: Port | None = None,
    initializer_available: bool = True,
    runtime_writable: bool = False,
    target_capabilities: object = _MISSING,
) -> tuple[Engine, DesignPoint]:
    engine = Engine()
    space = engine.validate(_spec())
    problem: dict[QualifiedPath, object] = {
        DEMAND: demand or _weight_port(),
        FULL_TILE: full_tile or _weight_port(),
        MVAUWeightSupplyProblemPaths.INITIALIZER_AVAILABLE: initializer_available,
        MVAUWeightSupplyProblemPaths.RUNTIME_WRITABLE: runtime_writable,
    }
    if target_capabilities is not _MISSING:
        problem[MVAUWeightSupplyProblemPaths.TARGET_MEMORY_CAPABILITIES] = target_capabilities
    return engine, engine.start(space, problem)


def _select(
    engine: Engine,
    point: DesignPoint,
    kernel: MVAUWeightSupplyKernelId,
    *,
    organization: WeightOrganization = WeightOrganization.AS_DEMANDED,
    ram_style: CyclicRamStyle | None = None,
    pumped_memory: bool | None = None,
) -> DesignPoint:
    paths = (
        FINN_RTL_MEMSTREAM_PATHS
        if kernel is MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM
        else FINNLIB_MEMSTREAM_PATHS
    )
    assignments: dict[QualifiedPath, object] = {
        PATHS.kernel: kernel.value,
        paths.organization: organization,
    }
    if ram_style is not None:
        assignments[paths.ram_style] = ram_style
    if pumped_memory is not None:
        assignments[paths.pumped_memory] = pumped_memory
    return engine.commit_assignments(point, assignments).point


def test_a_directly_connected_supplier_derives_the_exact_compute_demand() -> None:
    engine, point = _started()
    point = _select(
        engine,
        point,
        MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM,
        ram_style=CyclicRamStyle.BRAM,
        pumped_memory=False,
    )
    region = engine.query_property(point, PATHS.region)
    assert isinstance(region, Decided)
    assert isinstance(region.value, DataflowRegion)
    assert region.value.output_interface("weight").port == _weight_port()
    assert engine.query_property(point, PATHS.region_validation) == Decided(
        RegionValidationReport()
    )


def test_equal_supplier_regions_do_not_collapse_kernel_identity() -> None:
    left_engine, left = _started()
    left = _select(
        left_engine,
        left,
        MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM,
        ram_style=CyclicRamStyle.BRAM,
        pumped_memory=False,
    )
    right_engine, right = _started()
    right = _select(right_engine, right, MVAUWeightSupplyKernelId.FINNLIB_HLS_MEMSTREAM)
    left_region = left_engine.query_property(left, PATHS.region)
    right_region = right_engine.query_property(right, PATHS.region)
    assert isinstance(left_region, Decided) and isinstance(right_region, Decided)
    assert left_region.value == right_region.value
    assert left_engine.query_property(left, PATHS.selected_kernel) == Decided(
        SelectedKernel(SELECTION.name, MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value, "1")
    )
    assert right_engine.query_property(right, PATHS.selected_kernel) == Decided(
        SelectedKernel(SELECTION.name, MVAUWeightSupplyKernelId.FINNLIB_HLS_MEMSTREAM.value, "1")
    )


def test_initializer_presence_enables_but_does_not_force_a_supplier() -> None:
    engine, point = _started()
    state = engine.decision_state(point, PATHS.kernel)
    assert isinstance(state, Decided)
    assert state.value.status != "committed"
    candidates = engine.enumerate_candidates(point, PATHS.kernel)
    assert isinstance(candidates, Decided)
    assert NO_KERNEL in candidates.value
    unsupplied = engine.commit_assignments(point, {PATHS.kernel: NO_KERNEL}).point
    assert isinstance(engine.query_property(unsupplied, PATHS.region), Absent)


def test_finnlib_requires_an_initializer_and_owns_no_rtl_only_decisions() -> None:
    engine, point = _started(initializer_available=False, runtime_writable=True)
    point = _select(engine, point, MVAUWeightSupplyKernelId.FINNLIB_HLS_MEMSTREAM)
    assert isinstance(engine.decision_state(point, FINN_RTL_MEMSTREAM_PATHS.ram_style), Absent)
    assert isinstance(engine.decision_state(point, FINN_RTL_MEMSTREAM_PATHS.pumped_memory), Absent)
    assessment = engine.evaluate_constraint_set(point, SELECTION.feasibility_constraint_set)
    assert assessment.answers[
        FINNLIB_MEMSTREAM_PATHS.constraint("local_state_available")
    ] == Decided(False)


def test_runtime_writability_excludes_immutable_only_suppliers() -> None:
    engine, point = _started(initializer_available=False, runtime_writable=True)
    point = _select(
        engine,
        point,
        MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM,
        ram_style=CyclicRamStyle.BRAM,
        pumped_memory=False,
    )
    assert engine.evaluate_constraint_set(point, SELECTION.feasibility_constraint_set).verdict


def test_non_versal_initialized_uram_requires_runtime_writeability() -> None:
    capabilities = CyclicTargetMemoryCapabilities(supports_initialized_uram=False)
    engine, point = _started(target_capabilities=capabilities)
    point = _select(
        engine,
        point,
        MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM,
        ram_style=CyclicRamStyle.URAM,
        pumped_memory=False,
    )
    assessment = engine.evaluate_constraint_set(point, SELECTION.feasibility_constraint_set)
    assert assessment.answers[
        FINN_RTL_MEMSTREAM_PATHS.constraint("uram_initialization_supported")
    ] == Decided(False)

    writable_engine, writable = _started(runtime_writable=True, target_capabilities=capabilities)
    writable = _select(
        writable_engine,
        writable,
        MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM,
        ram_style=CyclicRamStyle.URAM,
        pumped_memory=False,
    )
    assert writable_engine.evaluate_constraint_set(
        writable, SELECTION.feasibility_constraint_set
    ).verdict


def test_missing_uram_target_fact_is_truthfully_unresolved() -> None:
    engine, point = _started(target_capabilities=_MISSING)
    point = _select(
        engine,
        point,
        MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM,
        ram_style=CyclicRamStyle.URAM,
        pumped_memory=False,
    )
    answer = engine.evaluate_constraint_set(point, SELECTION.feasibility_constraint_set).answers[
        FINN_RTL_MEMSTREAM_PATHS.constraint("uram_initialization_supported")
    ]
    assert isinstance(answer, Unresolved)


@pytest.mark.parametrize("elements_per_beat,expected", [(1, False), (4, True)])
def test_one_field_pumping_is_rejected_but_wider_pumping_is_supported(
    elements_per_beat: int, expected: bool
) -> None:
    port = _weight_port(elements_per_beat=elements_per_beat)
    engine, point = _started(demand=port, full_tile=port)
    point = _select(
        engine,
        point,
        MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM,
        ram_style=CyclicRamStyle.BRAM,
        pumped_memory=True,
    )
    assessment = engine.evaluate_constraint_set(point, SELECTION.feasibility_constraint_set)
    assert assessment.answers[FINN_RTL_MEMSTREAM_PATHS.constraint("pumping_supported")] == Decided(
        expected
    )


def test_independent_supplier_organization_produces_a_different_sequence() -> None:
    engine, point = _started(demand=_chunked_port(), full_tile=_weight_port())
    demanded = _select(
        engine,
        point,
        MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM,
        organization=WeightOrganization.AS_DEMANDED,
        ram_style=CyclicRamStyle.BRAM,
        pumped_memory=False,
    )
    independent = _select(
        engine,
        point,
        MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM,
        organization=WeightOrganization.STANDARD_FULL_TILE,
        ram_style=CyclicRamStyle.BRAM,
        pumped_memory=False,
    )
    left = engine.query_property(demanded, PATHS.export(OUTPUT_PORT_EXPORT))
    right = engine.query_property(independent, PATHS.export(OUTPUT_PORT_EXPORT))
    assert isinstance(left, Decided) and isinstance(right, Decided)
    assert left.value == _chunked_port()
    assert right.value == _weight_port()
    assert left.value != right.value


def test_the_supply_pool_declares_no_delivery_tile_decision() -> None:
    paths = {str(item.path) for item in SELECTION.build_spec().decisions}
    assert not any(path.endswith((".pe", ".simd")) for path in paths)
    assert CyclicParameterKernelPaths.RUNTIME_WRITABLE not in {
        item.path for item in SELECTION.build_spec().decisions
    }
