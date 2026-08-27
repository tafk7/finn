# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from finn.dataflow.design import Decided, DesignPoint, Engine, QualifiedPath
from finn.dataflow.mvau.regions import (
    construct_batch_interleaved_streamed_mvau_region,
    construct_standard_streamed_mvau_region,
)
from finn.dataflow.parameters.cyclic.definition import (
    CYCLIC_PARAMETER_KERNEL_SPEC,
    CyclicParameterBinding,
    CyclicParameterBindingWitness,
    CyclicParameterKernelPaths,
    CyclicParameterRegionDeclaration,
    CyclicRamStyle,
)
from finn.dataflow.region import DataflowRegion, NumericElementType, Port
from finn.dataflow.region_validation import RegionValidationReport

INT8 = NumericElementType("int", 8)
INT16 = NumericElementType("int", 16)


def _weight_port(declaration: CyclicParameterRegionDeclaration) -> Port:
    if declaration is CyclicParameterRegionDeclaration.FULL_TILE:
        region = construct_standard_streamed_mvau_region(2, 4, 4, INT8, INT8, INT16, 2, 2)
    else:
        region = construct_batch_interleaved_streamed_mvau_region(
            6, 4, 6, INT8, INT8, INT16, 3, 2, 3
        )
    return region.input_interface("weight").port


def _started(
    declaration: CyclicParameterRegionDeclaration,
    *,
    initializer_available: bool = True,
    runtime_writable: bool = False,
) -> tuple[Engine, DesignPoint]:
    engine = Engine()
    space = engine.validate(CYCLIC_PARAMETER_KERNEL_SPEC)
    point = engine.start(
        space,
        {
            str(CyclicParameterKernelPaths.OUTPUT_PORT): _weight_port(declaration),
            str(CyclicParameterKernelPaths.INITIALIZER_AVAILABLE): initializer_available,
            str(CyclicParameterKernelPaths.RUNTIME_WRITABLE): runtime_writable,
        },
    )
    return engine, point


def _commit_region(
    engine: Engine,
    point: DesignPoint,
    declaration: CyclicParameterRegionDeclaration,
) -> DesignPoint:
    return engine.commit_assignments(
        point, {CyclicParameterKernelPaths.REGION_DECLARATION: declaration}
    ).point


def _commit_binding(
    engine: Engine,
    point: DesignPoint,
    binding: CyclicParameterBinding,
    ram_style: CyclicRamStyle,
    pumped_memory: bool,
) -> DesignPoint:
    assignments: dict[QualifiedPath, object] = {
        CyclicParameterKernelPaths.BINDING: binding,
        CyclicParameterKernelPaths.RAM_STYLE: ram_style,
        CyclicParameterKernelPaths.PUMPED_MEMORY: pumped_memory,
    }
    return engine.commit_assignments(point, assignments).point


@pytest.mark.parametrize("declaration", tuple(CyclicParameterRegionDeclaration))
def test_cyclic_kernel_selects_an_exact_structurally_valid_region(
    declaration: CyclicParameterRegionDeclaration,
) -> None:
    engine, point = _started(declaration)
    point = _commit_region(engine, point, declaration)
    answer = engine.query_property(point, CyclicParameterKernelPaths.REGION)

    assert isinstance(answer, Decided)
    assert isinstance(answer.value, DataflowRegion)
    assert answer.value.output_interface("weight").port == _weight_port(declaration)
    assert engine.query_property(point, CyclicParameterKernelPaths.REGION_VALIDATION) == Decided(
        RegionValidationReport()
    )
    assert engine.check_readiness(point, "cyclic_model_structural").ready is True
    assert engine.check_readiness(point, "cyclic_binding_feasibility").ready is None


@pytest.mark.parametrize("binding", tuple(CyclicParameterBinding))
def test_memory_binding_identity_does_not_change_the_delivery_region(
    binding: CyclicParameterBinding,
) -> None:
    engine, point = _started(CyclicParameterRegionDeclaration.FULL_TILE)
    point = _commit_region(engine, point, CyclicParameterRegionDeclaration.FULL_TILE)
    expected = engine.query_property(point, CyclicParameterKernelPaths.REGION)
    point = _commit_binding(engine, point, binding, CyclicRamStyle.BRAM, False)

    assert engine.query_property(point, CyclicParameterKernelPaths.REGION) == expected
    witness = engine.query_property(point, CyclicParameterKernelPaths.BINDING_WITNESS)
    assert isinstance(witness, Decided)
    assert isinstance(witness.value, CyclicParameterBindingWitness)


def test_ram_style_pumping_and_runtime_writability_are_binding_owned() -> None:
    selected_regions = []
    for runtime_writable in (False, True):
        for ram_style in CyclicRamStyle:
            for pumped_memory in (False, True):
                engine, point = _started(
                    CyclicParameterRegionDeclaration.CHUNKED,
                    initializer_available=not runtime_writable,
                    runtime_writable=runtime_writable,
                )
                point = _commit_region(engine, point, CyclicParameterRegionDeclaration.CHUNKED)
                selected_regions.append(
                    engine.query_property(point, CyclicParameterKernelPaths.REGION)
                )
                point = _commit_binding(
                    engine,
                    point,
                    CyclicParameterBinding.FINN_RTL_MEMSTREAM,
                    ram_style,
                    pumped_memory,
                )
                assert engine.check_readiness(point, "cyclic_binding_feasibility").ready is True
    assert len(set(selected_regions)) == 1


def test_runtime_writability_changes_binding_feasibility_not_the_region() -> None:
    engine, point = _started(
        CyclicParameterRegionDeclaration.FULL_TILE,
        initializer_available=False,
        runtime_writable=False,
    )
    point = _commit_region(engine, point, CyclicParameterRegionDeclaration.FULL_TILE)
    region = engine.query_property(point, CyclicParameterKernelPaths.REGION)
    point = _commit_binding(
        engine,
        point,
        CyclicParameterBinding.FINN_RTL_MEMSTREAM,
        CyclicRamStyle.AUTO,
        False,
    )
    assessment = engine.evaluate_constraint_set(point, "cyclic_binding_feasibility")
    assert assessment.answers[CyclicParameterKernelPaths.LOCAL_STATE_AVAILABLE] == Decided(False)
    assert assessment.verdict is False
    assert engine.query_property(point, CyclicParameterKernelPaths.REGION) == region


def test_finnlib_hls_memstream_rejects_pumped_memory_without_changing_region() -> None:
    engine, point = _started(CyclicParameterRegionDeclaration.FULL_TILE)
    point = _commit_region(engine, point, CyclicParameterRegionDeclaration.FULL_TILE)
    region = engine.query_property(point, CyclicParameterKernelPaths.REGION)
    point = _commit_binding(
        engine,
        point,
        CyclicParameterBinding.FINNLIB_HLS_MEMSTREAM,
        CyclicRamStyle.BRAM,
        True,
    )
    assessment = engine.evaluate_constraint_set(point, "cyclic_binding_feasibility")
    assert assessment.answers[CyclicParameterKernelPaths.PUMPING_SUPPORTED] == Decided(False)
    assert engine.query_property(point, CyclicParameterKernelPaths.REGION) == region


def test_chunked_delivery_rejects_unvalidated_pumped_memory() -> None:
    engine, point = _started(CyclicParameterRegionDeclaration.CHUNKED)
    point = _commit_region(engine, point, CyclicParameterRegionDeclaration.CHUNKED)
    point = _commit_binding(
        engine,
        point,
        CyclicParameterBinding.FINN_RTL_MEMSTREAM,
        CyclicRamStyle.BRAM,
        True,
    )
    assessment = engine.evaluate_constraint_set(point, "cyclic_binding_feasibility")
    assert assessment.answers[CyclicParameterKernelPaths.PUMPING_SUPPORTED] == Decided(False)
