# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import replace

from finn.dataflow.design import Absent, Decided, DesignPoint, Engine, QualifiedPath, Unresolved
from finn.dataflow.mvau.regions import construct_standard_streamed_mvau_region
from finn.dataflow.parameters.cyclic.definition import (
    CYCLIC_PARAMETER_KERNEL_SPEC,
    CyclicParameterBinding,
    CyclicParameterBindingSelection,
    CyclicParameterKernelPaths,
    CyclicRamStyle,
    CyclicTargetMemoryCapabilities,
)
from finn.dataflow.region import DataflowRegion, NumericElementType, Port
from finn.dataflow.region_validation import RegionValidationReport

INT8 = NumericElementType("int", 8)
INT16 = NumericElementType("int", 16)
_MISSING = object()


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


def _started(
    *,
    output_port: Port | None = None,
    initializer_available: bool = True,
    runtime_writable: bool = False,
    target_capabilities: object = _MISSING,
) -> tuple[Engine, DesignPoint]:
    engine = Engine()
    space = engine.validate(CYCLIC_PARAMETER_KERNEL_SPEC)
    problem: dict[str, object] = {
        str(CyclicParameterKernelPaths.OUTPUT_PORT): output_port or _weight_port(),
        str(CyclicParameterKernelPaths.INITIALIZER_AVAILABLE): initializer_available,
        str(CyclicParameterKernelPaths.RUNTIME_WRITABLE): runtime_writable,
    }
    if target_capabilities is not _MISSING:
        problem[str(CyclicParameterKernelPaths.TARGET_MEMORY_CAPABILITIES)] = target_capabilities
    return engine, engine.start(space, problem)


def _commit_binding(
    engine: Engine,
    point: DesignPoint,
    binding: CyclicParameterBinding,
    *,
    ram_style: CyclicRamStyle | None = None,
    pumped_memory: bool | None = None,
) -> DesignPoint:
    assignments: dict[QualifiedPath, object] = {CyclicParameterKernelPaths.BINDING: binding}
    if ram_style is not None:
        assignments[CyclicParameterKernelPaths.RAM_STYLE] = ram_style
    if pumped_memory is not None:
        assignments[CyclicParameterKernelPaths.PUMPED_MEMORY] = pumped_memory
    return engine.commit_assignments(point, assignments).point


def test_cyclic_region_is_exact_and_structurally_ready_without_a_selector_or_binding() -> None:
    engine, point = _started()
    answer = engine.query_property(point, CyclicParameterKernelPaths.REGION)

    assert isinstance(answer, Decided)
    assert isinstance(answer.value, DataflowRegion)
    assert answer.value.output_interface("weight").port == _weight_port()
    assert engine.query_property(point, CyclicParameterKernelPaths.REGION_VALIDATION) == Decided(
        RegionValidationReport()
    )
    assert engine.check_readiness(point, "cyclic_model_structural").ready is True
    assert engine.check_readiness(point, "cyclic_binding_feasibility").ready is None


def test_finnlib_hls_requires_initializer_and_has_no_rtl_only_decisions() -> None:
    engine, point = _started(initializer_available=False, runtime_writable=True)
    point = _commit_binding(engine, point, CyclicParameterBinding.FINNLIB_HLS_MEMSTREAM)

    assert isinstance(engine.decision_state(point, CyclicParameterKernelPaths.RAM_STYLE), Absent)
    assert isinstance(
        engine.decision_state(point, CyclicParameterKernelPaths.PUMPED_MEMORY), Absent
    )
    assessment = engine.evaluate_constraint_set(point, "cyclic_binding_feasibility")
    assert assessment.answers[CyclicParameterKernelPaths.LOCAL_STATE_AVAILABLE] == Decided(False)
    assert engine.check_readiness(point, "cyclic_binding_feasibility").ready is True


def test_finn_rtl_accepts_runtime_writable_only_initialization() -> None:
    engine, point = _started(initializer_available=False, runtime_writable=True)
    point = _commit_binding(
        engine,
        point,
        CyclicParameterBinding.FINN_RTL_MEMSTREAM,
        ram_style=CyclicRamStyle.BRAM,
        pumped_memory=False,
    )
    assert engine.evaluate_constraint_set(point, "cyclic_binding_feasibility").verdict is True


def test_non_versal_initialized_uram_requires_runtime_writeability() -> None:
    capabilities = CyclicTargetMemoryCapabilities(supports_initialized_uram=False)
    engine, point = _started(target_capabilities=capabilities)
    point = _commit_binding(
        engine,
        point,
        CyclicParameterBinding.FINN_RTL_MEMSTREAM,
        ram_style=CyclicRamStyle.URAM,
        pumped_memory=False,
    )
    assessment = engine.evaluate_constraint_set(point, "cyclic_binding_feasibility")
    assert assessment.answers[CyclicParameterKernelPaths.URAM_INITIALIZATION_SUPPORTED] == Decided(
        False
    )

    writable_engine, writable = _started(runtime_writable=True, target_capabilities=capabilities)
    writable = _commit_binding(
        writable_engine,
        writable,
        CyclicParameterBinding.FINN_RTL_MEMSTREAM,
        ram_style=CyclicRamStyle.URAM,
        pumped_memory=False,
    )
    assert (
        writable_engine.evaluate_constraint_set(writable, "cyclic_binding_feasibility").verdict
        is True
    )


def test_missing_uram_target_fact_is_truthfully_unresolved() -> None:
    engine, point = _started(target_capabilities=_MISSING)
    point = _commit_binding(
        engine,
        point,
        CyclicParameterBinding.FINN_RTL_MEMSTREAM,
        ram_style=CyclicRamStyle.URAM,
        pumped_memory=False,
    )
    answer = engine.evaluate_constraint_set(point, "cyclic_binding_feasibility").answers[
        CyclicParameterKernelPaths.URAM_INITIALIZATION_SUPPORTED
    ]
    assert isinstance(answer, Unresolved)


def test_one_field_pumping_is_rejected_but_wider_rtl_pumping_is_supported() -> None:
    for elements_per_beat, expected in ((1, False), (4, True)):
        engine, point = _started(output_port=_weight_port(elements_per_beat=elements_per_beat))
        point = _commit_binding(
            engine,
            point,
            CyclicParameterBinding.FINN_RTL_MEMSTREAM,
            ram_style=CyclicRamStyle.BRAM,
            pumped_memory=True,
        )
        assessment = engine.evaluate_constraint_set(point, "cyclic_binding_feasibility")
        assert assessment.answers[CyclicParameterKernelPaths.PUMPING_SUPPORTED] == Decided(expected)


def test_binding_selection_does_not_claim_feasibility_or_change_the_region() -> None:
    engine, point = _started(initializer_available=False, runtime_writable=True)
    region = engine.query_property(point, CyclicParameterKernelPaths.REGION)
    point = _commit_binding(engine, point, CyclicParameterBinding.FINNLIB_HLS_MEMSTREAM)
    selection = engine.query_property(point, CyclicParameterKernelPaths.BINDING_SELECTION)

    assert isinstance(selection, Decided)
    assert isinstance(selection.value, CyclicParameterBindingSelection)
    assert selection.value.ram_style is None
    assert selection.value.pumped_memory is None
    assert engine.query_property(point, CyclicParameterKernelPaths.REGION) == region
    assert engine.evaluate_constraint_set(point, "cyclic_binding_feasibility").verdict is False
