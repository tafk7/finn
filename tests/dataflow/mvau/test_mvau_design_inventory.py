# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""D7 forcing cases for the reviewed MVAU design inventory."""

from __future__ import annotations

from typing import cast

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.design import Absent, Decided, DesignPoint, Engine, QualifiedPath
from finn.dataflow.mvau.associations import MVAUSourceAssociation
from finn.dataflow.mvau.designs.batch_interleaved import BatchInterleavedDesign
from finn.dataflow.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.mvau.designs.inventory import (
    MVAU_ARTIFACT_READINESS,
    MVAU_DESIGN_INVENTORY,
    MVAU_FEASIBILITY_CONSTRAINT_SET,
    MVAU_STRUCTURAL_READINESS,
    admissible_mvau_designs,
)
from finn.dataflow.mvau.hardware.dotp_axi import DotpAxiKernel
from finn.dataflow.mvau.hardware.replay_buffer import ReplayBufferKernel
from finn.dataflow.mvau.input_supply import EXTERNAL_SUPPLY, FINN_RTL_MEMSTREAM_SUPPLY
from finn.dataflow.mvau_problem import (
    MVAUComputationProfile,
    MVAUDspBlock,
    MVAUProblemPaths,
    MVAUSourceDescription,
)
from finn.dataflow.parameters.cyclic.definition import CyclicRamStyle
from finn.dataflow.resolution import NetworkRef

INT8 = DataType["INT8"]
INT16 = DataType["INT16"]


def _facts() -> dict[QualifiedPath, object]:
    return {
        MVAUProblemPaths.REPETITIONS: 6,
        MVAUProblemPaths.MATRIX_WIDTH: 4,
        MVAUProblemPaths.MATRIX_HEIGHT: 6,
        MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE: INT8,
        MVAUProblemPaths.WEIGHT_ELEMENT_TYPE: INT8,
        MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE: INT16,
        MVAUProblemPaths.OUTPUT_ELEMENT_TYPE: INT16,
        MVAUProblemPaths.COMPUTATION_PROFILE: MVAUComputationProfile.ACCUMULATOR_INTEGER,
        MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE: True,
        MVAUProblemPaths.RUNTIME_WRITABLE: False,
        MVAUProblemPaths.SOURCE_DESCRIPTION: MVAUSourceDescription(
            "mvau_inventory",
            "activation",
            "weights",
            "output",
            (6,),
        ),
        MVAUProblemPaths.TARGET_DSP_BLOCK: MVAUDspBlock.DSP58,
        MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS: 5.0,
        MVAUProblemPaths.TARGET_FPGA_PART: "xcvc1902-vsva2197-2MP-e-S",
    }


def _start() -> tuple[Engine, DesignPoint]:
    engine = Engine()
    return engine, engine.start(engine.validate(MVAU_DESIGN_INVENTORY.specification), _facts())


def _dot_product(supply: str = EXTERNAL_SUPPLY) -> tuple[Engine, DesignPoint]:
    assembly = MVAU_DESIGN_INVENTORY
    engine, point = _start()
    assert assembly.inventory.design_path is not None
    assignments: dict[QualifiedPath, object] = {
        assembly.inventory.design_path: DotProductDesign.id,
        assembly.dot_product.pe.path: 3,
        assembly.dot_product.simd.path: 2,
        assembly.compute_pumping.path: False,
        assembly.input_supply.declaration.choice.path: supply,
    }
    if supply == FINN_RTL_MEMSTREAM_SUPPLY:
        assignments.update(
            {
                assembly.input_supply.settings.ram_style.path: CyclicRamStyle.BRAM,
                assembly.input_supply.settings.pumped_memory.path: False,
            }
        )
    return engine, engine.commit_assignments(point, assignments).point


def _batch_interleaved() -> tuple[Engine, DesignPoint]:
    assembly = MVAU_DESIGN_INVENTORY
    engine, point = _start()
    assert assembly.inventory.design_path is not None
    return engine, engine.commit_assignments(
        point,
        {
            assembly.inventory.design_path: BatchInterleavedDesign.id,
            assembly.batch_interleaved.pe.path: 3,
            assembly.batch_interleaved.simd.path: 2,
            assembly.batch_interleaved.interleave.path: 3,
            assembly.input_supply.declaration.choice.path: EXTERNAL_SUPPLY,
        },
    ).point


def test_v6_inventory_has_only_the_frozen_decision_paths() -> None:
    assembly = MVAU_DESIGN_INVENTORY
    assert {item.path.value for item in assembly.specification.decisions} == {
        "mvau.design",
        "mvau.design.dot_product.pe",
        "mvau.design.dot_product.simd",
        "mvau.design.dot_product.compute.dotp_axi.compute_pumping",
        "mvau.design.batch_interleaved.pe",
        "mvau.design.batch_interleaved.simd",
        "mvau.design.batch_interleaved.interleave",
        "mvau.input.weight.supply",
        "mvau.input.weight.finn_rtl_memstream.ram_style",
        "mvau.input.weight.finn_rtl_memstream.pumped_memory",
    }


def test_unselected_design_declarations_are_absent() -> None:
    engine, point = _dot_product()
    batch = MVAU_DESIGN_INVENTORY.batch_interleaved
    assert isinstance(engine.decision_state(point, batch.pe.path), Absent)
    assert isinstance(engine.query_property(point, batch.network.path), Absent)


def test_dot_product_resolves_one_network_and_v11_ownership() -> None:
    engine, point = _dot_product()
    result = engine.query_property(point, MVAU_DESIGN_INVENTORY.result.path)
    association_answer = engine.query_property(point, MVAU_DESIGN_INVENTORY.source_association.path)
    assert isinstance(result, Decided) and isinstance(result.value, NetworkRef)
    assert isinstance(association_answer, Decided)
    association = cast(MVAUSourceAssociation, association_answer.value)
    assert result.value.source_association == association
    assert association.design_id == DotProductDesign.id
    assert association.compute_kernel_id == DotpAxiKernel.id
    assert association.kernel_ids == (DotpAxiKernel.id, ReplayBufferKernel.id)
    assert association.decision_paths == (
        QualifiedPath("mvau.design"),
        QualifiedPath("mvau.input.weight.supply"),
        MVAU_DESIGN_INVENTORY.dot_product.pe.path,
        MVAU_DESIGN_INVENTORY.dot_product.simd.path,
        MVAU_DESIGN_INVENTORY.compute_pumping.path,
    )
    assert engine.check_readiness(point, MVAU_STRUCTURAL_READINESS).ready is True
    assert engine.check_readiness(point, MVAU_ARTIFACT_READINESS).ready is True


def test_supplied_dot_product_v11_ownership_includes_delivery_kernel_and_choices() -> None:
    engine, point = _dot_product(FINN_RTL_MEMSTREAM_SUPPLY)
    answer = engine.query_property(point, MVAU_DESIGN_INVENTORY.source_association.path)
    assert isinstance(answer, Decided)
    association = cast(MVAUSourceAssociation, answer.value)
    assert association.supply_kernel_id == FINN_RTL_MEMSTREAM_SUPPLY
    assert association.kernel_ids == (
        DotpAxiKernel.id,
        ReplayBufferKernel.id,
        FINN_RTL_MEMSTREAM_SUPPLY,
    )
    assert association.decision_paths[-2:] == (
        MVAU_DESIGN_INVENTORY.input_supply.settings.ram_style.path,
        MVAU_DESIGN_INVENTORY.input_supply.settings.pumped_memory.path,
    )


def test_batch_interleaved_resolves_semantics_but_is_not_build_admitted() -> None:
    engine, point = _batch_interleaved()
    result = engine.query_property(point, MVAU_DESIGN_INVENTORY.result.path)
    association_answer = engine.query_property(point, MVAU_DESIGN_INVENTORY.source_association.path)
    assert isinstance(result, Decided) and isinstance(result.value, NetworkRef)
    assert isinstance(association_answer, Decided)
    association = cast(MVAUSourceAssociation, association_answer.value)
    assert association.design_id == BatchInterleavedDesign.id
    assert association.compute_kernel_id == ""
    assert association.kernel_ids == ()
    assert engine.check_readiness(point, MVAU_STRUCTURAL_READINESS).ready is True
    assert engine.evaluate_constraint_set(point, MVAU_FEASIBILITY_CONSTRAINT_SET).verdict is None
    assert engine.check_readiness(point, MVAU_ARTIFACT_READINESS).ready is None


def test_source_admission_names_semantic_designs_without_claiming_buildability() -> None:
    engine, point = _start()
    assert admissible_mvau_designs(engine, point) == (
        DotProductDesign.id,
        BatchInterleavedDesign.id,
    )
