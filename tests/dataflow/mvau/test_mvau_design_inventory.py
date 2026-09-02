# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""D7 forcing cases for the reviewed MVAU design inventory."""

from __future__ import annotations

from typing import cast

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.design import Absent, Decided, DesignPoint, Engine, QualifiedPath
from finn.dataflow.ops.mvau.associations import MVAUSourceAssociation
from finn.dataflow.ops.mvau.designs.batch_interleaved import BatchInterleavedDesign
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.ops.mvau.inventory import (
    MVAU_ARTIFACT_READINESS,
    MVAU_DESIGN_INVENTORY,
    MVAU_FEASIBILITY_CONSTRAINT_SET,
    MVAU_STRUCTURAL_READINESS,
    admissible_mvau_designs,
)
from finn.dataflow.ops.mvau.input_supply import EXTERNAL_SUPPLY, FINN_RTL_MEMSTREAM_SUPPLY
from finn.dataflow.kernels.dsp import DspBlock
from finn.dataflow.ops.mvau.problem import (
    MVAUComputationProfile,
    MVAUProblemPaths,
    MVAUSourceDescription,
)
from finn.dataflow.parameters.cyclic.definition import CyclicRamStyle
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.ops.mvau.op import MvauDataflowOp

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
        MVAUProblemPaths.TARGET_DSP_BLOCK: DspBlock.DSP58,
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


def test_mvau_op_uses_the_compiler_owned_inventory() -> None:
    compiled = MvauDataflowOp.compiled_dataflow_operation()
    assert compiled is not None and compiled.inventory is not None
    assert compiled.inventory.design_ids == MVAU_DESIGN_INVENTORY.inventory.design_ids
    assert MvauDataflowOp.build_design_space_spec() is compiled.specification
    assert MvauDataflowOp.result_path() == compiled.result.path
    assert MvauDataflowOp.source_association_path() == compiled.source_association.path


def test_structural_metadata_does_not_require_physical_kernel_choices() -> None:
    assembly = MVAU_DESIGN_INVENTORY
    engine, point = _start()
    assert assembly.inventory.design_selection is not None
    point = engine.commit_assignments(
        point,
        {
            assembly.inventory.design_selection.path: DotProductDesign.id,
            assembly.dot_product.pe.path: 3,
            assembly.dot_product.simd.path: 2,
            assembly.input_supply.declaration.choice.path: EXTERNAL_SUPPLY,
        },
    ).point

    association = engine.query_property(point, assembly.source_association.path)
    assert isinstance(association, Decided)
    assert assembly.compute_pumping.path not in point.assignments
    assert engine.check_readiness(point, MVAU_STRUCTURAL_READINESS).ready is True
    assert engine.check_readiness(point, MVAU_ARTIFACT_READINESS).ready is None


def test_unselected_design_declarations_are_absent() -> None:
    engine, point = _dot_product()
    batch = MVAU_DESIGN_INVENTORY.batch_interleaved
    assert isinstance(engine.decision_state(point, batch.pe.path), Absent)
    assert isinstance(engine.query_property(point, batch.network.path), Absent)


def test_dot_product_resolves_one_network_and_logical_association() -> None:
    engine, point = _dot_product()
    result = engine.query_property(point, MVAU_DESIGN_INVENTORY.result.path)
    association_answer = engine.query_property(point, MVAU_DESIGN_INVENTORY.source_association.path)
    assert isinstance(result, Decided) and isinstance(result.value, DataflowNetwork)
    assert isinstance(association_answer, Decided)
    association = cast(MVAUSourceAssociation, association_answer.value)
    assert set(vars(association)) == {
        "source_node_id",
        "fused_source_node_ids",
        "region_declaration_id",
        "parameter_topology",
        "operands",
    }
    assert engine.check_readiness(point, MVAU_STRUCTURAL_READINESS).ready is True
    assert engine.check_readiness(point, MVAU_ARTIFACT_READINESS).ready is True


def test_supplied_dot_product_changes_topology_without_physical_logical_metadata() -> None:
    engine, point = _dot_product(FINN_RTL_MEMSTREAM_SUPPLY)
    answer = engine.query_property(point, MVAU_DESIGN_INVENTORY.source_association.path)
    assert isinstance(answer, Decided)
    association = cast(MVAUSourceAssociation, answer.value)
    assert association.parameter_topology.value == "cyclic"


def test_batch_interleaved_resolves_semantics_but_is_not_build_admitted() -> None:
    engine, point = _batch_interleaved()
    result = engine.query_property(point, MVAU_DESIGN_INVENTORY.result.path)
    association_answer = engine.query_property(point, MVAU_DESIGN_INVENTORY.source_association.path)
    assert isinstance(result, Decided) and isinstance(result.value, DataflowNetwork)
    assert isinstance(association_answer, Decided)
    association = cast(MVAUSourceAssociation, association_answer.value)
    assert association.parameter_topology.value == "direct"
    assert engine.check_readiness(point, MVAU_STRUCTURAL_READINESS).ready is True
    assert engine.evaluate_constraint_set(point, MVAU_FEASIBILITY_CONSTRAINT_SET).verdict is None
    assert engine.check_readiness(point, MVAU_ARTIFACT_READINESS).ready is None


def test_source_admission_names_semantic_designs_without_claiming_buildability() -> None:
    engine, point = _start()
    assert admissible_mvau_designs(engine, point) == (
        DotProductDesign.id,
        BatchInterleavedDesign.id,
    )
