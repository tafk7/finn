# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""D6b normalization gates for the retained batch-interleaved design."""

from __future__ import annotations

from typing import cast

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.design import Decided, DesignPoint, Engine, QualifiedPath, Unresolved
from finn.dataflow.ops.mvau.associations import (
    BindingLocalStateDestination,
    MVAUParameterTopology,
    MVAUSourceAssociation,
    SemanticOperandDestination,
)
from finn.dataflow.ops.mvau.designs.batch_interleaved import (
    BATCH_INTERLEAVED_NODE,
    MVAU_BATCH_INTERLEAVED_DESIGN,
)
from finn.dataflow.ops.mvau.input_supply import EXTERNAL_SUPPLY, FINN_RTL_MEMSTREAM_SUPPLY
from finn.dataflow.ops.mvau.regions import MVAURegionDeclaration
from finn.dataflow.kernels.dsp import DspBlock
from finn.dataflow.ops.mvau.problem import (
    MVAUComputationProfile,
    MVAUProblemPaths,
    MVAUSourceDescription,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.parameters.cyclic.definition import CyclicRamStyle

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
            "mvau_batch",
            "activation",
            "weights",
            "output",
            (6,),
        ),
        MVAUProblemPaths.TARGET_DSP_BLOCK: DspBlock.DSP58,
        MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS: 5.0,
        MVAUProblemPaths.TARGET_FPGA_PART: "xcvc1902-vsva2197-2MP-e-S",
    }


def _point(supply: str = EXTERNAL_SUPPLY) -> tuple[Engine, DesignPoint]:
    assembly = MVAU_BATCH_INTERLEAVED_DESIGN
    engine = Engine()
    point = engine.start(engine.validate(assembly.specification), _facts())
    point = engine.commit_assignments(
        point,
        {
            assembly.semantics.pe.path: 3,
            assembly.semantics.simd.path: 2,
            assembly.semantics.interleave.path: 3,
            assembly.input_supply.declaration.choice.path: supply,
            **(
                {
                    assembly.input_supply.settings.ram_style.path: CyclicRamStyle.BRAM,
                    assembly.input_supply.settings.pumped_memory.path: False,
                }
                if supply == FINN_RTL_MEMSTREAM_SUPPLY
                else {}
            ),
        },
    ).point
    return engine, point


def _network(engine: Engine, point: DesignPoint) -> DataflowNetwork:
    answer = engine.query_property(point, MVAU_BATCH_INTERLEAVED_DESIGN.design.network.path)
    assert isinstance(answer, Decided)
    return cast(DataflowNetwork, answer.value)


def test_batch_interleaved_uses_the_frozen_design_decision_paths() -> None:
    semantics = MVAU_BATCH_INTERLEAVED_DESIGN.semantics
    assert semantics.pe.path == QualifiedPath("mvau.design.batch_interleaved.pe")
    assert semantics.simd.path == QualifiedPath("mvau.design.batch_interleaved.simd")
    assert semantics.interleave.path == QualifiedPath("mvau.design.batch_interleaved.interleave")


def test_batch_interleaved_is_a_canonical_singleton_network() -> None:
    engine, point = _point()
    network = _network(engine, point)
    assert tuple(node.id for node in network.nodes) == (BATCH_INTERLEAVED_NODE,)
    assert network.edges == ()
    assert tuple(boundary.id for boundary in network.boundaries) == (
        "input.activation",
        "input.weight",
        "output.output",
    )
    assert tuple(boundary.endpoint.port_id for boundary in network.boundaries) == (
        "activation",
        "weight",
        "output",
    )


def test_batch_interleaved_source_association_targets_the_singleton_node() -> None:
    engine, point = _point()
    answer = engine.query_property(point, MVAU_BATCH_INTERLEAVED_DESIGN.source_association.path)
    assert isinstance(answer, Decided)
    association = cast(MVAUSourceAssociation, answer.value)
    assert (
        association.region_declaration_id == MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED.value
    )
    assert association.parameter_topology is MVAUParameterTopology.DIRECT
    assert association.compute_kernel_id == ""
    assert all(
        isinstance(item.destination, SemanticOperandDestination) for item in association.operands
    )
    assert {
        item.destination.owner_id
        for item in association.operands
        if isinstance(item.destination, SemanticOperandDestination)
    } == {BATCH_INTERLEAVED_NODE}


def test_batch_interleaved_is_explicitly_semantic_only() -> None:
    engine, point = _point()
    answer = MVAU_BATCH_INTERLEAVED_DESIGN.inventory.realize(engine, point)
    assert isinstance(answer, Unresolved)
    assert "design-placement-has-no-kernel" in {item.code for item in answer.findings}


def test_common_memstream_supply_attaches_to_the_singleton_weight_boundary() -> None:
    engine, point = _point(FINN_RTL_MEMSTREAM_SUPPLY)
    network = _network(engine, point)
    assert {node.id for node in network.nodes} == {BATCH_INTERLEAVED_NODE, "delivery"}
    assert {edge.id for edge in network.edges} == {"weight"}
    assert {boundary.id for boundary in network.boundaries} == {
        "input.activation",
        "output.output",
    }
    association_answer = engine.query_property(
        point, MVAU_BATCH_INTERLEAVED_DESIGN.source_association.path
    )
    assert isinstance(association_answer, Decided)
    association = cast(MVAUSourceAssociation, association_answer.value)
    weight = next(item for item in association.operands if item.role == "weight")
    assert weight.destination == BindingLocalStateDestination("delivery", "weights")


def test_batch_interleaved_has_no_physical_compute_kernel_candidate() -> None:
    assembly = MVAU_BATCH_INTERLEAVED_DESIGN
    assert tuple(placement.name for placement in assembly.design.placements) == (
        "compute",
        "delivery",
    )
    compute = assembly.design.placement("compute")
    assert compute.candidates == ()
