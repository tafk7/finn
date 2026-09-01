# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""D6a semantic, physical, and association gates for MVAU input supply."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np  # type: ignore[import-not-found]
import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.design import Absent, Decided, DesignPoint, Engine, QualifiedPath, Unresolved
from finn.dataflow.artifacts import TargetIdentity, kernel_artifact_identity
from finn.dataflow.ops.mvau.associations import (
    BindingLocalStateDestination,
    MVAUParameterTopology,
    MVAUSourceAssociation,
)
from finn.dataflow.ops.mvau.designs.dot_product import MVAU_DOT_PRODUCT_DESIGN
from finn.dataflow.ops.mvau.binding import source_roots
from finn.dataflow.kernels.finn_rtl_memstream import (
    FINN_MEMSTREAM_MODULE,
    FINN_MEMSTREAM_SOURCES,
    FinnRtlMemstreamKernel,
)
from finn.dataflow.ops.mvau.artifacts._implementation import (
    MVAUDecomposedArtifactRequirements,
    package_decomposed_artifact,
    prepare_decomposed_synthesis,
    prepare_ip_package,
    staged_layout,
    write_decomposed_artifact,
)
from finn.dataflow.ops.mvau.artifacts.supplied import (
    build_supplied_artifact_requirements,
)
from finn.dataflow.ops.mvau.input_supply import (
    DELIVERY_EDGE,
    DELIVERY_NODE,
    EXTERNAL_SUPPLY,
    FINN_RTL_MEMSTREAM_SUPPLY,
)
from finn.dataflow.ops.mvau.elaboration import compose_dot_product_design
from finn.dataflow.ops.mvau.source import MVAUResolvedDesign, MVAUSourceProjection
from finn.dataflow.kernels.dsp import DspBlock
from finn.dataflow.ops.mvau.problem import (
    MVAUComputationProfile,
    MVAUProblemPaths,
    MVAUSourceDescription,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.ops.mvau import NetworkRef
from finn.dataflow.parameters.cyclic.definition import CyclicRamStyle

INT8 = DataType["INT8"]
INT16 = DataType["INT16"]


def _facts(
    *,
    initialized: bool = True,
    runtime_writable: bool = False,
    source_node_id: str = "mvau_node",
) -> dict[QualifiedPath, object]:
    return {
        MVAUProblemPaths.REPETITIONS: 2,
        MVAUProblemPaths.MATRIX_WIDTH: 4,
        MVAUProblemPaths.MATRIX_HEIGHT: 6,
        MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE: INT8,
        MVAUProblemPaths.WEIGHT_ELEMENT_TYPE: INT8,
        MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE: INT16,
        MVAUProblemPaths.OUTPUT_ELEMENT_TYPE: INT16,
        MVAUProblemPaths.COMPUTATION_PROFILE: MVAUComputationProfile.ACCUMULATOR_INTEGER,
        MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE: initialized,
        MVAUProblemPaths.RUNTIME_WRITABLE: runtime_writable,
        MVAUProblemPaths.SOURCE_DESCRIPTION: MVAUSourceDescription(
            source_node_id,
            "activation_tensor",
            "weight_tensor",
            "output_tensor",
            (2,),
        ),
        MVAUProblemPaths.TARGET_DSP_BLOCK: DspBlock.DSP58,
        MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS: 5.0,
        MVAUProblemPaths.TARGET_FPGA_PART: "xcvc1902-vsva2197-2MP-e-S",
    }


def _point(
    supply: str,
    *,
    initialized: bool = True,
    runtime_writable: bool = False,
    source_node_id: str = "mvau_node",
    ram_style: CyclicRamStyle = CyclicRamStyle.BRAM,
    pumped_memory: bool = False,
) -> tuple[Engine, DesignPoint]:
    assembly = MVAU_DOT_PRODUCT_DESIGN
    engine = Engine()
    point = engine.start(
        engine.validate(assembly.specification),
        _facts(
            initialized=initialized,
            runtime_writable=runtime_writable,
            source_node_id=source_node_id,
        ),
    )
    assignments = {
        assembly.semantics.pe.path: 2,
        assembly.semantics.simd.path: 2,
        assembly.compute_pumping.path: False,
        assembly.input_supply.declaration.choice.path: supply,
    }
    if supply == FINN_RTL_MEMSTREAM_SUPPLY:
        assignments.update(
            {
                assembly.input_supply.settings.ram_style.path: ram_style,
                assembly.input_supply.settings.pumped_memory.path: pumped_memory,
            }
        )
    return engine, engine.commit_assignments(point, assignments).point


def _realized(supply: str, **kwargs: object):  # type: ignore[no-untyped-def]
    engine, point = _point(supply, **kwargs)  # type: ignore[arg-type]
    answer = MVAU_DOT_PRODUCT_DESIGN.inventory.realize(engine, point)
    assert isinstance(answer, Decided)
    return engine, point, answer.value


def _resolved(engine: Engine, point: DesignPoint, network: DataflowNetwork) -> MVAUResolvedDesign:
    assembly = MVAU_DOT_PRODUCT_DESIGN
    association_answer = engine.query_property(point, assembly.source_association.path)
    assert isinstance(association_answer, Decided)
    association = cast(MVAUSourceAssociation, association_answer.value)
    facts = dict(point.problem)
    projection = MVAUSourceProjection(
        cast(MVAUSourceDescription, facts[MVAUProblemPaths.SOURCE_DESCRIPTION]),
        facts,
        {},
    )
    return MVAUResolvedDesign(
        engine,
        point,
        NetworkRef("mvau", network, association),
        association,
        "scope",
        projection,
    )


def test_mvau_supply_inventory_is_closed_finn_rtl_only_and_as_demanded() -> None:
    supply = MVAU_DOT_PRODUCT_DESIGN.input_supply
    assert supply.declaration.modes == (EXTERNAL_SUPPLY, FINN_RTL_MEMSTREAM_SUPPLY)
    assert tuple(item.id for item in supply.declaration.alternatives) == (
        FINN_RTL_MEMSTREAM_SUPPLY,
    )


def test_supply_and_memstream_decision_paths_are_the_frozen_v6_paths() -> None:
    supply = MVAU_DOT_PRODUCT_DESIGN.input_supply
    assert supply.declaration.choice.path.value == "mvau.input.weight.supply"
    assert supply.settings.ram_style.path.value == (
        "mvau.input.weight.finn_rtl_memstream.ram_style"
    )
    assert supply.settings.pumped_memory.path.value == (
        "mvau.input.weight.finn_rtl_memstream.pumped_memory"
    )


def test_existing_external_supply_network_and_association_are_unchanged() -> None:
    engine, point, realization = _realized(EXTERNAL_SUPPLY)
    assembly = MVAU_DOT_PRODUCT_DESIGN
    core_network = engine.query_property(point, assembly.semantics.network.path)
    core_association = engine.query_property(point, assembly.semantics.source_association.path)
    selected_association = engine.query_property(point, assembly.source_association.path)

    assert isinstance(core_network, Decided)
    assert realization.network == core_network.value
    assert isinstance(core_association, Decided)
    assert selected_association == core_association
    assert set(realization.kernels) == {"compute", "replay"}


def test_external_supply_makes_every_supplier_declaration_absent() -> None:
    engine, point = _point(EXTERNAL_SUPPLY)
    assembly = MVAU_DOT_PRODUCT_DESIGN
    delivery = next(node for node in assembly.design.nodes if node.node_id == DELIVERY_NODE)
    placement = assembly.design.placement("delivery")

    assert isinstance(engine.query_property(point, delivery.region.path), Absent)
    assert isinstance(engine.query_property(point, placement.selected_kernel.path), Absent)
    assert isinstance(
        engine.decision_state(point, assembly.input_supply.settings.ram_style.path), Absent
    )
    assert isinstance(
        engine.decision_state(point, assembly.input_supply.settings.pumped_memory.path), Absent
    )


def test_memstream_network_replaces_weight_boundary_with_delivery_edge() -> None:
    _engine, _point_value, realization = _realized(FINN_RTL_MEMSTREAM_SUPPLY)
    assert {node.id for node in realization.network.nodes} == {
        "replay",
        "compute",
        DELIVERY_NODE,
    }
    assert {edge.id for edge in realization.network.edges} == {
        "activation_replay",
        DELIVERY_EDGE,
    }
    assert {boundary.id for boundary in realization.network.boundaries} == {
        "activation",
        "output",
    }
    assert set(realization.kernels) == {"compute", "replay", "delivery"}


def test_memstream_output_sequence_exactly_equals_compute_demand() -> None:
    _engine, _point_value, realization = _realized(FINN_RTL_MEMSTREAM_SUPPLY)
    delivery = realization.network.node(DELIVERY_NODE).region.output_interface("weight").port
    demand = realization.network.node("compute").region.input_interface("weight").port
    assert delivery.operand == demand.operand
    assert delivery.beat_sequence == demand.beat_sequence


def test_memstream_parameters_match_the_resolved_design_point() -> None:
    _engine, _point_value, realization = _realized(
        FINN_RTL_MEMSTREAM_SUPPLY,
        ram_style=CyclicRamStyle.BRAM,
        pumped_memory=True,
    )
    delivery = realization.kernel("delivery")
    assert isinstance(delivery, FinnRtlMemstreamKernel)
    assert dict(delivery.parameters) == {
        "DEPTH": 6,
        "INITIALIZER_AVAILABLE": True,
        "INIT_FILE": "memblock.dat",
        "PUMPED_MEMORY": True,
        "RAM_STYLE": "block",
        "RUNTIME_WRITABLE": False,
        "SETS": 1,
        "WIDTH": 32,
    }
    assert delivery.components()[0].module == FINN_MEMSTREAM_MODULE
    assert tuple(source.path for source in delivery.sources) == FINN_MEMSTREAM_SOURCES


def test_memstream_source_association_moves_weights_to_delivery_local_state() -> None:
    engine, point, _realization_value = _realized(FINN_RTL_MEMSTREAM_SUPPLY)
    association = engine.query_property(point, MVAU_DOT_PRODUCT_DESIGN.source_association.path)
    assert isinstance(association, Decided)
    value = cast(MVAUSourceAssociation, association.value)
    weight = next(item for item in value.operands if item.role == "weight")
    assert value.parameter_topology is MVAUParameterTopology.CYCLIC
    assert value.supply_kernel_id == FINN_RTL_MEMSTREAM_SUPPLY
    assert weight.destination == BindingLocalStateDestination(DELIVERY_NODE, "weights")


@pytest.mark.parametrize("runtime_writable", (False, True))
def test_v6_memstream_requires_initializer_backed_supply(runtime_writable: bool) -> None:
    engine, point = _point(
        FINN_RTL_MEMSTREAM_SUPPLY,
        initialized=False,
        runtime_writable=runtime_writable,
    )
    answer = MVAU_DOT_PRODUCT_DESIGN.inventory.realize(engine, point)
    assert isinstance(answer, Unresolved)
    assert "hardware-coverage-refused" in {item.code for item in answer.findings}


def test_memstream_integrates_with_dot_product_physical_elaboration() -> None:
    engine, point, realization = _realized(FINN_RTL_MEMSTREAM_SUPPLY)
    resolved = _resolved(engine, point, realization.network)
    elaboration = compose_dot_product_design(resolved, realization)

    assert {item.module for item in elaboration.components} == {
        "finn.dataflow.mvau.decomposed_wrapper",
        "finn-rtllib.mvu.replay_buffer",
        "finnlib.rtl.dotp_axi",
        FINN_MEMSTREAM_MODULE,
    }
    delivery = elaboration.component("mvau_node.delivery.wrapper")
    assert dict(delivery.parameters)["INIT_FILE"] == "memblock.dat"
    connection = next(
        item for item in elaboration.connections if item.id == "network.delivery_to_compute"
    )
    assert connection.semantic_edge_ids == (DELIVERY_EDGE,)
    assert {item.id for item in elaboration.boundaries} == {"activation", "output"}


def test_memstream_artifact_identity_excludes_source_occurrence_and_placement() -> None:
    _engine, _point_value, realization = _realized(FINN_RTL_MEMSTREAM_SUPPLY)
    delivery = realization.kernel("delivery")
    moved_declaration = replace(
        delivery.declaration,
        namespace="elsewhere.design.delivery.finn_rtl_memstream",
    )
    moved = type(delivery)(
        moved_declaration,
        delivery.regions,
        delivery.edges,
        delivery.assignments,
        delivery.parameters,
    )
    roots = source_roots(Path(__file__).parents[3])
    assert kernel_artifact_identity(delivery, roots) == kernel_artifact_identity(moved, roots)


def test_supplied_artifact_identity_tracks_initializer_but_not_source_occurrence() -> None:
    weights = np.arange(24, dtype=np.float32).reshape(4, 6) - 12

    def build(source_node_id: str, values: np.ndarray) -> MVAUDecomposedArtifactRequirements:
        engine, point, realization = _realized(
            FINN_RTL_MEMSTREAM_SUPPLY,
            source_node_id=source_node_id,
        )
        resolved = _resolved(engine, point, realization.network)
        elaboration = compose_dot_product_design(resolved, realization)
        return build_supplied_artifact_requirements(
            resolved,
            realization,
            elaboration,
            values,
            Path(__file__).parents[3],
        )

    baseline = build("mvau_first", weights)
    another_occurrence = build("mvau_second", weights)
    changed_initializer = build("mvau_first", weights + 1)

    assert another_occurrence.identity == baseline.identity
    assert changed_initializer.identity != baseline.identity


def test_supplied_generated_packaged_synthesis_and_ipxact_inputs_are_complete(
    tmp_path: Path,
) -> None:
    engine, point, realization = _realized(FINN_RTL_MEMSTREAM_SUPPLY)
    resolved = _resolved(engine, point, realization.network)
    elaboration = compose_dot_product_design(resolved, realization)
    weights = np.arange(24, dtype=np.float32).reshape(4, 6) - 12
    requirements = build_supplied_artifact_requirements(
        resolved,
        realization,
        elaboration,
        weights,
        Path(__file__).parents[3],
    )

    assert tuple(name for name, _contents in requirements.data_files) == ("memblock.dat",)
    assert "memstream_wrapper" in requirements.wrapper_source
    assert "weight_tdata" in requirements.wrapper_source
    assert ".s_axis_0_tdata(s_axis_0_tdata)" in requirements.wrapper_source
    assert "input wire [4:0] s_axilite_AWADDR" in requirements.wrapper_source
    assert "input wire [4:0] s_axilite_ARADDR" in requirements.wrapper_source
    written = write_decomposed_artifact(requirements, tmp_path / "generated")
    assert tuple(Path(path).name for path in written) == staged_layout(requirements)
    assert (tmp_path / "generated" / "memblock.dat").read_text().strip()

    packaged = package_decomposed_artifact(requirements, tmp_path / "packages")
    target = TargetIdentity(requirements.target_fpga_part, requirements.clock_period_ns)
    synthesis = prepare_decomposed_synthesis(packaged, target, tmp_path / "synthesis")
    synthesis_script = Path(synthesis.script_path).read_text()
    assert "memblock.dat" not in synthesis_script
    assert requirements.top_module_name in synthesis_script
    ip_package = prepare_ip_package(packaged, target.fpga_part, tmp_path / "ip")
    assert requirements.top_module_name in Path(ip_package.script_path).read_text()
