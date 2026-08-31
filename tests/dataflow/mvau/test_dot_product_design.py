# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""D3 gate for ``DotProductDesign`` beside the legacy production path."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

import pytest

from finn.dataflow.authoring.design import DesignRealization
from finn.dataflow.design import Decided, DesignPoint, Engine, QualifiedPath, Unresolved
from finn.dataflow.hardware import composed_artifact_identity, kernel_artifact_identity
from finn.dataflow.kernels import NO_KERNEL
from finn.dataflow.mvau.associations import MVAUSourceAssociation
from finn.dataflow.mvau.compute_kernels import (
    DECOMPOSED_MVAU_KERNELS,
    MVAU_COMPUTE_SELECTION,
    MVAU_REPLAY_SELECTION,
)
from finn.dataflow.mvau.decomposed import ActivationReplayKernel, DotProductKernel
from finn.dataflow.mvau.designs.dot_product import (
    MVAU_DOT_PRODUCT_DESIGN,
    compose_dot_product_design,
    decomposed_bindings,
)
from finn.dataflow.mvau.hardware.binding import bind_decomposed, source_roots
from finn.dataflow.mvau.hardware.composition import (
    build_decomposed_artifact_requirements,
    decomposed_top_module_name,
    elaborate_decomposed,
    render_decomposed_wrapper,
    render_stitch_shim,
)
from finn.dataflow.mvau.hardware.dotp_axi import DotpAxiKernel
from finn.dataflow.mvau.hardware.replay_buffer import ReplayBufferKernel
from finn.dataflow.mvau.source import MVAUResolvedDesign, MVAUSourceProjection
from finn.dataflow.mvau_problem import (
    MVAUComputationProfile,
    MVAUDspBlock,
    MVAUProblemPaths,
    MVAUSourceDescription,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.ops.mvau import (
    MVAU_DATAFLOW_OP_SPEC,
    MVAU_WEIGHT_SUPPLY_SELECTION,
    MVAUDataflowOpPaths,
    NetworkRef,
)

INT8 = DataType["INT8"]
INT16 = DataType["INT16"]

GEOMETRIES = (
    (2, 4, 6, 2, 2),
    (1, 8, 8, 4, 4),
    (3, 6, 4, 1, 3),
    (2, 4, 4, 4, 4),
    (2, 4, 6, 2, 4),
    (1, 3, 5, 5, 3),
)


@dataclass(frozen=True)
class SideBySide:
    old_engine: Engine
    old_point: DesignPoint
    old_resolved: MVAUResolvedDesign
    new_engine: Engine
    new_point: DesignPoint
    realization: DesignRealization


def _facts(geometry: tuple[int, int, int, int, int]) -> dict[QualifiedPath, object]:
    repetitions, matrix_width, matrix_height, _pe, _simd = geometry
    return {
        MVAUProblemPaths.REPETITIONS: repetitions,
        MVAUProblemPaths.MATRIX_WIDTH: matrix_width,
        MVAUProblemPaths.MATRIX_HEIGHT: matrix_height,
        MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE: INT8,
        MVAUProblemPaths.WEIGHT_ELEMENT_TYPE: INT8,
        MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE: INT16,
        MVAUProblemPaths.OUTPUT_ELEMENT_TYPE: INT16,
        MVAUProblemPaths.COMPUTATION_PROFILE: MVAUComputationProfile.ACCUMULATOR_INTEGER,
        MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE: True,
        MVAUProblemPaths.RUNTIME_WRITABLE: False,
        MVAUProblemPaths.SOURCE_DESCRIPTION: MVAUSourceDescription(
            "mvau_node",
            "activation_tensor",
            "weight_tensor",
            "output_tensor",
            (repetitions,),
        ),
        MVAUProblemPaths.TARGET_DSP_BLOCK: MVAUDspBlock.DSP58,
        MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS: 5.0,
        MVAUProblemPaths.TARGET_FPGA_PART: "xcvc1902-vsva2197-2MP-e-S",
    }


def _side_by_side(geometry: tuple[int, int, int, int, int], *, pumping: bool = False) -> SideBySide:
    facts = _facts(geometry)
    pe, simd = geometry[3:]
    old_engine = Engine()
    old_point = old_engine.start(old_engine.validate(MVAU_DATAFLOW_OP_SPEC), facts)
    old_point = old_engine.commit_assignments(
        old_point,
        {
            MVAU_COMPUTE_SELECTION.paths.kernel: DotProductKernel.id,
            MVAU_REPLAY_SELECTION.paths.kernel: ActivationReplayKernel.id,
            MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel: NO_KERNEL,
            DECOMPOSED_MVAU_KERNELS.pe.path: pe,
            DECOMPOSED_MVAU_KERNELS.simd.path: simd,
            DECOMPOSED_MVAU_KERNELS.compute_pumping.path: pumping,
        },
    ).point
    old_network_answer = old_engine.query_property(old_point, MVAUDataflowOpPaths.NETWORK)
    old_association_answer = old_engine.query_property(
        old_point, MVAUDataflowOpPaths.SOURCE_ASSOCIATION
    )
    assert isinstance(old_network_answer, Decided)
    assert isinstance(old_association_answer, Decided)
    old_network = cast(DataflowNetwork, old_network_answer.value)
    old_association = cast(MVAUSourceAssociation, old_association_answer.value)
    projection = MVAUSourceProjection(
        cast(MVAUSourceDescription, facts[MVAUProblemPaths.SOURCE_DESCRIPTION]),
        facts,
        {},
    )
    old_resolved = MVAUResolvedDesign(
        old_engine,
        old_point,
        NetworkRef("mvau", old_network, old_association),
        old_association,
        "scope",
        projection,
    )

    assembly = MVAU_DOT_PRODUCT_DESIGN
    new_engine = Engine()
    new_point = new_engine.start(new_engine.validate(assembly.specification), facts)
    new_point = new_engine.commit_assignments(
        new_point,
        {
            assembly.semantics.pe.path: pe,
            assembly.semantics.simd.path: simd,
            assembly.compute_pumping.path: pumping,
        },
    ).point
    realized = assembly.inventory.realize(new_engine, new_point)
    assert isinstance(realized, Decided)
    return SideBySide(
        old_engine,
        old_point,
        old_resolved,
        new_engine,
        new_point,
        realized.value,
    )


def test_dot_product_is_selected_without_resolving_folding() -> None:
    assembly = MVAU_DOT_PRODUCT_DESIGN
    engine = Engine()
    point = engine.start(engine.validate(assembly.specification), _facts(GEOMETRIES[0]))

    selected = assembly.inventory.selected(point)
    assert isinstance(selected, Decided)
    assert selected.value.id == "dot_product"
    assert assembly.inventory.design_path is None
    assert isinstance(engine.query_property(point, assembly.design.network.path), Unresolved)


def test_dot_product_owns_two_one_candidate_placements_without_selection_decisions() -> None:
    assembly = MVAU_DOT_PRODUCT_DESIGN
    assert tuple(item.name for item in assembly.design.placements) == ("compute", "replay")
    assert tuple(placement.candidates[0].id for placement in assembly.design.placements) == (
        DotpAxiKernel.id,
        ReplayBufferKernel.id,
    )
    decisions = {item.path for item in assembly.specification.decisions}
    assert decisions == {
        assembly.semantics.pe.path,
        assembly.semantics.simd.path,
        assembly.compute_pumping.path,
    }
    assert assembly.compute_pumping.path == QualifiedPath(
        "mvau.design.dot_product.compute.dotp_axi.compute_pumping"
    )


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("pumping", (False, True))
def test_dot_product_realization_matches_legacy_kernel_parameters_and_sources(
    geometry: tuple[int, int, int, int, int], pumping: bool
) -> None:
    compared = _side_by_side(geometry, pumping=pumping)
    old = bind_decomposed(compared.old_resolved)
    new = decomposed_bindings(compared.realization)

    assert new.network == old.network
    assert new.compute.kernel_id == old.compute.kernel_id == DotpAxiKernel.id
    assert new.replay.kernel_id == old.replay.kernel_id == ReplayBufferKernel.id
    assert new.compute.parameters == old.compute.parameters
    assert new.replay.parameters == old.replay.parameters
    assert new.compute.kernel.sources == old.compute.kernel.sources
    assert new.replay.kernel.sources == old.replay.kernel.sources
    assert tuple(item.region for item in new.compute.regions) == tuple(
        item.region for item in old.compute.regions
    )
    assert tuple(item.region for item in new.replay.regions) == tuple(
        item.region for item in old.replay.regions
    )


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_dot_product_whole_design_realization_is_exact(
    geometry: tuple[int, int, int, int, int],
) -> None:
    realization = _side_by_side(geometry).realization
    assert {node for binding in realization.bindings for node in binding.node_ids} == {
        "compute",
        "replay",
    }
    assert realization.unabsorbed_edges == ("activation_replay",)
    assert realization.boundaries == ("activation", "output", "weight")


@pytest.mark.parametrize("pumping", (False, True))
def test_dot_product_calls_the_existing_decomposed_composer_without_structural_change(
    pumping: bool,
) -> None:
    compared = _side_by_side(GEOMETRIES[0], pumping=pumping)
    old = elaborate_decomposed(compared.old_resolved)
    new = compose_dot_product_design(compared.old_resolved, compared.realization)

    assert new.semantic_result == old.semantic_result
    assert new.components == old.components
    assert new.numeric_interfaces == old.numeric_interfaces
    assert new.control_interfaces == old.control_interfaces
    assert new.connections == old.connections
    assert new.boundaries == old.boundaries


def test_dot_product_kernel_artifact_identity_matches_legacy() -> None:
    compared = _side_by_side(GEOMETRIES[0], pumping=True)
    old = bind_decomposed(compared.old_resolved)
    new = decomposed_bindings(compared.realization)
    roots = source_roots(Path(__file__).parents[3])

    assert kernel_artifact_identity(new.compute, roots) == kernel_artifact_identity(
        old.compute, roots
    )
    assert kernel_artifact_identity(new.replay, roots) == kernel_artifact_identity(
        old.replay, roots
    )


def test_dot_product_wrapper_and_composed_artifact_identity_match_legacy() -> None:
    compared = _side_by_side(GEOMETRIES[0], pumping=True)
    old_elaboration = elaborate_decomposed(compared.old_resolved)
    old_requirements = build_decomposed_artifact_requirements(
        compared.old_resolved,
        old_elaboration,
        Path(__file__).parents[3],
    )
    bindings = decomposed_bindings(compared.realization)
    roots = source_roots(Path(__file__).parents[3])
    kernels = tuple(kernel_artifact_identity(binding, roots) for binding in bindings.bindings)
    top = decomposed_top_module_name(kernels)
    replay_region = bindings.replay.regions[0].region
    compute_region = bindings.compute.regions[0].region
    wrapper = render_decomposed_wrapper(
        top,
        dict(bindings.replay.parameters),
        dict(bindings.compute.parameters),
        activation_bits=replay_region.input_interface("activation_in").port.logical_beat_bits,
        weight_bits=compute_region.input_interface("weight").port.logical_beat_bits,
        output_bits=compute_region.output_interface("output").port.logical_beat_bits,
    )
    shim = render_stitch_shim(
        top,
        activation_bits=replay_region.input_interface("activation_in").port.logical_beat_bits,
        weight_bits=compute_region.input_interface("weight").port.logical_beat_bits,
        output_bits=compute_region.output_interface("output").port.logical_beat_bits,
    )
    assert wrapper == old_requirements.wrapper_source
    assert shim == old_requirements.stitch_source
    assert composed_artifact_identity(kernels, wrapper, shim) == old_requirements.identity


def test_production_mvau_has_not_imported_or_switched_to_dot_product_design() -> None:
    source = Path(__file__).parents[3] / "src" / "finn" / "dataflow" / "ops" / "mvau.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "finn.dataflow.mvau.designs.dot_product" not in imported
    production_decisions = {item.path for item in MVAU_DATAFLOW_OP_SPEC.decisions}
    assert MVAU_DOT_PRODUCT_DESIGN.semantics.pe.path not in production_decisions
    assert MVAU_DOT_PRODUCT_DESIGN.compute_pumping.path not in production_decisions
