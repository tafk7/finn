# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""D2 parity gate for the shared MVAU dot-product semantic declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

import pytest

from finn.dataflow.authoring import assemble_specs
from finn.dataflow.design import Decided, DependencyKind, DesignPoint, Engine, QualifiedPath
from finn.dataflow.datatypes import is_qonnx_datatype
from finn.dataflow.kernels import NO_KERNEL
from finn.dataflow.mvau.associations import MVAUSourceAssociation
from finn.dataflow.mvau.compute_kernels import (
    DECOMPOSED_MVAU_KERNELS,
    MVAU_COMPUTE_SELECTION,
    MVAU_REPLAY_SELECTION,
)
from finn.dataflow.mvau.decomposed import ActivationReplayKernel, DotProductKernel
from finn.dataflow.mvau.regions import MVAURegionDeclaration
from finn.dataflow.mvau.semantics import (
    DOT_PRODUCT_DESIGN_NAMESPACE,
    MVAUDotProductSemantics,
    declare_dot_product_semantics,
)
from finn.dataflow.mvau_problem import (
    MVAU_PROBLEM,
    MVAU_PROBLEM_SPEC,
    MVAUComputationProfile,
    MVAUProblemPaths,
    MVAUSourceDescription,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.mvau.compat.operation import (
    MVAU_LEGACY_DATAFLOW_OP_SPEC as MVAU_DATAFLOW_OP_SPEC,
    MVAU_WEIGHT_SUPPLY_SELECTION,
    MVAUDataflowOpPaths,
)
from finn.dataflow.region import DataflowRegion, Port

INT8 = DataType["INT8"]
INT16 = DataType["INT16"]
FLOAT32 = DataType["FLOAT32"]

GEOMETRIES = (
    (2, 4, 6, 2, 2),
    (1, 8, 8, 4, 4),
    (3, 6, 4, 1, 3),
    (2, 4, 4, 4, 4),
    (2, 4, 6, 2, 4),
    (1, 3, 5, 5, 3),
)


@dataclass(frozen=True)
class ComparedPoints:
    semantics: MVAUDotProductSemantics
    old_engine: Engine
    old_point: DesignPoint
    new_engine: Engine
    new_point: DesignPoint


def _problem(
    geometry: tuple[int, int, int, int, int],
    *,
    profile: object = MVAUComputationProfile.ACCUMULATOR_INTEGER,
    activation_type: object = INT8,
    weight_type: object = INT8,
    accumulator_type: object = INT16,
    output_type: object = INT16,
) -> dict[QualifiedPath, object]:
    repetitions, matrix_width, matrix_height, _pe, _simd = geometry
    return {
        MVAUProblemPaths.REPETITIONS: repetitions,
        MVAUProblemPaths.MATRIX_WIDTH: matrix_width,
        MVAUProblemPaths.MATRIX_HEIGHT: matrix_height,
        MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE: activation_type,
        MVAUProblemPaths.WEIGHT_ELEMENT_TYPE: weight_type,
        MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE: accumulator_type,
        MVAUProblemPaths.OUTPUT_ELEMENT_TYPE: output_type,
        MVAUProblemPaths.COMPUTATION_PROFILE: profile,
        MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE: True,
        MVAUProblemPaths.RUNTIME_WRITABLE: False,
        MVAUProblemPaths.SOURCE_DESCRIPTION: MVAUSourceDescription(
            "mvau_node",
            "activation_tensor",
            "weight_tensor",
            "output_tensor",
            (repetitions,),
        ),
    }


def _compared(
    geometry: tuple[int, int, int, int, int],
    **problem_overrides: object,
) -> ComparedPoints:
    repetitions, matrix_width, matrix_height, pe, simd = geometry
    del repetitions, matrix_width, matrix_height
    facts = _problem(geometry, **problem_overrides)

    old_engine = Engine()
    old_space = old_engine.validate(MVAU_DATAFLOW_OP_SPEC)
    old_point = old_engine.start(old_space, facts)
    old_point = old_engine.commit_assignments(
        old_point,
        {
            MVAU_COMPUTE_SELECTION.paths.kernel: DotProductKernel.id,
            MVAU_REPLAY_SELECTION.paths.kernel: ActivationReplayKernel.id,
            MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel: NO_KERNEL,
            DECOMPOSED_MVAU_KERNELS.pe.path: pe,
            DECOMPOSED_MVAU_KERNELS.simd.path: simd,
        },
    ).point

    semantics = declare_dot_product_semantics(MVAU_PROBLEM)
    new_engine = Engine()
    new_space = new_engine.validate(assemble_specs((MVAU_PROBLEM_SPEC, semantics.spec)))
    new_point = new_engine.start(new_space, facts)
    new_point = new_engine.commit_assignments(
        new_point,
        {semantics.pe.path: pe, semantics.simd.path: simd},
    ).point
    return ComparedPoints(semantics, old_engine, old_point, new_engine, new_point)


def _value(engine: Engine, point: DesignPoint, path: QualifiedPath) -> object:
    answer = engine.query_property(point, path)
    assert isinstance(answer, Decided)
    return answer.value


def test_shared_semantics_uses_the_frozen_dot_product_decision_paths() -> None:
    semantics = declare_dot_product_semantics(MVAU_PROBLEM)
    assert semantics.pe.path == QualifiedPath("mvau.design.dot_product.pe")
    assert semantics.simd.path == QualifiedPath("mvau.design.dot_product.simd")
    assert semantics.pe.kind is DependencyKind.DECISION
    assert semantics.simd.kind is DependencyKind.DECISION
    assert tuple(item.path for item in semantics.spec.decisions) == (
        semantics.pe.path,
        semantics.simd.path,
    )


def test_shared_semantics_declares_each_region_and_computation_once() -> None:
    semantics = declare_dot_product_semantics(MVAU_PROBLEM)
    assert semantics.replay_region.path == QualifiedPath(
        "semantic.mvau.design.dot_product.replay.region"
    )
    assert semantics.dot_product_region.path == QualifiedPath(
        "semantic.mvau.design.dot_product.compute.region"
    )
    assert semantics.network.path == QualifiedPath("semantic.mvau.design.dot_product.network")
    paths = tuple(item.path for item in semantics.spec.properties)
    for handle in (
        semantics.replay_region,
        semantics.replay_computation,
        semantics.dot_product_region,
        semantics.dot_product_computation,
        semantics.network,
    ):
        assert paths.count(handle.path) == 1


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_shared_regions_network_demands_exports_and_association_equal_legacy(
    geometry: tuple[int, int, int, int, int],
) -> None:
    compared = _compared(geometry)
    semantics = compared.semantics

    old_replay = _value(
        compared.old_engine, compared.old_point, DECOMPOSED_MVAU_KERNELS.replay_region.path
    )
    old_compute = _value(
        compared.old_engine,
        compared.old_point,
        DECOMPOSED_MVAU_KERNELS.dot_product_region.path,
    )
    new_replay = _value(compared.new_engine, compared.new_point, semantics.replay_region.path)
    new_compute = _value(compared.new_engine, compared.new_point, semantics.dot_product_region.path)
    assert cast(DataflowRegion, new_replay) == cast(DataflowRegion, old_replay)
    assert cast(DataflowRegion, new_compute) == cast(DataflowRegion, old_compute)

    old_network = _value(compared.old_engine, compared.old_point, MVAUDataflowOpPaths.NETWORK)
    new_network = _value(compared.new_engine, compared.new_point, semantics.network.path)
    assert cast(DataflowNetwork, new_network) == cast(DataflowNetwork, old_network)

    old_weight = _value(
        compared.old_engine,
        compared.old_point,
        MVAU_COMPUTE_SELECTION.paths.demand("weight"),
    )
    new_weight = _value(compared.new_engine, compared.new_point, semantics.weight_port.path)
    assert cast(Port, new_weight) == cast(Port, old_weight)
    assert semantics.demands[0].name == "weight"

    old_form = _value(
        compared.old_engine,
        compared.old_point,
        MVAU_COMPUTE_SELECTION.paths.export("region_form"),
    )
    new_form = _value(compared.new_engine, compared.new_point, semantics.region_form.path)
    assert new_form is old_form is MVAURegionDeclaration.DOT_PRODUCT_STREAMED
    assert semantics.exports[0].name == "region_form"

    old_association = _value(
        compared.old_engine,
        compared.old_point,
        MVAUDataflowOpPaths.SOURCE_ASSOCIATION,
    )
    new_association = _value(
        compared.new_engine,
        compared.new_point,
        semantics.source_association.path,
    )
    assert cast(MVAUSourceAssociation, new_association) == cast(
        MVAUSourceAssociation, old_association
    )
    assert (
        compared.old_engine.evaluate_constraints(
            compared.old_point,
            DECOMPOSED_MVAU_KERNELS.dot_product.feasibility_constraints,
        ).verdict
        is True
    )
    assert (
        compared.new_engine.evaluate_constraint_set(
            compared.new_point, semantics.feasibility_constraint_set
        ).verdict
        is True
    )


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_shared_computation_contracts_and_qonnx_identity_equal_legacy(
    geometry: tuple[int, int, int, int, int],
) -> None:
    compared = _compared(geometry)
    semantics = compared.semantics
    for old_path, new_path in (
        (
            DECOMPOSED_MVAU_KERNELS.replay_computation.path,
            semantics.replay_computation.path,
        ),
        (
            DECOMPOSED_MVAU_KERNELS.dot_product_computation.path,
            semantics.dot_product_computation.path,
        ),
    ):
        assert _value(compared.old_engine, compared.old_point, old_path) == _value(
            compared.new_engine, compared.new_point, new_path
        )
    network = cast(
        DataflowNetwork,
        _value(compared.new_engine, compared.new_point, semantics.network.path),
    )
    element_types = {
        interface.port.operand.element_type
        for node in network.nodes
        for interface in node.region.interfaces
    }
    assert INT8 in element_types
    assert all(is_qonnx_datatype(item) for item in element_types)
    assert all(type(item) is type(DataType[item.name]) for item in element_types)


def test_shared_and_legacy_folding_domains_are_identical() -> None:
    geometry = GEOMETRIES[0]
    facts = _problem(geometry)
    semantics = declare_dot_product_semantics(MVAU_PROBLEM)

    old_engine = Engine()
    old_point = old_engine.start(old_engine.validate(MVAU_DATAFLOW_OP_SPEC), facts)
    old_point = old_engine.commit_assignments(
        old_point,
        {
            MVAU_COMPUTE_SELECTION.paths.kernel: DotProductKernel.id,
            MVAU_REPLAY_SELECTION.paths.kernel: ActivationReplayKernel.id,
            MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel: NO_KERNEL,
        },
    ).point
    new_engine = Engine()
    new_point = new_engine.start(
        new_engine.validate(assemble_specs((MVAU_PROBLEM_SPEC, semantics.spec))), facts
    )

    assert old_engine.enumerate_candidates(
        old_point, DECOMPOSED_MVAU_KERNELS.pe.path
    ) == new_engine.enumerate_candidates(new_point, semantics.pe.path)
    assert old_engine.enumerate_candidates(
        old_point, DECOMPOSED_MVAU_KERNELS.simd.path
    ) == new_engine.enumerate_candidates(new_point, semantics.simd.path)


@pytest.mark.parametrize(
    "problem_overrides",
    (
        {"profile": MVAUComputationProfile.FUSED_THRESHOLD},
        {"output_type": INT8},
        {"activation_type": FLOAT32},
    ),
)
def test_shared_source_admission_matches_legacy(
    problem_overrides: dict[str, object],
) -> None:
    compared = _compared(GEOMETRIES[0], **problem_overrides)
    old_answers = compared.old_engine.evaluate_constraints(
        compared.old_point,
        DECOMPOSED_MVAU_KERNELS.dot_product.source_admission_constraints,
    ).answers
    new_answers = compared.new_engine.evaluate_constraints(
        compared.new_point,
        tuple(item.path for item in compared.semantics.source_constraints),
    ).answers

    assert tuple(
        answer.value for answer in old_answers.values() if isinstance(answer, Decided)
    ) == tuple(answer.value for answer in new_answers.values() if isinstance(answer, Decided))
    assert (
        compared.old_engine.evaluate_constraints(
            compared.old_point,
            DECOMPOSED_MVAU_KERNELS.dot_product.feasibility_constraints,
        ).verdict
        == compared.new_engine.evaluate_constraint_set(
            compared.new_point, compared.semantics.feasibility_constraint_set
        ).verdict
    )


def test_shared_semantic_readiness_matches_the_legacy_partial_state() -> None:
    geometry = GEOMETRIES[0]
    facts = _problem(geometry)
    semantics = declare_dot_product_semantics(MVAU_PROBLEM)

    old_engine = Engine()
    old_point = old_engine.start(old_engine.validate(MVAU_DATAFLOW_OP_SPEC), facts)
    old_point = old_engine.commit_assignments(
        old_point,
        {
            MVAU_COMPUTE_SELECTION.paths.kernel: DotProductKernel.id,
            MVAU_REPLAY_SELECTION.paths.kernel: ActivationReplayKernel.id,
            MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel: NO_KERNEL,
        },
    ).point
    new_engine = Engine()
    new_point = new_engine.start(
        new_engine.validate(assemble_specs((MVAU_PROBLEM_SPEC, semantics.spec))), facts
    )
    assert old_engine.check_readiness(old_point, "mvau_op_structural").ready is None
    assert new_engine.check_readiness(new_point, semantics.readiness_profile).ready is None

    pe, simd = geometry[3:]
    old_point = old_engine.commit_assignments(
        old_point,
        {DECOMPOSED_MVAU_KERNELS.pe.path: pe, DECOMPOSED_MVAU_KERNELS.simd.path: simd},
    ).point
    new_point = new_engine.commit_assignments(
        new_point, {semantics.pe.path: pe, semantics.simd.path: simd}
    ).point
    assert old_engine.check_readiness(old_point, "mvau_op_structural").ready is True
    assert new_engine.check_readiness(new_point, semantics.readiness_profile).ready is True


def test_new_semantics_remains_side_by_side_and_does_not_switch_production() -> None:
    production_decisions = {item.path for item in MVAU_DATAFLOW_OP_SPEC.decisions}
    assert QualifiedPath(f"{DOT_PRODUCT_DESIGN_NAMESPACE}.pe") not in production_decisions
    assert QualifiedPath(f"{DOT_PRODUCT_DESIGN_NAMESPACE}.simd") not in production_decisions
    assert DECOMPOSED_MVAU_KERNELS.pe.path in production_decisions
    assert DECOMPOSED_MVAU_KERNELS.simd.path in production_decisions
