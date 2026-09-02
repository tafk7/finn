# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Private pre-v7 MVAU inventory retained for migration-equivalence tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from finn.dataflow.authoring.admission import (
    GraphBuildAdmission,
    graph_stage_build_admission,
)
from finn.dataflow.authoring.inventory import (
    DataflowOpAuthoring,
    DataflowDesignEntry,
    DataflowDesignInventory,
    declare_dataflow_op_authoring,
    declare_dataflow_design_inventory,
)
from finn.dataflow.authoring.scope import Ref, Scope, unresolved
from finn.dataflow.design import (
    ABSENT,
    DATAFLOW_NETWORK_SEMANTICS,
    NETWORK_VALIDATION_REPORT_SEMANTICS,
    DesignPoint,
    DesignSpaceSpec,
    Engine,
    QualifiedPath,
)
from finn.dataflow.ops.mvau.associations import (
    MVAUSourceAssociation,
)
from finn.dataflow.ops.mvau.designs.batch_interleaved import (
    BatchInterleavedDesign,
    MVAUBatchInterleavedSemantics,
    _inputs as batch_interleaved_inputs,
    _semantics as batch_interleaved_semantics,
)
from finn.dataflow.ops.mvau.designs.dot_product import (
    DotProductDesign,
    _inputs as dot_product_inputs,
    _semantics as dot_product_semantics,
)
from finn.dataflow.kernels.dotp_axi import DotpAxiHandles
from finn.dataflow.ops.mvau.input_supply import (
    MVAUInputSupply,
    declare_mvau_input_supply,
)
from finn.dataflow.ops.mvau.semantics import MVAUDotProductSemantics
from finn.dataflow.ops.mvau.problem import (
    MVAU_EFFECTIVE_NARROW_WEIGHTS,
    MVAU_PROBLEM,
    MVAU_PROBLEM_PROVENANCE,
    MVAU_PROBLEM_SPEC,
    MVAUProblem,
    MVAUProblemPaths,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.network_validation import NetworkValidationReport, validate_network

MVAU_NETWORK_PATH = QualifiedPath("semantic.mvau.op.network")
MVAU_NETWORK_VALIDATION_PATH = QualifiedPath("semantic.mvau.op.network_validation")
MVAU_SOURCE_ASSOCIATION_PATH = QualifiedPath("semantic.mvau.op.source_association")
MVAU_RESULT_PATH = MVAU_NETWORK_PATH

MVAU_STRUCTURAL_CONSTRAINT_SET = "mvau_op_structural"
MVAU_FEASIBILITY_CONSTRAINT_SET = "mvau_op_feasibility"
MVAU_STRUCTURAL_READINESS = "mvau_op_structural"
MVAU_ARTIFACT_READINESS = "artifact_inputs"


class MVAUDataflowOpPaths:
    """Stable v6 operation paths and projected problem aliases."""

    SOURCE_DESCRIPTION = MVAUProblemPaths.SOURCE_DESCRIPTION
    ACCUMULATOR_TYPE_ANALYSIS_OWNER = MVAUProblemPaths.ACCUMULATOR_TYPE_ANALYSIS_OWNER
    WEIGHT_INITIALIZER_FINGERPRINT = MVAUProblemPaths.WEIGHT_INITIALIZER_FINGERPRINT
    THRESHOLD_INITIALIZER_FINGERPRINT = MVAUProblemPaths.THRESHOLD_INITIALIZER_FINGERPRINT
    EXTERNAL_WEIGHT_SEQUENCE = MVAUProblemPaths.EXTERNAL_WEIGHT_SEQUENCE
    TARGET_FPGA_PART = MVAUProblemPaths.TARGET_FPGA_PART
    TARGET_CLOCK_PERIOD_NS = MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS
    EFFECTIVE_NARROW_WEIGHTS = MVAUProblemPaths.EFFECTIVE_NARROW_WEIGHTS

    DESIGN = QualifiedPath("mvau.design")
    SOURCE_ASSOCIATION = MVAU_SOURCE_ASSOCIATION_PATH
    NETWORK = MVAU_NETWORK_PATH
    NETWORK_VALIDATION = MVAU_NETWORK_VALIDATION_PATH
    RESULT = MVAU_RESULT_PATH
    NETWORK_STRUCTURALLY_WELL_FORMED = QualifiedPath(
        "constraint.mvau.op.network_structurally_well_formed"
    )


def _selected_value(
    design: str,
    dot_product: object,
    batch_interleaved: object,
) -> object:
    selected = {
        DotProductDesign.id: dot_product,
        BatchInterleavedDesign.id: batch_interleaved,
    }[design]
    if selected is ABSENT:
        return unresolved(
            "mvau-selected-design-value-absent",
            f"selected design {design!r} did not produce its operation result",
        )
    return selected


def _validated_network(network: DataflowNetwork) -> NetworkValidationReport:
    return validate_network(network)


def _valid_network(report: NetworkValidationReport) -> bool:
    return not report


@dataclass(frozen=True)
class MVAUDesignInventoryAssembly:
    """The complete v6 operation-level declaration graph."""

    dot_product: MVAUDotProductSemantics
    batch_interleaved: MVAUBatchInterleavedSemantics
    input_supply: MVAUInputSupply
    inventory: DataflowDesignInventory
    dot_product_source_association: Ref[MVAUSourceAssociation]
    batch_interleaved_source_association: Ref[MVAUSourceAssociation]
    network: Ref[DataflowNetwork]
    network_validation: Ref[NetworkValidationReport]
    source_association: Ref[MVAUSourceAssociation]
    result: Ref[DataflowNetwork]
    compute_pumping: Ref[bool]
    authoring: DataflowOpAuthoring
    specification: DesignSpaceSpec


def declare_mvau_design_inventory(
    problem: MVAUProblem = MVAU_PROBLEM,
    *,
    narrow_weights: Ref[bool] = MVAU_EFFECTIVE_NARROW_WEIGHTS,
    problem_spec: DesignSpaceSpec = MVAU_PROBLEM_SPEC,
) -> MVAUDesignInventoryAssembly:
    """Declare the fresh v6 inventory without importing any legacy Kernel pool."""

    supply = declare_mvau_input_supply(problem)
    inventory = declare_dataflow_design_inventory(
        "mvau",
        (
            DataflowDesignEntry(
                DotProductDesign,
                dot_product_inputs(problem, narrow_weights, supply.declaration.choice),
            ),
            DataflowDesignEntry(
                BatchInterleavedDesign,
                batch_interleaved_inputs(problem, supply.declaration.choice),
            ),
        ),
        input_supplies=(supply.declaration,),
        shared_specs=(problem_spec,),
    )
    if inventory.design_selection is None:
        raise AssertionError("the MVAU inventory must expose its two-design choice")
    design = inventory.design_selection
    dot_declaration = inventory.declaration(DotProductDesign.id)
    batch_declaration = inventory.declaration(BatchInterleavedDesign.id)
    dot_product = dot_product_semantics(dot_declaration)
    batch_interleaved = batch_interleaved_semantics(batch_declaration)
    compute = dot_declaration.placement("compute").candidates[0]
    compute_pumping = compute.typed_handles(DotpAxiHandles).compute_pumping

    operation = Scope("mvau.op")
    network = cast(
        "Ref[DataflowNetwork]",
        operation.derived(
            "network",
            DATAFLOW_NETWORK_SEMANTICS,
            dependencies={
                "design": design,
                "dot_product": dot_declaration.network.allow_absent(),
                "batch_interleaved": batch_declaration.network.allow_absent(),
            },
            evaluate=_selected_value,
        ),
    )

    def selected_association(
        design: str,
        dot_product: object,
        batch_interleaved: object,
    ) -> object:
        return cast(
            MVAUSourceAssociation,
            _selected_value(design, dot_product, batch_interleaved),
        )

    source_association = operation.derived(
        "source_association",
        MVAUSourceAssociation,
        dependencies={
            "design": design,
            "dot_product": dot_product.source_association.allow_absent(),
            "batch_interleaved": batch_interleaved.source_association.allow_absent(),
        },
        evaluate=selected_association,
    )
    network_validation = cast(
        "Ref[NetworkValidationReport]",
        operation.derived(
            "network_validation",
            NETWORK_VALIDATION_REPORT_SEMANTICS,
            dependencies={"network": network},
            evaluate=_validated_network,
        ),
    )
    network_valid = operation.constraint(
        "network_structurally_well_formed",
        dependencies={"report": network_validation},
        evaluate=_valid_network,
        sets=(MVAU_STRUCTURAL_CONSTRAINT_SET, MVAU_FEASIBILITY_CONSTRAINT_SET),
    )
    result = network

    authoring = declare_dataflow_op_authoring(
        inventory,
        operation,
        result=result,
        source_association=cast("Ref[object]", source_association),
        structural_properties=(network, network_validation, source_association, result),
        structural_constraints=(network_valid,),
        structural_constraint_set=MVAU_STRUCTURAL_CONSTRAINT_SET,
        feasibility_constraint_set=MVAU_FEASIBILITY_CONSTRAINT_SET,
        structural_readiness_profile=MVAU_STRUCTURAL_READINESS,
        artifact_readiness_profile=MVAU_ARTIFACT_READINESS,
    )
    return MVAUDesignInventoryAssembly(
        dot_product,
        batch_interleaved,
        supply,
        inventory,
        dot_product.source_association,
        batch_interleaved.source_association,
        network,
        network_validation,
        source_association,
        result,
        compute_pumping,
        authoring,
        authoring.specification,
    )


MVAU_DESIGN_INVENTORY = declare_mvau_design_inventory()
MVAU_DATAFLOW_OP_SPEC = MVAU_DESIGN_INVENTORY.specification


def build_mvau_dataflow_op_spec() -> DesignSpaceSpec:
    """Return the reviewed v6 operation specification."""

    return MVAU_DATAFLOW_OP_SPEC


def admissible_mvau_designs(engine: Engine, point: DesignPoint) -> tuple[str, ...]:
    """Return semantically admissible designs without claiming physical coverage."""

    design_path = MVAU_DESIGN_INVENTORY.inventory.design_path
    assert design_path is not None
    candidates: list[str] = []
    for design_id, constraint_set in (
        (DotProductDesign.id, MVAU_DESIGN_INVENTORY.dot_product.source_constraint_set),
        (
            BatchInterleavedDesign.id,
            MVAU_DESIGN_INVENTORY.batch_interleaved.source_constraint_set,
        ),
    ):
        committed = engine.commit_assignments(point, {design_path: design_id}).point
        if engine.evaluate_constraint_set(committed, constraint_set).verdict is True:
            candidates.append(design_id)
    return tuple(candidates)


def mvau_build_admission(engine: Engine, point: DesignPoint) -> GraphBuildAdmission:
    """Return graph-stage build admission for semantically valid MVAU designs."""

    return graph_stage_build_admission(
        engine,
        point,
        MVAU_DESIGN_INVENTORY.inventory,
        MVAU_PROBLEM_PROVENANCE,
        design_ids=admissible_mvau_designs(engine, point),
    )


__all__ = [
    "MVAU_ARTIFACT_READINESS",
    "MVAU_DESIGN_INVENTORY",
    "MVAU_DATAFLOW_OP_SPEC",
    "MVAUDataflowOpPaths",
    "MVAU_FEASIBILITY_CONSTRAINT_SET",
    "MVAU_NETWORK_PATH",
    "MVAU_NETWORK_VALIDATION_PATH",
    "MVAU_RESULT_PATH",
    "MVAU_SOURCE_ASSOCIATION_PATH",
    "MVAU_STRUCTURAL_CONSTRAINT_SET",
    "MVAU_STRUCTURAL_READINESS",
    "MVAUDesignInventoryAssembly",
    "build_mvau_dataflow_op_spec",
    "admissible_mvau_designs",
    "declare_mvau_design_inventory",
    "mvau_build_admission",
]
