# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The reviewed MVAU ``DataflowDesign`` inventory and operation result."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    selected_design_metadata,
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
    MVAU_NETWORK_REF_SEMANTICS,
    MVAUNetworkRef,
    MVAUSourceAssociation,
)
from finn.dataflow.ops.mvau.designs.batch_interleaved import (
    BatchInterleavedDesign,
    BatchInterleavedDesignInputs,
    MVAUBatchInterleavedSemantics,
    declare_batch_interleaved_semantics,
)
from finn.dataflow.ops.mvau.designs.dot_product import (
    DotProductDesign,
    DotProductDesignInputs,
)
from finn.dataflow.ops.mvau.hardware.dotp_axi import DotpAxiHandles
from finn.dataflow.ops.mvau.input_supply import (
    MVAUInputSupply,
    declare_mvau_input_supply,
    declare_supplied_source_association,
)
from finn.dataflow.ops.mvau.semantics import MVAUDotProductSemantics, declare_dot_product_semantics
from finn.dataflow.ops.mvau.problem import (
    MVAU_EFFECTIVE_NARROW_WEIGHTS,
    MVAU_PROBLEM,
    MVAU_PROBLEM_PROVENANCE,
    MVAU_PROBLEM_SPEC,
    MVAUProblem,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.network_validation import NetworkValidationReport, validate_network

MVAU_NETWORK_PATH = QualifiedPath("semantic.mvau.op.network")
MVAU_NETWORK_VALIDATION_PATH = QualifiedPath("semantic.mvau.op.network_validation")
MVAU_SOURCE_ASSOCIATION_PATH = QualifiedPath("semantic.mvau.op.source_association")
MVAU_RESULT_PATH = QualifiedPath("semantic.mvau.op.result")

MVAU_STRUCTURAL_CONSTRAINT_SET = "mvau_op_structural"
MVAU_FEASIBILITY_CONSTRAINT_SET = "mvau_op_feasibility"
MVAU_STRUCTURAL_READINESS = "mvau_op_structural"
MVAU_ARTIFACT_READINESS = "artifact_inputs"


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
    result: Ref[MVAUNetworkRef]
    compute_pumping: Ref[bool]
    authoring: DataflowOpAuthoring
    specification: DesignSpaceSpec


def declare_mvau_design_inventory(
    problem: MVAUProblem = MVAU_PROBLEM,
    *,
    narrow_weights: Ref[bool] = MVAU_EFFECTIVE_NARROW_WEIGHTS,
) -> MVAUDesignInventoryAssembly:
    """Declare the fresh v6 inventory without importing any legacy Kernel pool."""

    dot_product = declare_dot_product_semantics(problem)
    batch_interleaved = declare_batch_interleaved_semantics(problem)
    supply = declare_mvau_input_supply(problem)
    dot_association, dot_association_spec = declare_supplied_source_association(
        "mvau.design.dot_product",
        dot_product.source_association,
        supply.declaration,
    )
    batch_association, batch_association_spec = declare_supplied_source_association(
        "mvau.design.batch_interleaved",
        batch_interleaved.source_association,
        supply.declaration,
    )
    inventory = declare_dataflow_design_inventory(
        "mvau",
        (
            DataflowDesignEntry(
                DotProductDesign,
                DotProductDesignInputs(problem, dot_product, narrow_weights),
                (dot_product.spec, dot_association_spec),
                (dot_product.pe, dot_product.simd),
                dot_product.feasibility_constraints,
            ),
            DataflowDesignEntry(
                BatchInterleavedDesign,
                BatchInterleavedDesignInputs(batch_interleaved),
                (batch_interleaved.spec, batch_association_spec),
                (
                    batch_interleaved.pe,
                    batch_interleaved.simd,
                    batch_interleaved.interleave,
                ),
                batch_interleaved.feasibility_constraints,
            ),
        ),
        input_supplies=(supply.declaration,),
        shared_specs=(MVAU_PROBLEM_SPEC,),
    )
    if inventory.design_selection is None:
        raise AssertionError("the MVAU inventory must expose its two-design choice")
    design = inventory.design_selection
    dot_declaration = inventory.declaration(DotProductDesign.id)
    batch_declaration = inventory.declaration(BatchInterleavedDesign.id)
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
        supply_mode: str,
        dot_product: object,
        batch_interleaved: object,
        dot_compute: object,
        dot_replay: object,
        dot_delivery: object,
        batch_compute: object,
        batch_delivery: object,
    ) -> object:
        selected = cast(
            MVAUSourceAssociation,
            _selected_value(design, dot_product, batch_interleaved),
        )
        metadata = selected_design_metadata(
            inventory,
            design,
            supply_modes={supply.declaration.source_operand: supply_mode},
            placement_selections={
                (DotProductDesign.id, "compute"): dot_compute,
                (DotProductDesign.id, "replay"): dot_replay,
                (DotProductDesign.id, "delivery"): dot_delivery,
                (BatchInterleavedDesign.id, "compute"): batch_compute,
                (BatchInterleavedDesign.id, "delivery"): batch_delivery,
            },
        )
        return replace(
            selected,
            compute_kernel_id=metadata.kernel_id("compute") or "",
            design_id=metadata.design_id,
            decision_paths=metadata.decision_paths,
            kernel_ids=metadata.kernel_ids,
        )

    source_association = operation.derived(
        "source_association",
        MVAUSourceAssociation,
        dependencies={
            "design": design,
            "supply_mode": supply.declaration.choice,
            "dot_product": dot_association.allow_absent(),
            "batch_interleaved": batch_association.allow_absent(),
            "dot_compute": dot_declaration.placement("compute").selected_kernel.allow_absent(),
            "dot_replay": dot_declaration.placement("replay").selected_kernel.allow_absent(),
            "dot_delivery": dot_declaration.placement("delivery").selected_kernel.allow_absent(),
            "batch_compute": batch_declaration.placement("compute").selected_kernel.allow_absent(),
            "batch_delivery": batch_declaration.placement(
                "delivery"
            ).selected_kernel.allow_absent(),
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
    result = operation.derived(
        "result",
        MVAU_NETWORK_REF_SEMANTICS,
        dependencies={"network": network, "source_association": source_association},
        evaluate=lambda network, source_association: MVAUNetworkRef(
            "mvau",
            cast(DataflowNetwork, network),
            cast(MVAUSourceAssociation, source_association),
        ),
    )

    authoring = declare_dataflow_op_authoring(
        inventory,
        operation,
        result=cast("Ref[object]", result),
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
        dot_association,
        batch_association,
        network,
        network_validation,
        source_association,
        result,
        compute_pumping,
        authoring,
        authoring.specification,
    )


MVAU_DESIGN_INVENTORY = declare_mvau_design_inventory()


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
    "MVAU_FEASIBILITY_CONSTRAINT_SET",
    "MVAU_NETWORK_PATH",
    "MVAU_NETWORK_VALIDATION_PATH",
    "MVAU_RESULT_PATH",
    "MVAU_SOURCE_ASSOCIATION_PATH",
    "MVAU_STRUCTURAL_CONSTRAINT_SET",
    "MVAU_STRUCTURAL_READINESS",
    "MVAUDesignInventoryAssembly",
    "admissible_mvau_designs",
    "declare_mvau_design_inventory",
    "mvau_build_admission",
]
