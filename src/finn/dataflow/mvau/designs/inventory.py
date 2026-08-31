# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The reviewed MVAU ``DataflowDesign`` inventory and operation result."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from finn.dataflow.authoring import assemble_specs
from finn.dataflow.authoring.design import (
    DataflowDesignEntry,
    DataflowDesignInventory,
    declare_dataflow_design_inventory,
)
from finn.dataflow.authoring.scope import ConstraintRef, Ref, Scope, unresolved
from finn.dataflow.design import (
    ABSENT,
    DATAFLOW_NETWORK_SEMANTICS,
    NETWORK_VALIDATION_REPORT_SEMANTICS,
    DependencyKind,
    DesignPoint,
    DesignSpaceSpec,
    Engine,
    QualifiedPath,
)
from finn.dataflow.mvau.associations import MVAUNetworkRef, MVAUSourceAssociation
from finn.dataflow.mvau.designs.batch_interleaved import (
    BatchInterleavedDesign,
    BatchInterleavedDesignInputs,
    MVAUBatchInterleavedSemantics,
    declare_batch_interleaved_semantics,
)
from finn.dataflow.mvau.designs.dot_product import (
    DotProductDesign,
    DotProductDesignInputs,
)
from finn.dataflow.mvau.hardware.dotp_axi import DotpAxiKernel
from finn.dataflow.mvau.hardware.replay_buffer import ReplayBufferKernel
from finn.dataflow.mvau.input_supply import (
    FINN_RTL_MEMSTREAM_SUPPLY,
    MVAUInputSupply,
    declare_mvau_input_supply,
    declare_supplied_source_association,
)
from finn.dataflow.mvau.semantics import MVAUDotProductSemantics, declare_dot_product_semantics
from finn.dataflow.mvau_problem import (
    MVAU_EFFECTIVE_NARROW_WEIGHTS,
    MVAU_PROBLEM,
    MVAU_PROBLEM_SPEC,
    MVAUProblem,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.network_validation import NetworkValidationReport, validate_network
from finn.dataflow.resolution import DATAFLOW_OP_RESULT_SEMANTICS

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
    result: Ref[object]
    compute_pumping: Ref[bool]
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
            ),
            DataflowDesignEntry(
                BatchInterleavedDesign,
                BatchInterleavedDesignInputs(batch_interleaved),
                (batch_interleaved.spec, batch_association_spec),
            ),
        ),
        input_supplies=(supply.declaration,),
        shared_specs=(MVAU_PROBLEM_SPEC,),
    )
    if inventory.design_path is None:
        raise AssertionError("the MVAU inventory must expose its two-design choice")
    design_decision = next(
        item for item in inventory.specification.decisions if item.path == inventory.design_path
    )
    design: Ref[str] = Ref(
        design_decision.path,
        DependencyKind.DECISION,
        design_decision.value_semantics,
    )
    dot_declaration = inventory.declaration(DotProductDesign.id)
    batch_declaration = inventory.declaration(BatchInterleavedDesign.id)
    compute = dot_declaration.placement("compute").candidates[0]
    pumping = tuple(
        item for item in compute.spec.decisions if item.path.value.endswith(".compute_pumping")
    )
    if len(pumping) != 1:
        raise AssertionError("DotpAxiKernel must declare exactly one compute-pumping choice")
    compute_pumping: Ref[bool] = Ref(
        pumping[0].path,
        DependencyKind.DECISION,
        pumping[0].value_semantics,
    )

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
    ) -> object:
        selected = cast(
            MVAUSourceAssociation,
            _selected_value(design, dot_product, batch_interleaved),
        )
        decisions = [design_decision.path, supply.declaration.choice.path]
        kernels: list[str] = []
        if design == DotProductDesign.id:
            decisions.extend((dot_product_semantics.pe.path, dot_product_semantics.simd.path))
            decisions.append(compute_pumping.path)
            kernels.extend((DotpAxiKernel.id, ReplayBufferKernel.id))
            compute_kernel_id = DotpAxiKernel.id
        else:
            decisions.extend(
                (
                    batch_interleaved_semantics.pe.path,
                    batch_interleaved_semantics.simd.path,
                    batch_interleaved_semantics.interleave.path,
                )
            )
            compute_kernel_id = ""
        if supply_mode == FINN_RTL_MEMSTREAM_SUPPLY:
            decisions.extend(
                (
                    supply.settings.ram_style.path,
                    supply.settings.pumped_memory.path,
                )
            )
            kernels.append(FINN_RTL_MEMSTREAM_SUPPLY)
        return replace(
            selected,
            compute_kernel_id=compute_kernel_id,
            design_id=design,
            decision_paths=tuple(decisions),
            kernel_ids=tuple(kernels),
        )

    dot_product_semantics = dot_product
    batch_interleaved_semantics = batch_interleaved
    source_association = operation.derived(
        "source_association",
        MVAUSourceAssociation,
        dependencies={
            "design": design,
            "supply_mode": supply.declaration.choice,
            "dot_product": dot_association.allow_absent(),
            "batch_interleaved": batch_association.allow_absent(),
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
        DATAFLOW_OP_RESULT_SEMANTICS,
        dependencies={"network": network, "source_association": source_association},
        evaluate=lambda network, source_association: MVAUNetworkRef(
            "mvau",
            cast(DataflowNetwork, network),
            cast(MVAUSourceAssociation, source_association),
        ),
    )

    all_design_constraints = tuple(
        ConstraintRef(item.path)
        for declaration in inventory.declarations
        for item in declaration.spec.constraints
    )
    operation.include_in(MVAU_FEASIBILITY_CONSTRAINT_SET, *all_design_constraints)
    structural_decisions = (
        design,
        dot_product.pe,
        dot_product.simd,
        batch_interleaved.pe,
        batch_interleaved.simd,
        batch_interleaved.interleave,
        supply.declaration.choice,
    )
    operation.readiness_profile(
        MVAU_STRUCTURAL_READINESS,
        decisions=structural_decisions,
        properties=(network, network_validation, source_association, result),
        constraints=(network_valid,),
    )
    operation.readiness_profile(
        MVAU_ARTIFACT_READINESS,
        decisions=(
            *structural_decisions,
            compute_pumping,
            supply.settings.ram_style,
            supply.settings.pumped_memory,
        ),
        properties=(network, network_validation, source_association, result),
        constraints=(*all_design_constraints, network_valid),
    )
    specification = assemble_specs((inventory.specification, operation.spec()))
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
        specification,
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
]
