# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Semantic-only batch-interleaved MVAU design as a singleton Network."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import cast

from finn.dataflow.authoring.design import (
    DataflowDesign,
    DataflowDesignDeclaration,
    DataflowDesignEntry,
    DataflowDesignInventory,
    DataflowDesignScope,
    declare_dataflow_design_inventory,
    singleton_network,
)
from finn.dataflow.authoring.scope import (
    ConstraintRef,
    Ref,
    Scope,
    divisors_of,
    domain,
    reject,
    unresolved,
)
from finn.dataflow.design import (
    DATAFLOW_NETWORK_SEMANTICS,
    DATAFLOW_REGION_SEMANTICS,
    NETWORK_VALIDATION_REPORT_SEMANTICS,
    DesignSpaceSpec,
)
from finn.dataflow.hardware import ComputationContract
from finn.dataflow.mvau.associations import (
    CoordinateMappingKind,
    MVAUParameterTopology,
    MVAUSourceAssociation,
    SemanticOperandDestination,
    SourceOperandAssociation,
)
from finn.dataflow.mvau.computation import DOT_PRODUCT_COMPUTATION
from finn.dataflow.mvau.input_supply import (
    MVAUInputSupply,
    declare_mvau_input_supply,
    declare_supplied_source_association,
)
from finn.dataflow.mvau.regions import (
    MVAURegionDeclaration,
    construct_batch_interleaved_streamed_mvau_region,
)
from finn.dataflow.mvau.semantics import (
    MVAUSemanticDemand,
    MVAUSemanticExport,
    accumulator_output_type_supported,
    dot_product_computation_supported,
)
from finn.dataflow.mvau_problem import (
    MVAU_PROBLEM,
    MVAU_PROBLEM_SPEC,
    MVAUProblem,
    MVAUSourceDescription,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.network_validation import NetworkValidationReport, validate_network
from finn.dataflow.region import DataflowRegion, Port

BATCH_INTERLEAVED_DESIGN_NAMESPACE = "mvau.design.batch_interleaved"
BATCH_INTERLEAVED_NODE = "compute"
BATCH_INTERLEAVED_READINESS = "mvau.design.batch_interleaved.semantic"


def _interleave_accepts(candidate: object, repetitions: int, pe: int, simd: int) -> object:
    if type(candidate) is not int:
        return False
    if candidate <= 1:
        return reject(
            "mvau-interleave-not-greater-than-one",
            "batch interleave must be greater than one",
            values={"candidate": candidate},
        )
    if repetitions % candidate:
        return reject(
            "mvau-interleave-does-not-divide-repetitions",
            "batch interleave must divide the repetition extent",
            values={"candidate": candidate, "repetitions": repetitions},
        )
    if (pe * simd) % candidate:
        return reject(
            "mvau-interleave-does-not-divide-weight-tile",
            "batch interleave must divide PE * SIMD",
            values={"candidate": candidate, "pe": pe, "simd": simd},
        )
    return True


def _interleave_candidates(repetitions: int, pe: int, simd: int) -> tuple[int, ...]:
    limit = gcd(repetitions, pe * simd)
    return tuple(value for value in range(2, limit + 1) if limit % value == 0)


def _interleave_available(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
) -> bool:
    return gcd(repetitions, matrix_width * matrix_height) > 1


def _physical_kernel_deferred() -> object:
    return unresolved(
        "mvau-batch-interleaved-kernel-deferred",
        "BatchInterleavedDesign is semantic-only until the TiledMvuKernel vertical slice",
    )


def _weight_port(region: DataflowRegion) -> Port:
    return region.input_interface("weight").port


def construct_external_batch_interleaved_source_association(
    description: MVAUSourceDescription,
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
) -> MVAUSourceAssociation:
    """Associate source tensors with the canonical singleton Network."""

    return MVAUSourceAssociation(
        description.source_node_id,
        description.fused_source_node_ids,
        MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED.value,
        MVAUParameterTopology.DIRECT,
        (
            SourceOperandAssociation(
                "activation",
                description.activation_operand_id,
                SemanticOperandDestination(BATCH_INTERLEAVED_NODE, "X"),
                CoordinateMappingKind.FLATTEN_LEADING,
                (*description.leading_shape, matrix_width),
                (repetitions, matrix_width),
            ),
            SourceOperandAssociation(
                "weight",
                description.weight_operand_id,
                SemanticOperandDestination(BATCH_INTERLEAVED_NODE, "W"),
                CoordinateMappingKind.TRANSPOSE_2D,
                (matrix_width, matrix_height),
                (matrix_height, matrix_width),
            ),
            SourceOperandAssociation(
                "output",
                description.output_operand_id,
                SemanticOperandDestination(BATCH_INTERLEAVED_NODE, "Y"),
                CoordinateMappingKind.FLATTEN_LEADING,
                (*description.leading_shape, matrix_height),
                (repetitions, matrix_height),
            ),
        ),
    )


@dataclass(frozen=True)
class MVAUBatchInterleavedSemantics:
    """All semantic declarations owned by ``BatchInterleavedDesign``."""

    spec: DesignSpaceSpec
    pe: Ref[int]
    simd: Ref[int]
    interleave: Ref[int]
    region: Ref[DataflowRegion]
    computation: Ref[ComputationContract]
    weight_port: Ref[Port]
    region_form: Ref[MVAURegionDeclaration]
    network: Ref[DataflowNetwork]
    network_validation: Ref[NetworkValidationReport]
    source_association: Ref[MVAUSourceAssociation]
    source_constraints: tuple[ConstraintRef, ...]
    feasibility_constraints: tuple[ConstraintRef, ...]
    source_constraint_set: str
    feasibility_constraint_set: str
    demands: tuple[MVAUSemanticDemand, ...]
    exports: tuple[MVAUSemanticExport, ...]
    readiness_profile: str


def declare_batch_interleaved_semantics(
    problem: MVAUProblem,
    *,
    namespace: str = BATCH_INTERLEAVED_DESIGN_NAMESPACE,
) -> MVAUBatchInterleavedSemantics:
    """Declare the retained batch-interleaved Region and its singleton Network."""

    scope = Scope(namespace)
    source_constraint_set = f"{namespace}.source_admission"
    feasibility_constraint_set = f"{namespace}.feasibility"
    pe = scope.decision("pe", int, domain=divisors_of(problem.matrix_height))
    simd = scope.decision("simd", int, domain=divisors_of(problem.matrix_width))
    interleave = scope.decision(
        "interleave",
        int,
        domain=domain(
            {"repetitions": problem.repetitions, "pe": pe, "simd": simd},
            accepts=_interleave_accepts,
            candidates=_interleave_candidates,
        ),
    )
    scope.constraint(
        "computation_supported",
        dependencies={"profile": problem.computation_profile},
        evaluate=dot_product_computation_supported,
        sets=(source_constraint_set, feasibility_constraint_set),
    )
    scope.constraint(
        "accumulator_output_type_supported",
        dependencies={
            "accumulator": problem.accumulator_element_type,
            "output": problem.output_element_type,
        },
        evaluate=accumulator_output_type_supported,
        sets=(source_constraint_set, feasibility_constraint_set),
    )
    scope.constraint(
        "interleave_available",
        dependencies={
            "repetitions": problem.repetitions,
            "matrix_width": problem.matrix_width,
            "matrix_height": problem.matrix_height,
        },
        evaluate=_interleave_available,
        sets=(source_constraint_set, feasibility_constraint_set),
    )
    region = cast(
        "Ref[DataflowRegion]",
        scope.derived(
            "compute.region",
            DATAFLOW_REGION_SEMANTICS,
            dependencies={
                "repetitions": problem.repetitions,
                "matrix_width": problem.matrix_width,
                "matrix_height": problem.matrix_height,
                "activation_element_type": problem.activation_element_type,
                "weight_element_type": problem.weight_element_type,
                "output_element_type": problem.output_element_type,
                "pe": pe,
                "simd": simd,
                "interleave": interleave,
            },
            evaluate=construct_batch_interleaved_streamed_mvau_region,
        ),
    )
    computation = scope.derived(
        "compute.computation",
        ComputationContract,
        dependencies={},
        evaluate=lambda: DOT_PRODUCT_COMPUTATION,
    )
    weight_port = scope.derived(
        "compute.weight_port",
        Port,
        dependencies={"region": region},
        evaluate=_weight_port,
    )
    region_form = scope.derived(
        "compute.region_form",
        MVAURegionDeclaration,
        dependencies={},
        evaluate=lambda: MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED,
    )
    network = cast(
        "Ref[DataflowNetwork]",
        scope.derived(
            "network",
            DATAFLOW_NETWORK_SEMANTICS,
            dependencies={"region": region},
            evaluate=lambda region: singleton_network(
                BATCH_INTERLEAVED_NODE, cast(DataflowRegion, region)
            ),
        ),
    )
    network_validation = cast(
        "Ref[NetworkValidationReport]",
        scope.derived(
            "network_validation",
            NETWORK_VALIDATION_REPORT_SEMANTICS,
            dependencies={"network": network},
            evaluate=lambda network: validate_network(cast(DataflowNetwork, network)),
        ),
    )
    network_valid = scope.constraint(
        "network_structurally_well_formed",
        dependencies={"report": network_validation},
        evaluate=lambda report: not cast(NetworkValidationReport, report),
        sets=(feasibility_constraint_set,),
    )
    scope.constraint(
        "physical_kernel_deferred",
        dependencies={},
        evaluate=_physical_kernel_deferred,
        sets=(feasibility_constraint_set,),
    )
    source_association = scope.derived(
        "core_source_association",
        MVAUSourceAssociation,
        dependencies={
            "description": problem.source_description,
            "repetitions": problem.repetitions,
            "matrix_width": problem.matrix_width,
            "matrix_height": problem.matrix_height,
        },
        evaluate=construct_external_batch_interleaved_source_association,
    )
    source_constraints = scope.constraints_in(source_constraint_set)
    feasibility_constraints = scope.constraints_in(feasibility_constraint_set)
    scope.readiness_profile(
        BATCH_INTERLEAVED_READINESS,
        decisions=(pe, simd, interleave),
        properties=(
            region,
            computation,
            weight_port,
            region_form,
            network,
            network_validation,
            source_association,
        ),
        constraints=(*source_constraints, network_valid),
    )
    return MVAUBatchInterleavedSemantics(
        scope.spec(),
        pe,
        simd,
        interleave,
        region,
        computation,
        weight_port,
        region_form,
        network,
        network_validation,
        source_association,
        source_constraints,
        feasibility_constraints,
        source_constraint_set,
        feasibility_constraint_set,
        (MVAUSemanticDemand("weight", weight_port),),
        (MVAUSemanticExport("region_form", cast("Ref[object]", region_form)),),
        BATCH_INTERLEAVED_READINESS,
    )


@dataclass(frozen=True)
class BatchInterleavedDesignInputs:
    semantics: MVAUBatchInterleavedSemantics


class BatchInterleavedDesign(DataflowDesign):
    """A genuine logical design whose physical vertical slice is deferred."""

    id = "batch_interleaved"
    version = "1"

    @classmethod
    def define(cls, design: DataflowDesignScope[BatchInterleavedDesignInputs]) -> None:
        semantics = design.inputs.semantics
        compute = design.node(
            "compute",
            node_id=BATCH_INTERLEAVED_NODE,
            region=semantics.region,
            computation=semantics.computation,
        )
        design.use_network(semantics.network)
        weight_interface = design.input_interface("compute.weight_interface", compute, "weight")
        design.map_input("weight", boundary_id="input.weight", consumer=weight_interface)
        # D12 deliberately keeps this design semantic-only.  TiledMvuKernel is
        # a separately reviewed future vertical slice, not a hidden fallback.
        design.kernels("compute", covers=(compute,), candidates=())


@dataclass(frozen=True)
class BatchInterleavedDesignAssembly:
    semantics: MVAUBatchInterleavedSemantics
    input_supply: MVAUInputSupply
    inventory: DataflowDesignInventory
    design: DataflowDesignDeclaration
    source_association: Ref[MVAUSourceAssociation]

    @property
    def specification(self) -> DesignSpaceSpec:
        return self.inventory.specification


def declare_batch_interleaved_design(
    problem: MVAUProblem = MVAU_PROBLEM,
) -> BatchInterleavedDesignAssembly:
    """Declare the normalized singleton design with the common MVAU supply policy."""

    semantics = declare_batch_interleaved_semantics(problem)
    supply = declare_mvau_input_supply(problem)
    source_association, association_spec = declare_supplied_source_association(
        BATCH_INTERLEAVED_DESIGN_NAMESPACE,
        semantics.source_association,
        supply.declaration,
    )
    inventory = declare_dataflow_design_inventory(
        "mvau",
        (
            DataflowDesignEntry(
                BatchInterleavedDesign,
                BatchInterleavedDesignInputs(semantics),
                (semantics.spec, association_spec),
                (semantics.pe, semantics.simd, semantics.interleave),
                semantics.feasibility_constraints,
            ),
        ),
        input_supplies=(supply.declaration,),
        shared_specs=(MVAU_PROBLEM_SPEC,),
    )
    return BatchInterleavedDesignAssembly(
        semantics,
        supply,
        inventory,
        inventory.declarations[0],
        source_association,
    )


MVAU_BATCH_INTERLEAVED_DESIGN = declare_batch_interleaved_design()


__all__ = [
    "BATCH_INTERLEAVED_DESIGN_NAMESPACE",
    "BATCH_INTERLEAVED_NODE",
    "BATCH_INTERLEAVED_READINESS",
    "BatchInterleavedDesign",
    "BatchInterleavedDesignAssembly",
    "BatchInterleavedDesignInputs",
    "MVAUBatchInterleavedSemantics",
    "MVAU_BATCH_INTERLEAVED_DESIGN",
    "construct_external_batch_interleaved_source_association",
    "declare_batch_interleaved_design",
    "declare_batch_interleaved_semantics",
]
