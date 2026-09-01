# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared semantic declarations for the ordinary MVAU dot-product design.

This is the single declaration API consumed by the new ``DataflowDesign``
path.  It owns folding, the replay and dot-product Regions, their computation
contracts, the external weight demand, the flat Network, source association,
source admission, and semantic readiness.  Physical Kernels are deliberately
absent.

The legacy semantic-Kernel path remains live during D2.  It calls the same
constructors and admission predicates, which lets tests compare both paths at
the same point before D3 begins consuming these handles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from finn.dataflow.authoring.scope import ConstraintRef, Ref, Scope, divisors_of
from finn.dataflow.design import (
    DATAFLOW_NETWORK_SEMANTICS,
    DATAFLOW_REGION_SEMANTICS,
    NETWORK_VALIDATION_REPORT_SEMANTICS,
    DesignSpaceSpec,
)
from finn.dataflow.hardware import ComputationContract
from finn.dataflow.ops.mvau.associations import (
    CoordinateMappingKind,
    MVAUParameterTopology,
    MVAUSourceAssociation,
    SemanticOperandDestination,
    SourceOperandAssociation,
)
from finn.dataflow.ops.mvau.computation import (
    ACTIVATION_REPLAY_COMPUTATION,
    DOT_PRODUCT_COMPUTATION,
)
from finn.dataflow.ops.mvau.regions import (
    MVAURegionDeclaration,
    construct_activation_replay_region,
    construct_dot_product_region,
    construct_standard_mvau_weight_port,
)
from finn.dataflow.ops.mvau.problem import (
    MVAUComputationProfile,
    MVAUProblem,
    MVAUSourceDescription,
)
from finn.dataflow.network import (
    BoundaryContract,
    DataflowNetwork,
    Edge,
    NetworkNode,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.network_validation import NetworkValidationReport, validate_network
from finn.dataflow.region import DataflowRegion, NumericElementType, Port

DOT_PRODUCT_DESIGN_NAMESPACE = "mvau.design.dot_product"
DOT_PRODUCT_SEMANTIC_READINESS = "mvau.design.dot_product.semantic"

REPLAY_NODE = "replay"
DOT_PRODUCT_NODE = "compute"
ACTIVATION_EDGE = "activation_replay"
WEIGHT_INTERFACE = "weight"
REGION_FORM_EXPORT = "region_form"


def dot_product_computation_supported(profile: MVAUComputationProfile) -> bool:
    """Whether the ordinary dot-product Region expresses this source computation."""

    return profile is MVAUComputationProfile.ACCUMULATOR_INTEGER


def accumulator_output_type_supported(
    accumulator: NumericElementType, output: NumericElementType
) -> bool:
    """The unfused dot product emits its accumulator value directly."""

    return accumulator == output


def construct_decomposed_mvau_network(
    replay_region: DataflowRegion, dot_product_region: DataflowRegion
) -> DataflowNetwork:
    """Assemble activation replay and dot product into one flat Network."""

    produced: Port = replay_region.output_interface("activation_out").port
    return DataflowNetwork(
        (
            NetworkNode(REPLAY_NODE, replay_region),
            NetworkNode(DOT_PRODUCT_NODE, dot_product_region),
        ),
        (
            Edge(
                ACTIVATION_EDGE,
                RegionEndpoint(REPLAY_NODE, "activation_out"),
                (
                    SinkContract(
                        RegionEndpoint(DOT_PRODUCT_NODE, "activation"),
                        PositionMap.identity(produced.beat_sequence.image),
                    ),
                ),
            ),
        ),
        (
            BoundaryContract(
                "activation",
                RegionEndpoint(REPLAY_NODE, "activation_in"),
                replay_region.input_interface("activation_in").port.beat_sequence,
            ),
            BoundaryContract(
                "weight",
                RegionEndpoint(DOT_PRODUCT_NODE, "weight"),
                dot_product_region.input_interface("weight").port.beat_sequence,
            ),
            BoundaryContract(
                "output",
                RegionEndpoint(DOT_PRODUCT_NODE, "output"),
                dot_product_region.output_interface("output").port.beat_sequence,
            ),
        ),
    )


def construct_external_dot_product_source_association(
    description: MVAUSourceDescription,
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
) -> MVAUSourceAssociation:
    """Associate source operands with the external-supply dot-product Network."""

    return MVAUSourceAssociation(
        description.source_node_id,
        description.fused_source_node_ids,
        MVAURegionDeclaration.DOT_PRODUCT_STREAMED.value,
        MVAUParameterTopology.DIRECT,
        (
            SourceOperandAssociation(
                "activation",
                description.activation_operand_id,
                SemanticOperandDestination(REPLAY_NODE, "X"),
                CoordinateMappingKind.FLATTEN_LEADING,
                (*description.leading_shape, matrix_width),
                (repetitions, matrix_width),
            ),
            SourceOperandAssociation(
                "weight",
                description.weight_operand_id,
                SemanticOperandDestination(DOT_PRODUCT_NODE, "W"),
                CoordinateMappingKind.TRANSPOSE_2D,
                (matrix_width, matrix_height),
                (matrix_height, matrix_width),
            ),
            SourceOperandAssociation(
                "output",
                description.output_operand_id,
                SemanticOperandDestination(DOT_PRODUCT_NODE, "Y"),
                CoordinateMappingKind.FLATTEN_LEADING,
                (*description.leading_shape, matrix_height),
                (repetitions, matrix_height),
            ),
        ),
        "dot_product",
    )


@dataclass(frozen=True)
class MVAUSemanticDemand:
    """One external logical input required by the design semantics."""

    name: str
    port: Ref[Port]


@dataclass(frozen=True)
class MVAUSemanticExport:
    """One design-derived value presented to the enclosing Operation."""

    name: str
    value: Ref[object]


@dataclass(frozen=True)
class MVAUDotProductSemantics:
    """Every shared declaration behind ``DotProductDesign``."""

    spec: DesignSpaceSpec
    pe: Ref[int]
    simd: Ref[int]
    replay_region: Ref[DataflowRegion]
    replay_computation: Ref[ComputationContract]
    dot_product_region: Ref[DataflowRegion]
    dot_product_computation: Ref[ComputationContract]
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


def declare_dot_product_semantics(
    problem: MVAUProblem,
    *,
    namespace: str = DOT_PRODUCT_DESIGN_NAMESPACE,
) -> MVAUDotProductSemantics:
    """Declare the complete shared semantics for the ordinary dot-product design."""

    scope = Scope(namespace)
    source_constraint_set = f"{namespace}.source_admission"
    feasibility_constraint_set = f"{namespace}.feasibility"
    pe = scope.decision("pe", int, domain=divisors_of(problem.matrix_height))
    simd = scope.decision("simd", int, domain=divisors_of(problem.matrix_width))

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
    dot_product_region = cast(
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
            },
            evaluate=construct_dot_product_region,
        ),
    )
    dot_product_computation = scope.derived(
        "compute.computation",
        ComputationContract,
        dependencies={},
        evaluate=lambda: DOT_PRODUCT_COMPUTATION,
    )
    replay_region = cast(
        "Ref[DataflowRegion]",
        scope.derived(
            "replay.region",
            DATAFLOW_REGION_SEMANTICS,
            dependencies={
                "repetitions": problem.repetitions,
                "matrix_width": problem.matrix_width,
                "matrix_height": problem.matrix_height,
                "activation_element_type": problem.activation_element_type,
                "pe": pe,
                "simd": simd,
            },
            evaluate=construct_activation_replay_region,
        ),
    )
    replay_computation = scope.derived(
        "replay.computation",
        ComputationContract,
        dependencies={},
        evaluate=lambda: ACTIVATION_REPLAY_COMPUTATION,
    )
    weight_port = scope.derived(
        "compute.weight_port",
        Port,
        dependencies={
            "repetitions": problem.repetitions,
            "matrix_width": problem.matrix_width,
            "matrix_height": problem.matrix_height,
            "weight_element_type": problem.weight_element_type,
            "pe": pe,
            "simd": simd,
        },
        evaluate=construct_standard_mvau_weight_port,
    )
    region_form = scope.derived(
        "compute.region_form",
        MVAURegionDeclaration,
        dependencies={"matrix_width": problem.matrix_width},
        evaluate=lambda matrix_width: MVAURegionDeclaration.DOT_PRODUCT_STREAMED,
    )
    network = cast(
        "Ref[DataflowNetwork]",
        scope.derived(
            "network",
            DATAFLOW_NETWORK_SEMANTICS,
            dependencies={
                "replay_region": replay_region,
                "dot_product_region": dot_product_region,
            },
            evaluate=construct_decomposed_mvau_network,
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
    source_association = scope.derived(
        "core_source_association",
        MVAUSourceAssociation,
        dependencies={
            "description": problem.source_description,
            "repetitions": problem.repetitions,
            "matrix_width": problem.matrix_width,
            "matrix_height": problem.matrix_height,
        },
        evaluate=construct_external_dot_product_source_association,
    )
    source_constraints = scope.constraints_in(source_constraint_set)
    feasibility_constraints = scope.constraints_in(feasibility_constraint_set)
    readiness = f"{namespace}.semantic"
    scope.readiness_profile(
        readiness,
        decisions=(pe, simd),
        properties=(
            replay_region,
            replay_computation,
            dot_product_region,
            dot_product_computation,
            weight_port,
            region_form,
            network,
            network_validation,
            source_association,
        ),
        constraints=(*source_constraints, network_valid),
    )
    return MVAUDotProductSemantics(
        scope.spec(),
        pe,
        simd,
        replay_region,
        replay_computation,
        dot_product_region,
        dot_product_computation,
        weight_port,
        region_form,
        network,
        network_validation,
        source_association,
        source_constraints,
        feasibility_constraints,
        source_constraint_set,
        feasibility_constraint_set,
        (MVAUSemanticDemand(WEIGHT_INTERFACE, weight_port),),
        (MVAUSemanticExport(REGION_FORM_EXPORT, cast("Ref[object]", region_form)),),
        readiness,
    )


__all__ = [
    "ACTIVATION_EDGE",
    "DOT_PRODUCT_DESIGN_NAMESPACE",
    "DOT_PRODUCT_NODE",
    "DOT_PRODUCT_SEMANTIC_READINESS",
    "MVAUDotProductSemantics",
    "MVAUSemanticDemand",
    "MVAUSemanticExport",
    "REPLAY_NODE",
    "accumulator_output_type_supported",
    "construct_decomposed_mvau_network",
    "construct_external_dot_product_source_association",
    "declare_dot_product_semantics",
    "dot_product_computation_supported",
]
