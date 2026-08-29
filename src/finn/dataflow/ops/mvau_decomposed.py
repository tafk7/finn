# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The MVAU operation over the decomposed compute pair.

``MvauDataflowOp`` selects one fused compute Kernel.  This operation selects
two -- an activation replay and a dot product -- and returns the Network they
assemble into.  Its external boundary is the same one the fused form presents,
which is the whole claim of the decomposition.

It stands beside the fused operation rather than replacing it because
``mvu_vvu_axi`` is still the equivalence oracle: the plan anticipated a
coexistence period in which both inventories are declared.  When the fused
compute identities retire this becomes *the* MVAU operation and the
distinction disappears.  Until then, keeping them apart means the migration
cannot destabilise the thing it is being checked against.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from finn.dataflow.authoring import (
    DataflowBuildConfigView,
    DataflowOp,
    NodeAttrCodec,
)
from finn.dataflow.design import (
    DATAFLOW_NETWORK_SEMANTICS,
    DATAFLOW_REGION_SEMANTICS,
    NETWORK_VALIDATION_REPORT_SEMANTICS,
    Answer,
    Constraint,
    ConstraintSet,
    Decided,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignSpaceSpec,
    EvaluatorSpec,
    QualifiedPath,
    ReadinessProfile,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.kernels import SELECTED_KERNEL_SEMANTICS, KernelSelection, SelectedKernel
from finn.dataflow.mvau.decomposed import (
    DOT_PRODUCT_NODE,
    REPLAY_NODE,
    build_decomposed_mvau_pools,
    construct_decomposed_mvau_network,
)
from finn.dataflow.mvau.source import MVAUResolvedDesign
from finn.dataflow.mvau_problem import MVAU_PROBLEM, MVAU_PROBLEM_SPEC, MVAUSourceDescription
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.network_validation import NetworkValidationReport, validate_network
from finn.dataflow.ops.mvau import (
    CoordinateMappingKind,
    MVAUParameterTopology,
    MVAUSourceAssociation,
    NetworkRef,
    SemanticOperandDestination,
    SourceOperandAssociation,
)
from finn.dataflow.ops.mvau_op import MvauDataflowOp
from finn.dataflow.region import DataflowRegion
from finn.dataflow.spec_algebra import assemble_specs

MVAU_DECOMPOSED_OP_FAMILY_VERSION = "mvau-decomposed-op-v1"

#: The two pools, and the folding handles the operation wired between them.
MVAU_DECOMPOSED_POOLS = build_decomposed_mvau_pools()

DOT_PRODUCT_SELECTION: KernelSelection = MVAU_DECOMPOSED_POOLS.dot_product
REPLAY_SELECTION: KernelSelection = MVAU_DECOMPOSED_POOLS.activation_replay

STRUCTURAL_CONSTRAINT_SET = "mvau.decomposed.structural"
FEASIBILITY_CONSTRAINT_SET = "mvau.decomposed.feasibility"

#: The Region declaration the assembled pair presents, for provenance.
DECOMPOSED_REGION_FORM = "dot_product.streamed"


class MVAUDecomposedOpPaths:
    """Paths this operation owns above its two pools."""

    NETWORK = QualifiedPath("semantic.mvau.decomposed.network")
    NETWORK_VALIDATION = QualifiedPath("semantic.mvau.decomposed.network_validation")
    SOURCE_ASSOCIATION = QualifiedPath("semantic.mvau.decomposed.source_association")
    RESULT = QualifiedPath("semantic.mvau.decomposed.result")

    NETWORK_STRUCTURALLY_WELL_FORMED = QualifiedPath(
        "constraint.mvau.decomposed.network_structurally_well_formed"
    )


_REGION = as_object_semantics(DATAFLOW_REGION_SEMANTICS)
_NETWORK = as_object_semantics(DATAFLOW_NETWORK_SEMANTICS)
_NETWORK_REPORT = as_object_semantics(NETWORK_VALIDATION_REPORT_SEMANTICS)
_SOURCE_ASSOCIATION = as_object_semantics(
    ValueSemantics.immutable_nominal(MVAUSourceAssociation, name="MVAUSourceAssociation")
)
_RESULT = as_object_semantics(ValueSemantics.immutable_nominal(NetworkRef, name="NetworkRef"))

_REPLAY_REGION = DependencyRef.property("replay_region", REPLAY_SELECTION.paths.region, _REGION)
_DOT_PRODUCT_REGION = DependencyRef.property(
    "dot_product_region", DOT_PRODUCT_SELECTION.paths.region, _REGION
)
_DOT_PRODUCT_SELECTED = DependencyRef.property(
    "dot_product_kernel", DOT_PRODUCT_SELECTION.paths.selected_kernel, SELECTED_KERNEL_SEMANTICS
)
_REPLAY_SELECTED = DependencyRef.property(
    "replay_kernel", REPLAY_SELECTION.paths.selected_kernel, SELECTED_KERNEL_SEMANTICS
)
_NETWORK_REF = DependencyRef.property("network", MVAUDecomposedOpPaths.NETWORK, _NETWORK)
_REPORT_REF = DependencyRef.property(
    "report", MVAUDecomposedOpPaths.NETWORK_VALIDATION, _NETWORK_REPORT
)
_ASSOCIATION_REF = DependencyRef.property(
    "source_association", MVAUDecomposedOpPaths.SOURCE_ASSOCIATION, _SOURCE_ASSOCIATION
)


def _derive_network(dependencies: DependencyView) -> Answer[object]:
    return Decided(
        construct_decomposed_mvau_network(
            cast(DataflowRegion, dependencies["replay_region"]),
            cast(DataflowRegion, dependencies["dot_product_region"]),
        )
    )


def _derive_network_validation(dependencies: DependencyView) -> Answer[object]:
    return Decided(validate_network(cast(DataflowNetwork, dependencies["network"])))


def _network_is_well_formed(dependencies: DependencyView) -> Answer[bool]:
    return Decided(not cast(NetworkValidationReport, dependencies["report"]))


def _derive_source_association(dependencies: DependencyView) -> Answer[object]:
    """Map the source tensors onto the two nodes that now carry them.

    The activation enters at the replay node; the weight and the result meet at
    the dot product.  That is the only thing the decomposition changes about
    provenance: the same three tensors, two owners instead of one.
    """

    description = cast(MVAUSourceDescription, dependencies["source_description"])
    repetitions = cast(int, dependencies["repetitions"])
    matrix_width = cast(int, dependencies["matrix_width"])
    matrix_height = cast(int, dependencies["matrix_height"])
    return Decided(
        MVAUSourceAssociation(
            description.source_node_id,
            description.fused_source_node_ids,
            DECOMPOSED_REGION_FORM,
            # Weights arrive at the boundary; no supplier is selected in this
            # slice, so the parameter topology is direct.
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
            cast(SelectedKernel, dependencies["dot_product_kernel"]).kernel_id,
            None,
            None,
        )
    )


def _derive_result(dependencies: DependencyView) -> Answer[object]:
    return Decided(
        NetworkRef(
            "mvau.decomposed",
            cast(DataflowNetwork, dependencies["network"]),
            cast(MVAUSourceAssociation, dependencies["source_association"]),
        )
    )


def _op_properties() -> tuple[DerivedProperty, ...]:
    return (
        DerivedProperty(
            MVAUDecomposedOpPaths.NETWORK,
            _NETWORK,
            EvaluatorSpec((_REPLAY_REGION, _DOT_PRODUCT_REGION), _derive_network),
        ),
        DerivedProperty(
            MVAUDecomposedOpPaths.NETWORK_VALIDATION,
            _NETWORK_REPORT,
            EvaluatorSpec((_NETWORK_REF,), _derive_network_validation),
        ),
        DerivedProperty(
            MVAUDecomposedOpPaths.SOURCE_ASSOCIATION,
            _SOURCE_ASSOCIATION,
            EvaluatorSpec(
                (
                    MVAU_PROBLEM.source_description.dependency("source_description"),
                    MVAU_PROBLEM.repetitions.dependency("repetitions"),
                    MVAU_PROBLEM.matrix_width.dependency("matrix_width"),
                    MVAU_PROBLEM.matrix_height.dependency("matrix_height"),
                    _DOT_PRODUCT_SELECTED,
                    _REPLAY_SELECTED,
                ),
                _derive_source_association,
            ),
        ),
        DerivedProperty(
            MVAUDecomposedOpPaths.RESULT,
            _RESULT,
            EvaluatorSpec((_NETWORK_REF, _ASSOCIATION_REF), _derive_result),
        ),
    )


def _op_constraints() -> tuple[Constraint, ...]:
    return (
        Constraint(
            MVAUDecomposedOpPaths.NETWORK_STRUCTURALLY_WELL_FORMED,
            EvaluatorSpec((_REPORT_REF,), _network_is_well_formed),
        ),
    )


_OP_PROPERTIES = (
    MVAUDecomposedOpPaths.NETWORK,
    MVAUDecomposedOpPaths.NETWORK_VALIDATION,
    MVAUDecomposedOpPaths.SOURCE_ASSOCIATION,
    MVAUDecomposedOpPaths.RESULT,
)
_OP_STRUCTURAL = (MVAUDecomposedOpPaths.NETWORK_STRUCTURALLY_WELL_FORMED,)


def build_decomposed_mvau_op_spec() -> DesignSpaceSpec:
    """One flat specification: the operation's problem, both pools, the Network."""

    additions = DesignSpaceSpec(
        properties=_op_properties(),
        constraints=_op_constraints(),
        constraint_sets=(
            ConstraintSet(STRUCTURAL_CONSTRAINT_SET, _OP_STRUCTURAL),
            ConstraintSet(
                FEASIBILITY_CONSTRAINT_SET,
                (
                    *_OP_STRUCTURAL,
                    *DOT_PRODUCT_SELECTION.feasibility_constraints(),
                    *REPLAY_SELECTION.feasibility_constraints(),
                ),
            ),
        ),
        readiness_profiles=(
            ReadinessProfile(
                "mvau_decomposed_structural",
                properties=_OP_PROPERTIES,
                constraints=_OP_STRUCTURAL,
            ),
            ReadinessProfile(
                "mvau_decomposed_artifacts",
                properties=_OP_PROPERTIES,
                constraints=_OP_STRUCTURAL,
            ),
        ),
    )
    return assemble_specs(
        (
            MVAU_PROBLEM_SPEC,
            DOT_PRODUCT_SELECTION.build_spec(),
            REPLAY_SELECTION.build_spec(),
            additions,
        )
    )


MVAU_DECOMPOSED_OP_SPEC = build_decomposed_mvau_op_spec()

MVAU_DECOMPOSED_SELECTIONS: tuple[KernelSelection, ...] = (
    DOT_PRODUCT_SELECTION,
    REPLAY_SELECTION,
)


class DecomposedMvauDataflowOp(MvauDataflowOp):
    """The MVAU as replay plus dot product, returning their Network.

    Graph and build projection, ONNX shape inference, and execution are all
    inherited: the decomposition is a statement about implementation structure,
    not about what the operation computes.
    """

    @classmethod
    def dataflow_family_id(cls) -> str:
        return "finn.dataflow.mvau.decomposed"

    @classmethod
    def dataflow_family_version(cls) -> str:
        return MVAU_DECOMPOSED_OP_FAMILY_VERSION

    @classmethod
    def build_design_space_spec(cls) -> DesignSpaceSpec:
        return MVAU_DECOMPOSED_OP_SPEC

    @classmethod
    def result_path(cls) -> QualifiedPath:
        return MVAUDecomposedOpPaths.RESULT

    @classmethod
    def source_association_path(cls) -> QualifiedPath:
        return MVAUDecomposedOpPaths.SOURCE_ASSOCIATION

    @classmethod
    def kernel_selections(cls) -> tuple[KernelSelection, ...]:
        return MVAU_DECOMPOSED_SELECTIONS

    @classmethod
    def selection_constraint_set(cls) -> str | None:
        return None

    @classmethod
    def structural_readiness_profile(cls) -> str | None:
        return "mvau_decomposed_structural"

    @classmethod
    def artifact_readiness_profile(cls) -> str | None:
        return "mvau_decomposed_artifacts"

    @classmethod
    def feasibility_constraint_sets(cls) -> tuple[str, ...]:
        return (FEASIBILITY_CONSTRAINT_SET,)

    @classmethod
    def decision_nodeattrs(cls) -> Mapping[QualifiedPath, NodeAttrCodec]:
        """What survives a save and reload: the two selections and the folding."""

        pools = MVAU_DECOMPOSED_POOLS
        return {
            DOT_PRODUCT_SELECTION.paths.kernel: NodeAttrCodec.string("dataflow_dot_product_kernel"),
            REPLAY_SELECTION.paths.kernel: NodeAttrCodec.string("dataflow_replay_kernel"),
            pools.pe.path: NodeAttrCodec.integer("dataflow_dot_product_pe"),
            pools.simd.path: NodeAttrCodec.integer("dataflow_dot_product_simd"),
            pools.compute_pumping.path: NodeAttrCodec.boolean("dataflow_dot_product_pumping"),
        }

    def resolve_dataflow(self, config: DataflowBuildConfigView) -> MVAUResolvedDesign:
        """Resolve generically, then present the MVAU-shaped result.

        ``MvauDataflowOp`` overrides resolution to reconstitute the fused
        compute pool's paths.  This operation has no such paths, so the generic
        implementation -- query whatever ``result_path`` and
        ``source_association_path`` name -- is the correct one.  The result is
        still an ``MVAUResolvedDesign``, because every MVAU consumer downstream
        expects that shape and the decomposition does not change it.
        """

        resolved = DataflowOp.resolve_dataflow(self, config)
        network_ref = cast(NetworkRef, resolved.result)
        return MVAUResolvedDesign(
            resolved.engine,
            resolved.point,
            network_ref,
            cast(MVAUSourceAssociation, resolved.source_association),
            resolved.source_scope_id,
            self._graph_projection(),
        )

    def project_build_problem(
        self, config: DataflowBuildConfigView
    ) -> Mapping[QualifiedPath, object]:
        problem = dict(super().project_build_problem(config))
        # The fused operation projects this so a supplier can match it.  This
        # slice covers external weights only and declares no supplier, so
        # nothing here reads it and the field is not part of this space.
        problem.pop(MVAU_PROBLEM.external_weight_sequence.path, None)
        return problem


__all__ = [
    "DECOMPOSED_REGION_FORM",
    "DOT_PRODUCT_SELECTION",
    "FEASIBILITY_CONSTRAINT_SET",
    "MVAU_DECOMPOSED_OP_FAMILY_VERSION",
    "MVAU_DECOMPOSED_OP_SPEC",
    "MVAU_DECOMPOSED_POOLS",
    "MVAU_DECOMPOSED_SELECTIONS",
    "REPLAY_SELECTION",
    "STRUCTURAL_CONSTRAINT_SET",
    "DecomposedMvauDataflowOp",
    "MVAUDecomposedOpPaths",
    "build_decomposed_mvau_op_spec",
]
