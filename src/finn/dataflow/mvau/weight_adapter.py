# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The first explicit MVAU weight-sequence adapter region."""

from __future__ import annotations

from typing import cast

from finn.dataflow.design import (
    DATAFLOW_REGION_SEMANTICS,
    Absent,
    Answer,
    ConstraintSet,
    Decided,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignSpaceSpec,
    EvaluatorSpec,
    Finding,
    FindingKind,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    ReadinessProfile,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.kernel import (
    KernelDefinition,
    RegionDeclaration,
    build_kernel_semantic_declarations,
)
from finn.dataflow.region import (
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    OutputInterface,
    Port,
    RequirementKey,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)


class MVAUWeightAdapterKernelPaths:
    """Stable paths for the first weight-sequence adapter Kernel."""

    SOURCE_PORT = QualifiedPath("problem.mvau.weight_adapter.source_port")
    SINK_PORT = QualifiedPath("problem.mvau.weight_adapter.sink_port")
    FULL_CHUNKED_REGION = QualifiedPath(
        "semantic.mvau.op.weight_adapter.region_declarations.full_chunked"
    )
    REGION = QualifiedPath("semantic.mvau.op.weight_adapter_region")
    REGION_VALIDATION = QualifiedPath("semantic.mvau.op.weight_adapter_region_validation")
    REGION_STRUCTURALLY_WELL_FORMED = QualifiedPath(
        "constraint.mvau.op.weight_adapter_region_structurally_well_formed"
    )


_PORT_SEMANTICS = as_object_semantics(ValueSemantics.immutable_nominal(Port, name="Port"))
_REGION_SEMANTICS = as_object_semantics(DATAFLOW_REGION_SEMANTICS)


def weight_sequence_adapter_applicable(source: Port, sink: Port) -> bool:
    """Return whether the evidenced adapter can preserve one weight pass."""
    return source.operand == sink.operand and (
        source.beat_sequence.image == sink.beat_sequence.image
    )


def construct_weight_sequence_adapter_region(source: Port, sink: Port) -> DataflowRegion:
    """Convert one fixed full-tile/chunked weight sequence into another.

    The schedule has one point per output beat.  Its requirements record every
    output field use at that point; the input boundary remains the independently
    selected producer sequence.  Buffering and physical packing are intentionally
    absent from this semantic declaration.
    """
    if not weight_sequence_adapter_applicable(source, sink):
        raise ValueError("weight adapter requires equal operands and beat-sequence images")
    requirements: dict[RequirementKey, int] = {}
    availability: dict[tuple[int, ...], tuple[int, ...]] = {}
    for ordinal, beat in enumerate(sink.beat_sequence.beats):
        iteration = (ordinal,)
        for position in beat:
            key = (iteration, position)
            requirements[key] = requirements.get(key, 0) + 1
            availability.setdefault(position, iteration)
    return DataflowRegion(
        LogicalSchedule((ScheduleLevel("output_beat", sink.beat_sequence.beat_count),)),
        (
            InputInterface(
                Port("weight_in", source.operand, source.beat_sequence),
                ScheduledInputRequirements(requirements),
            ),
        ),
        (
            OutputInterface(
                Port("weight_out", sink.operand, sink.beat_sequence),
                ScheduledOutputAvailability(availability),
            ),
        ),
    )


def build_mvau_weight_adapter_kernel_spec(
    source_dependency: DependencyRef | None = None,
    sink_dependency: DependencyRef | None = None,
    *,
    problem_fields_required: bool = True,
) -> DesignSpaceSpec:
    """Build the bindingless Kernel for the evidenced full/chunked conversion."""
    fields: list[ProblemField] = []
    if source_dependency is None:
        source_dependency = DependencyRef.problem(
            "source_port", MVAUWeightAdapterKernelPaths.SOURCE_PORT, _PORT_SEMANTICS
        )
        fields.append(
            ProblemField(
                MVAUWeightAdapterKernelPaths.SOURCE_PORT,
                _PORT_SEMANTICS,
                required=problem_fields_required,
            )
        )
    if sink_dependency is None:
        sink_dependency = DependencyRef.problem(
            "sink_port", MVAUWeightAdapterKernelPaths.SINK_PORT, _PORT_SEMANTICS
        )
        fields.append(
            ProblemField(
                MVAUWeightAdapterKernelPaths.SINK_PORT,
                _PORT_SEMANTICS,
                required=problem_fields_required,
            )
        )
    if source_dependency.name != "source_port" or sink_dependency.name != "sink_port":
        raise ValueError("adapter dependencies must be named source_port and sink_port")

    def derive(values: DependencyView) -> Answer[object]:
        source = cast(Port, values["source_port"])
        sink = cast(Port, values["sink_port"])
        if not weight_sequence_adapter_applicable(source, sink):
            return Absent(
                (
                    Finding(
                        FindingKind.REJECTION,
                        "mvau-weight-adapter-not-applicable",
                        MVAUWeightAdapterKernelPaths.FULL_CHUNKED_REGION,
                        "weight adapter requires equal operands and beat-sequence images",
                    ),
                )
            )
        return Decided(construct_weight_sequence_adapter_region(source, sink))

    declaration = RegionDeclaration(
        "full_chunked",
        MVAUWeightAdapterKernelPaths.FULL_CHUNKED_REGION,
    )
    semantic = build_kernel_semantic_declarations(
        (declaration,),
        selected_region_path=MVAUWeightAdapterKernelPaths.REGION,
        validation_report_path=MVAUWeightAdapterKernelPaths.REGION_VALIDATION,
        structural_constraint_path=(MVAUWeightAdapterKernelPaths.REGION_STRUCTURALLY_WELL_FORMED),
    )
    return DesignSpaceSpec(
        ProblemSchema(tuple(fields)),
        properties=(
            DerivedProperty(
                MVAUWeightAdapterKernelPaths.FULL_CHUNKED_REGION,
                _REGION_SEMANTICS,
                EvaluatorSpec((source_dependency, sink_dependency), derive),
            ),
            semantic.selected_region,
            semantic.validation_report,
        ),
        constraints=(semantic.structural_constraint,),
        constraint_sets=(
            ConstraintSet(
                "mvau_weight_adapter_structural",
                (MVAUWeightAdapterKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,),
            ),
        ),
        readiness_profiles=(
            ReadinessProfile(
                "mvau_weight_adapter_structural",
                properties=(
                    MVAUWeightAdapterKernelPaths.REGION,
                    MVAUWeightAdapterKernelPaths.REGION_VALIDATION,
                ),
                constraints=(MVAUWeightAdapterKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,),
            ),
        ),
    )


MVAU_WEIGHT_ADAPTER_KERNEL = KernelDefinition(
    id="mvau.weight_sequence_adapter",
    spec=build_mvau_weight_adapter_kernel_spec(),
    region_declarations=(
        RegionDeclaration("full_chunked", MVAUWeightAdapterKernelPaths.FULL_CHUNKED_REGION),
    ),
    binding_definitions=(),
    selected_region_path=MVAUWeightAdapterKernelPaths.REGION,
    binding_decision_path=None,
    binding_selection_path=None,
    structural_readiness_profile="mvau_weight_adapter_structural",
    binding_readiness_profile=None,
)


__all__ = [
    "MVAU_WEIGHT_ADAPTER_KERNEL",
    "MVAUWeightAdapterKernelPaths",
    "build_mvau_weight_adapter_kernel_spec",
    "construct_weight_sequence_adapter_region",
    "weight_sequence_adapter_applicable",
]
