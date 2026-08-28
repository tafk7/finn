# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The MVAU weight-sequence adapter expressed as an ordinary Kernel.

An adapter owns a complete logical schedule that turns one boundary sequence
into another, so it is a Kernel like any other.  It is never implied by a width
difference: assembly first tests exact endpoint compatibility and only then
considers this declared alternative.
"""

from __future__ import annotations

from typing import cast

from finn.dataflow.design import (
    DATAFLOW_REGION_SEMANTICS,
    Absent,
    Answer,
    Constraint,
    Decided,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignSpaceSpec,
    EvaluatorSpec,
    Finding,
    FindingKind,
    QualifiedPath,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.kernels import Kernel, KernelSelection
from finn.dataflow.mvau.weight_adapter import (
    construct_weight_sequence_adapter_region,
    weight_sequence_adapter_applicable,
)
from finn.dataflow.region import Port

#: Selection name and path root of the optional weight-adapter pool.
MVAU_WEIGHT_ADAPTER_SELECTION_NAME = "mvau.weight_adapter"

#: The single declared adapter alternative.
FULL_TILE_TO_CHUNKED = "full_tile_to_chunked"

_PORT = as_object_semantics(ValueSemantics.immutable_nominal(Port, name="Port"))
_REGION = as_object_semantics(DATAFLOW_REGION_SEMANTICS)

_REGION_PATH = QualifiedPath(
    f"semantic.{MVAU_WEIGHT_ADAPTER_SELECTION_NAME}.{FULL_TILE_TO_CHUNKED}.region"
)
_APPLICABLE_PATH = QualifiedPath(
    f"constraint.{MVAU_WEIGHT_ADAPTER_SELECTION_NAME}.{FULL_TILE_TO_CHUNKED}.applicable"
)


def build_mvau_weight_adapter_kernel(source_ref: DependencyRef, sink_ref: DependencyRef) -> Kernel:
    """Build the adapter Kernel over one supplier output and one compute demand."""

    if source_ref.name != "source_port" or sink_ref.name != "sink_port":
        raise ValueError("adapter dependencies must be named source_port and sink_port")

    def derive(dependencies: DependencyView) -> Answer[object]:
        source = cast(Port, dependencies["source_port"])
        sink = cast(Port, dependencies["sink_port"])
        if not weight_sequence_adapter_applicable(source, sink):
            return Absent(
                (
                    Finding(
                        FindingKind.REJECTION,
                        "mvau-weight-adapter-not-applicable",
                        _REGION_PATH,
                        "weight adapter requires equal operands and beat-sequence images",
                    ),
                )
            )
        return Decided(construct_weight_sequence_adapter_region(source, sink))

    def applicable(dependencies: DependencyView) -> Answer[bool]:
        return Decided(
            weight_sequence_adapter_applicable(
                cast(Port, dependencies["source_port"]), cast(Port, dependencies["sink_port"])
            )
        )

    spec = DesignSpaceSpec(
        properties=(
            DerivedProperty(_REGION_PATH, _REGION, EvaluatorSpec((source_ref, sink_ref), derive)),
        ),
        constraints=(
            Constraint(_APPLICABLE_PATH, EvaluatorSpec((source_ref, sink_ref), applicable)),
        ),
    )
    return Kernel(
        FULL_TILE_TO_CHUNKED,
        "1",
        spec,
        _REGION_PATH,
        feasibility_constraints=(_APPLICABLE_PATH,),
    )


def build_mvau_weight_adapter_selection(
    source_ref: DependencyRef,
    sink_ref: DependencyRef,
    *,
    applies_if: EvaluatorSpec[Answer[bool]] | None = None,
) -> KernelSelection:
    """Build the optional adapter pool between a supplier and a compute Kernel."""

    return KernelSelection(
        MVAU_WEIGHT_ADAPTER_SELECTION_NAME,
        (build_mvau_weight_adapter_kernel(source_ref, sink_ref),),
        optional=True,
        applies_if=applies_if,
    )


__all__ = [
    "FULL_TILE_TO_CHUNKED",
    "MVAU_WEIGHT_ADAPTER_SELECTION_NAME",
    "build_mvau_weight_adapter_kernel",
    "build_mvau_weight_adapter_selection",
]
