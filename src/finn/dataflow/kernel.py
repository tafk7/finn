# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Region-declaration authoring shared by Kernels and standalone specs.

The selectable Kernel surface lives in :mod:`finn.dataflow.kernels`.  What
remains here is the small spine a specification needs when one scope declares
several complete Regions and selects among them by applicability: the
declaration identity, the selected-Region property, its structural validation,
and the constraint that reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from finn.dataflow.design import (
    ABSENT,
    DATAFLOW_REGION_SEMANTICS,
    REGION_VALIDATION_REPORT_SEMANTICS,
    AbsenceMode,
    Answer,
    Constraint,
    Decided,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    EvaluatorSpec,
    QualifiedPath,
    as_object_semantics,
    validate_region,
)
from finn.dataflow.region import DataflowRegion
from finn.dataflow.region_validation import RegionValidationReport
from finn.dataflow.spec_algebra import SpecAuthoringError, SpecAuthoringIssue

_REGION_SEMANTICS = as_object_semantics(DATAFLOW_REGION_SEMANTICS)
_REPORT_SEMANTICS = as_object_semantics(REGION_VALIDATION_REPORT_SEMANTICS)


@dataclass(frozen=True, order=True)
class RegionDeclaration:
    """Identity and derived-property path of one complete region declaration."""

    id: str
    property_path: QualifiedPath

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("region declaration id must not be empty")


@dataclass(frozen=True)
class KernelSemanticDeclarations:
    """Generic selected-region and structural-validation declarations."""

    selected_region: DerivedProperty
    validation_report: DerivedProperty
    structural_constraint: Constraint


def build_kernel_semantic_declarations(
    region_declarations: tuple[RegionDeclaration, ...],
    *,
    selected_region_path: QualifiedPath,
    validation_report_path: QualifiedPath,
    structural_constraint_path: QualifiedPath,
) -> KernelSemanticDeclarations:
    """Build the shared selection and structural-validation spine for a scope."""

    if not region_declarations:
        raise SpecAuthoringError(
            (
                SpecAuthoringIssue(
                    "region-declaration-inventory-empty",
                    str(selected_region_path),
                    "a scope must declare at least one complete region",
                ),
            )
        )
    dependencies = tuple(
        DependencyRef.property(
            f"region_{index}",
            declaration.property_path,
            _REGION_SEMANTICS,
            absence=AbsenceMode.ALLOWS_ABSENT,
        )
        for index, declaration in enumerate(region_declarations)
    )

    def select_region(values: DependencyView) -> Answer[object]:
        candidates = tuple(value for value in values.values() if value is not ABSENT)
        if len(candidates) != 1:
            raise AssertionError("one region declaration must be applicable")
        return Decided(cast(DataflowRegion, candidates[0]))

    region_ref = DependencyRef.property("region", selected_region_path, _REGION_SEMANTICS)
    report_ref = DependencyRef.property("report", validation_report_path, _REPORT_SEMANTICS)

    def validate_selected_region(values: DependencyView) -> Answer[object]:
        return Decided(validate_region(cast(DataflowRegion, values["region"])))

    def structurally_well_formed(values: DependencyView) -> Answer[bool]:
        return Decided(not cast(RegionValidationReport, values["report"]))

    return KernelSemanticDeclarations(
        DerivedProperty(
            selected_region_path,
            _REGION_SEMANTICS,
            EvaluatorSpec(dependencies, select_region),
        ),
        DerivedProperty(
            validation_report_path,
            _REPORT_SEMANTICS,
            EvaluatorSpec((region_ref,), validate_selected_region),
        ),
        Constraint(
            structural_constraint_path,
            EvaluatorSpec((report_ref,), structurally_well_formed),
        ),
    )


__all__ = [
    "KernelSemanticDeclarations",
    "RegionDeclaration",
    "build_kernel_semantic_declarations",
]
