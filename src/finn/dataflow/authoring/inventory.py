# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Authoritative operation assembly over a ``DataflowDesign`` inventory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from finn.dataflow.authoring.design import (
    DataflowDesignEntry,
    DataflowDesignInventory,
    PlacementSelection,
    declare_dataflow_design_inventory,
)
from finn.dataflow.authoring.scope import ConstraintRef, Ref, Scope
from finn.dataflow.design import ABSENT, DesignSpaceSpec, QualifiedPath
from finn.dataflow.spec_algebra import assemble_specs


@dataclass(frozen=True)
class DesignSelectionMetadata:
    """Structural identity of one selected design and its active placements."""

    design_id: str
    decision_paths: tuple[QualifiedPath, ...]
    placements: tuple[tuple[str, str], ...]

    @property
    def kernel_ids(self) -> tuple[str, ...]:
        return tuple(kernel_id for _placement, kernel_id in self.placements)

    def kernel_id(self, placement: str) -> str | None:
        return next(
            (kernel_id for name, kernel_id in self.placements if name == placement),
            None,
        )


def selected_design_metadata(
    inventory: DataflowDesignInventory,
    design_id: str,
    *,
    supply_modes: Mapping[str, str],
    placement_selections: Mapping[tuple[str, str], object],
) -> DesignSelectionMetadata:
    """Describe selected declaration identity without inspecting compiled specs."""

    declaration = inventory.declaration(design_id)
    paths: list[QualifiedPath] = []
    if inventory.design_selection is not None:
        paths.append(inventory.design_selection.path)
    paths.extend(supply.choice.path for supply in inventory.input_supplies)
    paths.extend(handle.path for handle in declaration.decision_handles)

    placements: list[tuple[str, str]] = []
    for placement in declaration.placements:
        selected = placement_selections.get((design_id, placement.name), ABSENT)
        if selected is ABSENT:
            continue
        if not isinstance(selected, PlacementSelection):
            raise TypeError(
                f"placement {design_id}.{placement.name} produced "
                f"{type(selected).__name__}, not PlacementSelection"
            )
        if selected.kernel_id is None:
            continue
        candidate = placement.candidate(selected.kernel_id)
        placements.append((placement.name, candidate.id))
        paths.extend(handle.path for handle in candidate.decision_handles)

    for supply in inventory.input_supplies:
        mode = supply_modes[supply.source_operand]
        if mode == supply.external_id:
            continue
        paths.extend(
            handle.path for handle in supply.decision_handles if handle.path != supply.choice.path
        )

    return DesignSelectionMetadata(
        design_id,
        tuple(dict.fromkeys(paths)),
        tuple(placements),
    )


@dataclass(frozen=True)
class DataflowOpAuthoring:
    """Compiled operation declarations plus their typed runtime handles."""

    specification: DesignSpaceSpec
    result: Ref[object]
    source_association: Ref[object]
    selection_constraint_set: str | None
    structural_readiness_profile: str | None
    artifact_readiness_profile: str | None
    feasibility_constraint_sets: tuple[str, ...]
    inventory: DataflowDesignInventory


def declare_dataflow_op_authoring(
    inventory: DataflowDesignInventory,
    operation: Scope,
    *,
    result: Ref[object],
    source_association: Ref[object],
    structural_properties: Sequence[Ref[object]],
    structural_constraints: Sequence[ConstraintRef],
    structural_constraint_set: str,
    feasibility_constraint_set: str,
    structural_readiness_profile: str,
    artifact_readiness_profile: str,
    additional_structural_decisions: Sequence[Ref[object]] = (),
    additional_artifact_decisions: Sequence[Ref[object]] = (),
    additional_feasibility_constraints: Sequence[ConstraintRef] = (),
) -> DataflowOpAuthoring:
    """Compile the uniform operation boundary owned by an inventory.

    Operation code supplies only its genuinely local result properties and
    constraints.  Design decisions, Kernel choices, inventory constraints,
    and readiness aggregation remain authoritative here.
    """

    feasibility = (
        *inventory.constraint_handles,
        *additional_feasibility_constraints,
        *structural_constraints,
    )
    operation.include_in(structural_constraint_set, *structural_constraints)
    operation.include_in(feasibility_constraint_set, *feasibility)
    operation.readiness_profile(
        structural_readiness_profile,
        decisions=(*inventory.structural_decisions, *additional_structural_decisions),
        properties=structural_properties,
        constraints=structural_constraints,
    )
    operation.readiness_profile(
        artifact_readiness_profile,
        decisions=(
            *inventory.artifact_decisions,
            *additional_structural_decisions,
            *additional_artifact_decisions,
        ),
        properties=structural_properties,
        constraints=feasibility,
    )
    return DataflowOpAuthoring(
        assemble_specs((inventory.specification, operation.spec())),
        result,
        source_association,
        feasibility_constraint_set,
        structural_readiness_profile,
        artifact_readiness_profile,
        (feasibility_constraint_set,),
        inventory,
    )


__all__ = [
    "DataflowOpAuthoring",
    "DataflowDesignEntry",
    "DataflowDesignInventory",
    "DesignSelectionMetadata",
    "declare_dataflow_op_authoring",
    "declare_dataflow_design_inventory",
    "selected_design_metadata",
]
