# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Candidate-backed graph admission and resolved physical feasibility."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import cast

from finn.dataflow.authoring.design import (
    DataflowDesignInventory,
    KernelPlacement,
    PlacementSelection,
)
from finn.dataflow.authoring.op_design import GRAPH_OWNED, ProblemProvenance
from finn.dataflow.design import (
    Absent,
    Decided,
    DependencyKind,
    DependencyRef,
    DesignPoint,
    DesignSpaceSpec,
    Engine,
    Finding,
    QualifiedPath,
    RequestError,
    Unresolved,
)


class AdmissionVerdict(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class CandidateAdmission:
    candidate_id: str
    verdict: AdmissionVerdict
    graph_constraints: tuple[QualifiedPath, ...]
    deferred_constraints: tuple[QualifiedPath, ...]
    findings: tuple[Finding, ...] = ()


@dataclass(frozen=True)
class PlacementAdmission:
    placement: str
    verdict: AdmissionVerdict
    candidates: tuple[CandidateAdmission, ...]


@dataclass(frozen=True)
class DesignAdmissionTrial:
    design_id: str
    supply_modes: tuple[tuple[str, str], ...]
    verdict: AdmissionVerdict
    placements: tuple[PlacementAdmission, ...]


@dataclass(frozen=True)
class GraphBuildAdmission:
    admitted_designs: tuple[str, ...]
    unresolved_designs: tuple[str, ...]
    trials: tuple[DesignAdmissionTrial, ...]


@dataclass(frozen=True)
class ResolvedPhysicalFeasibility:
    verdict: bool | None
    findings: tuple[Finding, ...] = ()


def _dependency_is_graph_answerable(
    dependency: DependencyRef,
    *,
    specification: DesignSpaceSpec,
    provenance: ProblemProvenance,
    trial_decisions: frozenset[QualifiedPath],
    visiting: frozenset[QualifiedPath] = frozenset(),
) -> bool:
    if dependency.kind is DependencyKind.PROBLEM:
        return provenance.kind_of(dependency.path) in GRAPH_OWNED
    if dependency.kind is DependencyKind.DECISION:
        return dependency.path in trial_decisions
    if dependency.kind is not DependencyKind.PROPERTY or dependency.path in visiting:
        return False
    declaration = next(
        (item for item in specification.properties if item.path == dependency.path),
        None,
    )
    if declaration is None:
        return False
    dependencies = list(declaration.evaluator.dependencies)
    if declaration.applies_if is not None:
        dependencies.extend(declaration.applies_if.dependencies)
    nested = visiting | {dependency.path}
    return all(
        _dependency_is_graph_answerable(
            item,
            specification=specification,
            provenance=provenance,
            trial_decisions=trial_decisions,
            visiting=nested,
        )
        for item in dependencies
    )


def _constraint_is_graph_answerable(
    path: QualifiedPath,
    *,
    specification: DesignSpaceSpec,
    provenance: ProblemProvenance,
    trial_decisions: frozenset[QualifiedPath],
) -> bool:
    declaration = next(item for item in specification.constraints if item.path == path)
    dependencies = list(declaration.evaluator.dependencies)
    if declaration.applies_if is not None:
        dependencies.extend(declaration.applies_if.dependencies)
    return all(
        _dependency_is_graph_answerable(
            item,
            specification=specification,
            provenance=provenance,
            trial_decisions=trial_decisions,
        )
        for item in dependencies
    )


def _trial(
    engine: Engine,
    point: DesignPoint,
    assignments: Mapping[QualifiedPath, object],
) -> DesignPoint | None:
    requested = {
        path: value for path, value in assignments.items() if point.assignments.get(path) != value
    }
    if any(path in point.assignments for path in requested):
        return None
    committed = engine.try_commit_assignments(point, requested)
    if isinstance(committed, RequestError):
        return None
    if any(outcome.disposition not in {"committed", "unchanged"} for outcome in committed.outcomes):
        return None
    return committed.point


def _candidate_admission(
    engine: Engine,
    point: DesignPoint,
    inventory: DataflowDesignInventory,
    provenance: ProblemProvenance,
    placement: KernelPlacement,
    candidate_id: str,
    trial_decisions: frozenset[QualifiedPath],
) -> CandidateAdmission:
    candidate = placement.candidate(candidate_id)
    trial = point
    known = set(trial_decisions)
    if placement.kernel_choice is not None:
        selected_trial = _trial(engine, trial, {placement.kernel_choice.path: candidate_id})
        if selected_trial is None:
            return CandidateAdmission(candidate_id, AdmissionVerdict.REJECTED, (), ())
        trial = selected_trial
        known.add(placement.kernel_choice.path)
    graph_constraints = tuple(
        path
        for path in candidate.coverage_constraints
        if _constraint_is_graph_answerable(
            path,
            specification=inventory.specification,
            provenance=provenance,
            trial_decisions=frozenset(known),
        )
    )
    deferred = tuple(
        path for path in candidate.coverage_constraints if path not in graph_constraints
    )
    if not graph_constraints:
        return CandidateAdmission(
            candidate_id,
            AdmissionVerdict.ADMITTED,
            (),
            deferred,
        )
    assessment = engine.evaluate_constraints(trial, graph_constraints)
    findings: list[Finding] = []
    unresolved = False
    for answer in assessment.answers.values():
        if isinstance(answer, Decided):
            if answer.value is True:
                continue
            return CandidateAdmission(
                candidate_id,
                AdmissionVerdict.REJECTED,
                graph_constraints,
                deferred,
                tuple(findings),
            )
        findings.extend(answer.findings)
        if isinstance(answer, Unresolved):
            unresolved = True
            continue
        return CandidateAdmission(
            candidate_id,
            AdmissionVerdict.REJECTED,
            graph_constraints,
            deferred,
            tuple(findings),
        )
    return CandidateAdmission(
        candidate_id,
        AdmissionVerdict.UNRESOLVED if unresolved else AdmissionVerdict.ADMITTED,
        graph_constraints,
        deferred,
        tuple(findings),
    )


def _placement_admission(
    engine: Engine,
    point: DesignPoint,
    inventory: DataflowDesignInventory,
    provenance: ProblemProvenance,
    placement: KernelPlacement,
    trial_decisions: frozenset[QualifiedPath],
) -> PlacementAdmission | None:
    selected = engine.query_property(point, placement.selected_kernel.path)
    if isinstance(selected, Absent):
        return None
    candidates = tuple(
        _candidate_admission(
            engine,
            point,
            inventory,
            provenance,
            placement,
            candidate.id,
            trial_decisions,
        )
        for candidate in placement.candidates
    )
    if any(item.verdict is AdmissionVerdict.ADMITTED for item in candidates):
        verdict = AdmissionVerdict.ADMITTED
    elif any(item.verdict is AdmissionVerdict.UNRESOLVED for item in candidates):
        verdict = AdmissionVerdict.UNRESOLVED
    else:
        verdict = AdmissionVerdict.REJECTED
    return PlacementAdmission(placement.name, verdict, candidates)


def graph_stage_build_admission(
    engine: Engine,
    point: DesignPoint,
    inventory: DataflowDesignInventory,
    provenance: ProblemProvenance,
    *,
    design_ids: Iterable[str] | None = None,
) -> GraphBuildAdmission:
    """Project Kernel-owned coverage onto facts answerable from the graph."""

    requested_designs = tuple(design_ids) if design_ids is not None else inventory.design_ids
    trials: list[DesignAdmissionTrial] = []
    for design_id in requested_designs:
        declaration = inventory.declaration(design_id)
        supply_sets = tuple(supply.modes for supply in inventory.input_supplies)
        supply_trials = product(*supply_sets) if supply_sets else ((),)
        for modes in supply_trials:
            supply_modes = tuple(
                (supply.source_operand, mode)
                for supply, mode in zip(inventory.input_supplies, modes)
            )
            assignments: dict[QualifiedPath, object] = {
                supply.choice.path: mode for supply, mode in zip(inventory.input_supplies, modes)
            }
            if inventory.design_selection is not None:
                assignments[inventory.design_selection.path] = design_id
            trial = _trial(engine, point, assignments)
            if trial is None:
                trials.append(
                    DesignAdmissionTrial(
                        design_id,
                        supply_modes,
                        AdmissionVerdict.REJECTED,
                        (),
                    )
                )
                continue
            known = frozenset(assignments)
            placements = tuple(
                item
                for placement in declaration.placements
                for item in (
                    _placement_admission(
                        engine,
                        trial,
                        inventory,
                        provenance,
                        placement,
                        known,
                    ),
                )
                if item is not None
            )
            if placements and all(item.verdict is AdmissionVerdict.ADMITTED for item in placements):
                verdict = AdmissionVerdict.ADMITTED
            elif any(item.verdict is AdmissionVerdict.REJECTED for item in placements):
                verdict = AdmissionVerdict.REJECTED
            elif placements:
                verdict = AdmissionVerdict.UNRESOLVED
            else:
                verdict = AdmissionVerdict.REJECTED
            trials.append(DesignAdmissionTrial(design_id, supply_modes, verdict, placements))

    admitted = tuple(
        design_id
        for design_id in requested_designs
        if any(
            trial.design_id == design_id and trial.verdict is AdmissionVerdict.ADMITTED
            for trial in trials
        )
    )
    unresolved = tuple(
        design_id
        for design_id in requested_designs
        if design_id not in admitted
        and any(
            trial.design_id == design_id and trial.verdict is AdmissionVerdict.UNRESOLVED
            for trial in trials
        )
    )
    return GraphBuildAdmission(admitted, unresolved, tuple(trials))


def resolved_physical_feasibility(
    engine: Engine,
    point: DesignPoint,
    inventory: DataflowDesignInventory,
) -> ResolvedPhysicalFeasibility:
    """Require every selected active Kernel constraint to be positively true."""

    selected_design = inventory.selected(point)
    if not isinstance(selected_design, Decided):
        return ResolvedPhysicalFeasibility(None, selected_design.findings)
    findings: list[Finding] = []
    unresolved = False
    for placement in selected_design.value.placements:
        selected = engine.query_property(point, placement.selected_kernel.path)
        if isinstance(selected, Absent):
            continue
        if not isinstance(selected, Decided):
            findings.extend(selected.findings)
            unresolved = True
            continue
        selection = cast(PlacementSelection, selected.value)
        if selection.kernel_id is None:
            return ResolvedPhysicalFeasibility(False, tuple(findings))
        candidate = placement.candidate(selection.kernel_id)
        assessment = engine.evaluate_constraints(point, candidate.coverage_constraints)
        for answer in assessment.answers.values():
            if isinstance(answer, Decided):
                if answer.value is True:
                    continue
                return ResolvedPhysicalFeasibility(False, tuple(findings))
            findings.extend(answer.findings)
            if isinstance(answer, Unresolved):
                unresolved = True
                continue
            return ResolvedPhysicalFeasibility(False, tuple(findings))
    return ResolvedPhysicalFeasibility(None if unresolved else True, tuple(findings))


__all__ = [
    "AdmissionVerdict",
    "CandidateAdmission",
    "DesignAdmissionTrial",
    "GraphBuildAdmission",
    "PlacementAdmission",
    "ResolvedPhysicalFeasibility",
    "graph_stage_build_admission",
    "resolved_physical_feasibility",
]
