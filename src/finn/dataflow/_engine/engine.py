# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The public design-space engine facade."""

from __future__ import annotations

from collections.abc import Iterable

from .declarations import DeclarationKind, DesignSpaceSpec
from .errors import RequestError, ValidationError
from .evaluation import EvaluationKernel
from .points import CommitResult, DesignPoint, ProposalAdoptionResult, make_initial_point
from .primitives import PathMapping, QualifiedPath
from .requests import (
    normalize_assignments,
    normalize_constraints,
    normalize_proposal_targets,
    parse_problem_data,
    request_finding,
    resolve_constraint_set,
    resolve_readiness_profile,
)
from .results import (
    Answer,
    ConstraintAssessment,
    DecisionState,
    ProposalAdoptionMode,
    ReadinessAssessment,
)
from .validation import (
    DesignSpace,
    DesignSpaceInspection,
    inspect_design_space,
    validate_design_space,
)


class Engine:
    """Domain-free immutable design-space evaluation engine."""

    def __init__(self) -> None:
        self._kernel = EvaluationKernel()

    def inspect(self, space_or_spec: DesignSpaceSpec | DesignSpace) -> DesignSpaceInspection:
        return inspect_design_space(space_or_spec)

    def validate(self, specification: DesignSpaceSpec) -> DesignSpace:
        return validate_design_space(specification)

    def try_validate(self, specification: DesignSpaceSpec) -> DesignSpace | ValidationError:
        try:
            return self.validate(specification)
        except ValidationError as exc:
            return exc

    def start(
        self,
        design_space: DesignSpace,
        problem: PathMapping,
    ) -> DesignPoint:
        return make_initial_point(design_space, parse_problem_data(design_space, problem))

    def try_start(
        self,
        design_space: DesignSpace,
        problem: PathMapping,
    ) -> DesignPoint | RequestError:
        try:
            return self.start(design_space, problem)
        except RequestError as exc:
            return exc

    @staticmethod
    def _path(raw: QualifiedPath | str) -> QualifiedPath:
        try:
            return QualifiedPath.parse(raw)
        except (TypeError, ValueError) as exc:
            raise RequestError((request_finding("request-path", str(exc)),)) from exc

    def _declaration_path(
        self, point: DesignPoint, raw: QualifiedPath | str, kind: DeclarationKind
    ) -> QualifiedPath:
        path = self._path(raw)
        index = {
            DeclarationKind.DECISION: point.design_space.decisions,
            DeclarationKind.DERIVED_PROPERTY: point.design_space.properties,
            DeclarationKind.CONSTRAINT: point.design_space.constraints,
        }[kind]
        if path not in index:
            raise RequestError(
                (
                    request_finding(
                        f"{kind.value}-path",
                        f"no {kind.value} is declared at this path",
                        path,
                    ),
                )
            )
        return path

    def decision_state(
        self, point: DesignPoint, path: QualifiedPath | str
    ) -> Answer[DecisionState]:
        resolved = self._declaration_path(point, path, DeclarationKind.DECISION)
        return self._kernel.decision_state(point, resolved)

    def enumerate_candidates(
        self, point: DesignPoint, path: QualifiedPath | str
    ) -> Answer[tuple[object, ...]]:
        resolved = self._declaration_path(point, path, DeclarationKind.DECISION)
        return self._kernel.enumerate_candidates(point, resolved)

    def query_property(self, point: DesignPoint, path: QualifiedPath | str) -> Answer[object]:
        resolved = self._declaration_path(point, path, DeclarationKind.DERIVED_PROPERTY)
        return self._kernel.query_property(point, resolved)

    def commit_assignments(
        self,
        point: DesignPoint,
        assignments: PathMapping,
    ) -> CommitResult:
        normalized = normalize_assignments(point.design_space, assignments)
        return self._kernel.commit_assignments(point, normalized)

    def try_commit_assignments(
        self,
        point: DesignPoint,
        assignments: PathMapping,
    ) -> CommitResult | RequestError:
        try:
            return self.commit_assignments(point, assignments)
        except RequestError as exc:
            return exc

    def adopt_proposals(
        self,
        point: DesignPoint,
        targets: Iterable[QualifiedPath | str],
        mode: ProposalAdoptionMode = ProposalAdoptionMode.ONCE,
    ) -> ProposalAdoptionResult:
        if not isinstance(mode, ProposalAdoptionMode):
            raise RequestError(
                (
                    request_finding(
                        "proposal-mode",
                        "mode must be ProposalAdoptionMode.ONCE or TO_FIXPOINT",
                    ),
                )
            )
        normalized = normalize_proposal_targets(point.design_space, targets)
        return self._kernel.adopt_proposals(point, normalized, mode)

    def adopt_profile_proposals(
        self,
        point: DesignPoint,
        profile: str,
        mode: ProposalAdoptionMode = ProposalAdoptionMode.ONCE,
    ) -> ProposalAdoptionResult:
        resolved = resolve_readiness_profile(point.design_space, profile)
        return self.adopt_proposals(point, resolved.decisions, mode)

    def evaluate_constraints(
        self,
        point: DesignPoint,
        paths: Iterable[QualifiedPath | str] | None = None,
    ) -> ConstraintAssessment:
        normalized = normalize_constraints(point.design_space, paths)
        return self._kernel.evaluate_constraints(point, normalized)

    def evaluate_constraint_set(self, point: DesignPoint, name: str) -> ConstraintAssessment:
        paths = resolve_constraint_set(point.design_space, name)
        return self._kernel.evaluate_constraints(point, paths)

    def check_readiness(self, point: DesignPoint, profile: str) -> ReadinessAssessment:
        resolved = resolve_readiness_profile(point.design_space, profile)
        return self._kernel.check_readiness(
            point,
            resolved.name,
            tuple(sorted(resolved.decisions)),
            tuple(sorted(resolved.properties)),
            tuple(sorted(resolved.constraints)),
        )

    def _clear_cache(self, point: DesignPoint) -> None:
        """Testing hook proving cache state is semantically transparent."""

        self._kernel.clear_cache(point)


__all__ = ["Engine"]
