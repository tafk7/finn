# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Request validation and immutable ingress snapshots."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .declarations import DeclarationKind, ReadinessProfile
from .errors import EvaluationError, RequestError
from .primitives import PathMapping, QualifiedPath, ValueSemantics
from .results import Finding, FindingKind
from .validation import DesignSpace

_REQUEST_PATH = QualifiedPath("request")


def request_finding(code: str, message: str, path: QualifiedPath = _REQUEST_PATH) -> Finding:
    return Finding(FindingKind.REQUEST, code, path, message)


def _misresolved(design_space: DesignSpace, path: QualifiedPath, expected: DeclarationKind) -> str:
    indexes = {
        DeclarationKind.DECISION: design_space.decisions,
        DeclarationKind.DERIVED_PROPERTY: design_space.properties,
        DeclarationKind.CONSTRAINT: design_space.constraints,
    }
    actual = next((kind.value for kind, index in indexes.items() if path in index), None)
    if actual is None:
        return f"unknown path; no {expected.value} is declared at this path"
    return f"path is a {actual}, not a {expected.value}"


def _parse_path(raw: object) -> QualifiedPath | Finding:
    if isinstance(raw, QualifiedPath):
        return raw
    if not isinstance(raw, str):
        return request_finding("request-path-type", "path must be a string or QualifiedPath")
    try:
        return QualifiedPath(raw)
    except ValueError as exc:
        return request_finding("request-path-format", str(exc))


def _accepts(
    semantics: ValueSemantics[object], value: object, owner: QualifiedPath, role: str
) -> bool:
    try:
        return semantics.accepts(value)
    except Exception as exc:
        raise EvaluationError(owner, role, "value recognition raised an exception") from exc


def _freeze(
    semantics: ValueSemantics[object], value: object, owner: QualifiedPath, role: str
) -> object:
    try:
        return semantics.freeze(value)
    except Exception as exc:
        raise EvaluationError(owner, role, "value snapshot failed") from exc


def normalize_assignments(
    design_space: DesignSpace,
    assignments: PathMapping,
) -> dict[QualifiedPath, object]:
    if not isinstance(assignments, Mapping):
        raise RequestError(
            (request_finding("assignments-not-mapping", "assignments must be a mapping"),)
        )
    findings: list[Finding] = []
    values: dict[QualifiedPath, object] = {}
    for raw_path, value in assignments.items():
        path = _parse_path(raw_path)
        if isinstance(path, Finding):
            findings.append(path)
            continue
        if path in values:
            findings.append(request_finding("assignment-duplicate", "duplicate assignment", path))
            continue
        decision = design_space.decisions.get(path)
        if decision is None:
            findings.append(
                request_finding(
                    "assignment-path",
                    _misresolved(design_space, path, DeclarationKind.DECISION),
                    path,
                )
            )
            continue
        if not _accepts(decision.value_semantics, value, path, "assignment recognition"):
            findings.append(
                request_finding(
                    "assignment-type",
                    f"assignment requires nominal type {decision.value_semantics.name}",
                    path,
                )
            )
            continue
        values[path] = value
    if findings:
        raise RequestError(findings)
    return {
        path: _freeze(
            design_space.decisions[path].value_semantics,
            value,
            path,
            "assignment snapshot",
        )
        for path, value in values.items()
    }


def _normalize_paths(
    design_space: DesignSpace,
    raw_paths: object,
    expected: DeclarationKind,
    code: str,
) -> tuple[QualifiedPath, ...]:
    if isinstance(raw_paths, (str, QualifiedPath)):
        raise RequestError(
            (
                request_finding(
                    f"{code}-collection",
                    "paths must be an iterable of paths; wrap a single path in a tuple",
                ),
            )
        )
    if not isinstance(raw_paths, Iterable):
        raise RequestError((request_finding(f"{code}-collection", "paths must be an iterable"),))
    index = {
        DeclarationKind.DECISION: design_space.decisions,
        DeclarationKind.DERIVED_PROPERTY: design_space.properties,
        DeclarationKind.CONSTRAINT: design_space.constraints,
    }[expected]
    findings: list[Finding] = []
    paths: list[QualifiedPath] = []
    seen: set[QualifiedPath] = set()
    for raw_path in raw_paths:
        path = _parse_path(raw_path)
        if isinstance(path, Finding):
            findings.append(path)
            continue
        if path in seen:
            findings.append(request_finding(f"{code}-duplicate", "duplicate path", path))
            continue
        seen.add(path)
        if path not in index:
            findings.append(
                request_finding(f"{code}-path", _misresolved(design_space, path, expected), path)
            )
            continue
        paths.append(path)
    if findings:
        raise RequestError(findings)
    return tuple(sorted(paths))


def normalize_proposal_targets(
    design_space: DesignSpace, targets: object
) -> tuple[QualifiedPath, ...]:
    return _normalize_paths(design_space, targets, DeclarationKind.DECISION, "proposal-target")


def normalize_constraints(
    design_space: DesignSpace, paths: object | None
) -> tuple[QualifiedPath, ...]:
    if paths is None:
        return tuple(sorted(design_space.constraints))
    return _normalize_paths(design_space, paths, DeclarationKind.CONSTRAINT, "constraint")


def resolve_constraint_set(design_space: DesignSpace, name: str) -> tuple[QualifiedPath, ...]:
    if not isinstance(name, str):
        raise RequestError(
            (request_finding("constraint-set-name", "constraint-set name must be a string"),)
        )
    try:
        return design_space.constraint_sets[name]
    except KeyError:
        raise RequestError(
            (request_finding("constraint-set-unknown", f"unknown constraint set {name!r}"),)
        ) from None


def resolve_readiness_profile(design_space: DesignSpace, name: str) -> ReadinessProfile:
    if not isinstance(name, str):
        raise RequestError(
            (request_finding("profile-name", "readiness-profile name must be a string"),)
        )
    try:
        return design_space.readiness_profiles[name]
    except KeyError:
        raise RequestError(
            (request_finding("profile-unknown", f"unknown readiness profile {name!r}"),)
        ) from None


def parse_problem_data(
    design_space: DesignSpace, problem_data: PathMapping
) -> dict[QualifiedPath, object]:
    if not isinstance(problem_data, Mapping):
        raise RequestError(
            (request_finding("problem-not-mapping", "problem data must be a mapping"),)
        )
    findings: list[Finding] = []
    raw_values: dict[QualifiedPath, object] = {}
    for raw_path, value in problem_data.items():
        path = _parse_path(raw_path)
        if isinstance(path, Finding):
            findings.append(path)
            continue
        if path in raw_values:
            findings.append(request_finding("problem-duplicate", "duplicate problem field", path))
            continue
        raw_values[path] = value

    for path in raw_values:
        if path not in design_space.problem_fields:
            findings.append(request_finding("problem-unknown", "unknown problem field", path))
    for path, field in design_space.problem_fields.items():
        if field.required and path not in raw_values:
            findings.append(request_finding("problem-required", "required field is absent", path))
        if path in raw_values and not _accepts(
            field.value_semantics, raw_values[path], path, "problem recognition"
        ):
            findings.append(
                request_finding(
                    "problem-type",
                    f"problem field requires nominal type {field.value_semantics.name}",
                    path,
                )
            )
    if findings:
        raise RequestError(findings)

    frozen: dict[QualifiedPath, object] = {}
    for path, value in raw_values.items():
        field = design_space.problem_fields[path]
        snapshot = _freeze(field.value_semantics, value, path, "problem snapshot")
        if field.constraint is not None:
            try:
                valid = bool(field.constraint(snapshot))
            except Exception as exc:
                raise EvaluationError(path, "problem constraint", "constraint raised") from exc
            if not valid:
                description = field.constraint_description or "declared problem constraint"
                findings.append(
                    request_finding("problem-constraint", f"value violates {description}", path)
                )
                continue
        frozen[path] = snapshot
    if findings:
        raise RequestError(findings)
    return frozen


__all__ = [
    "normalize_assignments",
    "normalize_constraints",
    "normalize_proposal_targets",
    "parse_problem_data",
    "request_finding",
    "resolve_constraint_set",
    "resolve_readiness_profile",
]
