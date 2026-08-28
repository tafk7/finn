# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Commit Kernel and local design choices under a replaceable policy.

The engine owns decisions, constraints, points, and enumeration.  It does not
own preference, so preference lives here and only here.  A policy receives
complete coherent candidate points and returns the assignments to commit; it
never constructs a point one greedy decision at a time, and it never mutates a
node class or domain to express a specialization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.transformation.base import Transformation  # type: ignore[import-not-found]

from finn.dataflow.design import (
    ConstraintAssessment,
    Decided,
    DesignPoint,
    Engine,
    Finding,
    FindingKind,
    QualifiedPath,
    ReadinessAssessment,
)
from finn.dataflow.op import DataflowBuildConfigView, DataflowOp
from finn.dataflow.selection import enumerate_feasible_points

_SELECTION_PATH = QualifiedPath("compiler.dataflow.selection")


@dataclass(frozen=True)
class DataflowSelectionContext:
    """Everything a policy may read about one operation scope."""

    scope_id: str
    operation: DataflowOp
    engine: Engine
    point: DesignPoint
    decision_paths: tuple[QualifiedPath, ...]
    constraint_set: str

    def feasible_points(self) -> tuple[DesignPoint, ...]:
        """Enumerate coherent complete points, never one decision at a time."""

        return enumerate_feasible_points(
            self.engine,
            self.point,
            self.decision_paths,
            constraint_set=self.constraint_set,
        ).points


class DataflowSelectionPolicy(ABC):
    """A replaceable preference over feasible design points."""

    @abstractmethod
    def select(self, context: DataflowSelectionContext) -> Mapping[QualifiedPath, object] | None:
        """Return the assignments to commit, or None to leave the scope alone."""


class ExplicitAssignmentsPolicy(DataflowSelectionPolicy):
    """Commit exactly what the caller named, keyed by stable operation scope."""

    def __init__(self, assignments: Mapping[str, Mapping[QualifiedPath, object]]) -> None:
        self._assignments = {scope: dict(items) for scope, items in assignments.items()}

    def select(self, context: DataflowSelectionContext) -> Mapping[QualifiedPath, object] | None:
        return self._assignments.get(context.scope_id)


class FirstFeasiblePolicy(DataflowSelectionPolicy):
    """A deterministic reference policy, not a default and not a ranking.

    It exists so the seam can be exercised end to end.  It takes the first
    point in the enumeration's stable assignment order; that order is a
    tie-break, not a statement that the point is preferable.
    """

    def select(self, context: DataflowSelectionContext) -> Mapping[QualifiedPath, object] | None:
        points = context.feasible_points()
        if not points:
            return None
        return dict(points[0].assignments)


@dataclass(frozen=True)
class DataflowScopeReport:
    """Readiness and feasibility reported separately for one scope."""

    scope_id: str
    committed: tuple[QualifiedPath, ...] = ()
    structural_readiness: ReadinessAssessment | None = None
    artifact_readiness: ReadinessAssessment | None = None
    feasibility: Mapping[str, ConstraintAssessment] = field(default_factory=dict)
    findings: tuple[Finding, ...] = ()


@dataclass(frozen=True)
class DataflowSelectionReport:
    """What one selection pass committed and what it could not."""

    scopes: tuple[DataflowScopeReport, ...] = ()

    def scope(self, scope_id: str) -> DataflowScopeReport:
        return next(item for item in self.scopes if item.scope_id == scope_id)


class SelectDataflowDesign(Transformation):  # type: ignore[misc]
    """Apply one selection policy to every attached logical dataflow operation.

    The transform is operation-generic: it knows the ``DataflowOp`` boundary and
    nothing about MVAU.  Every commitment goes through the transactional
    assignment API, so invalid policy output leaves the node byte identical.
    """

    def __init__(
        self,
        policy: DataflowSelectionPolicy,
        config: DataflowBuildConfigView,
        *,
        constraint_set: str,
        structural_profile: str | None = None,
        artifact_profile: str | None = None,
        feasibility_sets: Iterable[str] = (),
        replace: bool = False,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.config = config
        self.constraint_set = constraint_set
        self.structural_profile = structural_profile
        self.artifact_profile = artifact_profile
        self.feasibility_sets = tuple(feasibility_sets)
        self.replace = replace
        self.report = DataflowSelectionReport()

    def _scope_report(
        self, operation: DataflowOp, scope_id: str, committed: tuple[QualifiedPath, ...]
    ) -> DataflowScopeReport:
        engine = Engine()
        point = operation.hydrate_dataflow_point(self.config)
        feasibility = {
            name: engine.evaluate_constraint_set(point, name) for name in self.feasibility_sets
        }
        return DataflowScopeReport(
            scope_id,
            committed,
            None
            if self.structural_profile is None
            else engine.check_readiness(point, self.structural_profile),
            None
            if self.artifact_profile is None
            else engine.check_readiness(point, self.artifact_profile),
            feasibility,
        )

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        scopes: list[DataflowScopeReport] = []
        for node in list(model.graph.node):
            operation = _attached_dataflow_op(model, node)
            if operation is None:
                continue
            scope_id = operation.dataflow_scope_id()
            engine = Engine()
            point = operation.hydrate_dataflow_point(self.config)
            context = DataflowSelectionContext(
                scope_id,
                operation,
                engine,
                point,
                tuple(sorted(type(operation).decision_nodeattrs())),
                self.constraint_set,
            )
            assignments = self.policy.select(context)
            if assignments is None:
                scopes.append(
                    DataflowScopeReport(
                        scope_id,
                        (),
                        findings=(
                            Finding(
                                FindingKind.LIMITATION,
                                "dataflow-selection-no-point",
                                _SELECTION_PATH,
                                "the policy returned no assignments for this scope",
                                (("scope_id", scope_id),),
                            ),
                        ),
                    )
                )
                continue
            snapshot = node.SerializeToString(deterministic=True)
            try:
                commit = (
                    operation.replace_dataflow_assignments(self.config, assignments)
                    if self.replace
                    else operation.commit_dataflow_assignments(self.config, assignments)
                )
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                restored = type(node)()
                restored.ParseFromString(snapshot)
                node.CopyFrom(restored)
                scopes.append(
                    DataflowScopeReport(
                        scope_id,
                        (),
                        findings=(
                            Finding(
                                FindingKind.REJECTION,
                                "dataflow-selection-commit-failed",
                                _SELECTION_PATH,
                                str(exc),
                                (("scope_id", scope_id),),
                            ),
                        ),
                    )
                )
                continue
            scopes.append(
                self._scope_report(operation, scope_id, tuple(sorted(commit.point.assignments)))
            )
        self.report = DataflowSelectionReport(tuple(scopes))
        return model, False


def _attached_dataflow_op(model: ModelWrapper, node: object) -> DataflowOp | None:
    try:
        operation = model.get_customop_wrapper(node)
    except Exception:  # noqa: BLE001 - a non-dataflow node is not an error
        return None
    return operation if isinstance(operation, DataflowOp) else None


def committed_kernel_ids(
    engine: Engine, point: DesignPoint, selected_kernel_paths: Iterable[QualifiedPath]
) -> Mapping[QualifiedPath, str]:
    """Read back which Kernel each pool committed, for reporting and tests."""

    identities: dict[QualifiedPath, str] = {}
    for path in selected_kernel_paths:
        answer = engine.query_property(point, path)
        if isinstance(answer, Decided):
            identities[path] = str(getattr(answer.value, "kernel_id", answer.value))
    return identities


__all__: Sequence[str] = [
    "DataflowScopeReport",
    "DataflowSelectionContext",
    "DataflowSelectionPolicy",
    "DataflowSelectionReport",
    "ExplicitAssignmentsPolicy",
    "FirstFeasiblePolicy",
    "SelectDataflowDesign",
    "committed_kernel_ids",
]
