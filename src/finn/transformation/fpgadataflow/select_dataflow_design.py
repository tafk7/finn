# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Commit Kernel and local design choices under a replaceable policy.

The engine owns decisions, constraints, points, and enumeration.  It does not
own preference, so preference lives here and only here.

A policy sees the whole model and every operation scope in one call, because
the choices are not independent: a later graph-global policy has to weigh one
scope's Kernel against another's.  It receives complete coherent candidate
points, never a point built one greedy decision at a time, and it never
mutates a node class or domain to express a specialization.

The transform itself is operation-generic.  Each ``DataflowOp`` family names
its own Kernel pools, constraint set, and readiness profiles, so heterogeneous
operations can be selected in one pass and no caller has to know that MVAU
exists.
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
    constraint_set: str | None

    def feasible_points(self) -> tuple[DesignPoint, ...]:
        """Enumerate coherent complete points, never one decision at a time."""

        if self.constraint_set is None:
            return ()
        return enumerate_feasible_points(
            self.engine,
            self.point,
            self.decision_paths,
            constraint_set=self.constraint_set,
        ).points


class DataflowSelectionPolicy(ABC):
    """A replaceable preference over feasible design points."""

    @abstractmethod
    def select(
        self, model: ModelWrapper, contexts: Sequence[DataflowSelectionContext]
    ) -> Mapping[str, Mapping[QualifiedPath, object]]:
        """Return the assignments to commit, keyed by operation scope.

        Every scope in the model is offered at once so a policy may coordinate
        across them.  Omitting a scope leaves it untouched.
        """


class ExplicitAssignmentsPolicy(DataflowSelectionPolicy):
    """Commit exactly what the caller named, keyed by stable operation scope."""

    def __init__(self, assignments: Mapping[str, Mapping[QualifiedPath, object]]) -> None:
        self._assignments = {scope: dict(items) for scope, items in assignments.items()}

    def select(
        self, model: ModelWrapper, contexts: Sequence[DataflowSelectionContext]
    ) -> Mapping[str, Mapping[QualifiedPath, object]]:
        del model
        return {
            context.scope_id: self._assignments[context.scope_id]
            for context in contexts
            if context.scope_id in self._assignments
        }


class FirstFeasiblePolicy(DataflowSelectionPolicy):
    """A deterministic reference policy, not a default and not a ranking.

    It exists so the seam can be exercised end to end.  It takes the first
    point in each scope's stable assignment order; that order is a tie-break,
    not a statement that the point is preferable.  It also makes no attempt to
    coordinate across scopes, which is exactly the limitation a real
    graph-global policy would exist to remove.
    """

    def select(
        self, model: ModelWrapper, contexts: Sequence[DataflowSelectionContext]
    ) -> Mapping[str, Mapping[QualifiedPath, object]]:
        del model
        selected: dict[str, Mapping[QualifiedPath, object]] = {}
        for context in contexts:
            points = context.feasible_points()
            if points:
                selected[context.scope_id] = dict(points[0].assignments)
        return selected


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

    Every commitment goes through the transactional assignment API, so invalid
    policy output leaves the node byte identical.
    """

    def __init__(
        self,
        policy: DataflowSelectionPolicy,
        config: DataflowBuildConfigView,
        *,
        replace: bool = False,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.config = config
        self.replace = replace
        self.report = DataflowSelectionReport()

    def _context(self, operation: DataflowOp) -> DataflowSelectionContext:
        family = type(operation)
        return DataflowSelectionContext(
            operation.dataflow_scope_id(),
            operation,
            Engine(),
            operation.hydrate_dataflow_point(self.config),
            tuple(sorted(family.decision_nodeattrs())),
            family.selection_constraint_set(),
        )

    def _scope_report(
        self, operation: DataflowOp, scope_id: str, committed: tuple[QualifiedPath, ...]
    ) -> DataflowScopeReport:
        family = type(operation)
        engine = Engine()
        point = operation.hydrate_dataflow_point(self.config)
        structural = family.structural_readiness_profile()
        artifact = family.artifact_readiness_profile()
        return DataflowScopeReport(
            scope_id,
            committed,
            None if structural is None else engine.check_readiness(point, structural),
            None if artifact is None else engine.check_readiness(point, artifact),
            {
                name: engine.evaluate_constraint_set(point, name)
                for name in family.feasibility_constraint_sets()
            },
        )

    def _commit(
        self,
        operation: DataflowOp,
        scope_id: str,
        assignments: Mapping[QualifiedPath, object],
    ) -> DataflowScopeReport:
        node = operation.onnx_node
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
            return DataflowScopeReport(
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
        return self._scope_report(operation, scope_id, tuple(sorted(commit.point.assignments)))

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        operations = [
            operation
            for node in list(model.graph.node)
            if (operation := _attached_dataflow_op(model, node)) is not None
        ]
        contexts = [self._context(operation) for operation in operations]
        selected = self.policy.select(model, contexts)
        scopes: list[DataflowScopeReport] = []
        for context in contexts:
            assignments = selected.get(context.scope_id)
            if assignments is None:
                scopes.append(
                    DataflowScopeReport(
                        context.scope_id,
                        (),
                        findings=(
                            Finding(
                                FindingKind.LIMITATION,
                                "dataflow-selection-no-point",
                                _SELECTION_PATH,
                                "the policy returned no assignments for this scope",
                                (("scope_id", context.scope_id),),
                            ),
                        ),
                    )
                )
                continue
            scopes.append(self._commit(context.operation, context.scope_id, assignments))
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
