# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit logical Views and their canonical structural validation."""

from __future__ import annotations
from collections.abc import Sequence
from typing import TypeVar, cast
from finn.kernels._engine import DesignPoint, Engine
from finn.parked.dataflow.model._authoring import GENERATED_MEMBERS, authored_member, generated_member
from finn.parked.dataflow.logical_values.results import LogicalResult, NetworkResult, RegionResult
from finn.parked.dataflow.logical_values.network import DataflowNetwork
from finn.parked.dataflow.logical_values.region import DataflowRegion
from finn.parked.dataflow.logical_values.region_validation import validate_region
from finn.parked.dataflow.logical_values.semantics import DATAFLOW_LOGICAL_RESULT_SEMANTICS
from finn.kernels.space.compiler import _CompiledSpace
from finn.kernels.space.declarations import (
    AuthoringError,
    Constraint,
    ConstraintGroup,
    Projection,
    Readiness,
    Space,
    ValueSource,
    declared_members,
    reject,
    reject_all,
)
from finn.kernels.space.occurrence import ProjectionAssessment, evaluate_projection

T_co = TypeVar("T_co", covariant=True)


class LogicalView(Projection[T_co]):
    """A typed logical capability using the common Projection runtime."""

    def __init__(
        self,
        output: ValueSource[T_co],
        *,
        applicable_if: ValueSource[bool] | None = None,
        readiness: Readiness,
        constraints: ConstraintGroup | Sequence[ConstraintGroup] = (),
        name: str | None = "logical",
    ) -> None:
        if output.value_semantics.type_token is not DATAFLOW_LOGICAL_RESULT_SEMANTICS.type_token:
            raise AuthoringError(
                "LogicalView output must use the standard LogicalResult value semantics"
            )
        super().__init__(
            output,
            applicable_if=applicable_if,
            readiness=readiness,
            constraints=constraints,
            name=name,
        )


class _LogicalValidityConstraint(Constraint):
    """Marker for the canonical validation every LogicalView receives."""


def _logical_value_valid(output: ValueSource[object]) -> _LogicalValidityConstraint:
    def evaluate(*, output: LogicalResult) -> object:
        if isinstance(output, RegionResult):
            issues = tuple(
                (issue.code, issue.message, issue.path)
                for issue in validate_region(output.region).issues
            )
            prefix = "kernel-region"
        elif isinstance(output, NetworkResult):
            from finn.parked.dataflow.logical_values.network_validation import validate_network  # noqa: PLC0415

            issues = tuple(
                (issue.code, issue.message, issue.path)
                for issue in validate_network(output.network).issues
            )
            prefix = "kernel-network"
        else:
            return reject(
                "kernel-logical-result-type",
                "LogicalView output must be RegionResult or NetworkResult",
            )
        if not issues:
            return True
        return reject_all(
            reject(f"{prefix}-{code}", message, values={"logical_path": path})
            for code, message, path in issues
        )

    return _LogicalValidityConstraint((("output", output),), evaluate)


def ensure_logical_view_validation(kernel_type: type[Space]) -> None:
    generated = set(
        cast("frozenset[str]", kernel_type.__dict__.get(GENERATED_MEMBERS, frozenset()))
    )
    for member_name, declaration in tuple(declared_members(kernel_type)):
        if not isinstance(declaration, LogicalView):
            continue
        if any(
            isinstance(constraint, _LogicalValidityConstraint)
            for group in declaration.constraints
            for constraint in group.constraints
        ):
            continue
        constraint_name = f"{member_name}_structurally_valid"
        group_name = f"{member_name}_domain_accepts"
        for name in (constraint_name, group_name):
            if authored_member(kernel_type, name) is not None:
                raise AuthoringError(
                    f"{kernel_type.__name__}.{name} conflicts with LogicalView domain validation"
                )
        validity = _logical_value_valid(cast("ValueSource[object]", declaration.output))
        group = ConstraintGroup(validity, name=group_name)
        generated_member(kernel_type, constraint_name, validity, generated)
        generated_member(kernel_type, group_name, group, generated)
        setattr(
            kernel_type,
            member_name,
            LogicalView(
                declaration.output,
                applicable_if=declaration.applicable_if,
                readiness=declaration.readiness,
                constraints=(*declaration.constraints, group),
                name=declaration.stable_name,
            ),
        )
    setattr(kernel_type, GENERATED_MEMBERS, frozenset(generated))


def kernel_dataflow(
    engine: Engine,
    compiled: _CompiledSpace[Space],
    point: DesignPoint,
) -> ProjectionAssessment[DataflowRegion | DataflowNetwork]:
    return cast(
        "ProjectionAssessment[DataflowRegion | DataflowNetwork]",
        evaluate_projection(engine, point, compiled.projection("dataflow")),
    )


__all__ = ["LogicalView", "kernel_dataflow"]
