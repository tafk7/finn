# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Logical Kernel views and their canonical validation attachment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar, cast

from finn.dataflow._engine import DesignPoint, Engine
from finn.dataflow.model._authoring import (
    GENERATED_MEMBERS,
    authored_member,
    generated_member,
    use_authored_or_generated,
)
from finn.dataflow.model.logical.authoring import (
    RegionDeclaration,
    composite_logical_property,
    network_property,
)
from finn.dataflow.model.logical.composition import LogicalResult, NetworkResult, RegionResult
from finn.dataflow.model.logical.network import DataflowNetwork
from finn.dataflow.model.logical.network_validation import validate_network
from finn.dataflow.model.logical.region import DataflowRegion
from finn.dataflow.model.logical.region_validation import validate_region
from finn.dataflow.model.logical.semantics import (
    DATAFLOW_LOGICAL_RESULT_SEMANTICS,
    DATAFLOW_NETWORK_SEMANTICS,
    DATAFLOW_REGION_SEMANTICS,
)
from finn.dataflow.space.compiler import _CompiledSpace
from finn.dataflow.space.declarations import (
    AuthoringError,
    Constraint,
    ConstraintGroup,
    Derived,
    Projection,
    Readiness,
    Space,
    ValueSource,
    declared_members,
    reject,
    reject_all,
    semantics_for,
)
from finn.dataflow.space.occurrence import ProjectionAssessment, evaluate_projection

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


def synchronize_generated_dataflow(kernel_type: type[Space], *, composite: bool) -> None:
    generated = set(
        cast("frozenset[str]", kernel_type.__dict__.get(GENERATED_MEMBERS, frozenset()))
    )
    if "dataflow" not in generated:
        return
    logical = getattr(kernel_type, "logical", None)
    if not isinstance(logical, Projection):
        raise AuthoringError(f"{kernel_type.__name__} generated dataflow without logical view")

    def evaluate(*, logical: LogicalResult) -> object:
        if composite and isinstance(logical, NetworkResult):
            return logical.network
        if not composite and isinstance(logical, RegionResult):
            return logical.region
        return reject(
            "kernel-dataflow-logical-type",
            "dataflow extraction does not match the authored logical capability",
        )

    dataflow_semantics = (
        semantics_for(DATAFLOW_NETWORK_SEMANTICS)
        if composite
        else semantics_for(DATAFLOW_REGION_SEMANTICS)
    )
    dataflow_result: Derived[object] = Derived(
        dataflow_semantics,
        None,
        (("logical", cast("ValueSource[object]", logical.output)),),
        evaluate,
    )
    generated_member(kernel_type, "dataflow_result", dataflow_result, generated)
    setattr(
        kernel_type,
        "dataflow",
        Projection(
            dataflow_result,
            applicable_if=logical.applicable_if,
            readiness=logical.readiness,
            constraints=logical.constraints,
            name="dataflow",
        ),
    )
    setattr(kernel_type, GENERATED_MEMBERS, frozenset(generated))


def _region_valid(region: RegionDeclaration) -> Constraint:
    def evaluate(*, region: DataflowRegion) -> object:
        report = validate_region(region)
        if not report.issues:
            return True
        return reject_all(
            reject(
                f"kernel-region-{issue.code}",
                issue.message,
                values={"region_path": issue.path},
            )
            for issue in report.issues
        )

    return _LogicalValidityConstraint((("region", cast("ValueSource[object]", region)),), evaluate)


def _network_valid(network: ValueSource[DataflowNetwork]) -> Constraint:
    def evaluate(*, network: DataflowNetwork) -> object:
        report = validate_network(network)
        if not report.issues:
            return True
        return reject_all(
            reject(f"kernel-network-{issue.code}", issue.message, values={"path": issue.path})
            for issue in report.issues
        )

    return _LogicalValidityConstraint(
        (("network", cast("ValueSource[object]", network)),), evaluate
    )


def _leaf_logical_property(region: RegionDeclaration) -> Derived[LogicalResult]:
    def evaluate(*, region: DataflowRegion) -> LogicalResult:
        return RegionResult(region)

    return Derived(
        semantics_for(DATAFLOW_LOGICAL_RESULT_SEMANTICS),
        None,
        (("region", cast("ValueSource[object]", region)),),
        evaluate,
    )


def attach_leaf_logical(kernel_type: type[Space], generated: set[str]) -> None:
    declarations = dict(declared_members(kernel_type))
    region = declarations.get("region")
    assert isinstance(region, RegionDeclaration)
    support = declarations.get("logical_support")
    if support is not None and not isinstance(support, ConstraintGroup):
        raise AuthoringError(f"{kernel_type.__name__}.logical_support must be a ConstraintGroup")
    region_valid = cast(
        Constraint,
        use_authored_or_generated(
            kernel_type,
            "region_structurally_valid",
            _region_valid(region),
            Constraint,
            generated,
        ),
    )
    logical_accepts = cast(
        ConstraintGroup,
        use_authored_or_generated(
            kernel_type,
            "logical_accepts",
            ConstraintGroup(
                region_valid,
                *(support.constraints if isinstance(support, ConstraintGroup) else ()),
                name="logical_accepts",
            ),
            ConstraintGroup,
            generated,
        ),
    )
    authored_logical = authored_member(kernel_type, "logical")
    if authored_logical is not None and not isinstance(authored_logical, Projection):
        raise AuthoringError(f"{kernel_type.__name__}.logical must be a Projection")
    if authored_logical is None:
        logical_result = cast(
            ValueSource[LogicalResult],
            use_authored_or_generated(
                kernel_type,
                "logical_result",
                _leaf_logical_property(region),
                ValueSource,
                generated,
            ),
        )
        logical_ready = cast(
            Readiness,
            use_authored_or_generated(
                kernel_type,
                "logical_ready",
                Readiness(properties=(logical_result,), constraints=logical_accepts),
                Readiness,
                generated,
            ),
        )
        generated_member(
            kernel_type,
            "logical",
            LogicalView(logical_result, readiness=logical_ready, constraints=logical_accepts),
            generated,
        )
    authored_dataflow = authored_member(kernel_type, "dataflow")
    if authored_dataflow is not None and not isinstance(authored_dataflow, Projection):
        raise AuthoringError(f"{kernel_type.__name__}.dataflow must be a Projection")
    if authored_dataflow is None:
        dataflow_ready = cast(
            Readiness,
            use_authored_or_generated(
                kernel_type,
                "dataflow_ready",
                Readiness(properties=(region,), constraints=logical_accepts),
                Readiness,
                generated,
            ),
        )
        generated_member(
            kernel_type,
            "dataflow",
            Projection(
                region,
                readiness=dataflow_ready,
                constraints=logical_accepts,
                name="dataflow",
            ),
            generated,
        )


def attach_composite_logical(kernel_type: type[Space], generated: set[str]) -> None:
    declarations = dict(declared_members(kernel_type))
    support = declarations.get("logical_support")
    if support is not None and not isinstance(support, ConstraintGroup):
        raise AuthoringError(f"{kernel_type.__name__}.logical_support must be a ConstraintGroup")
    authored_logical = authored_member(kernel_type, "logical")
    if authored_logical is not None and not isinstance(authored_logical, Projection):
        raise AuthoringError(f"{kernel_type.__name__}.logical must be a Projection")
    if authored_logical is None:
        logical_result = cast(
            ValueSource[LogicalResult],
            use_authored_or_generated(
                kernel_type,
                "logical_result",
                composite_logical_property(cast("type", kernel_type)),
                ValueSource,
                generated,
            ),
        )
    else:
        logical_result = cast("ValueSource[LogicalResult]", authored_logical.output)
    network = cast(
        ValueSource[DataflowNetwork],
        use_authored_or_generated(
            kernel_type,
            "network",
            network_property(logical_result),
            ValueSource,
            generated,
        ),
    )
    network_valid = cast(
        Constraint,
        use_authored_or_generated(
            kernel_type,
            "network_structurally_valid",
            _network_valid(network),
            Constraint,
            generated,
        ),
    )
    logical_accepts = cast(
        ConstraintGroup,
        use_authored_or_generated(
            kernel_type,
            "logical_accepts",
            ConstraintGroup(
                network_valid,
                *(support.constraints if isinstance(support, ConstraintGroup) else ()),
                name="logical_accepts",
            ),
            ConstraintGroup,
            generated,
        ),
    )
    if authored_logical is None:
        logical_ready = cast(
            Readiness,
            use_authored_or_generated(
                kernel_type,
                "logical_ready",
                Readiness(properties=(logical_result,), constraints=logical_accepts),
                Readiness,
                generated,
            ),
        )
        generated_member(
            kernel_type,
            "logical",
            LogicalView(logical_result, readiness=logical_ready, constraints=logical_accepts),
            generated,
        )
    authored_dataflow = authored_member(kernel_type, "dataflow")
    if authored_dataflow is not None and not isinstance(authored_dataflow, Projection):
        raise AuthoringError(f"{kernel_type.__name__}.dataflow must be a Projection")
    if authored_dataflow is None:
        dataflow_ready = cast(
            Readiness,
            use_authored_or_generated(
                kernel_type,
                "dataflow_ready",
                Readiness(properties=(network,), constraints=logical_accepts),
                Readiness,
                generated,
            ),
        )
        generated_member(
            kernel_type,
            "dataflow",
            Projection(
                network,
                readiness=dataflow_ready,
                constraints=logical_accepts,
                name="dataflow",
            ),
            generated,
        )


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
