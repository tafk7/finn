# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional automatic leaf/composite View authoring for conventional Kernels."""

from __future__ import annotations

from typing import cast

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

from finn.dataflow.model.logical.view import LogicalView, _LogicalValidityConstraint


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


__all__ = ["attach_leaf_logical", "attach_composite_logical", "synchronize_generated_dataflow"]
