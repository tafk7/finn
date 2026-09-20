# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Relation-view declarations and common stream-binding authoring."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar, cast

from finn.dataflow.artifacts.abi import Bus, Endpoint
from finn.dataflow.artifacts.build import ModuleABIRequirements
from finn.dataflow.model.logical.region import DataflowRegion, element_width
from finn.dataflow.model.physical.layout import (
    FieldPlacement,
    PackedBeatLayout,
    PeriodicLast,
    UnusedBitPolicy,
    UnusedBitRange,
)
from finn.dataflow.model.relations.validation import region_ports
from finn.dataflow.model.relations.validation import validate_logical_physical_relation
from finn.dataflow.model.relations.values import (
    CompositePhysicalFacts,
    KernelStreamBinding,
    LogicalPhysicalRelation,
)
from finn.dataflow.model._authoring import (
    authored_member,
    generated_member,
    use_authored_or_generated,
)
from finn.dataflow.model.logical.composition import LogicalResult, NetworkResult
from finn.dataflow.model.physical.structure import PhysicalStructureError
from finn.dataflow.space.declarations import (
    AuthoringError,
    ConstraintGroup,
    Derived,
    Projection,
    Readiness,
    ValueSource,
    reject,
    semantics_for,
)

T_co = TypeVar("T_co", covariant=True)


class RelationView(Projection[T_co]):
    """A typed logical/physical correspondence capability."""

    def __init__(
        self,
        output: ValueSource[T_co],
        *,
        applicable_if: ValueSource[bool] | None = None,
        readiness: Readiness,
        constraints: ConstraintGroup | Sequence[ConstraintGroup] = (),
        name: str | None = "physical_relation",
    ) -> None:
        super().__init__(
            output,
            applicable_if=applicable_if,
            readiness=readiness,
            constraints=constraints,
            name=name,
        )


def low_fields_binding(
    *,
    region: DataflowRegion,
    abi: ModuleABIRequirements,
    region_port_id: str,
    abi_bus_id: str,
    framing: PeriodicLast | None = None,
) -> KernelStreamBinding:
    """Author a low-field-first layout with explicit high-padding policy."""

    port, direction = next(
        (port, side) for port, side in region_ports(region) if port.id == region_port_id
    )
    bus = next(port for port in abi.ports if isinstance(port, Bus) and port.name == abi_bus_id)
    carrier = next(member.width for member in bus.signals if member.logical == "tdata")
    scalar = element_width(port.operand.element_type)
    logical = scalar * port.beat_sequence.elements_per_beat
    if logical > carrier:
        raise ValueError("logical payload exceeds physical carrier")
    return KernelStreamBinding(
        region_port_id,
        abi_bus_id,
        PackedBeatLayout(
            tuple(
                FieldPlacement(index, index * scalar, scalar)
                for index in range(port.beat_sequence.elements_per_beat)
            ),
            ()
            if logical == carrier
            else (
                UnusedBitRange(
                    logical,
                    carrier - logical,
                    UnusedBitPolicy.IGNORE_ON_RECEIVE
                    if direction is Endpoint.TARGET
                    else UnusedBitPolicy.DRIVE_ZERO,
                ),
            ),
        ),
        framing,
    )


def _composite_relation_property(
    logical: ValueSource[LogicalResult], physical: ValueSource[object]
) -> Derived[object]:
    def evaluate(*, logical: LogicalResult, physical: object) -> object:
        if not isinstance(logical, NetworkResult):
            return reject(
                "kernel-relation-logical-type",
                "a composite relation requires a NetworkResult logical capability",
            )
        if not isinstance(physical, CompositePhysicalFacts):
            return reject(
                "kernel-relation-physical-type",
                "a composite relation requires CompositePhysicalFacts",
            )
        try:
            if physical.structure is None:
                raise PhysicalStructureError(
                    "the local physical result carries no relation-validation structure"
                )
            validate_logical_physical_relation(logical.network, physical.structure, physical)
            return LogicalPhysicalRelation(logical.network, physical)
        except (TypeError, ValueError) as error:
            return reject("kernel-physical-relation-refused", str(error))

    return Derived(
        semantics_for(LogicalPhysicalRelation),
        None,
        (("logical", cast("ValueSource[object]", logical)), ("physical", physical)),
        evaluate,
    )


def attach_composite_relation(kernel_type: type[object], generated: set[str]) -> None:
    authored = authored_member(kernel_type, "physical_relation")
    if authored is not None and not isinstance(authored, Projection):
        raise AuthoringError(f"{kernel_type.__name__}.physical_relation must be a Projection")
    if authored is not None:
        return
    logical_view = cast(Projection[object], getattr(kernel_type, "logical"))
    physical_view = cast(Projection[object], getattr(kernel_type, "physical"))
    logical_result = cast("ValueSource[LogicalResult]", logical_view.output)
    physical_result = physical_view.output
    relation_result = cast(
        ValueSource[object],
        use_authored_or_generated(
            kernel_type,
            "relation_result",
            _composite_relation_property(logical_result, physical_result),
            ValueSource,
            generated,
        ),
    )
    relation_ready = cast(
        Readiness,
        use_authored_or_generated(
            kernel_type,
            "relation_ready",
            Readiness(
                decisions=tuple(
                    dict.fromkeys(
                        (*logical_view.readiness.decisions, *physical_view.readiness.decisions)
                    )
                ),
                properties=tuple(
                    dict.fromkeys(
                        (
                            relation_result,
                            *logical_view.readiness.properties,
                            *physical_view.readiness.properties,
                        )
                    )
                ),
                constraints=tuple(
                    dict.fromkeys(
                        (*logical_view.readiness.constraints, *physical_view.readiness.constraints)
                    )
                ),
            ),
            Readiness,
            generated,
        ),
    )
    generated_member(
        kernel_type,
        "physical_relation",
        RelationView(
            relation_result,
            readiness=relation_ready,
            constraints=tuple(
                dict.fromkeys((*logical_view.constraints, *physical_view.constraints))
            ),
        ),
        generated,
    )


__all__ = ["RelationView", "low_fields_binding"]
