# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Source-free capture of accepted logical/physical correspondence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from finn.dataflow._engine import Decided
from finn.dataflow.model.identity import ImplementationIdentity
from finn.dataflow.model.logical.composition import ImplementationPath, NetworkResult
from finn.dataflow.model.physical.capture import (
    LocalPhysicalCapture,
    PhysicalCaptureError,
    capture_local_physical,
)
from finn.dataflow.model.relations.values import LogicalPhysicalRelation
from finn.dataflow.space.declarations import Space
from finn.dataflow.space.occurrence import ProjectionAssessment


@dataclass(frozen=True)
class LocalRelationCapture:
    """A separately requested logical/physical correspondence claim."""

    implementation: ImplementationIdentity
    occurrence_path: ImplementationPath
    occurrence_token: int
    physical_point_fingerprint: str
    relation_fingerprint: str
    logical_fingerprint: str
    relation: object


def _digest(*values: object) -> str:
    return hashlib.sha256("\n".join(repr(value) for value in values).encode("utf-8")).hexdigest()


def capture_local_relation(
    implementation: object, physical: LocalPhysicalCapture
) -> LocalRelationCapture:
    if not isinstance(implementation, Space):
        raise TypeError("local relation capture requires a Space occurrence")
    if capture_local_physical(implementation) != physical:
        raise PhysicalCaptureError("physical capture belongs to a different projected point")
    assessment: ProjectionAssessment[Any] = implementation.assess_view("physical_relation")
    if not isinstance(assessment.accepted_answer, Decided):
        raise PhysicalCaptureError(
            "logical/physical relation is not accepted",
            tuple(assessment.accepted_answer.findings),
        )
    relation = assessment.accepted_answer.value
    if not isinstance(relation, LogicalPhysicalRelation):
        raise TypeError("physical relation capability returned the wrong value")
    if relation.physical.requirements != physical.requirements:
        raise PhysicalCaptureError("physical relation names different local requirements")
    logical: ProjectionAssessment[Any] = implementation.assess_view("logical")
    if not isinstance(logical.accepted_answer, Decided):
        raise PhysicalCaptureError(
            "logical capability is not accepted for relation capture",
            tuple(logical.accepted_answer.findings),
        )
    logical_value = logical.accepted_answer.value
    logical_network = (
        logical_value.network if isinstance(logical_value, NetworkResult) else logical_value
    )
    if relation.network != logical_network:
        raise PhysicalCaptureError("physical relation Network differs from the logical capability")
    logical_fingerprint = _digest(relation.network)
    relation_fingerprint = _digest(
        physical.point_fingerprint,
        logical_fingerprint,
        relation.physical.port_bindings,
        relation.physical.boundary_bindings,
        relation.physical.edge_bindings,
    )
    return LocalRelationCapture(
        physical.implementation,
        physical.occurrence_path,
        physical.occurrence_token,
        physical.point_fingerprint,
        relation_fingerprint,
        logical_fingerprint,
        relation,
    )


__all__ = ["LocalRelationCapture", "capture_local_relation"]
