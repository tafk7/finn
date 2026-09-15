# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Checked source-Op physical capture, preparation and component association.

Logical/source evidence stays in these per-use values. Reusable module inputs
contain no operation, model, logical graph or source occurrence identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakValueDictionary

from finn.dataflow._engine import Decided, Finding, FindingKind, QualifiedPath
from finn.dataflow.artifacts.build import (
    BlobSink,
    ModuleBuildRequirements,
    PreparedModuleBuild,
    module_build_fingerprint,
    module_source_derivation,
    portable_module_component,
    prepare_module_build,
    prepared_module_fingerprint,
)
from finn.dataflow.artifacts.packaging import PortableComponent
from finn.dataflow.artifacts.store import ArtifactStore
from finn.dataflow.ops.base import DataflowOpError
from finn.dataflow.space.occurrence import ProjectionAssessment

if TYPE_CHECKING:
    from finn.dataflow.designs.physical import BoundaryBinding, EdgeBinding, SemanticPortBinding  # noqa: PLC0415
    from finn.dataflow.ops.base import DataflowOp  # noqa: PLC0415
    from finn.dataflow.ops.graph_context import AcceptedLogicalCapture, GraphContext  # noqa: PLC0415


@dataclass(frozen=True)
class PhysicalBuildAssociation:
    logical: AcceptedLogicalCapture
    requirements_fingerprint: str
    port_bindings: tuple[SemanticPortBinding, ...]
    boundary_bindings: tuple[BoundaryBinding, ...]
    edge_bindings: tuple[EdgeBinding, ...]

    def __post_init__(self) -> None:
        for field in ("port_bindings", "boundary_bindings", "edge_bindings"):
            object.__setattr__(self, field, tuple(getattr(self, field)))


@dataclass(frozen=True)
class PhysicalBuildCapture:
    requirements: ModuleBuildRequirements
    association: PhysicalBuildAssociation

    def __post_init__(self) -> None:
        if not isinstance(self.requirements, ModuleBuildRequirements):
            raise TypeError("physical capture requires model-free module requirements")
        if not isinstance(self.association, PhysicalBuildAssociation):
            raise TypeError("physical capture requires its logical association")


@dataclass(frozen=True)
class PreparationReceipt:
    requirements_fingerprint: str
    prepared_fingerprint: str


# Receipts are transient service evidence, not persisted acceptance or cache
# identity. A pair enters this weak table only after actual checked preparation.
# Equal reconstituted values are allowed while the issuing request is alive;
# inventing a capture-A/prepared-B pair cannot manufacture preparation evidence.
_issued_receipts: WeakValueDictionary[tuple[str, str], PreparationReceipt] = WeakValueDictionary()


@dataclass(frozen=True)
class PreparedBuildRequest:
    capture: PhysicalBuildCapture
    prepared: PreparedModuleBuild
    receipt: PreparationReceipt


@dataclass(frozen=True)
class PhysicalInstanceAssociation:
    outer_instance_id: str
    component: PortableComponent
    association: PhysicalBuildAssociation
    prepared_fingerprint: str

    def __post_init__(self) -> None:
        if not self.outer_instance_id:
            raise ValueError("a physical instance requires an outer identity")


def _rejection(code: str, message: str) -> Finding:
    return Finding(FindingKind.REJECTION, code, QualifiedPath("physical"), message)


def op_physical(operation: DataflowOp) -> ProjectionAssessment[PhysicalBuildCapture]:
    """Ask only the selected Design, after frozen graph/source acceptance."""
    from finn.dataflow.designs.design import DataflowDesign  # noqa: PLC0415
    from finn.dataflow.ops.graph_context import capture_frozen_op_logical  # noqa: PLC0415

    graph = operation.graph_dataflow
    if not isinstance(graph.accepted_answer, Decided):
        return ProjectionAssessment(
            "physical",
            graph.readiness,
            graph.constraints,
            cast(Any, graph.accepted_answer),
            cast(Any, graph.accepted_answer),
        )
    design = operation.selected_design()
    if not isinstance(design, DataflowDesign):
        raise TypeError("selected_design must return the selected DataflowDesign occurrence")
    if design.root is not operation.root:
        raise DataflowOpError("selected physical Design belongs to a different operation point")
    selected_logical = design.dataflow.accepted_answer
    if (
        not isinstance(selected_logical, Decided)
        or selected_logical.value != graph.accepted_answer.value
    ):
        raise DataflowOpError(
            "selected physical Design does not realize the accepted operation Network"
        )
    physical = design.physical
    if not isinstance(physical.accepted_answer, Decided):
        return ProjectionAssessment(
            "physical",
            physical.readiness,
            (*graph.constraints, *physical.constraints),
            cast(Any, physical.output),
            cast(Any, physical.accepted_answer),
        )
    facts = physical.accepted_answer.value
    logical = capture_frozen_op_logical(operation)
    capture = PhysicalBuildCapture(
        facts.requirements,
        PhysicalBuildAssociation(
            logical,
            module_build_fingerprint(facts.requirements),
            facts.port_bindings,
            facts.boundary_bindings,
            facts.edge_bindings,
        ),
    )
    return ProjectionAssessment(
        "physical",
        physical.readiness,
        (*graph.constraints, *physical.constraints),
        Decided(capture),
        Decided(capture),
    )


def capture_op_physical(operation: DataflowOp) -> PhysicalBuildCapture:
    answer = operation.physical.accepted_answer
    if not isinstance(answer, Decided):
        raise DataflowOpError("operation physical projection is not accepted", answer.findings)
    return cast(PhysicalBuildCapture, answer.value)


def validate_physical_build_association(
    operation: DataflowOp,
    capture: PhysicalBuildCapture,
    *,
    model: Any,
    build: object,
    graph_context: GraphContext,
) -> tuple[Finding, ...]:
    from finn.dataflow.ops.graph_context import validate_frozen_op_logical  # noqa: PLC0415

    findings = validate_frozen_op_logical(
        operation,
        capture.association.logical,
        model=model,
        build=build,
        graph_context=graph_context,
    )
    if findings:
        return findings
    if (
        module_build_fingerprint(capture.requirements)
        != capture.association.requirements_fingerprint
    ):
        return (
            _rejection(
                "physical-requirements-mismatch",
                "requirements differ from the captured association",
            ),
        )
    answer = operation.physical.accepted_answer
    if not isinstance(answer, Decided):
        return cast("tuple[Finding, ...]", answer.findings)
    if answer.value != capture:
        return (
            _rejection(
                "physical-capture-mismatch",
                "capture is not the operation's same-point physical result",
            ),
        )
    return ()


def _require_valid(
    operation: DataflowOp,
    capture: PhysicalBuildCapture,
    *,
    model: Any,
    build: object,
    graph_context: GraphContext,
) -> None:
    findings = validate_physical_build_association(
        operation,
        capture,
        model=model,
        build=build,
        graph_context=graph_context,
    )
    if findings:
        raise DataflowOpError("physical build association is stale or invalid", findings)


def prepare_build_request(
    operation: DataflowOp,
    capture: PhysicalBuildCapture,
    *,
    model: Any,
    build: object,
    graph_context: GraphContext,
    roots: Mapping[str, Path],
    template_roots: Sequence[Path],
    blobs: BlobSink,
) -> PreparedBuildRequest:
    _require_valid(operation, capture, model=model, build=build, graph_context=graph_context)
    prepared = prepare_module_build(
        capture.requirements,
        roots=roots,
        template_roots=template_roots,
        blobs=blobs,
    )
    receipt = PreparationReceipt(
        module_build_fingerprint(capture.requirements),
        prepared_module_fingerprint(prepared),
    )
    key = (receipt.requirements_fingerprint, receipt.prepared_fingerprint)
    receipt = _issued_receipts.setdefault(key, receipt)
    return PreparedBuildRequest(capture, prepared, receipt)


def _validate_receipt(request: PreparedBuildRequest) -> None:
    expected = PreparationReceipt(
        module_build_fingerprint(request.capture.requirements),
        prepared_module_fingerprint(request.prepared),
    )
    if (
        request.receipt != expected
        or _issued_receipts.get((expected.requirements_fingerprint, expected.prepared_fingerprint))
        != request.receipt
    ):
        raise DataflowOpError(
            "request is not the requirements/prepared pair issued by checked preparation"
        )


def materialize_build_request(
    operation: DataflowOp,
    request: PreparedBuildRequest,
    *,
    model: Any,
    build: object,
    graph_context: GraphContext,
    store: ArtifactStore,
) -> PortableComponent:
    """Revalidate the occurrence immediately before source cache lookup/build."""
    from finn.dataflow.artifacts.build import materialize_module_sources  # noqa: PLC0415

    _require_valid(
        operation, request.capture, model=model, build=build, graph_context=graph_context
    )
    _validate_receipt(request)
    source = materialize_module_sources(request.prepared, store)
    return portable_module_component(request.prepared, source)


def install_physical_component(
    operation: DataflowOp,
    request: PreparedBuildRequest,
    component: PortableComponent,
    *,
    outer_instance_id: str,
    model: Any,
    build: object,
    graph_context: GraphContext,
    store: ArtifactStore,
) -> PhysicalInstanceAssociation:
    _require_valid(
        operation, request.capture, model=model, build=build, graph_context=graph_context
    )
    _validate_receipt(request)
    source = store.lookup(module_source_derivation(request.prepared))
    if source is None:
        raise DataflowOpError("prepared source artifact is not present in the supplied store")
    expected = portable_module_component(request.prepared, source)
    if component != expected:
        raise DataflowOpError("component differs from the store-verified prepared source and ABI")
    return PhysicalInstanceAssociation(
        outer_instance_id,
        component,
        request.capture.association,
        request.receipt.prepared_fingerprint,
    )
