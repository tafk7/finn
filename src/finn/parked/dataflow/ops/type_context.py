# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Fresh producer type queries for graph inputs and checked annotation updates.

Graph datatype annotations on DataflowOp outputs are derived caches. Reading a
consumer input therefore asks its producer's public contract, even when an old
annotation is present. This service owns no persistent cache and never mutates
the graph during a query.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast

from finn.kernels._engine import Answer, Finding, FindingKind, QualifiedPath, Unresolved
from finn.dataflow.datatypes import QONNXDataType

if TYPE_CHECKING:
    from finn.parked.dataflow.ops.model_effects import ModelReadSet
    from finn.parked.dataflow.ops.space import DataflowSpace


_active_types: ContextVar[tuple[tuple[int, str], ...]] = ContextVar(
    "dataflow_producer_type_queries", default=()
)


def _unavailable(code: str, message: str) -> Unresolved:
    return Unresolved(
        (Finding(FindingKind.LIMITATION, code, QualifiedPath("op.producer_type"), message),)
    )


def producer_type(
    model: Any, tensor: str, build: object | None = None
) -> Answer[QONNXDataType] | None:
    return producer_type_facts(model, tensor, build)[0]


def _type_choice_reads(operation: DataflowSpace, source_name: str) -> set[str]:
    """Native keys consumed by target resolution and the accepted type query."""
    from finn.kernels._engine import Decided, DependencyRef  # noqa: PLC0415
    from finn.parked.dataflow.model.logical.interface_authoring import operand_declaration  # noqa: PLC0415
    from finn.parked.dataflow.model.physical.capture import capture_assessment_dependencies  # noqa: PLC0415
    from finn.parked.dataflow.ops.binding import ImplementationBinding, OperandBinding  # noqa: PLC0415
    from finn.parked.dataflow.ops.native import choice_schema  # noqa: PLC0415
    from finn.kernels.space.compiler import _Ref, resolve_value_source  # noqa: PLC0415
    from finn.kernels.space.declarations import (  # noqa: PLC0415
        ConstraintGroup,
        Projection,
        Space,
        Subspace,
        SubspaceChoice,
    )
    from finn.kernels.space.occurrence import (  # noqa: PLC0415
        layer_runtime,
        occurrence_answer_at,
        occurrence_child,
        occurrence_choice,
    )

    route = cast(
        ImplementationBinding,
        getattr(type(operation), "interface_binding", None)
        or getattr(type(operation), "implementation_binding"),
    )
    binding = cast(
        OperandBinding,
        next(item for item in operation.operand_bindings if item.source == source_name),
    )
    paths: set[str] = set()

    def capture(owner: Space, subject: Projection[Any] | ConstraintGroup | DependencyRef) -> None:
        paths.update(
            item.path
            for item in capture_assessment_dependencies(owner, subject)
            if item.kind == "decision"
        )

    def capture_reference(owner: Space, reference: _Ref[Any]) -> None:
        capture(
            owner, DependencyRef("type-route", reference.path, reference.kind, reference.semantics)
        )

    current: Space = operation
    members = iter(route.members)
    resolved = True
    # Match ImplementationBinding.resolve's order: fixed-child guard, branch
    # guard/selector, explicit alternative, then the selected alternative guard.
    # An unresolved/inactive route consumes no candidate datatype or source group.
    for name in members:
        declaration = getattr(type(current), name)
        runtime = layer_runtime(current)
        if isinstance(declaration, Subspace):
            if declaration.when is not None:
                reference = resolve_value_source(runtime.compiled, declaration.when, "type route")
                capture_reference(current, reference)
                active = occurrence_answer_at(current, reference)
                if not isinstance(active, Decided) or not active.value:
                    resolved = False
                    break
            current = occurrence_child(current, declaration)
        elif isinstance(declaration, SubspaceChoice):
            branch = runtime.compiled.branch(name)
            if branch.active is not None:
                capture_reference(current, branch.active)
                active = occurrence_answer_at(current, branch.active)
                if not isinstance(active, Decided) or not active.value:
                    resolved = False
                    break
            if branch.selector is not None:
                capture_reference(current, branch.selector)
            choice = occurrence_choice(current, declaration)
            selected = choice.selected()
            if not isinstance(selected, Decided):
                resolved = False
                break
            explicit = next(members, None)
            if explicit is not None and explicit != selected.value:
                resolved = False
                break
            alternative = dict(declaration.alternatives)[selected.value]
            if alternative.when is not None:
                reference = resolve_value_source(runtime.compiled, alternative.when, "type route")
                capture_reference(current, reference)
                active = occurrence_answer_at(current, reference)
                if not isinstance(active, Decided) or not active.value:
                    resolved = False
                    break
            current = choice.alternative(selected.value)
        else:
            raise TypeError("type route does not name an authored child")
    if resolved:
        capture(current, operand_declaration(current, binding.role).datatype)
        if binding.output:
            capture(operation, type(operation).type_source_accepts)
    return {
        entry.name
        for entry in choice_schema(operation)
        if entry.choice.reference.path.value in paths
    }


def producer_type_facts(
    model: Any, tensor: str, build: object | None = None
) -> tuple[Answer[QONNXDataType] | None, ModelReadSet | None]:
    """Resolve a Dataflow producer's current type, independently of annotations.

    ``None`` means this value has no Dataflow producer and its external annotation
    remains an input fact. An unavailable producer returns an Answer, never None
    or a carrier-type fallback. Recursion is scoped to this synchronous query.
    """

    from finn.parked.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp, DataflowOpError  # noqa: PLC0415
    from finn.parked.dataflow.ops.space import source_declarations  # noqa: PLC0415
    from finn.parked.dataflow.ops.schema import OpOutput  # noqa: PLC0415

    producers = tuple(node for node in model.graph.node if tensor in node.output)
    if not any(node.domain == DATAFLOW_DOMAIN for node in producers):
        return None, None
    if len(producers) != 1:
        raise DataflowOpError(
            f"graph tensor {tensor!r} requires exactly one producer, found {len(producers)}"
        )
    node = producers[0]
    active = _active_types.get()
    key = (id(model), tensor)
    if key in active:
        return _unavailable(
            "producer-type-cycle", f"producer type dependencies cycle through {tensor!r}"
        ), None
    token = _active_types.set((*active, key))
    try:
        operation = model.get_customop_wrapper(node)
        if not isinstance(operation, DataflowOp):
            return _unavailable(
                "producer-type-definition-missing",
                f"no DataflowOp definition interprets producer of {tensor!r}",
            ), None
        outputs = tuple(
            name
            for name, declaration in source_declarations(operation.space_type)
            if isinstance(declaration, OpOutput)
            and declaration.index < len(node.output)
            and node.output[declaration.index] == tensor
        )
        if len(outputs) != 1:
            return _unavailable(
                "producer-type-output-ambiguous",
                f"producer does not declare exactly one public output for {tensor!r}",
            ), None
        from finn.parked.dataflow.ops.persistence import source_read_set  # noqa: PLC0415

        if build is not None:
            operation.set_context(build=build)
        current = operation.space
        answer = cast("Answer[QONNXDataType]", current.operand_type(outputs[0]))
        # Source type rules can consume semantic attributes and authenticated
        # initializer contents. Include their frozen reads recursively, without
        # promoting cached producer output annotations into dependencies.
        from finn.parked.dataflow.ops.native import SCHEMA_VERSION_ATTRIBUTE  # noqa: PLC0415

        names = _type_choice_reads(current, outputs[0])
        # Schema presence/meaning controls native interpretation even at a partial
        # point. Missing consumed keys are recorded as absent by source_read_set.
        names.add(SCHEMA_VERSION_ATTRIBUTE)
        return answer, source_read_set(
            current,
            expected_attributes={name: None for name in names},
            include_output_annotations=False,
            node_output_address=tensor,
        )
    finally:
        _active_types.reset(token)


__all__ = ["producer_type", "producer_type_facts"]
