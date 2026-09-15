# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Model-level, in-memory source analysis and operation reconstruction.

Use source_analysis for read passes on one ModelWrapper. For inference, use
ops.inference's pass owners: an outer context cannot cover a wrapper that
ModelWrapper.transform copies or preprocesses before callbacks. bind_operations
and analyze_sources establish their own context. Nested readers consult its
single result; no result is serialized or retained on the model. A later pass
always observes current initializer contents.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from qonnx.analysis.tensor_value_summary import TensorValueSummary  # type: ignore[import-not-found]

from finn.dataflow.ops.tensor_summary import FrozenInitializer

if TYPE_CHECKING:
    from finn.dataflow.ops.base import DataflowOp
    from finn.dataflow.ops.source import SourceNode


@dataclass(frozen=True, slots=True)
class SourceAnalysis(Mapping[str, TensorValueSummary]):
    summaries: Mapping[str, TensorValueSummary]
    initializers: Mapping[str, FrozenInitializer]

    def __getitem__(self, key: str) -> TensorValueSummary:
        return self.summaries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.summaries)

    def __len__(self) -> int:
        return len(self.summaries)


_active: ContextVar[tuple[Any, SourceAnalysis] | None] = ContextVar(
    "dataflow_source_analysis", default=None
)


def initializer_facts(model: Any) -> SourceAnalysis:
    """Capture every initializer's summary and detached payload in one traversal."""

    summaries: dict[str, TensorValueSummary] = {}
    initializers: dict[str, FrozenInitializer] = {}
    for tensor in model.graph.initializer:
        # Conversion happens once inside this constructor. The resulting
        # summary is the same QONNX content identity used by source facts.
        frozen = FrozenInitializer.from_tensor_proto(tensor)
        summaries[tensor.name] = frozen.summary
        initializers[tensor.name] = frozen
    return SourceAnalysis(MappingProxyType(summaries), MappingProxyType(initializers))


@contextmanager
def source_analysis(model: Any, *, fresh: bool = False) -> Iterator[SourceAnalysis]:
    """Analyze every initializer once for this pass, including unused ones."""

    current = _active.get()
    if not fresh and current is not None and current[0] is model:
        yield current[1]
        return
    summaries = initializer_facts(model)
    token = _active.set((model, summaries))
    try:
        yield summaries
    finally:
        _active.reset(token)


def _operations(model: Any, operations: Sequence[DataflowOp] | None) -> Sequence[DataflowOp]:
    if operations is not None:
        return operations
    from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp  # noqa: PLC0415

    result = []
    for node in model.graph.node:
        if node.domain == DATAFLOW_DOMAIN:
            operation = model.get_customop_wrapper(node)
            if isinstance(operation, DataflowOp):
                result.append(operation)
    return result


def bind_operations(
    model: Any,
    build: Any,
    *,
    operations: Sequence[DataflowOp] | None = None,
    graph_context: Any = None,
) -> tuple[DataflowOp, ...]:
    """Reconstruct all selected operations against one initializer analysis."""

    with source_analysis(model) as summaries:
        result = []
        for op in _operations(model, operations):
            context_read = None
            if graph_context is not None:
                from finn.dataflow.ops.graph_context import require_context_read  # noqa: PLC0415

                scope = op.recorded_scope_id()
                if not scope:
                    from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

                    raise DataflowOpError(
                        f"{op.onnx_node.name!r} has no dataflow scope id; "
                        "run AssignDataflowScopeIds first"
                    )
                context_read = require_context_read(
                    graph_context,
                    model,
                    build,
                    consumer_scope_id=scope,
                )
            result.append(
                op._bind_with(
                    model,
                    op._build_values(build),
                    summaries,
                    context_read=context_read,
                )
            )
        return tuple(result)


def bind_sources_only(
    model: Any, build: Any, *, operations: Sequence[DataflowOp] | None = None
) -> tuple[DataflowOp, ...]:
    """Bind current source facts without hydrating any persisted choices.

    This deliberately narrow entry point exists for explicit legacy migration.
    Ordinary bind/rebind remains strict and continues to reject old schemas.
    """

    with source_analysis(model) as summaries:
        return tuple(
            op._bind_with(model, op._build_values(build), summaries, recorded=False)
            for op in _operations(model, operations)
        )


def rebind_selected_graph(
    operation: DataflowOp,
    snapshot: Any,
    *,
    constructions: Any = None,
    update_origin: bool = True,
) -> Any:
    """Validate a detached selected artifact against one current source occurrence."""

    from finn.dataflow._engine import Decided  # noqa: PLC0415
    from finn.dataflow.designs.design import SelectedGraph  # noqa: PLC0415
    from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415
    from finn.dataflow.ops.native import (  # noqa: PLC0415
        choice_schema,
        choice_subset,
        resolve_choice_subset,
    )
    from finn.dataflow.ops.selected import (  # noqa: PLC0415
        ConstructionIdentity,
        EncodedSourceSemantics,
        RecordedChoice,
        SelectionFacts,
        SourceProvenance,
        build_selected_snapshot,
        decode_selected_graph,
        encode_selected_choices,
    )
    from finn.dataflow.ops.selected_registry import (  # noqa: PLC0415
        DEFAULT_SELECTED_CONSTRUCTIONS,
    )
    from finn.dataflow.model.network_validation import validate_network  # noqa: PLC0415

    registry = DEFAULT_SELECTED_CONSTRUCTIONS if constructions is None else constructions
    decoded = decode_selected_graph(snapshot, constructions=registry)
    if not isinstance(operation.dataflow.accepted_answer, Decided):
        raise DataflowOpError("current source has no accepted dataflow projection")
    selected_design = operation.selected_design()
    declaration = getattr(type(selected_design), "selected_graph", None)
    if not isinstance(declaration, SelectedGraph):
        raise DataflowOpError("current source Design has no selected construction")
    current_construction = declaration.construction
    if (current_construction.family, current_construction.version) != (
        decoded.declaration.construction.family,
        decoded.declaration.construction.version,
    ):
        raise DataflowOpError("selected construction family or version changed")
    artifact_identity = decoded.declaration.construction
    if artifact_identity.form not in current_construction.admitted_forms:
        raise DataflowOpError("current selected construction does not admit the artifact form")

    encoded = operation.selected_source_semantics()
    if not isinstance(encoded, EncodedSourceSemantics):
        raise DataflowOpError("current source returned invalid selected semantics")
    if (
        encoded.identity != current_construction.source_semantics_identity
        or encoded.version != current_construction.source_semantics_version
    ):
        raise DataflowOpError("current source-semantics codec differs from the construction")
    current_source = operation.selected_source_provenance(encoded)
    if not isinstance(current_source, SourceProvenance):
        raise DataflowOpError("current source returned invalid selected provenance")
    intrinsic_current = (
        current_source.family,
        current_source.family_version,
        current_source.operands,
        current_source.semantics,
    )
    intrinsic_selected = (
        decoded.declaration.source.family,
        decoded.declaration.source.family_version,
        decoded.declaration.source.operands,
        decoded.declaration.source.semantics,
    )
    if intrinsic_current != intrinsic_selected:
        raise DataflowOpError("selected artifact source semantics differ from current source")

    schema = choice_subset(choice_schema(operation), current_construction.choice_paths)
    selected_choices = []
    for item, answer in resolve_choice_subset(operation, schema):
        if not isinstance(answer, Decided):
            raise DataflowOpError(f"current source choice {item.choice.path!r} is unresolved")
        selected_choices.append(RecordedChoice(item.choice.path, answer.value))
    selected_choices = list(encode_selected_choices(schema, tuple(selected_choices)))
    if tuple(selected_choices) != decoded.selection_facts.choices:
        raise DataflowOpError("selected artifact logical choices differ from current source")

    semantics = current_construction.decode_source_semantics(encoded)
    current_identity = operation.selected_construction_identity(semantics)
    if not isinstance(current_identity, ConstructionIdentity) or (
        current_identity.family,
        current_identity.version,
    ) != (artifact_identity.family, artifact_identity.version):
        raise DataflowOpError("current source resolves a different selected construction")
    facts = current_construction.derive_facts(
        artifact_identity,
        current_source,
        semantics,
        tuple(selected_choices),
    )
    if not isinstance(facts, SelectionFacts):
        raise DataflowOpError("current selected construction returned invalid facts")
    verification_facts = replace(facts, source=decoded.declaration.source)
    try:
        projected = current_construction.project(verification_facts)
    except (TypeError, ValueError, KeyError) as error:
        raise DataflowOpError(
            f"current selected construction rejects the artifact form: {error}"
        ) from error
    report = validate_network(projected)
    if report.issues:
        issue = report.issues[0]
        raise DataflowOpError(
            f"current selected construction projects an invalid Network: "
            f"{issue.path}: {issue.message} [{issue.code}]"
        )
    if projected != decoded.network:
        raise DataflowOpError("current selected construction projects a different artifact Network")
    findings = current_construction.verify(snapshot, verification_facts)
    if findings:
        first = findings[0]
        raise DataflowOpError(
            f"current selected construction rejects the artifact graph: "
            f"{first.path}: {first.message} [{first.code}]"
        )
    if (
        facts.source_semantics != decoded.selection_facts.source_semantics
        or facts.choices != decoded.selection_facts.choices
        or facts.parameters != decoded.selection_facts.parameters
        or facts.selection_fingerprint != decoded.selection_facts.selection_fingerprint
    ):
        raise DataflowOpError("selected artifact facts differ from current source")
    if not update_origin or current_source.origin == decoded.declaration.source.origin:
        return decoded
    rebound = build_selected_snapshot(
        decoded.snapshot.model_copy(),
        replace(decoded.declaration, source=current_source),
    )
    return decode_selected_graph(rebound, constructions=registry)


def analyze_sources(model: Any) -> tuple[SourceNode, ...]:
    """Read the logical sources of the model without a build or hydration."""

    with source_analysis(model) as summaries:
        return tuple(
            op._read_source(model, op._live_node(model), summaries)
            for op in _operations(model, None)
        )


__all__ = [
    "SourceAnalysis",
    "analyze_sources",
    "bind_operations",
    "bind_sources_only",
    "initializer_facts",
    "rebind_selected_graph",
    "source_analysis",
]
