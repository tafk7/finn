# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Transactional native Decision writes with narrow source preconditions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from finn.dataflow._engine import Absent, Decided, Unresolved
from finn.dataflow.space.occurrence import ProjectionAssessment
from finn.dataflow.ops.base import SCOPE_ID_ATTRIBUTE, DataflowOp, DataflowOpError
from finn.dataflow.space.declarations import Problem
from finn.dataflow.ops.native import NativeAttribute
from finn.dataflow.ops.reconstruction import source_analysis


class CommitmentStage(Enum):
    """How far a caller is freezing the design when they write it down.

    ``Unresolved`` is allowed at every stage: a deliberately partial point is a
    legitimate thing to save, and refusing it would mean the only way to record
    progress is to finish.  A *final rejection* in a required projection is
    not, because a refused point recorded as a choice is a build that fails
    much later with nothing pointing back here.
    """

    #: The semantic frontier: the Network must not be refused.
    DATAFLOW = "dataflow"
    #: The semantic and physical frontiers both.  Available from U6.
    PHYSICAL = "physical"


@dataclass(frozen=True, slots=True)
class GraphEffects:
    """One complete, checked change to one dataflow node.

    Deliberately without reserved structural fields.  Node insertion, tensor
    creation and the rest are a later extension, and an empty ``inserts=()``
    sitting here now would read as "supported, none needed" rather than "not
    built yet".
    """

    scope_id: str
    commitment_stage: CommitmentStage
    expected_source_fingerprint: str
    expected_attributes: Mapping[str, bytes | None]
    remove_attributes: tuple[str, ...]
    set_attributes: Mapping[str, NativeAttribute]
    operation_type: type[DataflowOp]
    opset_version: int
    build_facts: Mapping[Problem[Any], object]
    expected_operator: tuple[str, str]
    expected_outputs: tuple[str, ...]
    tensor_datatypes: Mapping[str, Any] = field(default_factory=dict)
    tensor_shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    invalidates: tuple[str, ...] = ()


def allocate_scope_id() -> str:
    return f"dataflow_{uuid4().hex}"


def find_node(model: Any, scope_id: str) -> Any:
    """The one live node carrying this scope id.

    By scope id, not by object reference and not by name.  A plan may outlive
    the wrapper that produced it, a node may be renamed between planning and
    applying, and neither should silently write to the wrong node or to none.
    """

    found = [
        node
        for node in model.graph.node
        for attribute in node.attribute
        if attribute.name == SCOPE_ID_ATTRIBUTE and _text(attribute.s) == scope_id
    ]
    if not found:
        raise DataflowOpError(
            f"no node in this graph carries dataflow scope id {scope_id!r}; the plan was "
            "made against a different graph, or the node was replaced"
        )
    if len(found) > 1:
        raise DataflowOpError(
            f"{len(found)} nodes carry dataflow scope id {scope_id!r}; scope ids identify "
            "one operation each"
        )
    return found[0]


class AssignDataflowScopeIds:
    """Normalize the addressing identity of every dataflow node in one graph.

    The explicit upgrade transaction, named because ``bind`` names it.
    Identity belongs to whoever constructs the node; an imported graph that
    predates this layer has nodes with no owner to have done it, so the repair
    is an act a caller performs rather than something a query does behind their
    back on first read.

    **Duplicates matter as much as absences.**  Copying a ``NodeProto`` copies
    its attributes, so the ordinary way to clone a node also clones its scope
    id -- and two nodes sharing one identity is worse than a node having none:
    a change addressed to that id would find two candidates, and a recorded
    choice would be attributable to either.  The first occurrence in graph
    order keeps the id and every later one is reallocated, so a clone gets a
    fresh identity and the original is left alone.

    Idempotent: running it twice changes nothing the first run settled.
    """

    def __init__(self, domain: str) -> None:
        self.domain = domain

    def apply(self, model: Any) -> tuple[Any, bool]:
        """The QONNX transformation protocol: the model, and whether to re-run."""

        return model, bool(self.normalize(model))

    def normalize(self, model: Any) -> tuple[str, ...]:
        """Assign or reassign, returning every id this run had to allocate."""

        seen: set[str] = set()
        assigned: list[str] = []
        for node in model.graph.node:
            if node.domain != self.domain:
                continue
            existing = next(
                (
                    _text(attribute.s)
                    for attribute in node.attribute
                    if attribute.name == SCOPE_ID_ATTRIBUTE
                ),
                "",
            )
            if existing and existing not in seen:
                seen.add(existing)
                continue
            allocated = allocate_scope_id()
            _write_string(node, SCOPE_ID_ATTRIBUTE, allocated)
            seen.add(allocated)
            assigned.append(allocated)
        return tuple(assigned)


def assign_dataflow_scope_ids(model: Any, *, domain: str) -> tuple[str, ...]:
    """Run :class:`AssignDataflowScopeIds` over one graph."""

    return AssignDataflowScopeIds(domain).normalize(model)


def check_commitment(
    assessments: Mapping[CommitmentStage, ProjectionAssessment[Any]],
    require: CommitmentStage,
) -> None:
    """Refuse a final rejection in any projection at or below the given stage."""

    for stage in _stages_through(require):
        assessment = assessments.get(stage)
        if assessment is None:
            raise DataflowOpError(
                f"this operation offers no {stage.value!r} projection, so a commitment to "
                f"{require.value!r} cannot be checked"
            )
        answer = assessment.accepted_answer
        if isinstance(answer, Unresolved) or isinstance(answer, Decided):
            continue
        if isinstance(answer, Absent):
            codes = ", ".join(sorted({finding.code for finding in answer.findings}))
            raise DataflowOpError(
                f"the {stage.value} projection refuses this point ({codes or 'no findings'}); "
                "an unresolved point may be saved, a refused one may not",
                answer.findings,
            )


def _stages_through(require: CommitmentStage) -> tuple[CommitmentStage, ...]:
    if require is CommitmentStage.DATAFLOW:
        return (CommitmentStage.DATAFLOW,)
    return (CommitmentStage.DATAFLOW, CommitmentStage.PHYSICAL)


def apply_graph_effects(model: Any, effects: GraphEffects) -> Any:
    """Apply the whole change or none of it, returning the addressed live node."""

    return _apply_graph_effects(model, effects, lambda: find_node(model, effects.scope_id))


def _apply_graph_effects(model: Any, effects: GraphEffects, finish: Callable[[], Any]) -> Any:
    """Keep writes and their completion step inside one rollback boundary.

    Restoration is by serialized snapshot rather than by undoing each write:
    an undo list is a second description of the change and can disagree with
    the first. DataflowOp.commit supplies its post-write reconstruction as the
    completion step, so a hydration failure restores output repairs and native
    attributes together. No assertion carries this guarantee.
    """

    node = find_node(model, effects.scope_id)
    if (node.domain, node.op_type) != effects.expected_operator:
        raise DataflowOpError("source operator changed since this change was planned")
    if tuple(node.output) != effects.expected_outputs:
        raise DataflowOpError("output tensor identities changed since this change was planned")
    current: dict[str, bytes] = {}
    for item in node.attribute:
        if item.name in current:
            raise DataflowOpError(f"duplicate node attribute {item.name!r}")
        current[item.name] = item.SerializeToString(deterministic=True)
    for name, expected in effects.expected_attributes.items():
        if current.get(name) != expected:
            raise DataflowOpError(
                f"Decision attribute {name!r} changed since this change was planned; "
                "rebind and plan again"
            )
    # Re-read graph facts, including initializer content, before any mutation.
    # The plan carries detached build Problems, never an occurrence or model.
    with source_analysis(model, fresh=True) as summaries:
        fresh = effects.operation_type(node, effects.opset_version)._bind_with(
            model,
            effects.build_facts,
            summaries,
            recorded=False,
        )
    if fresh.problem_fingerprint != effects.expected_source_fingerprint:
        raise DataflowOpError(
            "source facts now describe a different problem; rebind and plan again"
        )

    # The *whole* model, not just the node.  Tensor datatypes live in graph
    # annotations and shapes in value_info, so restoring only the NodeProto
    # after a failed tensor write would leave the graph half-changed -- with
    # the node looking untouched, which is worse than an obvious failure.
    snapshot = model.model.SerializeToString(deterministic=True)
    try:
        for name in (*effects.remove_attributes, *effects.set_attributes):
            _drop(node, name)
        for name, attribute in sorted(effects.set_attributes.items()):
            node.attribute.append(attribute.proto(name))
        for tensor, datatype in effects.tensor_datatypes.items():
            model.set_tensor_datatype(tensor, datatype)
        for tensor, shape in effects.tensor_shapes.items():
            model.set_tensor_shape(tensor, list(shape))
        return finish()
    except Exception:
        model.model.ParseFromString(snapshot)
        raise


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _write_string(node: Any, name: str, value: str) -> None:
    from onnx import helper  # type: ignore[import-not-found] # noqa: PLC0415

    _drop(node, name)
    node.attribute.append(helper.make_attribute(name, value))


def _write_int(node: Any, name: str, value: int) -> None:
    from onnx import helper  # noqa: PLC0415 - deferred, as above

    _drop(node, name)
    node.attribute.append(helper.make_attribute(name, value))


def _drop(node: Any, name: str) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)


__all__ = [
    "AssignDataflowScopeIds",
    "CommitmentStage",
    "GraphEffects",
    "allocate_scope_id",
    "apply_graph_effects",
    "assign_dataflow_scope_ids",
    "check_commitment",
    "find_node",
]
