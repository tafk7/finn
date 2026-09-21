# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Native source checkpoint planning and checked graph transactions.

All model mutation is delegated to :mod:`finn.dataflow.ops.model_effects`.
This module owns source-specific planning, validation, reconstruction context,
and the public native lifecycle APIs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import uuid4

from finn.dataflow._engine import (
    Absent,
    Answer,
    Decided,
    Finding,
    FindingKind,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow._engine.results import ordered_findings
from finn.dataflow.ops.model_effects import (
    MODEL_READ_PRESENT,
    ModelEffects,
    ModelReadExpectation,
    ModelReadKind,
    ModelReadSet,
    apply_model_effects,
    merge_model_read_sets,
)
from finn.dataflow.ops.native import (
    SCOPE_ID_ATTRIBUTE,
    NativeAttribute,
    RecordedChoice,
    choice_schema,
)
from finn.dataflow.ops.schema import (
    Attribute,
    BuildFact,
    DatatypeAttribute,
    OpInput,
    attribute_name,
)
from finn.dataflow.space.declarations import Problem
from finn.dataflow.space.occurrence import ProjectionAssessment

if TYPE_CHECKING:
    from finn.dataflow.ops.space import DataflowSpace

T = TypeVar("T")


class CommitmentStage(Enum):
    """How far a caller is freezing the design when writing it down."""

    DATAFLOW = "dataflow"
    PHYSICAL = "physical"


@dataclass(frozen=True, slots=True)
class FrozenBuildFact:
    """One detached build input addressed by its stable class-member name."""

    member_name: str
    value: object


@dataclass(frozen=True, slots=True)
class SourcePublicationContext:
    """The nonserializable context needed to reconstruct one source operation."""

    operation_type: type[DataflowSpace]
    opset_version: int
    build_facts: tuple[FrozenBuildFact, ...]


@dataclass(frozen=True, slots=True)
class GraphEffects:
    """A source-node change lowered onto the shared model transaction engine."""

    scope_id: str
    commitment_stage: CommitmentStage
    expected_source_fingerprint: str
    expected_attributes: Mapping[str, bytes | None]
    remove_attributes: tuple[str, ...]
    set_attributes: Mapping[str, NativeAttribute]
    operation_type: type[DataflowSpace]
    opset_version: int
    build_facts: tuple[FrozenBuildFact, ...]
    expected_operator: tuple[str, str]
    expected_outputs: tuple[str, ...]
    expected_choices: tuple[tuple[str, object], ...] = ()
    read_set: ModelReadSet = ModelReadSet()
    tensor_datatypes: Mapping[str, Any] = field(default_factory=dict)
    tensor_shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    require_graph: bool = False
    expected_incoming: object | None = None

    def model_effects(self, *, read_set: ModelReadSet | None = None) -> ModelEffects:
        return ModelEffects(
            read_set=self.read_set if read_set is None else read_set,
            remove_attributes=tuple((self.scope_id, name) for name in self.remove_attributes),
            set_attributes=tuple(
                (self.scope_id, name, value) for name, value in sorted(self.set_attributes.items())
            ),
            tensor_datatypes=tuple(sorted(self.tensor_datatypes.items())),
            tensor_shapes=tuple(sorted(self.tensor_shapes.items())),
        )


def allocate_scope_id() -> str:
    return f"dataflow_{uuid4().hex}"


def save_unidentified_space(
    adapter: Any,
    proposal: Any,
    *,
    require: Any = None,
    require_graph: bool = False,
) -> DataflowSpace:
    """First-save identity transaction for exactly the supplied unidentified node."""
    from finn.dataflow.ops.model_effects import _wrapper_from_bytes  # noqa: PLC0415
    from finn.dataflow.ops.reconstruction import build_space  # noqa: PLC0415
    from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

    model = adapter._attached_model()
    indices = [index for index, node in enumerate(model.graph.node) if node is adapter.onnx_node]
    if len(indices) != 1:
        raise DataflowOpError("first save requires the exact supplied node in the attached model")
    index = indices[0]
    original = model.model.SerializeToString(deterministic=True)
    try:
        candidate = _wrapper_from_bytes(model, original)
        node = candidate.graph.node[index]
        scope = allocate_scope_id()
        _write_string(node, SCOPE_ID_ATTRIBUTE, scope)
        staged = type(adapter)(node, adapter.onnx_opset_version)
        staged._model = candidate
        staged._build, staged._graph_context = adapter._build, adapter._graph_context
        staged._space = build_space(
            adapter.space_type,
            candidate,
            node,
            build=adapter._build,
            graph_context=adapter._graph_context,
            recorded=False,
            opset_version=adapter.onnx_opset_version,
            fresh=True,
        )
        staged.save_space(proposal, require=require, require_graph=require_graph)
        prepared = candidate.model.SerializeToString(deterministic=True)
        if model.model.SerializeToString(deterministic=True) != original:
            raise DataflowOpError("the current graph changed during first-save validation")
        model.model.ParseFromString(prepared)
        saved = build_space(
            adapter.space_type,
            model,
            model.graph.node[index],
            build=adapter._build,
            graph_context=adapter._graph_context,
            opset_version=adapter.onnx_opset_version,
            fresh=True,
        )
        if model.model.SerializeToString(deterministic=True) != prepared:
            raise DataflowOpError("first-save final hydration mutated the graph")
        return saved
    except Exception:
        if model.model.SerializeToString(deterministic=True) != original:
            model.model.ParseFromString(original)
        # ParseFromString invalidates protobuf child references even after rollback.
        adapter.onnx_node = model.graph.node[index]
        raise


def find_node(model: Any, scope_id: str) -> Any:
    """Return the unique source node carrying a stable scope id."""

    found = [
        node
        for node in model.graph.node
        for attribute in node.attribute
        if attribute.name == SCOPE_ID_ATTRIBUTE and _text(attribute.s) == scope_id
    ]
    if not found:
        from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

        raise DataflowOpError(
            f"no node in this graph carries dataflow scope id {scope_id!r}; the plan was "
            "made against a different graph, or the node was replaced"
        )
    if len(found) > 1:
        from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

        raise DataflowOpError(
            f"{len(found)} nodes carry dataflow scope id {scope_id!r}; scope ids identify "
            "one operation each"
        )
    return found[0]


class AssignDataflowScopeIds:
    """Assign missing/duplicate scope ids without changing valid identities."""

    def __init__(self, domain: str) -> None:
        self.domain = domain

    def apply(self, model: Any) -> tuple[Any, bool]:
        return model, bool(self.normalize(model))

    def normalize(self, model: Any) -> tuple[str, ...]:
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
    return AssignDataflowScopeIds(domain).normalize(model)


def check_commitment(
    assessments: Mapping[CommitmentStage, ProjectionAssessment[Any]],
    require: CommitmentStage,
) -> None:
    for stage in _stages_through(require):
        assessment = assessments.get(stage)
        if assessment is None:
            from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

            raise DataflowOpError(
                f"this operation offers no {stage.value!r} projection, so a commitment to "
                f"{require.value!r} cannot be checked"
            )
        refused, findings = _constraint_refusals(stage, assessment)
        if refused:
            from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

            paths = ", ".join(str(path) for path in refused)
            raise DataflowOpError(
                f"the {stage.value} projection refuses this point because constraints "
                f"({paths}) refused; "
                "an unresolved point may be saved, a refused one may not",
                findings,
            )
        answer = assessment.accepted_answer
        if isinstance(answer, Unresolved) or isinstance(answer, Decided):
            continue
        if isinstance(answer, Absent):
            from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

            codes = ", ".join(sorted({finding.code for finding in answer.findings}))
            raise DataflowOpError(
                f"the {stage.value} projection refuses this point ({codes or 'no findings'}); "
                "an unresolved point may be saved, a refused one may not",
                answer.findings,
            )


def _constraint_refusals(
    stage: CommitmentStage,
    assessment: ProjectionAssessment[Any],
) -> tuple[tuple[QualifiedPath, ...], tuple[Finding, ...]]:
    answers: dict[QualifiedPath, Answer[bool]] = {}
    for constraint_assessment in assessment.constraints:
        for path in constraint_assessment.refused:
            answer = constraint_assessment.answers[path]
            previous = answers.get(path)
            if previous is None or (isinstance(previous, Decided) and isinstance(answer, Absent)):
                answers[path] = answer

    findings: list[Finding] = []
    for path, answer in sorted(answers.items()):
        if isinstance(answer, Absent):
            findings.extend(answer.findings)
            continue
        findings.append(
            Finding(
                FindingKind.REJECTION,
                "projection-constraint-refused",
                path,
                f"a {stage.value} commitment constraint refused this point",
            )
        )
    unique_findings = list(dict.fromkeys(findings))
    return tuple(sorted(answers)), ordered_findings(unique_findings)


def _stages_through(require: CommitmentStage) -> tuple[CommitmentStage, ...]:
    if require is CommitmentStage.DATAFLOW:
        return (CommitmentStage.DATAFLOW,)
    return (CommitmentStage.DATAFLOW, CommitmentStage.PHYSICAL)


def freeze_build_facts(operation: Any) -> tuple[FrozenBuildFact, ...]:
    """Detach present build Problems using stable member names and semantics."""

    from finn.dataflow.ops.base import source_declarations  # noqa: PLC0415

    result = []
    for name, declaration in source_declarations(type(operation)):
        if isinstance(declaration, BuildFact) and declaration in operation.problem_snapshot:
            result.append(
                FrozenBuildFact(
                    name,
                    declaration.value_semantics.freeze(operation.problem_snapshot[declaration]),
                )
            )
    return tuple(result)


def build_values(context: SourcePublicationContext) -> Mapping[Problem[Any], object]:
    """Resolve and validate a frozen named build context against current code."""

    from finn.dataflow.ops.base import DataflowOpError, source_declarations  # noqa: PLC0415

    by_name = {
        name: declaration
        for name, declaration in source_declarations(context.operation_type)
        if isinstance(declaration, BuildFact)
    }
    provided = tuple(item.member_name for item in context.build_facts)
    if len(provided) != len(set(provided)):
        raise DataflowOpError("frozen build facts contain duplicate member names")
    unknown = set(provided) - set(by_name)
    if unknown:
        raise DataflowOpError(f"frozen build facts name unknown members: {sorted(unknown)!r}")
    # A sparse checkpoint may omit target/build facts. The consuming physical
    # query reports their unavailability; independent logical/type commits do
    # not synthesize defaults or require unrelated generation inputs.
    result: dict[Problem[Any], object] = {}
    for item in context.build_facts:
        declaration = by_name[item.member_name]
        try:
            result[declaration] = declaration.value_semantics.freeze(item.value)
        except TypeError as error:
            raise DataflowOpError(
                f"frozen build fact {item.member_name!r} has the wrong value type"
            ) from error
    return MappingProxyType(result)


def source_publication_context(operation: Any) -> SourcePublicationContext:
    state = operation._bound_node()
    return SourcePublicationContext(
        type(operation),
        state.opset_version,
        freeze_build_facts(operation),
    )


def source_read_set(
    operation: Any,
    *,
    expected_attributes: Mapping[str, bytes | None],
    include_output_annotations: bool = True,
) -> ModelReadSet:
    """Capture only source facts and write targets used by a source plan."""

    from finn.dataflow.ops.base import source_declarations  # noqa: PLC0415

    state = operation._bound_node()
    node = state.materialize()
    expectations = [
        ModelReadExpectation(
            ModelReadKind.NODE,
            state.scope_id,
            "operator",
            json.dumps([node.domain, node.op_type], separators=(",", ":")).encode("utf-8"),
        ),
        ModelReadExpectation(
            ModelReadKind.OPSET,
            node.domain,
            None,
            state.source_opset_import,
        ),
    ]
    current_attributes = {
        item.name: bytes(item.SerializeToString(deterministic=True)) for item in node.attribute
    }
    source_attribute_names = {
        attribute_name(name, declaration)
        for name, declaration in source_declarations(type(operation))
        if isinstance(declaration, (Attribute, DatatypeAttribute))
    }
    for name in sorted(set(expected_attributes) | source_attribute_names):
        expectations.append(
            ModelReadExpectation(
                ModelReadKind.ATTRIBUTE,
                state.scope_id,
                name,
                current_attributes.get(name),
            )
        )
    source = operation.source
    input_indices = {
        name: declaration.index
        for name, declaration in source_declarations(type(operation))
        if isinstance(declaration, OpInput)
    }
    for item in source.inputs:
        index = input_indices[item.id]
        expectations.extend(
            (
                ModelReadExpectation(
                    ModelReadKind.OPERAND_SLOT,
                    state.scope_id,
                    f"input:{index}",
                    MODEL_READ_PRESENT,
                ),
                ModelReadExpectation(
                    ModelReadKind.TENSOR_FACT,
                    state.scope_id,
                    f"input:{index}:shape",
                    json.dumps(list(item.shape), separators=(",", ":")),
                ),
                ModelReadExpectation(
                    ModelReadKind.TENSOR_FACT,
                    state.scope_id,
                    f"input:{index}:carrier_dtype",
                    item.carrier_dtype,
                ),
                ModelReadExpectation(
                    ModelReadKind.TENSOR_FACT,
                    state.scope_id,
                    f"input:{index}:logical_datatype",
                    item.datatype_annotation
                    if item.producer_reads is not None
                    else item.datatype.name
                    if item.datatype_annotated and item.datatype is not None
                    else None,
                ),
                ModelReadExpectation(
                    ModelReadKind.INITIALIZER_CONTENT,
                    state.scope_id,
                    f"input:{index}",
                    item.initializer_digest,
                ),
            )
        )
    for index, item in enumerate(source.outputs):
        if not include_output_annotations:
            expectations.append(
                ModelReadExpectation(
                    ModelReadKind.OPERAND_SLOT,
                    state.scope_id,
                    f"output:{index}",
                    item.tensor,
                )
            )
            continue
        expectations.extend(
            (
                ModelReadExpectation(
                    ModelReadKind.OPERAND_SLOT,
                    state.scope_id,
                    f"output:{index}",
                    item.tensor,
                ),
                ModelReadExpectation(
                    ModelReadKind.VALUE_INFO,
                    item.tensor,
                    None,
                    state.output_value_info[index],
                ),
                ModelReadExpectation(
                    ModelReadKind.QUANTIZATION_ANNOTATION,
                    item.tensor,
                    None,
                    state.output_annotations[index],
                ),
            )
        )
    return merge_model_read_sets(
        ModelReadSet(tuple(expectations)),
        *(
            item.producer_reads
            for item in source.inputs
            if isinstance(item.producer_reads, ModelReadSet)
        ),
    )


def apply_graph_effects(
    model: Any,
    effects: GraphEffects,
    *,
    graph_context: Any = None,
    build: Any = None,
) -> Any:
    """Apply the source-node adapter through the shared transaction engine."""

    return _apply_graph_effects(
        model,
        effects,
        lambda current: find_node(current, effects.scope_id),
        graph_context=graph_context,
        build=build,
    )


def _apply_graph_effects(
    model: Any,
    effects: GraphEffects,
    finish: Callable[[Any], T],
    *,
    graph_context: Any = None,
    build: Any = None,
) -> T:
    """Validate current source facts, then apply through one rollback engine."""

    from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

    context = SourcePublicationContext(
        effects.operation_type,
        effects.opset_version,
        effects.build_facts,
    )
    source = _source_only_operation(
        model,
        context,
        effects.scope_id,
        incoming=effects.expected_incoming,
    )
    if source.local_problem_fingerprint != effects.expected_source_fingerprint:
        raise DataflowOpError(
            "source facts now describe a different problem; rebind and plan again"
        )
    records = _recorded_choices(effects.expected_choices)
    selected_source = _apply_expected_choices(source, records)
    schema = {item.choice.path: item for item in choice_schema(selected_source)}
    captured = tuple((schema[item.path], item.value) for item in records)
    canonical = selected_source.graph_effects(
        require=effects.commitment_stage,
        require_graph=effects.require_graph,
        _captured_choices=captured,
    )
    if effects != canonical:
        raise DataflowOpError("source graph effects differ from the validated source plan")

    current_read = None
    combined_reads = effects.read_set
    if effects.require_graph:
        from finn.dataflow.ops.graph_context import require_context_read  # noqa: PLC0415

        if graph_context is None:
            raise DataflowOpError("graph-required effects need a GraphContext")
        effective_build = _frozen_build_configuration(context) if build is None else build
        current_read = require_context_read(
            graph_context,
            model,
            effective_build,
            consumer_scope_id=effects.scope_id,
        )
        if current_read.incoming != effects.expected_incoming:
            raise DataflowOpError(
                "current incoming graph contracts differ from the validated source plan"
            )
        combined_reads = merge_model_read_sets(effects.read_set, current_read.model_reads)

    def validate(candidate: Any) -> None:
        from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415
        from finn.dataflow.ops.reconstruction import build_space  # noqa: PLC0415

        node = find_node(candidate, effects.scope_id)
        fresh = build_space(
            effects.operation_type,
            candidate,
            node,
            frozen_build=build_values(context),
            recorded=False,
            context_read=current_read,
            opset_version=context.opset_version,
            fresh=True,
        )
        if fresh.local_problem_fingerprint != effects.expected_source_fingerprint:
            raise DataflowOpError(
                "source facts now describe a different problem; rebind and plan again"
            )
        schema = {item.choice.path: item for item in choice_schema(fresh)}
        if any(path not in schema for path, _value in effects.expected_choices):
            raise DataflowOpError("recorded choice path is absent from the current schema")
        from finn.dataflow.space.occurrence import occurrence_commit_paths  # noqa: PLC0415

        committed = occurrence_commit_paths(
            fresh,
            {schema[path].choice.reference.path: value for path, value in effects.expected_choices},
        )
        if dict(committed.recorded()) != dict(effects.expected_choices):
            raise DataflowOpError("recorded choices differ after source reconstruction")
        hydrated = _hydrate_candidate(fresh)
        _assert_expected_choices(hydrated, _recorded_choices(effects.expected_choices))
        if effects.require_graph:
            graph_answer = hydrated.graph_dataflow.accepted_answer
            if not isinstance(graph_answer, Decided):
                raise DataflowOpError(
                    "written candidate has no accepted graph dataflow projection",
                    getattr(graph_answer, "findings", ()),
                )
        if effects.commitment_stage is CommitmentStage.PHYSICAL:
            physical_answer = hydrated.physical.accepted_answer
            if not isinstance(physical_answer, Decided):
                raise DataflowOpError(
                    "written candidate has no accepted physical projection",
                    getattr(physical_answer, "findings", ()),
                )

    model_effects = effects.model_effects(read_set=combined_reads)
    model_effects = _invalidate_downstream_types(model, model_effects)
    return apply_model_effects(
        model,
        model_effects,
        validate=validate,
        finish=finish,
    )


def _invalidate_downstream_types(model: Any, effects: ModelEffects) -> ModelEffects:
    """Clear dependent derived annotations in the same checked transaction.

    A later consumer rehydrates producer contracts. Its independent decisions
    stay untouched. The topology reads protect this closure against an added or
    rewired consumer between planning and application.
    """

    from dataclasses import replace  # noqa: PLC0415
    from finn.dataflow.ops.model_effects import _logical_datatype  # noqa: PLC0415

    datatypes = dict(effects.tensor_datatypes)
    pending = [
        tensor
        for tensor, datatype in datatypes.items()
        if _logical_datatype(model, tensor) != (None if datatype is None else datatype.name)
    ]
    visited: set[str] = set()
    reads: list[ModelReadExpectation] = []
    while pending:
        tensor = pending.pop()
        if tensor in visited:
            continue
        visited.add(tensor)
        users = tuple(node for node in model.graph.node if tensor in node.input)
        reads.append(
            ModelReadExpectation(
                ModelReadKind.VALUE_USERS,
                tensor,
                None,
                json.dumps(
                    sorted((tuple(node.input), tuple(node.output)) for node in users),
                    separators=(",", ":"),
                ),
            )
        )
        for node in users:
            for output in node.output:
                if output and output not in datatypes:
                    datatypes[output] = None
                    pending.append(output)
    return replace(
        effects,
        tensor_datatypes=tuple(sorted(datatypes.items())),
        read_set=merge_model_read_sets(effects.read_set, ModelReadSet(tuple(reads))),
    )


def _recorded_choices(values: tuple[tuple[str, object], ...]) -> tuple[RecordedChoice, ...]:
    return tuple(RecordedChoice(path, value) for path, value in values)


def _source_only_operation(
    model: Any,
    context: SourcePublicationContext,
    scope_id: str,
    *,
    incoming: object | None = None,
) -> Any:
    from finn.dataflow.ops.reconstruction import build_space  # noqa: PLC0415

    node = find_node(model, scope_id)
    context_read = None
    if incoming is not None:
        from finn.dataflow.ops.graph_context import (  # noqa: PLC0415
            ContextRead,
            IncomingGraphContext,
        )

        if not isinstance(incoming, IncomingGraphContext):
            from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

            raise DataflowOpError("graph effects contain an invalid incoming context")
        context_read = ContextRead(incoming, ModelReadSet())
    return build_space(
        context.operation_type,
        model,
        node,
        frozen_build=build_values(context),
        recorded=False,
        context_read=context_read,
        opset_version=context.opset_version,
        fresh=True,
    )


class _FrozenBuildConfiguration:
    def __init__(self, context: SourcePublicationContext) -> None:
        self._dataflow_frozen_build_values = MappingProxyType(
            {item.member_name: item.value for item in context.build_facts}
        )


def _frozen_build_configuration(context: SourcePublicationContext) -> object:
    return _FrozenBuildConfiguration(context)


def _apply_expected_choices(
    operation: DataflowSpace, choices: tuple[RecordedChoice, ...]
) -> DataflowSpace:
    from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415
    from finn.dataflow.space.occurrence import occurrence_commit_paths  # noqa: PLC0415

    schema = {item.choice.path: item for item in choice_schema(operation)}
    unknown = tuple(item.path for item in choices if item.path not in schema)
    if unknown:
        raise DataflowOpError(f"recorded choices are absent from the current schema: {unknown}")
    try:
        committed = occurrence_commit_paths(
            operation,
            {schema[item.path].choice.reference.path: item.value for item in choices},
        )
    except Exception as error:
        raise DataflowOpError(
            f"recorded choices are invalid for current source: {error}"
        ) from error
    _assert_expected_choices(committed, choices)
    return committed


def _assert_expected_choices(operation: DataflowSpace, choices: tuple[RecordedChoice, ...]) -> None:
    from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

    actual = tuple(operation.recorded().items())
    expected = tuple((item.path, item.value) for item in choices)
    if len(actual) != len(expected) or any(
        actual_path != expected_path
        or type(actual_value) is not type(expected_value)
        or actual_value != expected_value
        for (actual_path, actual_value), (expected_path, expected_value) in zip(actual, expected)
    ):
        raise DataflowOpError("written native choices differ from the publication plan")


def _hydrate_candidate(operation: DataflowSpace) -> DataflowSpace:
    from finn.dataflow.ops.native import hydrate  # noqa: PLC0415

    return cast("DataflowSpace", hydrate(operation))


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _write_string(node: Any, name: str, value: str) -> None:
    from onnx import helper  # type: ignore[import-not-found] # noqa: PLC0415

    _drop(node, name)
    node.attribute.append(helper.make_attribute(name, value))


def _drop(node: Any, name: str) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)


__all__ = [
    "AssignDataflowScopeIds",
    "CommitmentStage",
    "FrozenBuildFact",
    "GraphEffects",
    "MODEL_READ_PRESENT",
    "ModelEffects",
    "ModelReadExpectation",
    "ModelReadKind",
    "ModelReadSet",
    "SourcePublicationContext",
    "allocate_scope_id",
    "apply_graph_effects",
    "apply_model_effects",
    "assign_dataflow_scope_ids",
    "build_values",
    "check_commitment",
    "find_node",
    "freeze_build_facts",
    "source_publication_context",
    "source_read_set",
]
