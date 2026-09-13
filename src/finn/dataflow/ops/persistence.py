# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Source persistence and selected-publication lifecycle.

All model mutation is delegated to :mod:`finn.dataflow.ops.model_effects`.
This module owns source-specific planning, validation, reconstruction context,
and the public native/selected lifecycle APIs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import uuid4

from finn.dataflow._engine import Absent, Decided, Unresolved
from finn.dataflow.ops.model_effects import (
    MODEL_READ_PRESENT,
    ModelEffects,
    ModelReadExpectation,
    ModelReadKind,
    ModelReadSet,
    apply_model_effects,
)
from finn.dataflow.ops.native import (
    SCOPE_ID_ATTRIBUTE,
    NativeAttribute,
    capture_decided_choices,
    captured_choice_mapping,
    choice_schema,
    encode_choice_value,
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
    from finn.dataflow.ops.base import DataflowOp
    from finn.dataflow.ops.legacy import LegacySelection
    from finn.dataflow.ops.selected import (
        DecodedSelectedGraph,
        RecordedChoice,
        SelectedGraphSnapshot,
    )

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

    operation_type: type[DataflowOp]
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
    operation_type: type[DataflowOp]
    opset_version: int
    build_facts: tuple[FrozenBuildFact, ...]
    expected_operator: tuple[str, str]
    expected_outputs: tuple[str, ...]
    expected_choices: tuple[tuple[str, object], ...] = ()
    read_set: ModelReadSet = ModelReadSet()
    tensor_datatypes: Mapping[str, Any] = field(default_factory=dict)
    tensor_shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    def model_effects(self) -> ModelEffects:
        return ModelEffects(
            read_set=self.read_set,
            remove_attributes=tuple((self.scope_id, name) for name in self.remove_attributes),
            set_attributes=tuple(
                (self.scope_id, name, value) for name, value in sorted(self.set_attributes.items())
            ),
            tensor_datatypes=tuple(sorted(self.tensor_datatypes.items())),
            tensor_shapes=tuple(sorted(self.tensor_shapes.items())),
        )


@dataclass(frozen=True, slots=True)
class SelectedPublicationPlan:
    source_effects: ModelEffects
    context: SourcePublicationContext
    expected_problem_fingerprint: str
    expected_choices: tuple[RecordedChoice, ...]
    candidate: SelectedGraphSnapshot


@dataclass(frozen=True, slots=True)
class PublishedSelection:
    operation: DataflowOp
    selected: DecodedSelectedGraph[Any, Any]


@dataclass(frozen=True, slots=True)
class SelectionMigrationPlan:
    source_effects: ModelEffects
    context: SourcePublicationContext
    legacy: LegacySelection
    replacement_choices: tuple[RecordedChoice, ...]
    selected_candidate: SelectedGraphSnapshot | None


def allocate_scope_id() -> str:
    return f"dataflow_{uuid4().hex}"


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
    missing = {
        name
        for name, declaration in by_name.items()
        if declaration.build_required and name not in provided
    }
    if missing:
        raise DataflowOpError(f"frozen build facts omit required members: {sorted(missing)!r}")
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
                    item.datatype.name if item.datatype_annotated else None,
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
    return ModelReadSet(tuple(expectations))


def apply_graph_effects(model: Any, effects: GraphEffects) -> Any:
    """Apply the source-node adapter through the shared transaction engine."""

    return _apply_graph_effects(
        model,
        effects,
        lambda current: find_node(current, effects.scope_id),
    )


def _apply_graph_effects(
    model: Any,
    effects: GraphEffects,
    finish: Callable[[Any], T],
) -> T:
    """Validate current source facts, then apply through one rollback engine."""

    from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

    context = SourcePublicationContext(
        effects.operation_type,
        effects.opset_version,
        effects.build_facts,
    )
    source = _source_only_operation(model, context, effects.scope_id)
    if source.problem_fingerprint != effects.expected_source_fingerprint:
        raise DataflowOpError(
            "source facts now describe a different problem; rebind and plan again"
        )
    records = _recorded_choices(effects.expected_choices)
    selected_source = _apply_expected_choices(source, records)
    schema = {item.choice.path: item for item in choice_schema(selected_source)}
    captured = tuple((schema[item.path], item.value) for item in records)
    canonical = selected_source.graph_effects(
        require=effects.commitment_stage,
        _captured_choices=captured,
    )
    if effects != canonical:
        raise DataflowOpError("source graph effects differ from the validated source plan")

    def validate(candidate: Any) -> None:
        from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415
        from finn.dataflow.ops.reconstruction import source_analysis  # noqa: PLC0415

        node = find_node(candidate, effects.scope_id)
        with source_analysis(candidate, fresh=True) as summaries:
            fresh = effects.operation_type(node, context.opset_version)._bind_with(
                candidate,
                build_values(context),
                summaries,
                recorded=False,
            )
        if fresh.problem_fingerprint != effects.expected_source_fingerprint:
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

    return apply_model_effects(
        model,
        effects.model_effects(),
        validate=validate,
        finish=finish,
    )


def _captured_records(
    captured: tuple[tuple[Any, object], ...],
) -> tuple[RecordedChoice, ...]:
    from finn.dataflow.ops.selected import RecordedChoice  # noqa: PLC0415

    return tuple(
        RecordedChoice(item.choice.path, value, encode_choice_value(item, value))
        for item, value in captured
    )


def _recorded_choices(values: tuple[tuple[str, object], ...]) -> tuple[RecordedChoice, ...]:
    from finn.dataflow.ops.selected import RecordedChoice  # noqa: PLC0415

    return tuple(RecordedChoice(path, value) for path, value in values)


def _source_scope(effects: ModelEffects) -> str:
    owners = {owner for owner, name, _value in effects.set_attributes if name == SCOPE_ID_ATTRIBUTE}
    if len(owners) != 1:
        from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

        raise DataflowOpError("source effects do not identify one source scope")
    return next(iter(owners))


def _written_source_fingerprint(effects: ModelEffects) -> str:
    values = tuple(
        value.value
        for _owner, name, value in effects.set_attributes
        if name == "dataflow_problem_fingerprint"
    )
    if len(values) != 1 or type(values[0]) is not str:
        from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

        raise DataflowOpError("source effects do not contain one problem fingerprint")
    return values[0]


def _source_only_operation(model: Any, context: SourcePublicationContext, scope_id: str) -> Any:
    from finn.dataflow.ops.reconstruction import source_analysis  # noqa: PLC0415

    node = find_node(model, scope_id)
    with source_analysis(model, fresh=True) as summaries:
        return context.operation_type(node, context.opset_version)._bind_with(
            model,
            build_values(context),
            summaries,
            recorded=False,
        )


def _apply_expected_choices(
    operation: DataflowOp, choices: tuple[RecordedChoice, ...]
) -> DataflowOp:
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


def _assert_expected_choices(operation: DataflowOp, choices: tuple[RecordedChoice, ...]) -> None:
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


def _hydrate_candidate(operation: DataflowOp) -> DataflowOp:
    from finn.dataflow.ops.native import hydrate  # noqa: PLC0415

    return cast("DataflowOp", hydrate(operation))


def _captured_for_records(
    operation: DataflowOp, choices: tuple[RecordedChoice, ...]
) -> tuple[tuple[Any, object], ...]:
    schema = {item.choice.path: item for item in choice_schema(operation)}
    return tuple((schema[item.path], item.value) for item in choices)


def _canonical_publication_effects(
    operation: DataflowOp,
    choices: tuple[RecordedChoice, ...],
) -> ModelEffects:
    selected = _apply_expected_choices(operation, choices)
    return cast(
        ModelEffects,
        selected.graph_effects(
            require=CommitmentStage.DATAFLOW,
            _captured_choices=_captured_for_records(selected, choices),
        ).model_effects(),
    )


def _infer_selected_candidate(snapshot: SelectedGraphSnapshot) -> SelectedGraphSnapshot:
    from finn.dataflow.ops.inference import InferDataTypes, InferShapes  # noqa: PLC0415
    from finn.dataflow.ops.selected import (  # noqa: PLC0415
        build_selected_snapshot,
        decode_selected_graph,
    )

    model = snapshot.model_copy()
    original_docs = {
        item.node_id: (
            model.graph.node[item.index].HasField("doc_string"),
            model.graph.node[item.index].doc_string,
        )
        for item in snapshot.declaration.graph_nodes
    }
    marker_prefix = "__finn_selected_inference_node__:"
    for item in snapshot.declaration.graph_nodes:
        model.graph.node[item.index].doc_string = marker_prefix + item.node_id
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    by_id = {
        node.doc_string[len(marker_prefix) :]: index
        for index, node in enumerate(model.graph.node)
        if node.doc_string.startswith(marker_prefix)
    }
    if len(by_id) != len(model.graph.node) or set(by_id) != set(original_docs):
        from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

        raise DataflowOpError("selected inference lost stable construction node identities")
    for node_id, index in by_id.items():
        present, value = original_docs[node_id]
        if present:
            model.graph.node[index].doc_string = value
        else:
            model.graph.node[index].ClearField("doc_string")
    declaration = replace(
        snapshot.declaration,
        graph_nodes=tuple(
            replace(item, index=by_id[item.node_id]) for item in snapshot.declaration.graph_nodes
        ),
    )
    inferred = build_selected_snapshot(model, declaration)
    decode_selected_graph(inferred)
    return inferred


def plan_selected_publication(operation: DataflowOp) -> SelectedPublicationPlan:
    """Freeze one executable selected candidate and its source-side transaction."""

    from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415
    from finn.dataflow.designs.design import (  # noqa: PLC0415
        DataflowDesign,
        selected_graph_for,
    )

    captured = capture_decided_choices(operation)
    choice_values = captured_choice_mapping(captured)
    design = operation.selected_design()
    if not isinstance(design, DataflowDesign):
        raise DataflowOpError("selected operation did not resolve a DataflowDesign")
    selected = selected_graph_for(design, captured_choices=choice_values)
    if not isinstance(selected.accepted_answer, Decided):
        findings = getattr(selected.accepted_answer, "findings", ())
        codes = ", ".join(sorted({item.code for item in findings})) or "not ready"
        raise DataflowOpError(f"selected publication is unavailable ({codes})", findings)
    candidate = _infer_selected_candidate(selected.accepted_answer.value)
    decoded = candidate.declaration
    if decoded.source.problem_fingerprint != operation.problem_fingerprint:
        raise DataflowOpError("selected candidate source differs from the publication source")
    graph_effects = operation.graph_effects(
        require=CommitmentStage.DATAFLOW,
        _captured_choices=captured,
    )
    return SelectedPublicationPlan(
        graph_effects.model_effects(),
        source_publication_context(operation),
        operation.problem_fingerprint,
        _captured_records(captured),
        candidate,
    )


def apply_selected_publication(model: Any, plan: SelectedPublicationPlan) -> PublishedSelection:
    """Atomically commit native choices and finish with fresh source/selected decode."""

    from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415
    from finn.dataflow.ops.reconstruction import rebind_selected_graph  # noqa: PLC0415
    from finn.dataflow.ops.selected import decode_selected_graph  # noqa: PLC0415

    scope_id = _source_scope(plan.source_effects)
    current = _source_only_operation(model, plan.context, scope_id)
    if current.problem_fingerprint != plan.expected_problem_fingerprint:
        raise DataflowOpError(
            "source facts now describe a different problem; rebind and plan again"
        )
    if _canonical_publication_effects(current, plan.expected_choices) != plan.source_effects:
        raise DataflowOpError("publication effects differ from the validated source plan")

    def validate(candidate_model: Any) -> None:
        source = _source_only_operation(candidate_model, plan.context, scope_id)
        if source.problem_fingerprint != plan.expected_problem_fingerprint:
            raise DataflowOpError(
                "source facts now describe a different problem; rebind and plan again"
            )
        selected_source = _apply_expected_choices(source, plan.expected_choices)
        check_commitment(
            {CommitmentStage.DATAFLOW: selected_source.dataflow},
            CommitmentStage.DATAFLOW,
        )
        hydrated = _hydrate_candidate(source)
        _assert_expected_choices(hydrated, plan.expected_choices)
        rebind_selected_graph(selected_source, plan.candidate, update_origin=False)

    def finish(current: Any) -> PublishedSelection:
        source = _source_only_operation(current, plan.context, scope_id)
        source = _apply_expected_choices(source, plan.expected_choices)
        hydrated = cast("DataflowOp", source.rebind(current))
        _assert_expected_choices(hydrated, plan.expected_choices)
        selected = decode_selected_graph(plan.candidate)
        rebound = rebind_selected_graph(hydrated, selected.snapshot)
        return PublishedSelection(hydrated, rebound)

    return cast(
        "PublishedSelection | DataflowOp",
        apply_model_effects(
            model,
            plan.source_effects,
            validate=validate,
            finish=finish,
        ),
    )


def plan_selection_migration(
    operation: DataflowOp,
    legacy: LegacySelection,
    assignments: Mapping[str, object],
    *,
    publish_selected: bool = False,
) -> SelectionMigrationPlan:
    """Validate explicit legacy assignments and prepare one atomic migration."""

    from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415
    from finn.dataflow.ops.legacy import inspect_legacy_selection  # noqa: PLC0415
    from finn.dataflow.ops.selected import RecordedChoice  # noqa: PLC0415

    inspected = inspect_legacy_selection(operation)
    if inspected != legacy:
        raise DataflowOpError("legacy selection changed since it was inspected")
    missing = {item.path for item in legacy.choices} - set(assignments)
    if missing:
        raise DataflowOpError(f"migration assignments omit legacy choices: {sorted(missing)!r}")
    schema = {item.choice.path: item for item in choice_schema(operation)}
    unknown = set(assignments) - set(schema)
    if unknown:
        raise DataflowOpError(f"migration assignments name unknown choices: {sorted(unknown)!r}")
    migrated = _apply_expected_choices(
        operation,
        tuple(
            RecordedChoice(
                path,
                assignments[path],
                encode_choice_value(schema[path], assignments[path]),
            )
            for path in schema
            if path in assignments
        ),
    )
    check_commitment(
        {CommitmentStage.DATAFLOW: migrated.dataflow},
        CommitmentStage.DATAFLOW,
    )
    captured = capture_decided_choices(migrated)
    effects = _migration_effects(operation, legacy, _captured_records(captured))
    selected_candidate = None
    if publish_selected:
        publication = plan_selected_publication(migrated)
        selected_candidate = publication.candidate
    return SelectionMigrationPlan(
        effects,
        source_publication_context(operation),
        legacy,
        _captured_records(captured),
        selected_candidate,
    )


def _migration_effects(
    operation: DataflowOp,
    legacy: LegacySelection,
    choices: tuple[RecordedChoice, ...],
) -> ModelEffects:
    selected = _apply_expected_choices(operation, choices)
    effects = cast(
        ModelEffects,
        selected.graph_effects(
            require=CommitmentStage.DATAFLOW,
            _captured_choices=_captured_for_records(selected, choices),
        ).model_effects(),
    )
    expected_old = dict(legacy.owned_attribute_bytes)
    scope_id = _source_scope(effects)
    read_by_key = {
        (item.kind, item.owner, item.field): item for item in effects.read_set.expectations
    }
    for name in legacy.owned_attributes:
        item = ModelReadExpectation(
            ModelReadKind.ATTRIBUTE,
            scope_id,
            name,
            expected_old.get(name),
        )
        read_by_key[(item.kind, item.owner, item.field)] = item
    set_names = {name for _owner, name, _value in effects.set_attributes}
    remove_attributes = {
        *effects.remove_attributes,
        *((scope_id, name) for name in legacy.owned_attributes if name not in set_names),
    }
    effects = replace(
        effects,
        read_set=ModelReadSet(tuple(read_by_key.values())),
        remove_attributes=tuple(sorted(remove_attributes)),
    )
    return effects


def apply_selection_migration(
    model: Any,
    plan: SelectionMigrationPlan,
) -> PublishedSelection | DataflowOp:
    """Apply an explicit native migration, optionally publishing selected v2."""

    from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415
    from finn.dataflow.ops.legacy import inspect_legacy_selection  # noqa: PLC0415
    from finn.dataflow.ops.reconstruction import rebind_selected_graph  # noqa: PLC0415
    from finn.dataflow.ops.selected import decode_selected_graph  # noqa: PLC0415

    scope_id = _source_scope(plan.source_effects)
    original = _source_only_operation(model, plan.context, scope_id)
    if inspect_legacy_selection(original) != plan.legacy:
        raise DataflowOpError("legacy source changed since migration was planned")
    expected_effects = _migration_effects(
        original,
        plan.legacy,
        plan.replacement_choices,
    )
    if expected_effects != plan.source_effects:
        raise DataflowOpError("migration effects differ from the validated migration plan")

    def validate(candidate_model: Any) -> None:
        source = _source_only_operation(candidate_model, plan.context, scope_id)
        if source.problem_fingerprint != _written_source_fingerprint(plan.source_effects):
            raise DataflowOpError("migrated source facts differ from the planned problem")
        migrated = _apply_expected_choices(source, plan.replacement_choices)
        check_commitment(
            {CommitmentStage.DATAFLOW: migrated.dataflow},
            CommitmentStage.DATAFLOW,
        )
        hydrated = _hydrate_candidate(migrated)
        _assert_expected_choices(hydrated, plan.replacement_choices)
        if plan.selected_candidate is not None:
            rebind_selected_graph(migrated, plan.selected_candidate, update_origin=False)

    def finish(current: Any) -> PublishedSelection | DataflowOp:
        source = _source_only_operation(current, plan.context, scope_id)
        migrated = _apply_expected_choices(source, plan.replacement_choices)
        hydrated = migrated.rebind(current)
        _assert_expected_choices(hydrated, plan.replacement_choices)
        if plan.selected_candidate is None:
            return cast("DataflowOp", hydrated)
        decoded = decode_selected_graph(plan.selected_candidate)
        rebound = rebind_selected_graph(hydrated, decoded.snapshot)
        return PublishedSelection(hydrated, rebound)

    return apply_model_effects(
        model,
        plan.source_effects,
        validate=validate,
        finish=finish,
    )


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
    "PublishedSelection",
    "SelectionMigrationPlan",
    "SelectedPublicationPlan",
    "SourcePublicationContext",
    "allocate_scope_id",
    "apply_graph_effects",
    "apply_model_effects",
    "apply_selected_publication",
    "apply_selection_migration",
    "assign_dataflow_scope_ids",
    "build_values",
    "check_commitment",
    "find_node",
    "freeze_build_facts",
    "plan_selected_publication",
    "plan_selection_migration",
    "source_publication_context",
    "source_read_set",
]
