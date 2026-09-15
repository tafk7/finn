# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One bounded transaction engine for ONNX model effects.

The records in this module are detached and operation-neutral. Source commit,
selected publication, migration, and selected transforms lower their own plans
to :class:`ModelEffects`; this module owns the one implementation that checks
their reads, preflights the prepared final graph, applies the writes, and keeps
final hydration or decoding inside complete-model rollback.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast, TypeVar

from finn.dataflow.model.datatypes import canonical_qonnx_datatype
from finn.dataflow.ops.base import DataflowOpError
from finn.dataflow.ops.native import NativeAttribute, SCOPE_ID_ATTRIBUTE, read_attributes

T = TypeVar("T")


class ObservationMutationError(DataflowOpError):
    """An observational callback wrote through its detached or retained alias."""


class ModelReadKind(str, Enum):
    """One supported address form for a plan's live-model preconditions."""

    NODE = "node"
    NODE_ORDER = "node_order"
    ATTRIBUTE = "attribute"
    OPERAND_SLOT = "operand_slot"
    GRAPH_INPUT = "graph_input"
    VALUE_INFO = "value_info"
    TENSOR_FACT = "tensor_fact"
    QUANTIZATION_ANNOTATION = "quantization_annotation"
    INITIALIZER_CONTENT = "initializer_content"
    METADATA = "metadata"
    OPSET = "opset"
    SNAPSHOT_DIGEST = "snapshot_digest"


MODEL_READ_PRESENT = b"__finn_model_read_present_v1__"


@dataclass(frozen=True, slots=True)
class ModelReadExpectation:
    """One exact fact read while a mutation plan was prepared.

    Node-owned reads use a stable source scope id or selected-v2 node id.
    ``OPERAND_SLOT`` fields are ``input:N`` or ``output:N``. Ordinal tensor
    reads use ``input:N:shape``, ``input:N:carrier_dtype``, or
    ``input:N:logical_datatype`` so a coherent input rename can remain valid.
    ``INITIALIZER_CONTENT`` likewise uses an ``input:N`` field.
    """

    kind: ModelReadKind
    owner: str
    field: str | int | None
    expected: str | bytes | int | None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ModelReadKind):
            raise TypeError("model read kind must be ModelReadKind")
        if type(self.owner) is not str:
            raise TypeError("model read owner must be str")
        if self.field is not None and type(self.field) not in (str, int):
            raise TypeError("model read field must be str, int, or None")
        if self.expected is not None and type(self.expected) not in (str, bytes, int):
            raise TypeError("model read expected value must be str, bytes, int, or None")


@dataclass(frozen=True, slots=True)
class ModelReadSet:
    expectations: tuple[ModelReadExpectation, ...] = ()

    def __post_init__(self) -> None:
        expectations = tuple(self.expectations)
        if any(not isinstance(item, ModelReadExpectation) for item in expectations):
            raise TypeError("model read sets contain ModelReadExpectation values")
        keys = tuple((item.kind, item.owner, item.field) for item in expectations)
        if len(keys) != len(set(keys)):
            raise ValueError("model read expectations must have unique kind/owner/field keys")
        object.__setattr__(self, "expectations", expectations)


@dataclass(frozen=True, slots=True)
class ModelEffects:
    """The complete supported write set for one atomic model change."""

    read_set: ModelReadSet = ModelReadSet()
    remove_nodes: tuple[str, ...] = ()
    replace_nodes: tuple[tuple[str, bytes], ...] = ()
    insert_nodes: tuple[tuple[int, bytes], ...] = ()
    node_order: tuple[str, ...] | None = None
    rename_values: tuple[tuple[str, str], ...] = ()
    remove_graph_inputs: tuple[str, ...] = ()
    remove_value_info: tuple[str, ...] = ()
    remove_initializers: tuple[str, ...] = ()
    set_initializers: tuple[tuple[str, bytes], ...] = ()
    replace_quantization_annotations: tuple[bytes, ...] | None = None
    remove_metadata: tuple[str, ...] = ()
    set_metadata: tuple[tuple[str, str], ...] = ()
    remove_attributes: tuple[tuple[str, str], ...] = ()
    set_attributes: tuple[tuple[str, str, NativeAttribute], ...] = ()
    tensor_datatypes: tuple[tuple[str, Any | None], ...] = ()
    tensor_shapes: tuple[tuple[str, tuple[int, ...] | None], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.read_set, ModelReadSet):
            raise TypeError("read_set must be a ModelReadSet")
        object.__setattr__(self, "remove_nodes", _freeze_names(self.remove_nodes, "remove_nodes"))
        object.__setattr__(
            self,
            "replace_nodes",
            tuple(
                (
                    _record_name(record, 0, "replace_nodes"),
                    _record_bytes(record, 1, "replace_nodes"),
                )
                for record in self.replace_nodes
            ),
        )
        object.__setattr__(
            self,
            "insert_nodes",
            tuple(
                (
                    _record_int(record, 0, "insert_nodes"),
                    _record_bytes(record, 1, "insert_nodes"),
                )
                for record in self.insert_nodes
            ),
        )
        object.__setattr__(
            self,
            "rename_values",
            tuple(
                (
                    _record_name(record, 0, "rename_values"),
                    _record_name(record, 1, "rename_values"),
                )
                for record in self.rename_values
            ),
        )
        object.__setattr__(
            self,
            "remove_graph_inputs",
            _freeze_names(self.remove_graph_inputs, "remove_graph_inputs"),
        )
        object.__setattr__(
            self,
            "remove_value_info",
            _freeze_names(self.remove_value_info, "remove_value_info"),
        )
        object.__setattr__(
            self,
            "remove_initializers",
            _freeze_names(self.remove_initializers, "remove_initializers"),
        )
        object.__setattr__(
            self,
            "set_initializers",
            tuple(
                (
                    _record_name(record, 0, "set_initializers"),
                    _record_bytes(record, 1, "set_initializers"),
                )
                for record in self.set_initializers
            ),
        )
        object.__setattr__(
            self, "remove_metadata", _freeze_names(self.remove_metadata, "remove_metadata")
        )
        object.__setattr__(
            self,
            "set_metadata",
            tuple(
                (
                    _record_name(record, 0, "set_metadata"),
                    _record_string(record, 1, "set_metadata"),
                )
                for record in self.set_metadata
            ),
        )
        object.__setattr__(
            self,
            "remove_attributes",
            tuple(
                (
                    _record_name(record, 0, "remove_attributes"),
                    _record_name(record, 1, "remove_attributes"),
                )
                for record in self.remove_attributes
            ),
        )
        object.__setattr__(
            self,
            "set_attributes",
            tuple(
                (
                    _record_name(record, 0, "set_attributes"),
                    _record_name(record, 1, "set_attributes"),
                    _freeze_native_attribute(_record_item(record, 2, 3, "set_attributes")),
                )
                for record in self.set_attributes
            ),
        )
        object.__setattr__(
            self,
            "tensor_datatypes",
            tuple(
                (
                    _record_name(record, 0, "tensor_datatypes"),
                    _freeze_datatype(_record_item(record, 1, 2, "tensor_datatypes")),
                )
                for record in self.tensor_datatypes
            ),
        )
        object.__setattr__(
            self,
            "tensor_shapes",
            tuple(
                (
                    _record_name(record, 0, "tensor_shapes"),
                    _freeze_shape(_record_item(record, 1, 2, "tensor_shapes")),
                )
                for record in self.tensor_shapes
            ),
        )
        if self.node_order is not None:
            object.__setattr__(self, "node_order", _freeze_names(self.node_order, "node_order"))
        if self.replace_quantization_annotations is not None:
            object.__setattr__(
                self,
                "replace_quantization_annotations",
                tuple(
                    _freeze_bytes(value, "replace_quantization_annotations")
                    for value in self.replace_quantization_annotations
                ),
            )


_WRITE_RECORD_LENGTHS = {
    "replace_nodes": 2,
    "insert_nodes": 2,
    "rename_values": 2,
    "set_initializers": 2,
    "set_metadata": 2,
    "remove_attributes": 2,
    "set_attributes": 3,
    "tensor_datatypes": 2,
    "tensor_shapes": 2,
}


def _record_item(value: object, index: int, length: int, what: str) -> object:
    if type(value) not in (tuple, list) or len(value) != length:  # type: ignore[arg-type]
        raise TypeError(f"{what} records must be {length}-item tuples or lists")
    return value[index]  # type: ignore[index]


def _record_name(value: object, index: int, what: str) -> str:
    item = _record_string(value, index, what)
    if not item:
        raise ValueError(f"{what} names must be non-empty")
    return item


def _record_string(value: object, index: int, what: str) -> str:
    item = _record_item(value, index, _WRITE_RECORD_LENGTHS[what], what)
    if type(item) is not str:
        raise TypeError(f"{what} string fields must be str")
    return item


def _record_bytes(value: object, index: int, what: str) -> bytes:
    item = _record_item(value, index, _WRITE_RECORD_LENGTHS[what], what)
    return _freeze_bytes(item, what)


def _record_int(value: object, index: int, what: str) -> int:
    item = _record_item(value, index, _WRITE_RECORD_LENGTHS[what], what)
    if type(item) is not int:
        raise TypeError(f"{what} integer fields must be int")
    return item


def _freeze_names(values: Sequence[object], what: str) -> tuple[str, ...]:
    result = []
    for value in values:
        if type(value) is not str:
            raise TypeError(f"{what} values must be str")
        if not value:
            raise ValueError(f"{what} values must be non-empty")
        result.append(value)
    return tuple(result)


def _freeze_bytes(value: object, what: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{what} protobuf values must be bytes-like")
    return bytes(value)


def _freeze_native_attribute(value: object) -> NativeAttribute:
    if not isinstance(value, NativeAttribute):
        raise TypeError("set_attributes values must be NativeAttribute")
    from onnx import NodeProto  # type: ignore[import-not-found] # noqa: PLC0415

    node = NodeProto()
    try:
        node.attribute.append(value.proto("value"))
        return read_attributes(node)["value"]
    except Exception as error:
        raise ValueError(f"invalid NativeAttribute: {error}") from error


def _freeze_datatype(value: object) -> Any | None:
    return None if value is None else canonical_qonnx_datatype(value)


def _freeze_shape(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)):
        raise TypeError("tensor_shapes values must be tuples, lists, or None")
    if any(type(extent) is not int for extent in value):
        raise TypeError("tensor_shapes extents must be int")
    return tuple(cast(int, extent) for extent in value)


def model_snapshot_digest(model: Any) -> str:
    """The immutable selected-input stale guard used by ``SNAPSHOT_DIGEST``."""

    encoded = model.model.SerializeToString(deterministic=True)
    return hashlib.sha256(encoded).hexdigest()


def merge_model_read_sets(*read_sets: ModelReadSet) -> ModelReadSet:
    """Merge exact observations without weakening or last-wins behavior.

    Equal keyed reads coalesce.  For one ``OPERAND_SLOT`` key, an exact tensor
    name is stronger than the presence marker and replaces it.  Every other
    unequal duplicate is contradictory and refuses the plan.
    """

    by_key: dict[tuple[ModelReadKind, str, str | int | None], ModelReadExpectation] = {}
    for read_set in read_sets:
        if not isinstance(read_set, ModelReadSet):
            raise TypeError("merge_model_read_sets accepts ModelReadSet values")
        for expectation in read_set.expectations:
            key = (expectation.kind, expectation.owner, expectation.field)
            previous = by_key.get(key)
            if previous is None:
                by_key[key] = expectation
                continue
            if previous.expected == expectation.expected and type(previous.expected) is type(
                expectation.expected
            ):
                continue
            if expectation.kind is ModelReadKind.OPERAND_SLOT:
                if previous.expected == MODEL_READ_PRESENT and type(expectation.expected) is str:
                    by_key[key] = expectation
                    continue
                if expectation.expected == MODEL_READ_PRESENT and type(previous.expected) is str:
                    continue
            raise DataflowOpError(
                f"contradictory model reads for {expectation.kind.value} "
                f"{expectation.owner!r} field {expectation.field!r}"
            )
    ordered = tuple(
        by_key[key]
        for key in sorted(
            by_key,
            key=lambda item: (
                item[0].value,
                item[1].encode("utf-8"),
                "" if item[2] is None else f"{type(item[2]).__name__}:{item[2]}",
            ),
        )
    )
    return ModelReadSet(ordered)


def validate_model_read_set(model: Any, read_set: ModelReadSet) -> None:
    """Check a detached read set against the current wrapper."""

    _check_read_set(model, read_set)


def checked_model_observation(
    model: Any,
    observe: Callable[[Any], T],
    *,
    what: str = "model observer",
) -> T:
    """Run an observational callback on a detached view and guard live aliases."""

    original = model.model.SerializeToString(deterministic=True)
    detached = _wrapper_from_bytes(model, original)
    detached_original = detached.model.SerializeToString(deterministic=True)
    try:
        try:
            result = observe(detached)
        except Exception as error:
            if (
                detached.model.SerializeToString(deterministic=True) != detached_original
                or model.model.SerializeToString(deterministic=True) != original
            ):
                raise ObservationMutationError(
                    f"the {what} mutated a model while raising"
                ) from error
            raise
        if detached.model.SerializeToString(deterministic=True) != detached_original:
            raise ObservationMutationError(f"the {what} mutated its detached validation model")
        if model.model.SerializeToString(deterministic=True) != original:
            raise ObservationMutationError(f"the {what} mutated the live model through an alias")
        return result
    except Exception:
        if model.model.SerializeToString(deterministic=True) != original:
            model.model.ParseFromString(original)
        raise


def apply_model_effects(
    model: Any,
    effects: ModelEffects,
    *,
    validate: Callable[[Any], object],
    finish: Callable[[Any], T],
) -> T:
    """Apply one checked model change and return only after final hydration.

    Unsupported structure and the caller's validation run on a detached
    prepared candidate before the live model is touched. The same writes are
    then applied to the live wrapper, and ``finish`` runs inside the full-model
    rollback boundary. Both callbacks are read-only: a callback that mutates
    its wrapper is itself a transaction failure.
    """

    original = model.model.SerializeToString(deterministic=True)
    try:
        prepared = _prepare_effects(model, effects)
        _check_read_set(model, effects.read_set)
        candidate = _wrapper_from_bytes(model, original)
        _apply_prepared(candidate, effects, prepared)
        _check_supported_final_state(candidate, effects, prepared)
        prepared_bytes = candidate.model.SerializeToString(deterministic=True)
        validate(candidate)
        if candidate.model.SerializeToString(deterministic=True) != prepared_bytes:
            raise DataflowOpError("the model-effects validate callback mutated the candidate")
        if model.model.SerializeToString(deterministic=True) != original:
            raise DataflowOpError("the model-effects validate callback mutated the live model")

        _apply_prepared(model, effects, prepared)
        _check_supported_final_state(model, effects, prepared)
        if model.model.SerializeToString(deterministic=True) != prepared_bytes:
            raise DataflowOpError("live model effects differ from the prepared candidate")
        result = finish(model)
        if model.model.SerializeToString(deterministic=True) != prepared_bytes:
            raise DataflowOpError("the model-effects finish callback mutated the candidate")
        return result
    except Exception:
        if model.model.SerializeToString(deterministic=True) != original:
            model.model.ParseFromString(original)
        raise


@dataclass(frozen=True, slots=True)
class _PreparedEffects:
    old_node_ids: Mapping[str, int]
    replacement_nodes: Mapping[str, Any]
    inserted_nodes: tuple[tuple[int, Any], ...]
    initializers: Mapping[str, Any]
    annotations: tuple[Any, ...] | None
    old_opsets: bytes
    old_outputs: tuple[str, ...]
    old_output_producers: tuple[tuple[str, int] | None, ...]
    old_external_inputs: frozenset[str]
    output_producers_require_stable_ids: bool


def _prepare_effects(model: Any, effects: ModelEffects) -> _PreparedEffects:
    from onnx import NodeProto, TensorAnnotation, TensorProto  # noqa: PLC0415

    _unique("remove_nodes", effects.remove_nodes)
    _unique_first("replace_nodes", effects.replace_nodes)
    _unique_first("insert_nodes", effects.insert_nodes)
    _unique_first("rename_values", effects.rename_values)
    _unique("remove_graph_inputs", effects.remove_graph_inputs)
    _unique("remove_value_info", effects.remove_value_info)
    _unique("remove_initializers", effects.remove_initializers)
    _unique_first("set_initializers", effects.set_initializers)
    _unique("remove_metadata", effects.remove_metadata)
    _unique_first("set_metadata", effects.set_metadata)
    _unique("remove_attributes", effects.remove_attributes)
    _unique("set_attributes", tuple((owner, name) for owner, name, _ in effects.set_attributes))
    _unique_first("tensor_datatypes", effects.tensor_datatypes)
    _unique_first("tensor_shapes", effects.tensor_shapes)

    remove_nodes = set(effects.remove_nodes)
    replace_nodes = {owner for owner, _ in effects.replace_nodes}
    if remove_nodes & replace_nodes:
        raise DataflowOpError("one node cannot be both removed and replaced")
    if effects.node_order is not None and effects.insert_nodes:
        raise DataflowOpError("node_order cannot address inserted nodes in ModelEffects v1")
    if effects.node_order is not None and len(effects.node_order) != len(set(effects.node_order)):
        raise DataflowOpError("node_order contains duplicate stable node ids")

    renames = dict(effects.rename_values)
    if any(not old or not new for old, new in effects.rename_values):
        raise DataflowOpError("value renames require non-empty names")
    if len(set(renames.values())) != len(renames):
        raise DataflowOpError("value renames may not merge distinct values")
    existing_values = _all_value_names(model)
    missing_rename_sources = set(renames) - existing_values
    if missing_rename_sources:
        raise DataflowOpError(
            f"value rename sources are absent: {sorted(missing_rename_sources)!r}"
        )
    collisions = set(renames.values()) & (existing_values - set(renames))
    if collisions:
        raise DataflowOpError(f"value rename target already exists: {sorted(collisions)!r}")
    removed_names = (
        set(effects.remove_graph_inputs)
        | set(effects.remove_value_info)
        | set(effects.remove_initializers)
    )
    if set(renames) & removed_names:
        raise DataflowOpError("a value cannot be renamed and removed by the same effect")

    graph_inputs = _unique_named(model.graph.input, "graph input")
    initializer_names = _unique_named(model.graph.initializer, "initializer")
    missing_inputs = set(effects.remove_graph_inputs) - graph_inputs
    if missing_inputs:
        raise DataflowOpError(f"graph inputs are absent: {sorted(missing_inputs)!r}")
    nonstatic_inputs = set(effects.remove_graph_inputs) - initializer_names
    if nonstatic_inputs:
        raise DataflowOpError(
            "ModelEffects v1 removes only static initializer-backed graph inputs: "
            f"{sorted(nonstatic_inputs)!r}"
        )
    if set(effects.remove_graph_inputs) & set(effects.remove_initializers):
        raise DataflowOpError("static graph-input removal must retain its initializer")

    value_info_names = _unique_named(model.graph.value_info, "value_info")
    missing_value_info = set(effects.remove_value_info) - value_info_names
    if missing_value_info:
        raise DataflowOpError(f"value_info entries are absent: {sorted(missing_value_info)!r}")

    old_node_ids = _node_id_indices(model)
    attribute_targets = {owner for owner, _ in effects.remove_attributes} | {
        owner for owner, _, _ in effects.set_attributes
    }
    referenced = remove_nodes | replace_nodes | attribute_targets
    missing = referenced - set(old_node_ids)
    if missing:
        raise DataflowOpError(f"unknown stable node id(s): {sorted(missing)!r}")
    if remove_nodes & attribute_targets:
        raise DataflowOpError("attributes cannot target a node removed by the same effect")

    has_subgraphs, nested_values = _nested_graph_values(model)
    topology_change = bool(
        effects.remove_nodes
        or effects.replace_nodes
        or effects.insert_nodes
        or effects.node_order is not None
    )
    if has_subgraphs and topology_change:
        raise DataflowOpError(
            "ModelEffects v1 supports structural node writes only on flat graphs without subgraphs"
        )
    renamed_values = set(renames) | set(renames.values())
    if renamed_values & nested_values:
        raise DataflowOpError(
            "ModelEffects v1 cannot rename values referenced by or declared in subgraphs"
        )
    if set(effects.remove_initializers) & nested_values:
        raise DataflowOpError("ModelEffects v1 cannot remove initializers referenced by subgraphs")
    for owner, name in (
        *effects.remove_attributes,
        *((owner, name) for owner, name, _attribute in effects.set_attributes),
    ):
        if _attribute_is_subgraph(model.graph.node[old_node_ids[owner]], name):
            raise DataflowOpError(
                f"ModelEffects v1 cannot remove or replace GRAPH attribute {name!r}"
            )

    replacements: dict[str, Any] = {}
    for owner, encoded in effects.replace_nodes:
        node = NodeProto()
        try:
            node.ParseFromString(encoded)
        except Exception as error:
            raise DataflowOpError(f"replacement node {owner!r} is not a NodeProto") from error
        if not node.op_type:
            raise DataflowOpError(f"replacement node {owner!r} has no operator type")
        _reject_subgraphs(node)
        old_node = model.graph.node[old_node_ids[owner]]
        _reject_subgraphs(old_node)
        old_scope = _scope_id(old_node)
        if old_scope is not None and _scope_id(node) != old_scope:
            raise DataflowOpError(
                f"replacement for source node {owner!r} must preserve its stable scope id"
            )
        replacements[owner] = node

    inserted: list[tuple[int, Any]] = []
    for index, encoded in effects.insert_nodes:
        if type(index) is not int or index < 0:
            raise DataflowOpError("node insertion indices must be nonnegative integers")
        node = NodeProto()
        try:
            node.ParseFromString(encoded)
        except Exception as error:
            raise DataflowOpError(f"inserted node at {index} is not a NodeProto") from error
        if not node.op_type:
            raise DataflowOpError(f"inserted node at {index} has no operator type")
        _reject_subgraphs(node)
        inserted.append((index, node))

    initializers: dict[str, Any] = {}
    for name, encoded in effects.set_initializers:
        tensor = TensorProto()
        try:
            tensor.ParseFromString(encoded)
        except Exception as error:
            raise DataflowOpError(f"initializer {name!r} is not a TensorProto") from error
        if not name or tensor.name != name:
            raise DataflowOpError(f"initializer key {name!r} differs from TensorProto name")
        if int(tensor.data_type) <= 0:
            raise DataflowOpError(f"initializer {name!r} has no element type")
        initializers[name] = tensor

    annotations = None
    if effects.replace_quantization_annotations is not None:
        decoded = []
        for index, encoded in enumerate(effects.replace_quantization_annotations):
            annotation = TensorAnnotation()
            try:
                annotation.ParseFromString(encoded)
            except Exception as error:
                raise DataflowOpError(
                    f"quantization annotation {index} is not a TensorAnnotation"
                ) from error
            if not annotation.tensor_name:
                raise DataflowOpError(f"quantization annotation {index} has no tensor name")
            decoded.append(annotation)
        names = tuple(item.tensor_name for item in decoded)
        if len(names) != len(set(names)):
            raise DataflowOpError("replacement quantization annotations contain duplicates")
        annotations = tuple(decoded)

    if set(effects.remove_metadata) & {name for name, _ in effects.set_metadata}:
        raise DataflowOpError("metadata cannot be both removed and set")
    if set(effects.remove_initializers) & set(initializers):
        raise DataflowOpError("an initializer cannot be both removed and set")

    structural = bool(
        effects.remove_nodes
        or effects.replace_nodes
        or effects.insert_nodes
        or effects.node_order is not None
    )
    old_outputs = tuple(item.name for item in model.graph.output)
    old_producers = _output_producers(
        model,
        old_outputs,
        old_node_ids,
        require_stable_ids=structural,
    )
    return _PreparedEffects(
        old_node_ids,
        replacements,
        tuple(inserted),
        initializers,
        annotations,
        _opset_bytes(model),
        old_outputs,
        old_producers,
        frozenset(_external_inputs(model)),
        structural,
    )


def _apply_prepared(model: Any, effects: ModelEffects, prepared: _PreparedEffects) -> None:
    final_nodes: list[tuple[str | None, Any]] = []
    by_index = {index: owner for owner, index in prepared.old_node_ids.items()}
    removed = set(effects.remove_nodes)
    for index, node in enumerate(model.graph.node):
        owner = by_index.get(index)
        if owner in removed:
            continue
        replacement = prepared.replacement_nodes.get(owner or "")
        final_nodes.append((owner, _copy_proto(replacement if replacement is not None else node)))
    for index, node in sorted(prepared.inserted_nodes, key=lambda item: item[0]):
        if index > len(final_nodes):
            raise DataflowOpError(f"node insertion index {index} exceeds final node count")
        final_nodes.insert(index, (None, _copy_proto(node)))
    if effects.node_order is not None:
        available = {owner: node for owner, node in final_nodes if owner is not None}
        if set(effects.node_order) != set(available) or len(available) != len(final_nodes):
            raise DataflowOpError("node_order must cover every final stable node id exactly")
        final_nodes = [(owner, available[owner]) for owner in effects.node_order]
    del model.graph.node[:]
    model.graph.node.extend(node for _owner, node in final_nodes)
    final_by_id = {
        owner: model.graph.node[index]
        for index, (owner, _node) in enumerate(final_nodes)
        if owner is not None
    }

    _rename_values(model, dict(effects.rename_values))
    _remove_named(model.graph.input, set(effects.remove_graph_inputs))
    _remove_named(model.graph.value_info, set(effects.remove_value_info))
    _remove_named(model.graph.initializer, set(effects.remove_initializers))
    _set_initializers(model, prepared.initializers)
    if prepared.annotations is not None:
        del model.graph.quantization_annotation[:]
        model.graph.quantization_annotation.extend(
            _copy_proto(item) for item in prepared.annotations
        )
    _apply_metadata(model, effects)
    _apply_attributes(final_by_id, effects)
    for tensor, datatype in effects.tensor_datatypes:
        _set_logical_datatype(model, tensor, datatype)
    for tensor, shape in effects.tensor_shapes:
        _set_shape(model, tensor, shape)


def _check_supported_final_state(
    model: Any, effects: ModelEffects, prepared: _PreparedEffects
) -> None:
    if _opset_bytes(model) != prepared.old_opsets:
        raise DataflowOpError("ModelEffects v1 does not support opset changes")
    rename = dict(effects.rename_values)
    expected_outputs = tuple(rename.get(name, name) for name in prepared.old_outputs)
    current_outputs = tuple(item.name for item in model.graph.output)
    if current_outputs != expected_outputs:
        raise DataflowOpError(
            "ModelEffects v1 does not support graph-output membership or order changes"
        )
    current_ids = _node_id_indices(model)
    current_producers = _output_producers(
        model,
        current_outputs,
        current_ids,
        require_stable_ids=prepared.output_producers_require_stable_ids,
    )
    if current_producers != prepared.old_output_producers:
        raise DataflowOpError("ModelEffects v1 does not support graph-output producer changes")

    removed_initializers = set(effects.remove_initializers)
    still_used_initializers = removed_initializers & {
        value for node in model.graph.node for value in node.input if value
    }
    still_used_initializers.update(removed_initializers & {item.name for item in model.graph.input})
    still_used_initializers.update(
        removed_initializers & {item.name for item in model.graph.output}
    )
    if still_used_initializers:
        raise DataflowOpError(
            f"ModelEffects v1 removes only unused initializers: {sorted(still_used_initializers)!r}"
        )

    removed_value_info = set(effects.remove_value_info)
    final_references = {
        value for node in model.graph.node for value in (*node.input, *node.output) if value
    }
    final_references.update(item.name for item in model.graph.input)
    final_references.update(item.name for item in model.graph.output)
    final_references.update(item.name for item in model.graph.initializer)
    still_used_value_info = removed_value_info & final_references
    if still_used_value_info:
        raise DataflowOpError(
            "ModelEffects v1 removes only final-state-unused value_info entries: "
            f"{sorted(still_used_value_info)!r}"
        )

    expected_external = {rename.get(name, name) for name in prepared.old_external_inputs}
    new_external = _external_inputs(model) - expected_external
    if new_external:
        raise DataflowOpError(
            f"ModelEffects v1 does not support new external inputs: {sorted(new_external)!r}"
        )


def _check_read_set(model: Any, read_set: ModelReadSet) -> None:
    node_ids = _node_id_indices(model)
    for expectation in read_set.expectations:
        actual = _read_expectation(model, node_ids, expectation)
        if actual != expectation.expected:
            if expectation.kind is ModelReadKind.ATTRIBUTE:
                raise DataflowOpError(
                    f"Decision attribute {expectation.field!r} changed since this change was "
                    "planned; rebind and plan again"
                )
            raise DataflowOpError(
                f"model read changed for {expectation.kind.value} "
                f"{expectation.owner!r} field {expectation.field!r}; replan"
            )


def _read_expectation(
    model: Any, node_ids: Mapping[str, int], expectation: ModelReadExpectation
) -> str | bytes | int | None:
    kind = expectation.kind
    owner = expectation.owner
    field = expectation.field
    if kind is ModelReadKind.NODE:
        node = _node_for_id(model, node_ids, owner)
        if field == "operator":
            return _operator_identity(node.domain, node.op_type)
        if field == "bytes":
            return bytes(node.SerializeToString(deterministic=True))
        if field == "present":
            return MODEL_READ_PRESENT
        raise DataflowOpError(f"unsupported NODE read field {field!r}")
    if kind is ModelReadKind.NODE_ORDER:
        if owner != model.graph.name or field is not None:
            raise DataflowOpError("NODE_ORDER reads use the graph name and no field")
        inverse = {index: stable_id for stable_id, index in node_ids.items()}
        if len(inverse) != len(model.graph.node):
            raise DataflowOpError("NODE_ORDER requires every node to have a stable compiler id")
        return json.dumps(
            [inverse[index] for index in range(len(model.graph.node))],
            separators=(",", ":"),
        )
    if kind is ModelReadKind.ATTRIBUTE:
        if not isinstance(field, str):
            raise DataflowOpError("ATTRIBUTE reads require an attribute name")
        node = _node_for_id(model, node_ids, owner)
        values = [item for item in node.attribute if item.name == field]
        _at_most_one(values, f"node attribute {field!r}")
        return None if not values else values[0].SerializeToString(deterministic=True)
    if kind is ModelReadKind.OPERAND_SLOT:
        direction, index = _operand_slot(field)
        value = _operand_name(_node_for_id(model, node_ids, owner), direction, index)
        if value is None:
            return None
        return MODEL_READ_PRESENT if expectation.expected == MODEL_READ_PRESENT else value
    if kind is ModelReadKind.GRAPH_INPUT:
        values = model.graph.input
        if isinstance(field, int):
            return (
                None
                if field < 0 or field >= len(values)
                else values[field].SerializeToString(deterministic=True)
            )
        name = owner if field is None else str(field)
        matches = [item for item in values if item.name == name]
        _at_most_one(matches, f"graph input {name!r}")
        return None if not matches else matches[0].SerializeToString(deterministic=True)
    if kind is ModelReadKind.VALUE_INFO:
        matches = _value_info_matches(model, owner, field)
        _at_most_one(matches, f"value information {owner!r}")
        return None if not matches else matches[0].SerializeToString(deterministic=True)
    if kind is ModelReadKind.TENSOR_FACT:
        direction, index, fact = _ordinal_fact(field)
        tensor = _operand_name(_node_for_id(model, node_ids, owner), direction, index)
        if tensor is None:
            return None
        if fact == "shape":
            shape = model.get_tensor_shape(tensor)
            return None if shape is None else json.dumps(list(shape), separators=(",", ":"))
        if fact == "logical_datatype":
            return _logical_datatype(model, tensor)
        if fact == "carrier_dtype":
            return _carrier_dtype(model, tensor)
        raise DataflowOpError(f"unsupported TENSOR_FACT field {fact!r}")
    if kind is ModelReadKind.QUANTIZATION_ANNOTATION:
        values = [item for item in model.graph.quantization_annotation if item.tensor_name == owner]
        _at_most_one(values, f"quantization annotation {owner!r}")
        return None if not values else values[0].SerializeToString(deterministic=True)
    if kind is ModelReadKind.INITIALIZER_CONTENT:
        direction, index = _operand_slot(field)
        if direction != "input":
            raise DataflowOpError("INITIALIZER_CONTENT reads require an input ordinal")
        tensor = _operand_name(_node_for_id(model, node_ids, owner), direction, index)
        if tensor is None:
            return None
        return _initializer_digest(model, tensor)
    if kind is ModelReadKind.METADATA:
        if field is not None:
            raise DataflowOpError("METADATA reads use the key as owner and no field")
        values = [item.value for item in model.graph.metadata_props if item.key == owner]
        _at_most_one(values, f"metadata {owner!r}")
        return None if not values else values[0]
    if kind is ModelReadKind.OPSET:
        if field is not None:
            raise DataflowOpError("OPSET reads use the domain as owner and no field")
        values = [int(item.version) for item in model.model.opset_import if item.domain == owner]
        _at_most_one(values, f"opset {owner!r}")
        return None if not values else values[0]
    if kind is ModelReadKind.SNAPSHOT_DIGEST:
        if owner or field is not None:
            raise DataflowOpError("SNAPSHOT_DIGEST reads use an empty owner and no field")
        return model_snapshot_digest(model)
    raise DataflowOpError(f"unsupported model read kind {kind!r}")


def _node_id_indices(model: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    by_index: dict[int, str] = {}
    for index, node in enumerate(model.graph.node):
        ids = [
            _text(attribute.s)
            for attribute in node.attribute
            if attribute.name == SCOPE_ID_ATTRIBUTE
        ]
        if len(ids) > 1:
            raise DataflowOpError("a node has duplicate dataflow scope-id attributes")
        if ids:
            _add_node_id(result, by_index, ids[0], index)

    try:
        from finn.dataflow.ops.selected import (  # noqa: PLC0415
            SELECTED_METADATA_KEY,
            decode_selected_declaration,
        )

        selected_values = [
            item.value for item in model.graph.metadata_props if item.key == SELECTED_METADATA_KEY
        ]
        if len(selected_values) > 1:
            raise DataflowOpError("selected metadata is duplicated")
        if selected_values:
            declaration = decode_selected_declaration(selected_values[0])
            if len(declaration.graph_nodes) != len(model.graph.node):
                raise DataflowOpError("selected stable node bindings do not cover the graph")
            for record in declaration.graph_nodes:
                if record.index < 0 or record.index >= len(model.graph.node):
                    raise DataflowOpError(
                        f"selected node id {record.node_id!r} has an invalid index"
                    )
                _add_node_id(result, by_index, record.node_id, record.index)
    except DataflowOpError:
        raise
    except Exception as error:
        raise DataflowOpError(f"cannot resolve selected stable node ids: {error}") from error
    return result


def _add_node_id(
    result: dict[str, int], by_index: dict[int, str], stable_id: str, index: int
) -> None:
    if not stable_id:
        raise DataflowOpError("stable node ids must be non-empty")
    if stable_id in result and result[stable_id] != index:
        raise DataflowOpError(f"stable node id {stable_id!r} is duplicated")
    if index in by_index and by_index[index] != stable_id:
        raise DataflowOpError(
            f"node index {index} has two stable ids: {by_index[index]!r} and {stable_id!r}"
        )
    result[stable_id] = index
    by_index[index] = stable_id


def _node_for_id(model: Any, node_ids: Mapping[str, int], stable_id: str) -> Any:
    try:
        return model.graph.node[node_ids[stable_id]]
    except KeyError as error:
        raise DataflowOpError(f"unknown stable node id {stable_id!r}") from error


def _output_producers(
    model: Any,
    output_names: Sequence[str],
    node_ids: Mapping[str, int],
    *,
    require_stable_ids: bool,
) -> tuple[tuple[str, int] | None, ...]:
    inverse = {index: stable_id for stable_id, index in node_ids.items()}
    result: list[tuple[str, int] | None] = []
    for value in output_names:
        matches = [
            (node_index, inverse.get(node_index), slot)
            for node_index, node in enumerate(model.graph.node)
            for slot, output in enumerate(node.output)
            if output == value
        ]
        if len(matches) > 1:
            raise DataflowOpError(f"graph output {value!r} has multiple producers")
        if not matches:
            result.append(None)
            continue
        node_index, stable_id, slot = matches[0]
        if stable_id is None:
            if require_stable_ids:
                raise DataflowOpError(
                    f"graph output producer for {value!r} has no stable compiler id"
                )
            stable_id = f"@ordinal:{node_index}"
        result.append((stable_id, slot))
    return tuple(result)


def _rename_values(model: Any, renames: Mapping[str, str]) -> None:
    if not renames:
        return
    for values in (model.graph.input, model.graph.output, model.graph.value_info):
        for item in values:
            item.name = renames.get(item.name, item.name)
    for tensor in model.graph.initializer:
        tensor.name = renames.get(tensor.name, tensor.name)
    for annotation in model.graph.quantization_annotation:
        annotation.tensor_name = renames.get(annotation.tensor_name, annotation.tensor_name)
        for parameter in annotation.quant_parameter_tensor_names:
            if parameter.key != "finn_datatype":
                parameter.value = renames.get(parameter.value, parameter.value)
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            node.input[index] = renames.get(value, value)
        for index, value in enumerate(node.output):
            node.output[index] = renames.get(value, value)


def _remove_named(values: Any, names: set[str]) -> None:
    if not names:
        return
    present = [item.name for item in values]
    if len(present) != len(set(present)):
        raise DataflowOpError("named model entries must be unique")
    missing = names - set(present)
    if missing:
        raise DataflowOpError(f"named model entries are absent: {sorted(missing)!r}")
    kept = [item for item in values if item.name not in names]
    del values[:]
    values.extend(kept)


def _set_initializers(model: Any, initializers: Mapping[str, Any]) -> None:
    if not initializers:
        return
    indices = {item.name: index for index, item in enumerate(model.graph.initializer)}
    if len(indices) != len(model.graph.initializer):
        raise DataflowOpError("initializer names must be unique")
    for name, tensor in initializers.items():
        copied = _copy_proto(tensor)
        if name in indices:
            model.graph.initializer[indices[name]].CopyFrom(copied)
        else:
            model.graph.initializer.append(copied)


def _apply_metadata(model: Any, effects: ModelEffects) -> None:
    remove = set(effects.remove_metadata)
    existing: dict[str, Any] = {}
    for item in model.graph.metadata_props:
        if item.key in existing:
            raise DataflowOpError(f"duplicate metadata key {item.key!r}")
        existing[item.key] = item
    missing = remove - set(existing)
    if missing:
        raise DataflowOpError(f"metadata keys are absent: {sorted(missing)!r}")
    kept = [item for item in model.graph.metadata_props if item.key not in remove]
    del model.graph.metadata_props[:]
    model.graph.metadata_props.extend(kept)
    for key, value in effects.set_metadata:
        matches = [item for item in model.graph.metadata_props if item.key == key]
        if matches:
            matches[0].value = value
        else:
            item = model.graph.metadata_props.add()
            item.key = key
            item.value = value


def _apply_attributes(nodes: Mapping[str, Any], effects: ModelEffects) -> None:
    for owner, name in effects.remove_attributes:
        _drop(nodes[owner], name)
    for owner, name, attribute in effects.set_attributes:
        node = nodes[owner]
        _drop(node, name)
        node.attribute.append(attribute.proto(name))


def _set_logical_datatype(model: Any, tensor: str, datatype: Any | None) -> None:
    if tensor not in _all_value_names(model):
        raise DataflowOpError(f"logical datatype target {tensor!r} is not a graph value")
    annotations = [
        item for item in model.graph.quantization_annotation if item.tensor_name == tensor
    ]
    _at_most_one(annotations, f"quantization annotation {tensor!r}")
    annotation = None if not annotations else annotations[0]
    if annotation is not None:
        entries = [
            item for item in annotation.quant_parameter_tensor_names if item.key == "finn_datatype"
        ]
        _at_most_one(entries, f"finn_datatype entry for {tensor!r}")
    else:
        entries = []
    if datatype is None:
        if annotation is None or not entries:
            return
        kept = [
            item for item in annotation.quant_parameter_tensor_names if item.key != "finn_datatype"
        ]
        del annotation.quant_parameter_tensor_names[:]
        annotation.quant_parameter_tensor_names.extend(kept)
        if not annotation.quant_parameter_tensor_names:
            kept_annotations = [
                item for item in model.graph.quantization_annotation if item.tensor_name != tensor
            ]
            del model.graph.quantization_annotation[:]
            model.graph.quantization_annotation.extend(kept_annotations)
        return
    if annotation is None:
        annotation = model.graph.quantization_annotation.add()
        annotation.tensor_name = tensor
    if entries:
        entries[0].value = datatype.name
    else:
        entry = annotation.quant_parameter_tensor_names.add()
        entry.key = "finn_datatype"
        entry.value = datatype.name


def _set_shape(model: Any, tensor: str, shape: tuple[int, ...] | None) -> None:
    info = model.get_tensor_valueinfo(tensor)
    if info is None:
        raise DataflowOpError(
            f"shape target {tensor!r} has no existing graph input/output/value_info"
        )
    if shape is None:
        info.type.tensor_type.ClearField("shape")
        return
    model.set_tensor_shape(tensor, list(shape))


def _logical_datatype(model: Any, tensor: str) -> str | None:
    annotations = [
        item for item in model.graph.quantization_annotation if item.tensor_name == tensor
    ]
    _at_most_one(annotations, f"quantization annotation {tensor!r}")
    if not annotations:
        return None
    values = [
        item.value
        for item in annotations[0].quant_parameter_tensor_names
        if item.key == "finn_datatype"
    ]
    _at_most_one(values, f"finn_datatype entry for {tensor!r}")
    return None if not values else values[0]


def _carrier_dtype(model: Any, tensor: str) -> int | None:
    initializers = [item for item in model.graph.initializer if item.name == tensor]
    _at_most_one(initializers, f"initializer {tensor!r}")
    if initializers:
        return int(initializers[0].data_type)
    info = model.get_tensor_valueinfo(tensor)
    if info is None:
        return None
    value = int(info.type.tensor_type.elem_type)
    return None if value <= 0 else value


def _initializer_digest(model: Any, tensor: str) -> str | None:
    from finn.dataflow.ops.tensor_summary import FrozenInitializer  # noqa: PLC0415

    values = [item for item in model.graph.initializer if item.name == tensor]
    _at_most_one(values, f"initializer {tensor!r}")
    return (
        None
        if not values
        else FrozenInitializer.from_tensor_proto(values[0]).summary.content_digest
    )


def _operand_slot(field: str | int | None) -> tuple[str, int]:
    if not isinstance(field, str):
        raise DataflowOpError("operand reads require 'input:N' or 'output:N'")
    parts = field.split(":")
    if len(parts) != 2 or parts[0] not in {"input", "output"}:
        raise DataflowOpError("operand reads require 'input:N' or 'output:N'")
    try:
        index = int(parts[1])
    except ValueError as error:
        raise DataflowOpError(f"invalid operand index {parts[1]!r}") from error
    if index < 0:
        raise DataflowOpError("operand indices must be nonnegative")
    return parts[0], index


def _ordinal_fact(field: str | int | None) -> tuple[str, int, str]:
    if not isinstance(field, str):
        raise DataflowOpError("ordinal tensor facts require 'input:N:fact'")
    parts = field.split(":")
    if len(parts) != 3:
        raise DataflowOpError("ordinal tensor facts require 'input:N:fact'")
    direction, index = _operand_slot(":".join(parts[:2]))
    if parts[2] not in {"shape", "carrier_dtype", "logical_datatype"}:
        raise DataflowOpError(f"unsupported ordinal tensor fact {parts[2]!r}")
    return direction, index, parts[2]


def _operand_name(node: Any, direction: str, index: int) -> str | None:
    values = node.input if direction == "input" else node.output
    if index >= len(values) or not values[index]:
        return None
    return str(values[index])


def _value_info_matches(model: Any, owner: str, field: str | int | None) -> list[Any]:
    containers = {
        "input": model.graph.input,
        "output": model.graph.output,
        "value_info": model.graph.value_info,
    }
    if field is None:
        return [item for values in containers.values() for item in values if item.name == owner]
    if not isinstance(field, str) or field not in containers:
        raise DataflowOpError("VALUE_INFO fields are 'input', 'output', 'value_info', or None")
    return [item for item in containers[field] if item.name == owner]


def _external_inputs(model: Any) -> set[str]:
    produced = {value for node in model.graph.node for value in node.output if value}
    initialized = {item.name for item in model.graph.initializer}
    declared = {item.name for item in model.graph.input}
    used = {value for node in model.graph.node for value in node.input if value}
    return (declared - initialized) | (used - produced - initialized)


def _all_value_names(model: Any) -> set[str]:
    values = {value for node in model.graph.node for value in (*node.input, *node.output) if value}
    values.update(item.name for item in model.graph.input)
    values.update(item.name for item in model.graph.output)
    values.update(item.name for item in model.graph.value_info)
    values.update(item.name for item in model.graph.initializer)
    values.update(item.tensor_name for item in model.graph.quantization_annotation)
    values.update(
        entry.value
        for annotation in model.graph.quantization_annotation
        for entry in annotation.quant_parameter_tensor_names
        if entry.key != "finn_datatype"
    )
    return values


def _nested_graph_values(model: Any) -> tuple[bool, set[str]]:
    from onnx import AttributeProto  # noqa: PLC0415

    found = False
    values: set[str] = set()

    def visit_node(node: Any) -> None:
        nonlocal found
        for attribute in node.attribute:
            graphs = (
                (attribute.g,)
                if attribute.type == AttributeProto.GRAPH
                else tuple(attribute.graphs)
                if attribute.type == AttributeProto.GRAPHS
                else ()
            )
            for graph in graphs:
                found = True
                values.update(item.name for item in graph.input)
                values.update(item.name for item in graph.output)
                values.update(item.name for item in graph.value_info)
                values.update(item.name for item in graph.initializer)
                values.update(
                    value
                    for nested in graph.node
                    for value in (*nested.input, *nested.output)
                    if value
                )
                values.update(item.tensor_name for item in graph.quantization_annotation)
                for nested in graph.node:
                    visit_node(nested)

    for node in model.graph.node:
        visit_node(node)
    return found, values


def _attribute_is_subgraph(node: Any, name: str) -> bool:
    from onnx import AttributeProto  # noqa: PLC0415

    return any(
        item.name == name and item.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS)
        for item in node.attribute
    )


def _reject_subgraphs(node: Any) -> None:
    from onnx import AttributeProto  # noqa: PLC0415

    if any(item.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS) for item in node.attribute):
        raise DataflowOpError("ModelEffects v1 does not support editing nodes with subgraphs")


def _scope_id(node: Any) -> str | None:
    values = [
        _text(attribute.s) for attribute in node.attribute if attribute.name == SCOPE_ID_ATTRIBUTE
    ]
    if len(values) > 1:
        raise DataflowOpError("a node has duplicate dataflow scope-id attributes")
    return None if not values else values[0]


def _wrapper_from_bytes(model: Any, encoded: bytes) -> Any:
    from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found] # noqa: PLC0415

    proto = model.model.__class__()
    proto.ParseFromString(encoded)
    wrapper = ModelWrapper(proto)
    # ModelWrapper may repair missing initializer value-info while it starts.
    # A private transaction candidate must begin at the exact snapshotted bytes;
    # otherwise its implicit repair would become an undeclared write.
    if wrapper.model.SerializeToString(deterministic=True) != encoded:
        wrapper.model.ParseFromString(encoded)
    return wrapper


def _copy_proto(value: Any) -> Any:
    copied = value.__class__()
    copied.ParseFromString(value.SerializeToString(deterministic=True))
    return copied


def _opset_bytes(model: Any) -> bytes:
    return b"".join(
        len(encoded).to_bytes(8, "big") + encoded
        for encoded in (
            item.SerializeToString(deterministic=True) for item in model.model.opset_import
        )
    )


def _operator_identity(domain: str, op_type: str) -> bytes:
    return json.dumps([domain, op_type], separators=(",", ":")).encode("utf-8")


def _unique(what: str, values: Sequence[object]) -> None:
    if len(values) != len(set(values)):
        raise DataflowOpError(f"{what} contains duplicates")


def _unique_first(what: str, values: Sequence[tuple[Any, ...]]) -> None:
    keys = tuple(value[0] for value in values)
    if len(keys) != len(set(keys)):
        raise DataflowOpError(f"{what} contains duplicate targets")


def _unique_named(values: Sequence[Any], what: str) -> set[str]:
    names = [item.name for item in values]
    if len(names) != len(set(names)):
        raise DataflowOpError(f"duplicate {what} name")
    return set(names)


def _at_most_one(values: Sequence[object], what: str) -> None:
    if len(values) > 1:
        raise DataflowOpError(f"duplicate {what}")


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _drop(node: Any, name: str) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)


__all__ = [
    "MODEL_READ_PRESENT",
    "ModelEffects",
    "ModelReadExpectation",
    "ModelReadKind",
    "ModelReadSet",
    "ObservationMutationError",
    "apply_model_effects",
    "checked_model_observation",
    "merge_model_read_sets",
    "model_snapshot_digest",
    "validate_model_read_set",
]
