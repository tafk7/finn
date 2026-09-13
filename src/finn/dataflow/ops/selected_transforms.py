# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded transformations of immutable selected-graph snapshots."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Any, NoReturn, cast

from onnx import ModelProto, TensorProto  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow.model.maps import IdentityCoordinateMap
from finn.dataflow.model.network import DataflowNetwork, DirectConnection, Edge, RegionEndpoint
from finn.dataflow.model.region import InputInterface
from finn.dataflow.ops.model_effects import (
    ModelEffects,
    ModelReadExpectation,
    ModelReadKind,
    ModelReadSet,
    apply_model_effects,
)
from finn.dataflow.ops.selected import (
    SELECTED_METADATA_KEY,
    ConstructionIdentity,
    ConstructionRegistry,
    DecodedSelectedGraph,
    GraphNodeBinding,
    GraphSlotKind,
    GraphSlotRef,
    OwnerKind,
    SelectedGraphDeclaration,
    SelectedGraphSnapshot,
    build_selected_snapshot,
    decode_selected_graph,
    encode_selected_declaration,
)

READABLE_NAMES_TRANSFORM = "selected.readable_names"
BOUNDED_CLEANUP_TRANSFORM = "selected.bounded_cleanup"
ELIDE_EQUAL_WIDTH_IDENTITY_TRANSFORM = "selected.elide_equal_width_identity"
TRANSFORM_VERSION = 1
ELIDED_IDENTITY_FORM = "identity_elided"


class SelectedTransformError(ValueError):
    """A selected transform request is malformed, stale, or unsupported."""

    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(f"{path}: {message} [{code}]")
        self.code = code
        self.path = path


def _fail(code: str, path: str, message: str) -> NoReturn:
    raise SelectedTransformError(code, path, message)


def _freeze_correspondence(
    values: Iterable[tuple[str, str]], what: str
) -> tuple[tuple[str, str], ...]:
    result = []
    for value in values:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise TypeError(f"{what} records must be two-item tuples or lists")
        source, target = value
        if type(source) is not str or not source or type(target) is not str or not target:
            raise TypeError(f"{what} values must be non-empty strings")
        result.append((source, target))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SelectedTransformAuthorization:
    transform_id: str
    transform_version: int
    source_forms: tuple[str, ...]
    target_form: str | None
    allowed_effect_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.transform_id) is not str or not self.transform_id:
            raise TypeError("transform_id must be a non-empty string")
        if type(self.transform_version) is not int or self.transform_version < 1:
            raise TypeError("transform_version must be a positive integer")
        source_forms = tuple(self.source_forms)
        allowed = tuple(self.allowed_effect_fields)
        if not source_forms or any(type(item) is not str or not item for item in source_forms):
            raise TypeError("source_forms must contain non-empty strings")
        if len(source_forms) != len(set(source_forms)):
            raise ValueError("source_forms must be unique")
        if self.target_form is not None and (
            type(self.target_form) is not str or not self.target_form
        ):
            raise TypeError("target_form must be None or a non-empty string")
        if any(type(item) is not str or not item for item in allowed):
            raise TypeError("allowed_effect_fields must contain non-empty strings")
        if len(allowed) != len(set(allowed)):
            raise ValueError("allowed_effect_fields must be unique")
        known = {item.name for item in fields(ModelEffects)} - {"read_set", "set_metadata"}
        unknown = set(allowed) - known
        if unknown:
            raise ValueError(f"unknown ModelEffects fields {sorted(unknown)!r}")
        object.__setattr__(self, "source_forms", source_forms)
        object.__setattr__(self, "allowed_effect_fields", allowed)


@dataclass(frozen=True, slots=True)
class SelectedTransformRegistry:
    entries: Mapping[tuple[str, str], tuple[SelectedTransformAuthorization, ...]]

    def __post_init__(self) -> None:
        copied = {key: tuple(value) for key, value in self.entries.items()}
        for key, values in copied.items():
            identities = tuple((item.transform_id, item.transform_version) for item in values)
            if len(identities) != len(set(identities)):
                raise ValueError(f"duplicate selected transform authorization for {key!r}")
        object.__setattr__(self, "entries", MappingProxyType(copied))

    def resolve(
        self,
        source: ConstructionIdentity,
        transform_id: str,
        transform_version: int,
        target: ConstructionIdentity,
    ) -> SelectedTransformAuthorization:
        if type(transform_id) is not str or not transform_id:
            _fail("selected.transform.identity", "plan.transform_id", "invalid transform id")
        if type(transform_version) is not int or transform_version < 1:
            _fail(
                "selected.transform.version",
                "plan.transform_version",
                "transform version must be a positive integer",
            )
        matches = tuple(
            item
            for item in self.entries.get((source.family, source.version), ())
            if (item.transform_id, item.transform_version) == (transform_id, transform_version)
        )
        if len(matches) != 1:
            _fail(
                "selected.transform.unsupported",
                "plan.transform_id",
                "construction does not authorize this transform and version",
            )
        result = matches[0]
        if source.form not in result.source_forms:
            _fail(
                "selected.transform.source_form",
                "declaration.construction.form",
                "transform is not admitted from the current construction form",
            )
        if (target.family, target.version) != (source.family, source.version):
            _fail(
                "selected.transform.construction",
                "plan.target_construction",
                "a transform may not change construction family or version",
            )
        expected_form = source.form if result.target_form is None else result.target_form
        if target.form != expected_form:
            _fail(
                "selected.transform.target_form",
                "plan.target_construction.form",
                f"construction requires target form {expected_form!r}",
            )
        if result.target_form is None and target != source:
            _fail(
                "selected.transform.form_changed",
                "plan.target_construction",
                "this transform must preserve the complete construction form",
            )
        return result


@dataclass(frozen=True, slots=True)
class SelectedTransformPlan:
    transform_id: str
    transform_version: int
    target_construction: ConstructionIdentity
    expected_snapshot_digest: str
    node_correspondence: tuple[tuple[str, str], ...]
    value_correspondence: tuple[tuple[str, str], ...]
    effects: ModelEffects

    def __post_init__(self) -> None:
        if type(self.transform_id) is not str or not self.transform_id:
            raise TypeError("transform_id must be a non-empty string")
        if type(self.transform_version) is not int or self.transform_version < 1:
            raise TypeError("transform_version must be a positive integer")
        if not isinstance(self.target_construction, ConstructionIdentity):
            raise TypeError("target_construction must be ConstructionIdentity")
        if type(self.expected_snapshot_digest) is not str or not self.expected_snapshot_digest:
            raise TypeError("expected_snapshot_digest must be a non-empty string")
        if not isinstance(self.effects, ModelEffects):
            raise TypeError("effects must be ModelEffects")
        object.__setattr__(
            self,
            "node_correspondence",
            _freeze_correspondence(self.node_correspondence, "node_correspondence"),
        )
        object.__setattr__(
            self,
            "value_correspondence",
            _freeze_correspondence(self.value_correspondence, "value_correspondence"),
        )

    @property
    def target_form(self) -> str:
        return self.target_construction.form


def apply_selected_transform(
    snapshot: SelectedGraphSnapshot,
    plan: SelectedTransformPlan,
    *,
    constructions: ConstructionRegistry | None = None,
    transforms: SelectedTransformRegistry | None = None,
) -> DecodedSelectedGraph[Any, Any]:
    """Apply one authorized plan to a private copy and return its decoded result."""

    construction_registry, transform_registry = _registries(constructions, transforms)
    if snapshot.content_digest != plan.expected_snapshot_digest:
        _fail(
            "selected.transform.snapshot_stale",
            "plan.expected_snapshot_digest",
            "transform plan was made for a different selected snapshot",
        )
    before = decode_selected_graph(snapshot, constructions=construction_registry)
    authorization = transform_registry.resolve(
        before.declaration.construction,
        plan.transform_id,
        plan.transform_version,
        plan.target_construction,
    )
    _validate_effect_surface(plan, authorization)
    model = snapshot.model_copy()

    def decode_candidate(candidate: ModelWrapper) -> DecodedSelectedGraph[Any, Any]:
        decoded = decode_selected_graph(
            SelectedGraphSnapshot.capture(candidate), constructions=construction_registry
        )
        _validate_successor(before, decoded, plan)
        return decoded

    return apply_model_effects(
        model,
        plan.effects,
        validate=decode_candidate,
        finish=decode_candidate,
    )


def plan_readable_names(
    snapshot: SelectedGraphSnapshot,
    *,
    constructions: ConstructionRegistry | None = None,
    transforms: SelectedTransformRegistry | None = None,
) -> SelectedTransformPlan:
    """Plan deterministic readable node names and a total tensor rename."""

    construction_registry, transform_registry = _registries(constructions, transforms)
    decoded = decode_selected_graph(snapshot, constructions=construction_registry)
    target = decoded.declaration.construction
    authorization = transform_registry.resolve(
        target, READABLE_NAMES_TRANSFORM, TRANSFORM_VERSION, target
    )
    model = snapshot.model_copy()
    records = tuple(sorted(decoded.declaration.graph_nodes, key=lambda item: item.index))
    node_names: dict[str, str] = {}
    counts: dict[str, int] = {}
    for record in records:
        node = model.graph.node[record.index]
        index = counts.get(node.op_type, 0)
        counts[node.op_type] = index + 1
        node_names[record.node_id] = f"selected_{node.op_type}_{index}"
    value_names = _readable_value_names(model, records, node_names)
    final_model = _copy_wrapper(model)
    for record in records:
        final_model.graph.node[record.index].name = node_names[record.node_id]
    _rename_model_values(final_model, value_names)
    node_map = {record.node_id: record.node_id for record in records}
    input_slots = {
        (record.node_id, index): (record.node_id, index)
        for record in records
        for index in range(len(model.graph.node[record.index].input))
    }
    output_slots = {
        (record.node_id, index): (record.node_id, index)
        for record in records
        for index in range(len(model.graph.node[record.index].output))
    }
    declaration = _repair_declaration(
        decoded,
        final_model,
        target=target,
        final_node_ids=tuple(record.node_id for record in records),
        node_map=node_map,
        value_map=value_names,
        input_slots=input_slots,
        output_slots=output_slots,
    )
    effects = ModelEffects(
        replace_nodes=tuple(
            (
                record.node_id,
                final_model.graph.node[record.index].SerializeToString(deterministic=True),
            )
            for record in records
        ),
        rename_values=tuple((old, new) for old, new in sorted(value_names.items()) if old != new),
    )
    return _finalize_plan(
        snapshot,
        decoded,
        final_model,
        declaration,
        effects,
        authorization,
        tuple((record.node_id, record.node_id) for record in records),
        tuple(sorted(value_names.items())),
        construction_registry,
    )


def plan_bounded_cleanup(
    snapshot: SelectedGraphSnapshot,
    *,
    constructions: ConstructionRegistry | None = None,
    transforms: SelectedTransformRegistry | None = None,
) -> SelectedTransformPlan:
    """Plan the exact cleanup subset admitted by ``ModelEffects`` version 1."""

    construction_registry, transform_registry = _registries(constructions, transforms)
    decoded = decode_selected_graph(snapshot, constructions=construction_registry)
    target = decoded.declaration.construction
    authorization = transform_registry.resolve(
        target, BOUNDED_CLEANUP_TRANSFORM, TRANSFORM_VERSION, target
    )
    model = snapshot.model_copy()
    final_model = _copy_wrapper(model)
    records = tuple(sorted(decoded.declaration.graph_nodes, key=lambda item: item.index))
    record_by_id = {record.node_id: record for record in records}
    ordered_ids = _topological_node_ids(model, records)
    ordered_nodes = [model.graph.node[record_by_id[node_id].index] for node_id in ordered_ids]
    del final_model.graph.node[:]
    final_model.graph.node.extend(ordered_nodes)
    final_indices = {node_id: index for index, node_id in enumerate(ordered_ids)}

    source_roots = _source_initializer_roots(decoded, model)
    construction_constants = {
        item.name for item in model.graph.initializer if item.name not in source_roots
    }
    initializer_names = {item.name for item in final_model.graph.initializer}
    annotations_before = tuple(
        item.SerializeToString(deterministic=True) for item in model.graph.quantization_annotation
    )
    input_slots: dict[tuple[str, int], tuple[str, int]] = {}
    copied_values: list[tuple[str, str]] = []
    new_initializers: list[tuple[str, bytes]] = []
    seen_parameters: set[str] = set()
    duplicate_index: dict[str, int] = {}
    for node_id in ordered_ids:
        node = final_model.graph.node[final_indices[node_id]]
        old_inputs = tuple(node.input)
        indices = tuple(range(len(old_inputs)))
        if node.op_type in {"Add", "Mul", "And", "Or", "Xor", "Sum"}:
            indices = (
                *(index for index in indices if old_inputs[index] not in initializer_names),
                *(index for index in indices if old_inputs[index] in initializer_names),
            )
        new_inputs = []
        for new_index, old_index in enumerate(indices):
            value = old_inputs[old_index]
            new_value = value
            if value in initializer_names:
                if value in seen_parameters and value in construction_constants:
                    count = duplicate_index.get(value, 0) + 1
                    duplicate_index[value] = count
                    new_value = _fresh_name(
                        f"{value}_unique_{count}", _all_value_names(final_model)
                    )
                    tensor = next(
                        item for item in final_model.graph.initializer if item.name == value
                    )
                    copied = tensor.__class__()
                    copied.ParseFromString(tensor.SerializeToString(deterministic=True))
                    copied.name = new_value
                    final_model.graph.initializer.append(copied)
                    initializer_names.add(new_value)
                    source_annotations = tuple(
                        item
                        for item in final_model.graph.quantization_annotation
                        if item.tensor_name == value
                    )
                    if len(source_annotations) > 1:
                        _fail(
                            "selected.transform.cleanup_annotation",
                            f"tensor.{value}",
                            "construction constant has duplicate annotations",
                        )
                    if source_annotations:
                        copied_annotation = source_annotations[0].__class__()
                        copied_annotation.CopyFrom(source_annotations[0])
                        copied_annotation.tensor_name = new_value
                        final_model.graph.quantization_annotation.append(copied_annotation)
                    new_initializers.append(
                        (new_value, copied.SerializeToString(deterministic=True))
                    )
                    copied_values.append((value, new_value))
                seen_parameters.add(value)
            new_inputs.append(new_value)
            input_slots[(node_id, old_index)] = (node_id, new_index)
        del node.input[:]
        node.input.extend(new_inputs)

    static_inputs = tuple(
        item.name for item in final_model.graph.input if item.name in initializer_names
    )
    used = {value for node in final_model.graph.node for value in node.input if value}
    used.update(item.name for item in final_model.graph.output)
    structural_before_cleanup = _structural_value_names(final_model)
    used.update(
        item.value
        for annotation in final_model.graph.quantization_annotation
        for item in annotation.quant_parameter_tensor_names
        if item.key != "finn_datatype" and item.value in structural_before_cleanup
    )
    remove_initializers = tuple(
        item.name
        for item in final_model.graph.initializer
        if item.name not in used and item.name not in static_inputs
    )
    if remove_initializers:
        kept = [
            item for item in final_model.graph.initializer if item.name not in remove_initializers
        ]
        del final_model.graph.initializer[:]
        final_model.graph.initializer.extend(kept)
    if static_inputs:
        kept = [item for item in final_model.graph.input if item.name not in static_inputs]
        del final_model.graph.input[:]
        final_model.graph.input.extend(kept)

    final_references = {
        value for node in final_model.graph.node for value in (*node.input, *node.output) if value
    }
    final_references.update(item.name for item in final_model.graph.input)
    final_references.update(item.name for item in final_model.graph.output)
    final_references.update(item.name for item in final_model.graph.initializer)
    final_references.update(
        item.value
        for annotation in final_model.graph.quantization_annotation
        for item in annotation.quant_parameter_tensor_names
        if item.key != "finn_datatype"
    )
    remove_value_info = tuple(
        item.name for item in final_model.graph.value_info if item.name not in final_references
    )
    if remove_value_info:
        kept = [item for item in final_model.graph.value_info if item.name not in remove_value_info]
        del final_model.graph.value_info[:]
        final_model.graph.value_info.extend(kept)
    retained = _structural_value_names(final_model)
    kept_annotations = [
        item for item in final_model.graph.quantization_annotation if item.tensor_name in retained
    ]
    del final_model.graph.quantization_annotation[:]
    final_model.graph.quantization_annotation.extend(kept_annotations)
    annotations_after = tuple(
        item.SerializeToString(deterministic=True)
        for item in final_model.graph.quantization_annotation
    )

    output_slots = {
        (record.node_id, index): (record.node_id, index)
        for record in records
        for index in range(len(model.graph.node[record.index].output))
    }
    value_map = {name: name for name in _all_value_names(model)}
    declaration = _repair_declaration(
        decoded,
        final_model,
        target=target,
        final_node_ids=ordered_ids,
        node_map={record.node_id: record.node_id for record in records},
        value_map=value_map,
        input_slots=input_slots,
        output_slots=output_slots,
        dropped_graph_inputs=frozenset(static_inputs),
    )
    replacements = tuple(
        (
            node_id,
            final_model.graph.node[final_indices[node_id]].SerializeToString(deterministic=True),
        )
        for node_id in ordered_ids
        if model.graph.node[record_by_id[node_id].index].SerializeToString(deterministic=True)
        != final_model.graph.node[final_indices[node_id]].SerializeToString(deterministic=True)
    )
    effects = ModelEffects(
        replace_nodes=replacements,
        node_order=ordered_ids
        if ordered_ids != tuple(record.node_id for record in records)
        else None,
        remove_graph_inputs=static_inputs,
        remove_value_info=remove_value_info,
        remove_initializers=remove_initializers,
        set_initializers=tuple(new_initializers),
        replace_quantization_annotations=annotations_after
        if annotations_after != annotations_before
        else None,
    )
    correspondence = (
        *tuple((name, name) for name in sorted(_all_value_names(model))),
        *tuple(copied_values),
    )
    return _finalize_plan(
        snapshot,
        decoded,
        final_model,
        declaration,
        effects,
        authorization,
        tuple((record.node_id, record.node_id) for record in records),
        correspondence,
        construction_registry,
    )


def plan_equal_width_identity_elision(
    snapshot: SelectedGraphSnapshot,
    region_id: str,
    *,
    constructions: ConstructionRegistry | None = None,
    transforms: SelectedTransformRegistry | None = None,
) -> SelectedTransformPlan:
    """Remove one exact interior equal-width Identity Region."""

    if type(region_id) is not str or not region_id:
        _fail(
            "selected.transform.identity_region",
            "region_id",
            "Identity-elision region_id must be a non-empty string",
        )
    construction_registry, transform_registry = _registries(constructions, transforms)
    decoded = decode_selected_graph(snapshot, constructions=construction_registry)
    target = replace(
        decoded.declaration.construction,
        form=ELIDED_IDENTITY_FORM,
        form_version=1,
        form_arguments=(("region_id", region_id),),
    )
    authorization = transform_registry.resolve(
        decoded.declaration.construction,
        ELIDE_EQUAL_WIDTH_IDENTITY_TRANSFORM,
        TRANSFORM_VERSION,
        target,
    )
    owners = tuple(
        item
        for item in decoded.declaration.ownership
        if item.kind is OwnerKind.REGION and item.owner_id == region_id
    )
    if len(owners) != 1 or len(owners[0].node_ids) != 1:
        _fail(
            "selected.transform.identity_owner",
            f"region.{region_id}",
            "bounded elision requires one single-node Region owner",
        )
    node_id = owners[0].node_ids[0]
    records = {item.node_id: item for item in decoded.declaration.graph_nodes}
    record = records.get(node_id)
    if record is None:
        _fail("selected.transform.identity_missing", f"node.{node_id}", "node is absent")
    model = snapshot.model_copy()
    node = model.graph.node[record.index]
    if node.domain or node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
        _fail(
            "selected.transform.identity_profile",
            f"node.{node_id}",
            "bounded elision requires one standard one-input/one-output Identity",
        )
    input_value, output_value = node.input[0], node.output[0]
    if output_value in {item.name for item in model.graph.output}:
        _fail(
            "selected.transform.graph_output_producer",
            f"node.{node_id}",
            "bounded elision does not change a graph-output producer",
        )
    input_shape = model.get_tensor_shape(input_value)
    output_shape = model.get_tensor_shape(output_value)
    if input_shape is None or output_shape is None or tuple(input_shape) != tuple(output_shape):
        _fail(
            "selected.transform.identity_width_change_requires_adapter",
            f"node.{node_id}",
            "Identity input and output tensor shapes differ",
        )
    if model.get_tensor_datatype(input_value) != model.get_tensor_datatype(output_value):
        _fail(
            "selected.transform.identity_width_change_requires_adapter",
            f"node.{node_id}",
            "Identity input and output logical datatypes differ",
        )

    try:
        region = decoded.network.node(region_id).region
    except KeyError as error:
        raise SelectedTransformError(
            "selected.transform.identity_owner",
            f"region.{region_id}",
            "owned Identity Region is absent from the projected Network",
        ) from error
    if len(region.inputs) != 1 or len(region.outputs) != 1:
        _fail(
            "selected.transform.identity_region",
            f"region.{region_id}",
            "bounded elision requires one input and one output interface",
        )
    input_interface = region.inputs[0]
    output_interface = region.outputs[0]
    if not isinstance(input_interface, InputInterface):
        _fail(
            "selected.transform.identity_region",
            f"region.{region_id}",
            "bounded elision does not remove an internally supplied adapter",
        )
    if (
        input_interface.port.operand.position_domain
        != output_interface.port.operand.position_domain
        or input_interface.port.operand.element_type != output_interface.port.operand.element_type
        or input_interface.port.beat_sequence != output_interface.port.beat_sequence
    ):
        _fail(
            "selected.transform.identity_width_change_requires_adapter",
            f"region.{region_id}",
            "Identity does not preserve the complete logical port contract",
        )
    incoming = tuple(
        (edge, sink)
        for edge in decoded.network.edges
        for sink in edge.sinks
        if sink.endpoint == RegionEndpoint(region_id, input_interface.port.id)
    )
    outgoing = tuple(
        edge
        for edge in decoded.network.edges
        if edge.source == RegionEndpoint(region_id, output_interface.port.id)
    )
    if (
        len(incoming) != 1
        or len(outgoing) != 1
        or len(incoming[0][0].sinks) != 1
        or len(outgoing[0].sinks) != 1
    ):
        _fail(
            "selected.transform.identity_topology",
            f"region.{region_id}",
            "bounded elision requires one incoming and outgoing point-to-point edge",
        )
    if any(boundary.endpoint.node_id == region_id for boundary in decoded.network.boundaries):
        _fail(
            "selected.transform.identity_boundary",
            f"region.{region_id}",
            "bounded elision does not remove a boundary endpoint",
        )
    incoming_edge, incoming_sink = incoming[0]
    outgoing_edge = outgoing[0]
    outgoing_sink = outgoing_edge.sinks[0]
    if not isinstance(
        incoming_sink.position_map.coordinate_map, IdentityCoordinateMap
    ) or not isinstance(outgoing_sink.position_map.coordinate_map, IdentityCoordinateMap):
        _fail(
            "selected.transform.identity_position_map",
            f"region.{region_id}",
            "bounded elision requires identity edge maps",
        )
    if (
        incoming_edge.fanout != outgoing_edge.fanout
        or incoming_edge.pass_correspondence != outgoing_edge.pass_correspondence
        or incoming_edge.transport != outgoing_edge.transport
        or not isinstance(incoming_edge.transport, DirectConnection)
    ):
        _fail(
            "selected.transform.identity_transport",
            f"region.{region_id}",
            "bounded elision requires equal direct edge transport contracts",
        )
    if any(supply.required_input.node_id == region_id for supply in decoded.declaration.supplies):
        _fail(
            "selected.transform.identity_supply",
            f"region.{region_id}",
            "bounded elision does not remove a supplied internal input",
        )

    final_model = _copy_wrapper(model)
    del final_model.graph.node[record.index]
    changed_node_ids = []
    for candidate_id, candidate_record in records.items():
        if candidate_id == node_id:
            continue
        adjusted = candidate_record.index - (1 if candidate_record.index > record.index else 0)
        candidate = final_model.graph.node[adjusted]
        changed = False
        for index, value in enumerate(candidate.input):
            if value == output_value:
                candidate.input[index] = input_value
                changed = True
        if changed:
            changed_node_ids.append(candidate_id)

    kept_info = [item for item in final_model.graph.value_info if item.name != output_value]
    removed_value_info = (
        (output_value,) if len(kept_info) != len(final_model.graph.value_info) else ()
    )
    del final_model.graph.value_info[:]
    final_model.graph.value_info.extend(kept_info)
    annotations_before = tuple(
        item.SerializeToString(deterministic=True)
        for item in final_model.graph.quantization_annotation
    )
    kept_annotations = [
        item
        for item in final_model.graph.quantization_annotation
        if item.tensor_name != output_value
    ]
    for annotation in kept_annotations:
        for parameter in annotation.quant_parameter_tensor_names:
            if parameter.value == output_value:
                parameter.value = input_value
    del final_model.graph.quantization_annotation[:]
    final_model.graph.quantization_annotation.extend(kept_annotations)
    annotations_after = tuple(
        item.SerializeToString(deterministic=True)
        for item in final_model.graph.quantization_annotation
    )

    final_node_ids = tuple(
        item.node_id
        for item in sorted(decoded.declaration.graph_nodes, key=lambda item: item.index)
        if item.node_id != node_id
    )
    node_map = {item: item for item in final_node_ids}
    value_map = {name: name for name in _all_value_names(model)}
    value_map[output_value] = input_value
    input_slots = {}
    output_slots = {}
    final_indices = {item: index for index, item in enumerate(final_node_ids)}
    for candidate_id in final_node_ids:
        old_node = model.graph.node[records[candidate_id].index]
        for index in range(len(old_node.input)):
            input_slots[(candidate_id, index)] = (candidate_id, index)
        for index in range(len(old_node.output)):
            output_slots[(candidate_id, index)] = (candidate_id, index)
    declaration = _repair_declaration(
        decoded,
        final_model,
        target=target,
        final_node_ids=final_node_ids,
        node_map=node_map,
        value_map=value_map,
        input_slots=input_slots,
        output_slots=output_slots,
        slot_replacements={
            (GraphSlotKind.NODE_OUTPUT, node_id, 0): (
                _defining_slot(model, decoded.declaration.graph_nodes, input_value),
            )
        },
        removed_region=region_id,
    )
    replacements = tuple(
        (
            candidate_id,
            final_model.graph.node[final_indices[candidate_id]].SerializeToString(
                deterministic=True
            ),
        )
        for candidate_id in changed_node_ids
    )
    effects = ModelEffects(
        remove_nodes=(node_id,),
        replace_nodes=replacements,
        remove_value_info=removed_value_info,
        replace_quantization_annotations=annotations_after
        if annotations_after != annotations_before
        else None,
    )
    correspondence = tuple(
        (name, input_value if name == output_value else name)
        for name in sorted(_all_value_names(model))
    )
    return _finalize_plan(
        snapshot,
        decoded,
        final_model,
        declaration,
        effects,
        authorization,
        tuple((item, item) for item in final_node_ids),
        correspondence,
        construction_registry,
    )


def _registries(
    constructions: ConstructionRegistry | None,
    transforms: SelectedTransformRegistry | None,
) -> tuple[ConstructionRegistry, SelectedTransformRegistry]:
    if constructions is None:
        from finn.dataflow.ops.selected_registry import (  # noqa: PLC0415
            DEFAULT_SELECTED_CONSTRUCTIONS,
        )

        constructions = DEFAULT_SELECTED_CONSTRUCTIONS
    if transforms is None:
        from finn.dataflow.ops.selected_transform_registry import (  # noqa: PLC0415
            DEFAULT_SELECTED_TRANSFORMS,
        )

        transforms = DEFAULT_SELECTED_TRANSFORMS
    return constructions, transforms


def _finalize_plan(
    snapshot: SelectedGraphSnapshot,
    before: DecodedSelectedGraph[Any, Any],
    final_model: ModelWrapper,
    declaration: SelectedGraphDeclaration,
    structural: ModelEffects,
    authorization: SelectedTransformAuthorization,
    node_correspondence: tuple[tuple[str, str], ...],
    value_correspondence: tuple[tuple[str, str], ...],
    constructions: ConstructionRegistry,
) -> SelectedTransformPlan:
    final_snapshot = build_selected_snapshot(final_model, declaration)
    metadata = encode_selected_declaration(final_snapshot.declaration)
    effects = replace(
        structural,
        read_set=ModelReadSet(
            (
                ModelReadExpectation(
                    ModelReadKind.SNAPSHOT_DIGEST,
                    "",
                    None,
                    snapshot.content_digest,
                ),
            )
        ),
        set_metadata=((SELECTED_METADATA_KEY, metadata),),
    )
    plan = SelectedTransformPlan(
        authorization.transform_id,
        authorization.transform_version,
        declaration.construction,
        snapshot.content_digest,
        node_correspondence,
        value_correspondence,
        effects,
    )
    _validate_effect_surface(plan, authorization)
    after = decode_selected_graph(final_snapshot, constructions=constructions)
    _validate_successor(before, after, plan)
    return plan


def _validate_effect_surface(
    plan: SelectedTransformPlan, authorization: SelectedTransformAuthorization
) -> None:
    expected_read_set = ModelReadSet(
        (
            ModelReadExpectation(
                ModelReadKind.SNAPSHOT_DIGEST,
                "",
                None,
                plan.expected_snapshot_digest,
            ),
        )
    )
    if plan.effects.read_set != expected_read_set:
        _fail(
            "selected.transform.read_set",
            "plan.effects.read_set",
            "selected transform must read the exact input snapshot digest",
        )
    allowed = {"read_set", "set_metadata", *authorization.allowed_effect_fields}
    nonempty = set()
    for item in fields(plan.effects):
        value = getattr(plan.effects, item.name)
        if item.name == "read_set":
            if value.expectations:
                nonempty.add(item.name)
        elif value not in ((), None):
            nonempty.add(item.name)
    unsupported = nonempty - allowed
    if unsupported:
        _fail(
            "selected.transform.effects_unsupported",
            "plan.effects",
            f"transform requested unsupported write fields {sorted(unsupported)!r}",
        )
    if (
        len(plan.effects.set_metadata) != 1
        or plan.effects.set_metadata[0][0] != SELECTED_METADATA_KEY
    ):
        _fail(
            "selected.transform.metadata_unauthorized",
            "plan.effects.set_metadata",
            "transform must replace exactly the selected declaration metadata key",
        )


def _validate_successor(
    before: DecodedSelectedGraph[Any, Any],
    after: DecodedSelectedGraph[Any, Any],
    plan: SelectedTransformPlan,
) -> None:
    if after.declaration.construction != plan.target_construction:
        _fail(
            "selected.transform.form_not_persisted",
            "declaration.construction",
            "successor did not persist the planned construction identity",
        )
    if (
        before.selection_facts.source != after.selection_facts.source
        or before.selection_facts.source_semantics != after.selection_facts.source_semantics
        or before.selection_facts.choices != after.selection_facts.choices
    ):
        _fail(
            "selected.transform.selection_changed",
            "selection_facts",
            "transform changed retained source meaning or logical choices",
        )
    if before.selection_facts.parameters != after.selection_facts.parameters:
        _fail(
            "selected.transform.parameters_changed",
            "selection_facts.parameters",
            "transform changed derived parameters",
        )
    _validate_network_successor(before.network, after.network, before, plan)

    node_map = dict(plan.node_correspondence)
    if len(node_map) != len(plan.node_correspondence) or len(set(node_map.values())) != len(
        node_map
    ):
        _fail(
            "selected.transform.node_correspondence",
            "plan.node_correspondence",
            "node correspondence is duplicate or ambiguous",
        )
    old_nodes = {item.node_id for item in before.declaration.graph_nodes}
    removed_nodes = set(plan.effects.remove_nodes)
    new_nodes = {item.node_id for item in after.declaration.graph_nodes}
    if set(node_map) != old_nodes - removed_nodes or set(node_map.values()) != new_nodes:
        _fail(
            "selected.transform.node_correspondence",
            "plan.node_correspondence",
            "node correspondence does not cover every surviving node exactly once",
        )
    if set(plan.node_correspondence) != {
        (node_id, node_id) for node_id in old_nodes - removed_nodes
    }:
        _fail(
            "selected.transform.node_correspondence",
            "plan.node_correspondence",
            "stable node correspondence disagrees with the actual transform",
        )

    if len(set(plan.value_correspondence)) != len(plan.value_correspondence):
        _fail(
            "selected.transform.value_correspondence",
            "plan.value_correspondence",
            "value correspondence contains duplicate pairs",
        )
    relations: dict[str, set[str]] = {}
    for old, new in plan.value_correspondence:
        relations.setdefault(old, set()).add(new)
    old_values = _graph_value_inventory(before.snapshot.model_copy())
    new_values = _graph_value_inventory(after.snapshot.model_copy())
    if set(relations) != old_values:
        _fail(
            "selected.transform.value_correspondence",
            "plan.value_correspondence",
            "value correspondence does not cover the complete input inventory",
        )
    targets = {target for values in relations.values() for target in values}
    if not new_values.issubset(targets):
        _fail(
            "selected.transform.value_correspondence",
            "plan.value_correspondence",
            f"successor values lack correspondence {sorted(new_values - targets)!r}",
        )
    removed_values = old_values - new_values
    dangling = targets - new_values
    if any(
        old not in removed_values or values - new_values != {old}
        for old, values in relations.items()
        if values - new_values
    ):
        _fail(
            "selected.transform.value_correspondence",
            "plan.value_correspondence",
            f"correspondence has unexplained targets {sorted(dangling)!r}",
        )
    if plan.transform_id == READABLE_NAMES_TRANSFORM:
        if any(len(values) != 1 for values in relations.values()) or len(targets) != len(relations):
            _fail(
                "selected.transform.value_correspondence",
                "plan.value_correspondence",
                "readable naming requires one-to-one total value correspondence",
            )
        renames = dict(plan.effects.rename_values)
        expected_pairs = {(name, renames.get(name, name)) for name in old_values}
        if set(plan.value_correspondence) != expected_pairs:
            _fail(
                "selected.transform.value_correspondence",
                "plan.value_correspondence",
                "readable-name correspondence disagrees with the actual rename effects",
            )
    elif plan.transform_id == BOUNDED_CLEANUP_TRANSFORM:
        constants = _construction_constants(before)
        old_model = before.snapshot.model_copy()
        old_payloads = {name: _initializer_payload(old_model, name) for name in constants}
        created = {}
        for name, encoded in plan.effects.set_initializers:
            tensor = TensorProto()
            tensor.ParseFromString(encoded)
            if tensor.name != name:
                _fail(
                    "selected.transform.value_correspondence",
                    "plan.effects.set_initializers",
                    "initializer key differs from its TensorProto name",
                )
            copied = tensor.__class__()
            copied.CopyFrom(tensor)
            copied.name = ""
            created[name] = copied.SerializeToString(deterministic=True)
        expected_pairs = {(name, name) for name in old_values}
        for target, payload in created.items():
            candidates = {
                source
                for source, source_payload in old_payloads.items()
                if source_payload == payload
            }
            chosen = {
                source
                for source, candidate_target in plan.value_correspondence
                if candidate_target == target
            }
            if len(chosen) != 1 or not chosen <= candidates:
                _fail(
                    "selected.transform.value_correspondence",
                    "plan.value_correspondence",
                    "initializer copy lacks one exact construction-constant source",
                )
            expected_pairs.add((next(iter(chosen)), target))
        if set(plan.value_correspondence) != expected_pairs:
            _fail(
                "selected.transform.value_correspondence",
                "plan.value_correspondence",
                "cleanup correspondence disagrees with actual retained and copied values",
            )
    elif plan.transform_id == ELIDE_EQUAL_WIDTH_IDENTITY_TRANSFORM:
        if len(plan.effects.remove_nodes) != 1:
            _fail(
                "selected.transform.value_correspondence",
                "plan.effects.remove_nodes",
                "Identity elision must remove exactly one stable node",
            )
        records = {item.node_id: item for item in before.declaration.graph_nodes}
        removed = records[plan.effects.remove_nodes[0]]
        node = before.snapshot.model_copy().graph.node[removed.index]
        if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
            _fail(
                "selected.transform.value_correspondence",
                "plan.effects.remove_nodes",
                "removed node is not the planned one-input/one-output Identity",
            )
        source, output = node.input[0], node.output[0]
        expected_pairs = {(name, source if name == output else name) for name in old_values}
        if set(plan.value_correspondence) != expected_pairs:
            _fail(
                "selected.transform.value_correspondence",
                "plan.value_correspondence",
                "Identity-elision correspondence disagrees with the removed value",
            )


def _validate_network_successor(
    before_network: DataflowNetwork,
    after_network: DataflowNetwork,
    before: DecodedSelectedGraph[Any, Any],
    plan: SelectedTransformPlan,
) -> None:
    if before.declaration.construction == plan.target_construction:
        if before_network != after_network:
            _fail(
                "selected.transform.network_changed",
                "network",
                "same-form transform changed the projected logical Network",
            )
        return
    if plan.transform_id != ELIDE_EQUAL_WIDTH_IDENTITY_TRANSFORM:
        _fail(
            "selected.transform.network_changed",
            "network",
            "form-changing transform has no admitted Network correspondence",
        )

    arguments = dict(plan.target_construction.form_arguments)
    region_id = arguments.get("region_id")
    if type(region_id) is not str or not region_id:
        _fail(
            "selected.transform.network_changed",
            "network",
            "Identity elision has no exact removed Region identity",
        )
    before_nodes = {item.id: item for item in before_network.nodes}
    after_nodes = {item.id: item for item in after_network.nodes}
    if region_id not in before_nodes or set(after_nodes) != set(before_nodes) - {region_id}:
        _fail(
            "selected.transform.network_changed",
            "network.nodes",
            "Identity elision changed the surviving Region identities",
        )
    if any(
        after_nodes[node_id] != node
        for node_id, node in before_nodes.items()
        if node_id != region_id
    ):
        _fail(
            "selected.transform.network_changed",
            "network.nodes",
            "Identity elision changed a surviving Region contract",
        )
    if after_network.boundaries != before_network.boundaries:
        _fail(
            "selected.transform.network_changed",
            "network.boundaries",
            "Identity elision changed a boundary contract",
        )

    incoming = tuple(
        edge
        for edge in before_network.edges
        if any(sink.endpoint.node_id == region_id for sink in edge.sinks)
    )
    outgoing = tuple(edge for edge in before_network.edges if edge.source.node_id == region_id)
    if (
        len(incoming) != 1
        or len(outgoing) != 1
        or len(incoming[0].sinks) != 1
        or incoming[0].sinks[0].endpoint.node_id != region_id
        or len(outgoing[0].sinks) != 1
    ):
        _fail(
            "selected.transform.network_changed",
            "network.edges",
            "Identity elision does not replace one exact incoming/outgoing edge pair",
        )
    incoming_edge, outgoing_edge = incoming[0], outgoing[0]
    removed_edge_ids = {incoming_edge.id, outgoing_edge.id}
    unaffected = tuple(edge for edge in before_network.edges if edge.id not in removed_edge_ids)
    after_by_id = {edge.id: edge for edge in after_network.edges}
    if any(after_by_id.get(edge.id) != edge for edge in unaffected):
        _fail(
            "selected.transform.network_changed",
            "network.edges",
            "Identity elision changed an unaffected edge contract",
        )
    unaffected_ids = {edge.id for edge in unaffected}
    replacements = tuple(edge for edge in after_network.edges if edge.id not in unaffected_ids)
    if len(replacements) != 1 or len(after_network.edges) != len(unaffected) + 1:
        _fail(
            "selected.transform.network_changed",
            "network.edges",
            "Identity elision did not replace only the adjacent edge pair",
        )
    bypass = replacements[0]
    if not _is_identity_bypass(bypass, incoming_edge, outgoing_edge):
        _fail(
            "selected.transform.network_changed",
            "network.edges",
            "Identity elision changed the bypass edge contract",
        )


def _is_identity_bypass(bypass: Edge, incoming: Edge, outgoing: Edge) -> bool:
    return (
        bypass.source == incoming.source
        and bypass.sinks == outgoing.sinks
        and bypass.sinks[0].position_map == incoming.sinks[0].position_map
        and bypass.fanout == incoming.fanout == outgoing.fanout
        and bypass.pass_correspondence
        == incoming.pass_correspondence
        == outgoing.pass_correspondence
        and bypass.transport == incoming.transport == outgoing.transport
    )


def _repair_declaration(
    decoded: DecodedSelectedGraph[Any, Any],
    final_model: ModelWrapper,
    *,
    target: ConstructionIdentity,
    final_node_ids: tuple[str, ...],
    node_map: Mapping[str, str],
    value_map: Mapping[str, str],
    input_slots: Mapping[tuple[str, int], tuple[str, int]],
    output_slots: Mapping[tuple[str, int], tuple[str, int]],
    dropped_graph_inputs: frozenset[str] = frozenset(),
    slot_replacements: Mapping[
        tuple[GraphSlotKind, str, int], tuple[GraphSlotRef, ...]
    ] = MappingProxyType({}),
    removed_region: str | None = None,
) -> SelectedGraphDeclaration:
    declaration = decoded.declaration
    old_model = decoded.snapshot.model_copy()
    old_records = {item.node_id: item for item in declaration.graph_nodes}
    final_indices = {node_id: index for index, node_id in enumerate(final_node_ids)}

    def repair_slot(slot: GraphSlotRef) -> GraphSlotRef | None:
        if slot.index is not None:
            replacements = slot_replacements.get((slot.kind, slot.owner, slot.index))
            if replacements is not None:
                if len(replacements) != 1:
                    _fail(
                        "selected.transform.slot_repair",
                        slot.owner,
                        "slot replacement is not unique",
                    )
                return repair_slot(replacements[0])
        old_value = _slot_value(old_model, old_records, slot)
        value = value_map.get(old_value, old_value)
        if slot.kind is GraphSlotKind.GRAPH_INPUT:
            if old_value in dropped_graph_inputs:
                return None
            matches = [
                index for index, item in enumerate(final_model.graph.input) if item.name == value
            ]
            if len(matches) != 1:
                _fail("selected.transform.slot_repair", old_value, "graph input is absent")
            return GraphSlotRef(slot.kind, final_model.graph.name, matches[0])
        if slot.kind is GraphSlotKind.GRAPH_OUTPUT:
            matches = [
                index for index, item in enumerate(final_model.graph.output) if item.name == value
            ]
            if len(matches) != 1:
                _fail("selected.transform.slot_repair", old_value, "graph output is absent")
            return GraphSlotRef(slot.kind, final_model.graph.name, matches[0])
        if slot.kind is GraphSlotKind.INITIALIZER:
            if value not in {item.name for item in final_model.graph.initializer}:
                return None
            return GraphSlotRef(slot.kind, value, None)
        if slot.index is None:
            _fail("selected.transform.slot_repair", slot.owner, "node slot has no index")
        table = input_slots if slot.kind is GraphSlotKind.NODE_INPUT else output_slots
        mapped = table.get((slot.owner, slot.index))
        if mapped is None:
            return None
        owner, index = mapped
        if owner not in final_indices:
            return None
        values = (
            final_model.graph.node[final_indices[owner]].input
            if slot.kind is GraphSlotKind.NODE_INPUT
            else final_model.graph.node[final_indices[owner]].output
        )
        if index < 0 or index >= len(values) or values[index] != value:
            _fail(
                "selected.transform.slot_repair",
                slot.owner,
                "node slot does not contain the corresponding value",
            )
        return GraphSlotRef(slot.kind, owner, index)

    def repair_slots(slots: Iterable[GraphSlotRef]) -> tuple[GraphSlotRef, ...]:
        values = tuple(repair_slot(slot) for slot in slots)
        return tuple(item for item in values if item is not None)

    bindings = tuple(
        replace(
            item,
            graph_value=value_map.get(item.graph_value, item.graph_value),
            anchors=repair_slots(item.anchors),
        )
        for item in declaration.interface_bindings
        if item.interface.node_id != removed_region
    )
    source_bindings = tuple(
        replace(
            item,
            graph_value=value_map.get(item.graph_value, item.graph_value),
            anchors=repair_slots(item.anchors),
        )
        for item in declaration.source_bindings
    )
    supplies = tuple(
        replace(
            item,
            root_graph_value=value_map.get(item.root_graph_value, item.root_graph_value),
            derivation_nodes=tuple(
                node_map[node] for node in item.derivation_nodes if node in node_map
            ),
        )
        for item in declaration.supplies
    )
    ownership = tuple(
        replace(
            item,
            node_ids=tuple(node_map[node] for node in item.node_ids if node in node_map),
        )
        for item in declaration.ownership
        if item.owner_id != removed_region and any(node in node_map for node in item.node_ids)
    )
    return replace(
        declaration,
        construction=target,
        graph_nodes=tuple(
            GraphNodeBinding(node_id, index) for index, node_id in enumerate(final_node_ids)
        ),
        interface_bindings=bindings,
        source_bindings=source_bindings,
        supplies=supplies,
        ownership=ownership,
    )


def _slot_value(
    model: ModelWrapper,
    records: Mapping[str, GraphNodeBinding],
    slot: GraphSlotRef,
) -> str:
    if slot.kind is GraphSlotKind.GRAPH_INPUT:
        values = model.graph.input
    elif slot.kind is GraphSlotKind.GRAPH_OUTPUT:
        values = model.graph.output
    elif slot.kind is GraphSlotKind.INITIALIZER:
        if slot.index is not None or slot.owner not in {
            item.name for item in model.graph.initializer
        }:
            _fail("selected.transform.slot", slot.owner, "initializer slot is absent")
        return slot.owner
    else:
        record = records.get(slot.owner)
        if record is None or slot.index is None:
            _fail("selected.transform.slot", slot.owner, "node slot is absent")
        node = model.graph.node[record.index]
        values = node.input if slot.kind is GraphSlotKind.NODE_INPUT else node.output
    if slot.index is None or slot.index < 0 or slot.index >= len(values):
        _fail("selected.transform.slot", slot.owner, "slot index is out of range")
    value = values[slot.index]
    return value.name if hasattr(value, "name") else cast(str, value)


def _defining_slot(
    model: ModelWrapper, records: tuple[GraphNodeBinding, ...], value: str
) -> GraphSlotRef:
    result: list[GraphSlotRef] = []
    result.extend(
        GraphSlotRef(GraphSlotKind.GRAPH_INPUT, model.graph.name, index)
        for index, item in enumerate(model.graph.input)
        if item.name == value
    )
    result.extend(
        GraphSlotRef(GraphSlotKind.INITIALIZER, value, None)
        for item in model.graph.initializer
        if item.name == value
    )
    for record in records:
        node = model.graph.node[record.index]
        result.extend(
            GraphSlotRef(GraphSlotKind.NODE_OUTPUT, record.node_id, index)
            for index, output in enumerate(node.output)
            if output == value
        )
    if len(result) != 1:
        _fail(
            "selected.transform.identity_source_binding",
            value,
            "removed Identity output has no unique surviving upstream definition",
        )
    return result[0]


def _source_initializer_roots(
    decoded: DecodedSelectedGraph[Any, Any], model: ModelWrapper
) -> set[str]:
    initializer_names = {item.name for item in model.graph.initializer}
    roots = {item.root_graph_value for item in decoded.declaration.supplies}
    roots.update(
        item.graph_value
        for item in decoded.declaration.source_bindings
        if item.graph_value in initializer_names
    )
    return roots


def _construction_constants(decoded: DecodedSelectedGraph[Any, Any]) -> set[str]:
    model = decoded.snapshot.model_copy()
    roots = _source_initializer_roots(decoded, model)
    used = {value for node in model.graph.node for value in node.input if value}
    return {item.name for item in model.graph.initializer if item.name in used} - roots


def _initializer_payload(model: ModelWrapper, name: str) -> bytes:
    matches = tuple(item for item in model.graph.initializer if item.name == name)
    if len(matches) != 1:
        _fail(
            "selected.transform.value_correspondence",
            f"initializer.{name}",
            "construction initializer does not resolve uniquely",
        )
    copied = TensorProto()
    copied.CopyFrom(matches[0])
    copied.name = ""
    return bytes(copied.SerializeToString(deterministic=True))


def _readable_value_names(
    model: ModelWrapper,
    records: tuple[GraphNodeBinding, ...],
    node_names: Mapping[str, str],
) -> dict[str, str]:
    old_names = _ordered_value_names(model)
    reserved = set(old_names)
    result: dict[str, str] = {}
    used: set[str] = set()

    def assign(old: str, base: str) -> None:
        if old in result:
            return
        candidate = _fresh_name(base, used | (reserved - set(result) - {old}))
        result[old] = candidate
        used.add(candidate)

    for index, item in enumerate(model.graph.input):
        assign(item.name, "selected_input" if index == 0 else f"selected_input_{index}")
    for index, item in enumerate(model.graph.output):
        assign(item.name, "selected_output" if index == 0 else f"selected_output_{index}")
    initializer_names = {item.name for item in model.graph.initializer}
    parameter_counts: dict[str, int] = {}
    for record in records:
        node = model.graph.node[record.index]
        for output_index, value in enumerate(node.output):
            if value:
                assign(value, f"{node_names[record.node_id]}_out{output_index}")
        for value in node.input:
            if value in initializer_names:
                count = parameter_counts.get(record.node_id, 0)
                parameter_counts[record.node_id] = count + 1
                assign(value, f"{node_names[record.node_id]}_param{count}")
    for index, value in enumerate(old_names):
        assign(value, f"selected_value_{index}")
    return result


def _ordered_value_names(model: ModelWrapper) -> tuple[str, ...]:
    values = [item.name for item in model.graph.input]
    values.extend(item.name for item in model.graph.initializer)
    values.extend(
        value for node in model.graph.node for value in (*node.input, *node.output) if value
    )
    values.extend(item.name for item in model.graph.output)
    values.extend(item.name for item in model.graph.value_info)
    values.extend(item.tensor_name for item in model.graph.quantization_annotation)
    return tuple(dict.fromkeys(values))


def _all_value_names(model: ModelWrapper) -> set[str]:
    return set(_ordered_value_names(model))


def _structural_value_names(model: ModelWrapper) -> set[str]:
    values = {item.name for item in model.graph.input}
    values.update(item.name for item in model.graph.output)
    values.update(item.name for item in model.graph.value_info)
    values.update(item.name for item in model.graph.initializer)
    values.update(
        value for node in model.graph.node for value in (*node.input, *node.output) if value
    )
    return values


def _fresh_name(base: str, unavailable: set[str]) -> str:
    if base not in unavailable:
        return base
    index = 1
    while f"{base}_{index}" in unavailable:
        index += 1
    return f"{base}_{index}"


def _rename_model_values(model: ModelWrapper, names: Mapping[str, str]) -> None:
    for values in (model.graph.input, model.graph.output, model.graph.value_info):
        for item in values:
            item.name = names.get(item.name, item.name)
    for item in model.graph.initializer:
        item.name = names.get(item.name, item.name)
    for annotation in model.graph.quantization_annotation:
        annotation.tensor_name = names.get(annotation.tensor_name, annotation.tensor_name)
        for parameter in annotation.quant_parameter_tensor_names:
            if parameter.key != "finn_datatype":
                parameter.value = names.get(parameter.value, parameter.value)
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            node.input[index] = names.get(value, value)
        for index, value in enumerate(node.output):
            node.output[index] = names.get(value, value)


def _topological_node_ids(
    model: ModelWrapper, records: tuple[GraphNodeBinding, ...]
) -> tuple[str, ...]:
    ids = tuple(item.node_id for item in records)
    id_by_index = {item.index: item.node_id for item in records}
    producer = {
        value: id_by_index[index]
        for index, node in enumerate(model.graph.node)
        for value in node.output
        if value
    }
    dependencies = {
        item.node_id: {
            producer[value]
            for value in model.graph.node[item.index].input
            if value in producer and producer[value] != item.node_id
        }
        for item in records
    }
    result: list[str] = []
    pending = list(ids)
    while pending:
        ready = [node_id for node_id in pending if dependencies[node_id] <= set(result)]
        if not ready:
            _fail(
                "selected.transform.cleanup_topology",
                "declaration.graph_nodes",
                "selected graph is cyclic or has unresolved producers",
            )
        for node_id in ready:
            result.append(node_id)
            pending.remove(node_id)
    return tuple(result)


def _graph_value_inventory(model: ModelWrapper) -> set[str]:
    result = _structural_value_names(model)
    for annotation in model.graph.quantization_annotation:
        result.add(annotation.tensor_name)
        result.update(
            item.value for item in annotation.quant_parameter_tensor_names if item.value in result
        )
    return result


def _copy_wrapper(model: ModelWrapper) -> ModelWrapper:
    copied = ModelProto()
    copied.ParseFromString(model.model.SerializeToString(deterministic=True))
    return ModelWrapper(copied)


__all__ = [
    "BOUNDED_CLEANUP_TRANSFORM",
    "ELIDED_IDENTITY_FORM",
    "ELIDE_EQUAL_WIDTH_IDENTITY_TRANSFORM",
    "READABLE_NAMES_TRANSFORM",
    "TRANSFORM_VERSION",
    "SelectedTransformAuthorization",
    "SelectedTransformError",
    "SelectedTransformPlan",
    "SelectedTransformRegistry",
    "apply_selected_transform",
    "plan_bounded_cleanup",
    "plan_equal_width_identity_elision",
    "plan_readable_names",
]
