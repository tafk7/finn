# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure, name-independent verification helpers for selected ONNX artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

from qonnx.analysis.tensor_value_summary import (  # type: ignore[import-not-found]
    initializer_value_summaries,
)
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow._engine import Finding, FindingKind, QualifiedPath
from finn.dataflow.ops.selected import (
    SELECTED_METADATA_KEY,
    GraphSlotKind,
    GraphSlotRef,
    SelectedGraphDeclaration,
    SelectedGraphSnapshot,
    SourceOperandKey,
)
from finn.dataflow.ops.tensor_summary import FrozenInitializer


@dataclass(frozen=True, slots=True)
class _NormalizedSelectedSnapshot:
    core: tuple[object, ...]
    static_graph_inputs: tuple[object, ...]
    unused_initializers: tuple[object, ...]
    unused_value_info: tuple[object, ...]
    unused_annotations: tuple[object, ...]


def verify_normalized_selected_snapshot(
    snapshot: SelectedGraphSnapshot,
    expected: SelectedGraphSnapshot,
    *,
    finding_code: str,
    path: str,
    message: str,
) -> tuple[Finding, ...]:
    """Compare graph meaning while admitting bounded naming/cleanup normal forms."""

    try:
        actual_signature = _normalized_snapshot(snapshot)
        expected_signature = _normalized_snapshot(expected)
    except (IndexError, KeyError, TypeError, ValueError) as error:
        return (
            Finding(
                FindingKind.REJECTION,
                finding_code,
                QualifiedPath(path),
                f"{message}: normalization failed: {error}",
            ),
        )
    if actual_signature.core != expected_signature.core:
        return (Finding(FindingKind.REJECTION, finding_code, QualifiedPath(path), message),)
    for actual, canonical in (
        (actual_signature.static_graph_inputs, expected_signature.static_graph_inputs),
        (actual_signature.unused_initializers, expected_signature.unused_initializers),
        (actual_signature.unused_value_info, expected_signature.unused_value_info),
        (actual_signature.unused_annotations, expected_signature.unused_annotations),
    ):
        if not Counter(actual) <= Counter(canonical):
            return (Finding(FindingKind.REJECTION, finding_code, QualifiedPath(path), message),)
    return ()


def frozen_initializer_for_source(
    snapshot: SelectedGraphSnapshot, source: SourceOperandKey
) -> FrozenInitializer | None:
    """Resolve one selected initializer root by retained source identity."""

    declaration = snapshot.declaration
    source_values = tuple(item for item in declaration.source.operands if item.key == source)
    bindings = tuple(item for item in declaration.source_bindings if item.source == source)
    if len(source_values) != 1 or len(bindings) != 1:
        return None
    source_value = source_values[0]
    if source_value.initializer_content_digest is None:
        return None

    model = snapshot.model_copy()
    initializer_names = {item.name for item in model.graph.initializer}
    roots = {item.root_graph_value for item in declaration.supplies if item.source == source}
    if bindings[0].graph_value in initializer_names:
        roots.add(bindings[0].graph_value)
    if len(roots) != 1:
        return None
    root = next(iter(roots))
    matches = tuple(item for item in model.graph.initializer if item.name == root)
    if len(matches) != 1:
        return None
    frozen = FrozenInitializer.from_tensor_proto(matches[0])
    if (
        frozen.shape != source_value.shape
        or frozen.carrier_dtype != source_value.carrier_dtype
        or frozen.summary.content_digest != source_value.initializer_content_digest
    ):
        return None
    return frozen


def _normalized_snapshot(snapshot: SelectedGraphSnapshot) -> _NormalizedSelectedSnapshot:
    model = snapshot.model_copy()
    declaration = snapshot.declaration
    nodes = _resolve_nodes(model, declaration)
    node_id_by_index = {record.index: record.node_id for record in declaration.graph_nodes}

    summaries = initializer_value_summaries(model)
    initializers = {item.name: item for item in model.graph.initializer}
    initializer_names = set(initializers)
    if len(initializers) != len(model.graph.initializer):
        raise ValueError("initializer names are not unique")
    dynamic_inputs = [item.name for item in model.graph.input if item.name not in initializer_names]
    graph_outputs = [item.name for item in model.graph.output]
    input_roles = {name: ("graph_input", index) for index, name in enumerate(dynamic_inputs)}
    producer_roles = {
        value: ("node_output", node_id_by_index[index], slot)
        for index, node in enumerate(model.graph.node)
        for slot, value in enumerate(node.output)
        if value
    }
    produced_count = sum(bool(value) for node in model.graph.node for value in node.output)
    if len(producer_roles) != produced_count:
        raise ValueError("selected graph value has several producers")

    structural_names = {
        value for node in model.graph.node for value in (*node.input, *node.output) if value
    }
    structural_names.update(item.name for item in model.graph.input)
    structural_names.update(item.name for item in model.graph.output)
    structural_names.update(item.name for item in model.graph.value_info)
    structural_names.update(initializer_names)

    source_roots: dict[str, set[SourceOperandKey]] = {}
    for binding in declaration.source_bindings:
        if binding.graph_value in initializer_names:
            source_roots.setdefault(binding.graph_value, set()).add(binding.source)
    for supply in declaration.supplies:
        source_roots.setdefault(supply.root_graph_value, set()).add(supply.source)

    def initializer_payload(name: str) -> tuple[object, ...] | None:
        tensor = initializers.get(name)
        summary = summaries.get(name)
        if tensor is None or summary is None:
            return None
        return (
            int(tensor.data_type),
            tuple(int(extent) for extent in tensor.dims),
            summary.content_digest,
        )

    def base_value_key(name: str) -> tuple[object, ...]:
        payload = initializer_payload(name)
        if payload is not None:
            sources = tuple(sorted(source_roots.get(name, ()), key=repr))
            return (
                ("source_initializer", sources, *payload)
                if sources
                else (
                    "initializer",
                    *payload,
                )
            )
        if name in producer_roles:
            return producer_roles[name]
        if name in input_roles:
            return input_roles[name]
        raise ValueError(f"graph value {name!r} has no stable structural origin")

    annotations: dict[str, tuple[tuple[str, str], ...]] = {}
    for annotation in model.graph.quantization_annotation:
        if annotation.tensor_name in annotations:
            raise ValueError(f"graph value {annotation.tensor_name!r} has duplicate annotations")
        annotations[annotation.tensor_name] = tuple(
            sorted((item.key, item.value) for item in annotation.quant_parameter_tensor_names)
        )

    def annotation_key(name: str) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                key,
                ("value", base_value_key(value))
                if key != "finn_datatype" and value in structural_names
                else ("literal", value),
            )
            for key, value in annotations.get(name, ())
        )

    def value_key(name: str) -> tuple[object, ...]:
        return base_value_key(name), annotation_key(name)

    def shape_and_carrier(name: str) -> tuple[tuple[int, ...], int]:
        tensor = initializers.get(name)
        if tensor is not None:
            return tuple(int(extent) for extent in tensor.dims), int(tensor.data_type)
        shape = model.get_tensor_shape(name)
        info = model.get_tensor_valueinfo(name)
        if shape is None or info is None:
            raise ValueError(f"graph value {name!r} has incomplete static facts")
        return (
            tuple(int(extent) for extent in shape),
            int(info.type.tensor_type.elem_type),
        )

    def value_fact(name: str) -> tuple[object, ...]:
        shape, carrier = shape_and_carrier(name)
        return value_key(name), shape, carrier

    def normalized_inputs(node: Any) -> tuple[tuple[object, ...], ...]:
        values = [
            (
                value_key(value) if value else (("absent_input", index), ()),
                value in initializer_names if value else False,
            )
            for index, value in enumerate(node.input)
        ]
        if node.op_type in {"Add", "Mul", "And", "Or", "Xor", "Sum"}:
            values = [
                *(item for item in values if not item[1]),
                *(item for item in values if item[1]),
            ]
        return tuple(item[0] for item in values)

    node_signatures = []
    for node_id, node in nodes.items():
        core = node.__class__()
        core.CopyFrom(node)
        core.name = ""
        del core.input[:]
        del core.output[:]
        node_signatures.append(
            (
                node_id,
                core.SerializeToString(deterministic=True),
                normalized_inputs(node),
                tuple(("node_output", node_id, index) for index, _value in enumerate(node.output)),
            )
        )

    referenced = {
        value for node in model.graph.node for value in (*node.input, *node.output) if value
    }
    referenced.update(dynamic_inputs)
    referenced.update(graph_outputs)
    facts: dict[tuple[object, ...], tuple[object, ...]] = {}
    for name in referenced:
        fact = value_fact(name)
        key = cast("tuple[object, ...]", fact[0])
        previous = facts.get(key)
        if previous is not None and previous != fact:
            raise ValueError(f"stable value role for {name!r} has conflicting facts")
        facts[key] = fact

    def slot_key(slot: GraphSlotRef) -> tuple[object, ...]:
        if slot.kind is GraphSlotKind.GRAPH_INPUT:
            if slot.index is None or slot.index < 0 or slot.index >= len(model.graph.input):
                raise ValueError("graph-input slot does not resolve")
            name = model.graph.input[slot.index].name
            return slot.kind.value, value_key(name)
        if slot.kind is GraphSlotKind.GRAPH_OUTPUT:
            if slot.index is None or slot.index < 0 or slot.index >= len(model.graph.output):
                raise ValueError("graph-output slot does not resolve")
            name = model.graph.output[slot.index].name
            return slot.kind.value, value_key(name)
        if slot.kind is GraphSlotKind.INITIALIZER:
            if slot.index is not None or slot.owner not in initializer_names:
                raise ValueError("initializer slot does not resolve")
            return slot.kind.value, value_key(slot.owner)
        node = nodes.get(slot.owner)
        if node is None or slot.index is None:
            raise ValueError("node slot does not resolve")
        values = node.input if slot.kind is GraphSlotKind.NODE_INPUT else node.output
        if slot.index < 0 or slot.index >= len(values):
            raise ValueError("node slot index is out of range")
        return slot.kind.value, slot.owner, slot.index, value_key(values[slot.index])

    interface_bindings = tuple(
        sorted(
            [
                (
                    item.interface,
                    value_key(item.graph_value),
                    item.relation,
                    tuple(slot_key(slot) for slot in item.anchors),
                )
                for item in declaration.interface_bindings
            ],
            key=repr,
        )
    )
    source_bindings = tuple(
        sorted(
            [
                (
                    item.source,
                    value_key(item.graph_value),
                    item.relation,
                    tuple(slot_key(slot) for slot in item.anchors),
                )
                for item in declaration.source_bindings
            ],
            key=repr,
        )
    )
    supplies = tuple(
        sorted(
            [
                (
                    item.required_input,
                    value_key(item.root_graph_value),
                    item.source,
                    item.derivation_nodes,
                )
                for item in declaration.supplies
            ],
            key=repr,
        )
    )

    used_names = {
        value for node in model.graph.node for value in (*node.input, *node.output) if value
    }
    used_names.update(dynamic_inputs)
    used_names.update(graph_outputs)
    unused_initializers = tuple(
        sorted(
            (value_fact(name) for name in initializer_names if name not in used_names),
            key=repr,
        )
    )
    unused_value_info = tuple(
        sorted(
            [
                (
                    tuple(int(extent.dim_value) for extent in item.type.tensor_type.shape.dim),
                    int(item.type.tensor_type.elem_type),
                    annotation_key(item.name),
                )
                for item in model.graph.value_info
                if item.name not in used_names
            ],
            key=repr,
        )
    )
    unused_annotations = tuple(
        sorted(
            (annotation_key(name) for name in annotations if name not in used_names),
            key=repr,
        )
    )
    static_graph_inputs = tuple(
        value_fact(item.name) for item in model.graph.input if item.name in initializer_names
    )
    metadata = tuple(
        sorted(
            (item.key, item.value)
            for item in model.graph.metadata_props
            if item.key != SELECTED_METADATA_KEY
        )
    )
    core = (
        int(model.model.ir_version),
        tuple(sorted((item.domain, int(item.version)) for item in model.model.opset_import)),
        metadata,
        tuple(value_fact(name) for name in dynamic_inputs),
        tuple(value_fact(name) for name in graph_outputs),
        tuple(sorted(node_signatures, key=repr)),
        tuple(sorted(facts.values(), key=repr)),
        declaration.identity,
        declaration.version,
        declaration.construction,
        declaration.source,
        declaration.choices,
        interface_bindings,
        source_bindings,
        supplies,
        declaration.ownership,
    )
    return _NormalizedSelectedSnapshot(
        core,
        static_graph_inputs,
        unused_initializers,
        unused_value_info,
        unused_annotations,
    )


def _resolve_nodes(model: ModelWrapper, declaration: SelectedGraphDeclaration) -> dict[str, Any]:
    if len(declaration.graph_nodes) != len(model.graph.node):
        raise ValueError("graph-node bindings are incomplete")
    result: dict[str, Any] = {}
    indices: set[int] = set()
    for record in declaration.graph_nodes:
        if record.node_id in result or record.index in indices:
            raise ValueError("graph-node bindings are duplicate")
        if record.index < 0 or record.index >= len(model.graph.node):
            raise ValueError("graph-node binding index is out of range")
        result[record.node_id] = model.graph.node[record.index]
        indices.add(record.index)
    return result


__all__ = ["frozen_initializer_for_source", "verify_normalized_selected_snapshot"]
