# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only inspection of the two explicitly supported legacy native schemas."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any

from onnx import AttributeProto, helper  # type: ignore[import-not-found]

from finn.dataflow.ops.base import DataflowOp
from finn.dataflow.ops.native import (
    FINGERPRINT_ATTRIBUTE,
    SCHEMA_VERSION_ATTRIBUTE,
    NativeAttribute,
    operation_choice_schema,
)
from finn.dataflow.ops.selected import RecordedChoice
from finn.dataflow.ops.source import SourceOperand
from finn.dataflow.space.declarations import Problem, check_canonical, declared_members


class LegacySelectionError(ValueError):
    """A legacy selection cannot be interpreted by its explicit adapter."""


@dataclass(frozen=True, slots=True)
class LegacySelection:
    adapter_id: str
    adapter_version: int
    from_schema_version: int
    to_schema_version: int
    problem_fingerprint: str
    choices: tuple[RecordedChoice, ...]
    owned_attributes: tuple[str, ...]
    owned_attribute_bytes: tuple[tuple[str, bytes], ...]

    def __post_init__(self) -> None:
        if type(self.adapter_id) is not str or not self.adapter_id:
            raise TypeError("legacy adapter id must be a non-empty string")
        for name, value in (
            ("adapter_version", self.adapter_version),
            ("from_schema_version", self.from_schema_version),
            ("to_schema_version", self.to_schema_version),
        ):
            if type(value) is not int or value < 1:
                raise TypeError(f"{name} must be a positive integer")
        if self.from_schema_version >= self.to_schema_version:
            raise ValueError("legacy adapter schema versions must advance")
        if type(self.problem_fingerprint) is not str or not self.problem_fingerprint:
            raise TypeError("legacy problem fingerprint must be a non-empty string")
        choices = tuple(self.choices)
        owned = tuple(self.owned_attributes)
        raw = tuple((name, bytes(value)) for name, value in self.owned_attribute_bytes)
        if len(owned) != len(set(owned)) or any(
            type(name) is not str or not name for name in owned
        ):
            raise ValueError("legacy owned attribute names must be unique non-empty strings")
        raw_names = tuple(name for name, _value in raw)
        if len(raw_names) != len(set(raw_names)) or not set(raw_names).issubset(owned):
            raise ValueError("legacy owned attribute bytes must be unique and owned")
        object.__setattr__(self, "choices", choices)
        object.__setattr__(self, "owned_attributes", owned)
        object.__setattr__(self, "owned_attribute_bytes", raw)


@dataclass(frozen=True, slots=True)
class _LegacyChoice:
    path: str
    attribute: str
    kind: str


@dataclass(frozen=True, slots=True)
class _LegacyAdapter:
    adapter_id: str
    adapter_version: int
    family: str
    from_schema_version: int
    to_schema_version: int
    choices: tuple[_LegacyChoice, ...]

    @property
    def owned_attributes(self) -> tuple[str, ...]:
        return (
            SCHEMA_VERSION_ATTRIBUTE,
            FINGERPRINT_ATTRIBUTE,
            *(item.attribute for item in self.choices),
        )


def _choice(path: str, attribute: str, kind: str) -> _LegacyChoice:
    return _LegacyChoice(path, attribute, kind)


_REPLAY = _LegacyAdapter(
    "finn.dataflow.legacy.activation_replay",
    1,
    "finn.dataflow.activation_replay",
    2,
    3,
    (
        _choice("design.pe", "design__pe", "i"),
        _choice("design.simd", "design__simd", "i"),
    ),
)

_MVAU = _LegacyAdapter(
    "finn.dataflow.legacy.mvau",
    1,
    "finn.dataflow.mvau",
    3,
    4,
    (
        _choice("design.case", "design__case", "s"),
        _choice("design.dot_product.pe", "design__dot_product__pe", "i"),
        _choice("design.dot_product.simd", "design__dot_product__simd", "i"),
        _choice(
            "design.dot_product.weight_supply",
            "design__dot_product__weight_supply",
            "s",
        ),
        _choice(
            "design.dot_product.compute.kernel",
            "design__dot_product__compute__kernel",
            "s",
        ),
        _choice(
            "design.dot_product.compute.dotp_axi.compute_pumping",
            "design__dot_product__compute__dotp_axi__compute_pumping",
            "i",
        ),
        _choice(
            "design.dot_product.compute.dotp_axi_embedded.compute_pumping",
            "design__dot_product__compute__dotp_axi_embedded__compute_pumping",
            "i",
        ),
        _choice("design.batch_interleaved.pe", "design__batch_interleaved__pe", "i"),
        _choice("design.batch_interleaved.simd", "design__batch_interleaved__simd", "i"),
        _choice(
            "design.batch_interleaved.interleave",
            "design__batch_interleaved__interleave",
            "i",
        ),
        _choice(
            "design.batch_interleaved.compute.dotp_axi_batch_interleaved.compute_pumping",
            "design__batch_interleaved__compute__dotp_axi_batch_interleaved__compute_pumping",
            "i",
        ),
    ),
)

_ADAPTERS = MappingProxyType({_REPLAY.family: _REPLAY, _MVAU.family: _MVAU})


def inspect_legacy_selection(operation: DataflowOp) -> LegacySelection:
    """Decode one source-only-bound old native selection without modifying it."""

    if not isinstance(operation, DataflowOp) or not operation.is_bound:
        raise LegacySelectionError("legacy inspection requires a source-only-bound DataflowOp")
    family = type(operation).family
    try:
        adapter = _ADAPTERS[family]
    except KeyError as error:
        raise LegacySelectionError(f"no legacy selection adapter for family {family!r}") from error
    if type(operation).schema_version != adapter.to_schema_version:
        raise LegacySelectionError(
            f"adapter target schema {adapter.to_schema_version} differs from current "
            f"schema {type(operation).schema_version}"
        )

    node = operation.onnx_node
    choice_names = {item.attribute for item in adapter.choices}
    owned = set(adapter.owned_attributes)
    present: dict[str, Any] = {}
    for attribute in node.attribute:
        name = attribute.name
        if name.startswith("design__") and name not in choice_names:
            raise LegacySelectionError(f"unknown legacy choice attribute {name!r}")
        if name not in owned:
            continue
        if name in present:
            raise LegacySelectionError(f"duplicate legacy attribute {name!r}")
        present[name] = attribute

    schema = _decode_attribute(present.get(SCHEMA_VERSION_ATTRIBUTE), "i", SCHEMA_VERSION_ATTRIBUTE)
    if schema.value != adapter.from_schema_version:
        raise LegacySelectionError(
            f"adapter expects schema {adapter.from_schema_version}, found {schema.value!r}"
        )
    fingerprint = _decode_attribute(present.get(FINGERPRINT_ATTRIBUTE), "s", FINGERPRINT_ATTRIBUTE)
    if type(fingerprint.value) is not str or not fingerprint.value:
        raise LegacySelectionError("legacy problem fingerprint must be a non-empty string")
    expected_fingerprint = _historical_problem_fingerprint(operation)
    if fingerprint.value != expected_fingerprint:
        raise LegacySelectionError(
            "legacy choices were recorded for different source/build facts: "
            f"expected historical fingerprint {expected_fingerprint}, found {fingerprint.value}"
        )

    current = {item.choice.path: item for item in operation_choice_schema(type(operation))}
    choices = []
    for legacy in adapter.choices:
        attribute = present.get(legacy.attribute)
        if attribute is None:
            continue
        try:
            declaration = current[legacy.path]
        except KeyError as error:
            raise LegacySelectionError(
                f"legacy choice path {legacy.path!r} is absent from the current declaration"
            ) from error
        if declaration.name != legacy.attribute:
            raise LegacySelectionError(
                f"legacy choice path {legacy.path!r} no longer has attribute {legacy.attribute!r}"
            )
        encoded = _decode_attribute(attribute, legacy.kind, legacy.attribute)
        try:
            value = declaration.decode(encoded)
        except (TypeError, ValueError, UnicodeError) as error:
            raise LegacySelectionError(
                f"cannot decode legacy choice {legacy.path!r}: {error}"
            ) from error
        choices.append(RecordedChoice(legacy.path, value))

    raw = tuple(
        (
            name,
            present[name].SerializeToString(deterministic=True),
        )
        for name in adapter.owned_attributes
        if name in present
    )
    return LegacySelection(
        adapter.adapter_id,
        adapter.adapter_version,
        adapter.from_schema_version,
        adapter.to_schema_version,
        fingerprint.value,
        tuple(choices),
        adapter.owned_attributes,
        raw,
    )


def _decode_attribute(attribute: Any | None, kind: str, name: str) -> NativeAttribute:
    if attribute is None:
        raise LegacySelectionError(f"legacy selection is missing attribute {name!r}")
    expected_type = {
        "i": AttributeProto.INT,
        "f": AttributeProto.FLOAT,
        "s": AttributeProto.STRING,
        "ints": AttributeProto.INTS,
        "floats": AttributeProto.FLOATS,
        "strings": AttributeProto.STRINGS,
    }[kind]
    if attribute.type != expected_type:
        raise LegacySelectionError(f"legacy attribute {name!r} must use ONNX kind {kind!r}")
    value = helper.get_attribute_value(attribute)
    try:
        if kind == "s":
            value = value.decode("utf-8")
        elif kind == "strings":
            value = tuple(item.decode("utf-8") for item in value)
        elif kind in {"ints", "floats"}:
            value = tuple(value)
    except UnicodeError as error:
        raise LegacySelectionError(f"legacy attribute {name!r} is not UTF-8") from error
    return NativeAttribute(kind, value)


def _historical_problem_fingerprint(operation: DataflowOp) -> str:
    snapshot = operation.problem_snapshot
    problem = []
    for name, declaration in declared_members(type(operation)):
        if not isinstance(declaration, Problem):
            continue
        source_operand = declaration.canonical.identity == "finn.dataflow.source_operand"
        codec = (
            "finn.dataflow.source_operand@2"
            if source_operand
            else f"{declaration.canonical.identity}@{declaration.canonical.version}"
        )
        if declaration in snapshot:
            value = snapshot[declaration]
            if source_operand:
                encoded = check_canonical(
                    _encode_source_operand_v2(value),
                    f"historical source operand {type(operation).__name__}.{name}",
                )
            else:
                encoded = check_canonical(
                    declaration.canonical.encode(value),
                    f"historical problem {type(operation).__name__}.{name}",
                )
            wrapped = {"present": encoded}
        else:
            wrapped = {"absent": True}
        problem.append(
            {
                "name": name if declaration.stable_name is None else declaration.stable_name,
                "semantics": declaration.value_semantics.name,
                "codec": codec,
                "value": wrapped,
            }
        )
    payload = {
        "space": f"{type(operation).__module__}.{type(operation).__qualname__}",
        "problem": problem,
    }
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload_bytes).hexdigest()


def _encode_source_operand_v2(value: object) -> dict[str, object]:
    if not isinstance(value, SourceOperand):
        raise LegacySelectionError("historical source operand has the wrong nominal type")
    return {
        "operand": value.id,
        "shape": list(value.shape),
        "datatype": value.datatype.name,
        "initializer": value.initializer,
        "digest": value.initializer_digest,
    }


__all__ = ["LegacySelection", "LegacySelectionError", "inspect_legacy_selection"]
