# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit portable selection schemas and application-owned value codecs."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, TypeAlias, TypeVar, Union, cast

from .compiler import SpaceModel
from .declarations import Decision, DecisionRef, Space
from .errors import DefinitionError, RequestError
from .inspection import DecisionInfo, decision_info, decisions
from .selections import Selection, _recognize, _semantics, _snapshot

T = TypeVar("T")
S = TypeVar("S", bound=Space)
JSONValue: TypeAlias = Union[None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"]]


def _identity(name: str, version: int, *, label: str) -> None:
    if type(name) is not str or not name:
        raise DefinitionError(f"{label} identity must be a nonempty string")
    if type(version) is not int or version < 1:
        raise DefinitionError(f"{label} version must be a positive integer")


def _json_copy(value: object, *, label: str, seen: set[int] | None = None) -> JSONValue:
    """Detach JSON trees, rejecting nonportable values and cyclic containers."""
    if value is None or type(value) in (bool, int, str):
        return cast(JSONValue, value)
    if type(value) is float:
        if not math.isfinite(value):
            raise RequestError(f"{label}: nonfinite floats are not portable JSON values")
        return value
    if type(value) not in (dict, list):
        raise RequestError(f"{label}: expected JSON primitives, lists, or string-keyed objects")
    active = set() if seen is None else seen
    identity = id(value)
    if identity in active:
        raise RequestError(f"{label}: a portable document cannot contain a cycle")
    active.add(identity)
    try:
        if isinstance(value, list):
            return [_json_copy(item, label=label, seen=active) for item in value]
        assert isinstance(value, dict)
        if any(type(key) is not str for key in value):
            raise RequestError(f"{label}: JSON object keys must be strings")
        return {key: _json_copy(item, label=label, seen=active) for key, item in value.items()}
    finally:
        active.remove(identity)


@dataclass(frozen=True)
class ValueCodec(Generic[T]):
    """One explicit codec identity; no registry or default codec is consulted."""

    id: str
    version: int
    encode: Callable[[T], JSONValue]
    decode: Callable[[JSONValue], T]

    def __post_init__(self) -> None:
        _identity(self.id, self.version, label="codec")
        if not callable(self.encode) or not callable(self.decode):
            raise DefinitionError("codec encode and decode must be callable")


@dataclass(frozen=True, slots=True)
class CodecBinding:
    """A type-checked pairing, constructed with codec_for()."""

    reference: Decision[object] | DecisionRef[object]
    codec: ValueCodec[object]


def codec_for(reference: Decision[T] | DecisionRef[T], codec: ValueCodec[T]) -> CodecBinding:
    return CodecBinding(
        cast("Decision[object] | DecisionRef[object]", reference), cast(ValueCodec[object], codec)
    )


@dataclass(frozen=True, slots=True)
class _SchemaEntry:
    info: DecisionInfo[object]
    codec: ValueCodec[object]


@dataclass(frozen=True, slots=True, init=False)
class SelectionSchema:
    """An explicit portable family/version and codecs for known stable choice keys."""

    family: str
    version: int
    owned_keys: frozenset[str]
    _model: SpaceModel[Space] = field(repr=False)
    _entries: Mapping[str, _SchemaEntry] = field(repr=False)
    _decisions: Mapping[str, DecisionInfo[object]] = field(repr=False)

    def __init__(
        self,
        model: SpaceModel[S],
        *,
        family: str,
        version: int,
        bindings: Iterable[CodecBinding],
        owned_keys: Iterable[str] = (),
    ) -> None:
        _identity(family, version, label="selection schema")
        if not isinstance(model, SpaceModel):
            raise DefinitionError("a selection schema requires a compiled model")
        declared = {info.key: info for info in decisions(model)}
        entries: dict[str, _SchemaEntry] = {}
        for binding in bindings:
            if not isinstance(binding, CodecBinding) or not isinstance(binding.codec, ValueCodec):
                raise DefinitionError("schema bindings must come from codec_for()")
            info = decision_info(model, binding.reference)
            if info.key in entries:
                raise DefinitionError(f"duplicate codec binding for {info.key!r}")
            entries[info.key] = _SchemaEntry(info, binding.codec)
        owned = frozenset(owned_keys)
        if any(type(key) is not str or not key for key in owned):
            raise DefinitionError("historical owned keys must be nonempty strings")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "owned_keys", owned | declared.keys())
        object.__setattr__(self, "_model", cast(SpaceModel[Space], model))
        object.__setattr__(self, "_entries", MappingProxyType(entries))
        object.__setattr__(self, "_decisions", MappingProxyType(declared))


def encode(selection: Selection, schema: SelectionSchema) -> dict[str, JSONValue]:
    """Encode commitments and verify each codec preserves declared value equality."""
    if not isinstance(selection, Selection) or not isinstance(schema, SelectionSchema):
        raise RequestError("encode requires a Selection and SelectionSchema")
    if selection._model.linked is not schema._model.linked:
        raise RequestError("selection and schema belong to different compiled models")
    for key in selection.keys:
        if key not in schema._entries:
            raise RequestError(f"{key}: encoding a committed choice requires an explicit codec")
    selected_entries = tuple((entry.info.key, entry.value) for entry in selection._entries)
    for key, value in selected_entries:
        info = schema._entries[key].info
        if info.selector and (type(value) is not str or value not in info.cases):
            raise RequestError(f"{key}: unknown structural case {value!r}")
    entries: list[JSONValue] = []
    for key, value in selected_entries:
        bound = schema._entries[key]
        codec = bound.codec
        try:
            expected = _snapshot(bound.info, value)
            encoded = codec.encode(_snapshot(bound.info, value))
        except Exception as cause:
            raise RequestError(f"{key}: codec {codec.id!r} encoding failed: {cause}") from cause
        payload = _json_copy(encoded, label=f"{key} codec {codec.id!r} encoded value")
        try:
            decoded = codec.decode(_json_copy(payload, label=f"{key} round-trip value"))
            _recognize(bound.info, decoded)
            restored = _snapshot(bound.info, decoded)
            equal = _semantics(bound.info).values_equal(
                _snapshot(bound.info, expected), _snapshot(bound.info, restored)
            )
        except Exception as cause:
            raise RequestError(
                f"{key}: codec {codec.id!r} round-trip validation failed: {cause}"
            ) from cause
        if not equal:
            raise RequestError(
                f"{key}: codec {codec.id!r} changes the committed value on round-trip"
            )
        entries.append(
            {
                "key": key,
                "codec": codec.id,
                "codec_version": codec.version,
                "value": payload,
            }
        )
    return {"family": schema.family, "version": schema.version, "entries": entries}


def decode(document: object, schema: SelectionSchema) -> Selection:
    """Check all document structure before decoding values; replay remains separate."""
    if not isinstance(schema, SelectionSchema):
        raise RequestError("decode requires an explicit SelectionSchema")
    copied = _json_copy(document, label="selection document")
    if not isinstance(copied, dict) or copied.keys() != {"family", "version", "entries"}:
        raise RequestError("selection document requires exactly family, version, and entries")
    if copied["family"] != schema.family or type(copied["family"]) is not str:
        raise RequestError("selection family is incompatible with this schema")
    if copied["version"] != schema.version or type(copied["version"]) is not int:
        raise RequestError("selection schema version is incompatible")
    raw_entries = copied["entries"]
    if not isinstance(raw_entries, list):
        raise RequestError("selection entries must be a list")
    pending: list[tuple[_SchemaEntry, JSONValue]] = []
    seen: set[str] = set()
    for entry in raw_entries:
        if not isinstance(entry, dict) or entry.keys() != {
            "key",
            "codec",
            "codec_version",
            "value",
        }:
            raise RequestError("each selection entry requires key, codec, codec_version, and value")
        key = entry["key"]
        if type(key) is not str or key not in schema._decisions:
            raise RequestError(f"unknown selection key {key!r}")
        if key in seen:
            raise RequestError(f"duplicate selection key {key!r}")
        seen.add(key)
        bound = schema._entries.get(key)
        if bound is None:
            raise RequestError(f"{key}: no explicit codec is available")
        if entry["codec"] != bound.codec.id or type(entry["codec"]) is not str:
            raise RequestError(f"{key}: incompatible codec identity")
        if entry["codec_version"] != bound.codec.version or type(entry["codec_version"]) is not int:
            raise RequestError(f"{key}: incompatible codec version")
        pending.append((bound, entry["value"]))
    values: list[tuple[DecisionInfo[object], object]] = []
    for bound, encoded in pending:
        try:
            value = bound.codec.decode(encoded)
        except Exception as cause:
            raise RequestError(
                f"{bound.info.key}: codec {bound.codec.id!r} decoding failed: {cause}"
            ) from cause
        if bound.info.selector and (type(value) is not str or value not in bound.info.cases):
            raise RequestError(f"{bound.info.key}: unknown structural case {value!r}")
        values.append((bound.info, value))
    return Selection._from_values(schema._model, values)


__all__ = [
    "CodecBinding",
    "JSONValue",
    "SelectionSchema",
    "ValueCodec",
    "codec_for",
    "decode",
    "encode",
]
