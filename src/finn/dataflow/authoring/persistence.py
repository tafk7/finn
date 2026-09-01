# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Portable operation-choice codecs and class-local persistence declarations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
import math
from types import MappingProxyType
from typing import Protocol, cast, get_args, get_origin, get_type_hints

from finn.dataflow.authoring.declarations import CompiledClassDeclarations
from finn.dataflow.authoring.scope import AuthoringError, Ref
from finn.dataflow.datatypes import (
    QONNX_DATATYPE_TOKEN,
    decode_datatype,
    encode_datatype,
    is_qonnx_datatype,
)
from finn.dataflow.design import QualifiedPath
from finn.dataflow.op_contracts import NodeAttrCodec, NodeAttributeType


class DecisionStorageCodec(Protocol):
    @property
    def attribute_name(self) -> str: ...

    @property
    def value_type(self) -> object: ...

    @property
    def nodeattr_definition(self) -> NodeAttributeType: ...

    def encode(self, value: object) -> int | float | str: ...

    def decode(self, value: object) -> object: ...


def _type_token(value_type: type[object]) -> str:
    token = getattr(
        value_type,
        "__dataflow_identity_token__",
        f"{value_type.__module__}.{value_type.__qualname__}",
    )
    if not isinstance(token, str) or not token:
        raise TypeError(f"invalid stable identity token for {value_type.__name__}")
    return token


def _encode(value: object) -> object:
    if value is None:
        return {"optional": "absent"}
    if type(value) is bool:
        return {"bool": value}
    if type(value) is int:
        return {"int": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("portable persistence refuses non-finite floats")
        return {"float": value.hex()}
    if type(value) is str:
        return {"str": value}
    if is_qonnx_datatype(value):
        return encode_datatype(value)
    if isinstance(value, Enum):
        return {
            "enum_type": _type_token(type(value)),
            "member": value.name,
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "structured_type": _type_token(type(value)),
            "fields": [
                [field.name, _encode(getattr(value, field.name))] for field in fields(value)
            ],
        }
    if isinstance(value, Mapping):
        encoded = [[_encode(key), _encode(item)] for key, item in value.items()]
        encoded.sort(key=lambda item: json.dumps(item[0], sort_keys=True, separators=(",", ":")))
        return {"mapping": encoded}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {"sequence": [_encode(item) for item in value]}
    raise TypeError(f"no portable persistence encoding for {type(value).__name__}")


def _decode(payload: object, expected: object) -> object:
    origin = get_origin(expected)
    arguments = get_args(expected)
    if origin is not None and type(None) in arguments:
        if payload == {"optional": "absent"}:
            return None
        present = tuple(item for item in arguments if item is not type(None))
        if len(present) != 1:
            raise TypeError("portable optional values require one present type")
        return _decode(payload, present[0])
    if expected is bool and isinstance(payload, dict) and set(payload) == {"bool"}:
        value = payload["bool"]
        if type(value) is bool:
            return value
    if expected is int and isinstance(payload, dict) and set(payload) == {"int"}:
        value = payload["int"]
        if type(value) is int:
            return value
    if expected is float and isinstance(payload, dict) and set(payload) == {"float"}:
        value = payload["float"]
        if isinstance(value, str):
            decoded = float.fromhex(value)
            if math.isfinite(decoded):
                return decoded
    if expected is str and isinstance(payload, dict) and set(payload) == {"str"}:
        value = payload["str"]
        if type(value) is str:
            return value
    if expected is QONNX_DATATYPE_TOKEN:
        return decode_datatype(payload)
    if isinstance(expected, type) and issubclass(expected, Enum):
        if (
            isinstance(payload, dict)
            and set(payload) == {"enum_type", "member"}
            and payload["enum_type"] == _type_token(expected)
            and isinstance(payload["member"], str)
        ):
            return expected[payload["member"]]
    if isinstance(expected, type) and is_dataclass(expected):
        if (
            not isinstance(payload, dict)
            or set(payload) != {"structured_type", "fields"}
            or payload["structured_type"] != _type_token(expected)
            or not isinstance(payload["fields"], list)
        ):
            raise ValueError("structured persistence payload has the wrong type or fields")
        annotations = get_type_hints(expected)
        encoded_fields = payload["fields"]
        if any(not isinstance(item, list) or len(item) != 2 for item in encoded_fields):
            raise ValueError("structured persistence fields are malformed")
        by_name = {cast(str, item[0]): item[1] for item in encoded_fields}
        expected_names = tuple(field.name for field in fields(expected))
        if tuple(item[0] for item in encoded_fields) != expected_names:
            raise ValueError("structured persistence fields are not exact or ordered")
        return expected(
            **{name: _decode(by_name[name], annotations[name]) for name in expected_names}
        )
    if (
        origin in {tuple, list, Sequence}
        and isinstance(payload, dict)
        and set(payload) == {"sequence"}
    ):
        raw = payload["sequence"]
        if not isinstance(raw, list):
            raise ValueError("portable sequence payload is malformed")
        item_type = arguments[0] if arguments else object
        values = tuple(
            _decode(item, item_type) if item_type is not object else item for item in raw
        )
        return list(values) if origin is list else values
    if origin in {dict, Mapping} and isinstance(payload, dict) and set(payload) == {"mapping"}:
        raw = payload["mapping"]
        if not isinstance(raw, list):
            raise ValueError("portable mapping payload is malformed")
        key_type, value_type = arguments if len(arguments) == 2 else (object, object)
        return {
            _decode(item[0], key_type) if key_type is not object else item[0]: _decode(
                item[1], value_type
            )
            if value_type is not object
            else item[1]
            for item in raw
        }
    raise ValueError(f"portable payload does not encode {expected!r}")


@dataclass(frozen=True, slots=True)
class PortableCodec:
    """Canonical JSON-compatible codec for one declared runtime type."""

    value_type: object

    def encode(self, value: object) -> object:
        return _encode(value)

    def decode(self, payload: object) -> object:
        return _decode(payload, self.value_type)

    def dumps(self, value: object) -> str:
        return json.dumps(self.encode(value), sort_keys=True, separators=(",", ":"))

    def loads(self, payload: str) -> object:
        return self.decode(json.loads(payload))


@dataclass(frozen=True, slots=True)
class JsonNodeAttrCodec:
    """Store a portable value in one deterministic ONNX string attribute."""

    attribute_name: str
    value_type: object
    portable: PortableCodec

    @property
    def nodeattr_definition(self) -> NodeAttributeType:
        return ("s", False, "", None)

    def encode(self, value: object) -> str:
        return self.portable.dumps(value)

    def decode(self, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError(f"{self.attribute_name} must contain a JSON string")
        return self.portable.loads(value)


@dataclass(frozen=True, slots=True)
class Persist:
    """Operation-owned storage name for one class-declared decision."""

    decision: object
    attribute_name: str
    codec: DecisionStorageCodec | None = None

    def __post_init__(self) -> None:
        if not self.attribute_name:
            raise ValueError("a persistence attribute name must not be empty")


def _default_codec(attribute_name: str, value_type: object) -> DecisionStorageCodec:
    if value_type is bool:
        return NodeAttrCodec.boolean(attribute_name)
    if value_type is int:
        return NodeAttrCodec.integer(attribute_name)
    if value_type is str:
        return NodeAttrCodec.string(attribute_name)
    if isinstance(value_type, type) and issubclass(value_type, Enum):
        tokens = {
            str(member.value) if isinstance(member.value, str) else member.name: member
            for member in value_type
        }
        return NodeAttrCodec.finite_enum(attribute_name, value_type, tokens)
    return JsonNodeAttrCodec(attribute_name, value_type, PortableCodec(value_type))


def compile_persistence(
    declarations: CompiledClassDeclarations,
    persistence: Sequence[Persist],
    *,
    external: Mapping[int, Ref[object]] = MappingProxyType({}),
) -> Mapping[QualifiedPath, DecisionStorageCodec]:
    result: dict[QualifiedPath, DecisionStorageCodec] = {}
    storage_names: set[str] = set()
    for item in persistence:
        member_name = declarations.template_members.get(id(item.decision))
        ref = declarations.ref(member_name) if member_name is not None else None
        external_ref = external.get(id(item.decision))
        if ref is None and external_ref is None:
            raise AuthoringError("a persisted decision is not declared by the operation class")
        resolved_ref = ref if ref is not None else cast(Ref[object], external_ref)
        path = resolved_ref.path
        if ref is not None and ref.kind.value != "decision":
            raise AuthoringError(f"persisted member {member_name!r} is not a decision")
        if item.attribute_name in storage_names:
            raise AuthoringError(
                f"persistent node-attribute name {item.attribute_name!r} is duplicated"
            )
        storage_names.add(item.attribute_name)
        semantics = resolved_ref.semantics
        codec = item.codec or _default_codec(item.attribute_name, semantics.type_token)
        if codec.value_type is not semantics.type_token:
            raise AuthoringError(
                f"persistence codec for {member_name!r} has incompatible value semantics"
            )
        result[path] = codec
    return MappingProxyType(result)


__all__ = [
    "DecisionStorageCodec",
    "JsonNodeAttrCodec",
    "Persist",
    "PortableCodec",
    "compile_persistence",
]
