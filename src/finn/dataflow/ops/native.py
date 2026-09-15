# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Native ONNX choice attributes, discovered from compiled declaration names.

Dots in root-relative stable paths become double underscores. All other path
characters are preserved. Collisions (including with source or metadata names)
are authoring errors. Encoding changes require the operation schema version to
change; individual values carry no codec tags.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast, TYPE_CHECKING

from finn.dataflow._engine import Answer, Decided, QualifiedPath, RequestError
from finn.dataflow.space.declarations import (
    AuthoringError,
    CanonicalValue,
    PersistentCodec,
    check_canonical,
)
from finn.dataflow.space.occurrence import (
    PersistableChoice,
    occurrence_answer_at,
    occurrence_commit_paths,
    occurrence_persistable,
)

if TYPE_CHECKING:
    from finn.dataflow.ops.base import DataflowOp

SCOPE_ID_ATTRIBUTE = "dataflow_scope_id"
FINGERPRINT_ATTRIBUTE = "dataflow_problem_fingerprint"
SCHEMA_VERSION_ATTRIBUTE = "dataflow_schema_version"
RESERVED_ATTRIBUTES = frozenset(
    {SCOPE_ID_ATTRIBUTE, FINGERPRINT_ATTRIBUTE, SCHEMA_VERSION_ATTRIBUTE}
)


class DecodeError(ValueError):
    """A native attribute does not match its declaration's encoding."""


@dataclass(frozen=True, slots=True)
class AttributeCodec(PersistentCodec[Any]):
    """An explicit native kind and two functions for a structured Decision.

    The inherited identity/version describe the authoring codec. The operation's
    schema_version is the durable compatibility guard, so changing this codec
    requires bumping that schema. No per-value codec tag is stored.
    """

    kind: str = "s"

    def __post_init__(self) -> None:
        PersistentCodec.__post_init__(self)
        if self.kind not in {"i", "f", "s", "ints", "floats", "strings"}:
            raise AuthoringError(f"unsupported native attribute kind {self.kind!r}")


@dataclass(frozen=True, slots=True)
class NativeAttribute:
    kind: str
    value: Any

    def proto(self, name: str) -> Any:
        from onnx import AttributeProto, helper  # type: ignore[import-not-found] # noqa: PLC0415

        kinds = {
            "i": AttributeProto.INT,
            "f": AttributeProto.FLOAT,
            "s": AttributeProto.STRING,
            "ints": AttributeProto.INTS,
            "floats": AttributeProto.FLOATS,
            "strings": AttributeProto.STRINGS,
        }
        return helper.make_attribute(name, self.value, attr_type=kinds[self.kind])


def attribute_name(path: str) -> str:
    return path.replace(".", "__")


def _exact(value: Any, token: type, what: str) -> Any:
    if type(value) is not token:
        raise DecodeError(f"{what} requires {token.__name__}, got {type(value).__name__}")
    return value


def _boolean(value: Any) -> bool:
    _exact(value, int, "boolean attribute")
    if value not in (0, 1):
        raise DecodeError("a boolean attribute is exactly 0 or 1")
    return bool(value)


def _native(kind: str, value: Any) -> NativeAttribute:
    if kind in ("ints", "floats", "strings"):
        if not isinstance(value, (tuple, list)):
            raise DecodeError(f"{kind} requires a sequence")
        scalar = {"ints": "i", "floats": "f", "strings": "s"}[kind]
        return NativeAttribute(kind, tuple(_native(scalar, item).value for item in value))
    if kind == "i":
        _exact(value, int, "integer attribute")
        if not -(2**63) <= value < 2**63:
            raise DecodeError("an ONNX integer attribute must fit signed 64 bits")
    elif kind == "f":
        _exact(value, float, "float attribute")
        if not math.isfinite(value):
            raise DecodeError("ordinary float Decisions must be finite")
        try:
            exact = struct.unpack("f", struct.pack("f", value))[0]
        except OverflowError as error:
            raise DecodeError("float Decision does not fit an ONNX float") from error
        if exact != value:
            raise DecodeError(
                "float Decision loses precision in ONNX FLOAT; declare an AttributeCodec"
            )
    elif kind == "s":
        _exact(value, str, "string attribute")
    else:
        raise DecodeError(f"unsupported native kind {kind!r}")
    return NativeAttribute(kind, value)


@dataclass(frozen=True, slots=True)
class ChoiceAttribute:
    choice: PersistableChoice
    name: str
    codec: AttributeCodec | None
    token: Any

    def encode(self, value: Any) -> NativeAttribute:
        encoded = encode_choice_value(self, value)
        if self.codec is not None:
            return _native(self.codec.kind, encoded)
        if self.choice.selector or self.token is str:
            return _native("s", encoded)
        if self.token is bool:
            return _native("i", int(cast(bool, encoded)))
        if self.token is int:
            return _native("i", encoded)
        if self.token is float:
            return _native("f", encoded)
        if isinstance(self.token, type) and issubclass(self.token, Enum):
            return _native("s", encoded)
        if not isinstance(encoded, list):
            raise DecodeError("tuple Decision did not encode as a list")
        item_type = type(encoded[0]) if encoded else int
        kinds = {int: "ints", float: "floats", str: "strings"}
        if item_type not in kinds:
            raise DecodeError("a tuple Decision needs a homogeneous native list or AttributeCodec")
        return _native(kinds[item_type], encoded)

    def decode(self, attribute: NativeAttribute) -> object:
        if self.codec is not None:
            if attribute.kind != self.codec.kind:
                raise DecodeError(f"expected {self.codec.kind}, got {attribute.kind}")
            encoded: object = (
                list(attribute.value)
                if attribute.kind in {"ints", "floats", "strings"}
                else attribute.value
            )
            return decode_choice_value(self, encoded)
        if self.token is tuple:
            if attribute.kind not in {"ints", "floats", "strings"}:
                raise DecodeError("a tuple Decision needs a native list")
            return decode_choice_value(self, list(attribute.value))
        expected = "i" if self.token in (int, bool) else "f" if self.token is float else "s"
        if attribute.kind != expected:
            raise DecodeError(f"expected native {expected}, got {attribute.kind}")
        if self.token is bool:
            return decode_choice_value(self, _boolean(attribute.value))
        return decode_choice_value(self, attribute.value)


def encode_choice_value(choice: ChoiceAttribute, value: object) -> CanonicalValue:
    """Encode one semantic Decision value with its declaration-owned codec.

    The result is transport-neutral canonical data. Native ONNX attributes and
    selected declarations add only their own container representation around
    this value, so both persistence paths share nominal checks and codecs.
    """

    semantics = choice.choice.reference.semantics
    try:
        frozen = semantics.freeze(value)
    except TypeError as error:
        raise DecodeError(str(error)) from error
    if choice.codec is not None:
        encoded = check_canonical(
            choice.codec.encode(frozen),
            f"{choice.choice.path} codec {choice.codec.identity}@{choice.codec.version}",
        )
    elif choice.choice.selector or choice.token is str:
        encoded = _exact(frozen, str, "string Decision")
    elif choice.token is bool:
        encoded = _exact(frozen, bool, "boolean Decision")
    elif choice.token is int:
        encoded = _exact(frozen, int, "integer Decision")
    elif choice.token is float:
        encoded = _exact(frozen, float, "float Decision")
        if not math.isfinite(encoded):
            raise DecodeError("ordinary float Decisions must be finite")
    elif isinstance(choice.token, type) and issubclass(choice.token, Enum):
        member = _exact(frozen, choice.token, "Enum Decision")
        encoded = str(member.value)
    else:
        values = _exact(frozen, tuple, "tuple Decision")
        item_type = type(values[0]) if values else int
        if item_type not in (int, float, str) or any(
            type(item) is not item_type for item in values
        ):
            raise DecodeError("a tuple Decision needs a homogeneous native list or AttributeCodec")
        encoded = check_canonical(values, f"{choice.choice.path} tuple Decision")
    _validate_choice_encoding(choice, encoded)
    try:
        decoded = decode_choice_value(choice, encoded)
    except (TypeError, ValueError) as error:
        raise DecodeError(f"choice codec does not decode its encoded value: {error}") from error
    if not semantics.values_equal(decoded, frozen) or type(decoded) is not type(frozen):
        raise DecodeError("choice codec does not round-trip the Decision value")
    return encoded


def decode_choice_value(choice: ChoiceAttribute, encoded: object) -> object:
    """Decode canonical choice data and enforce the compiled nominal semantics."""

    _validate_choice_encoding(choice, encoded)
    if choice.codec is not None:
        value = choice.codec.decode(check_canonical(encoded, choice.choice.path))
    elif choice.choice.selector or choice.token is str:
        value = _exact(encoded, str, "string Decision")
    elif choice.token is bool:
        value = _exact(encoded, bool, "boolean Decision")
    elif choice.token is int:
        value = _exact(encoded, int, "integer Decision")
    elif choice.token is float:
        value = _exact(encoded, float, "float Decision")
        if not math.isfinite(value):
            raise DecodeError("ordinary float Decisions must be finite")
    elif isinstance(choice.token, type) and issubclass(choice.token, Enum):
        raw = _exact(encoded, str, "Enum Decision")
        matches = [item for item in choice.token if str(item.value) == raw]
        if len(matches) != 1:
            raise DecodeError(f"unknown {choice.token.__name__} member {raw!r}")
        value = matches[0]
    else:
        if not isinstance(encoded, (tuple, list)):
            raise DecodeError("a tuple Decision needs a list")
        value = tuple(encoded)
    try:
        return choice.choice.reference.semantics.freeze(value)
    except TypeError as error:
        raise DecodeError(str(error)) from error


def _validate_choice_encoding(choice: ChoiceAttribute, encoded: object) -> None:
    """Require canonical data representable by the declaration's native codec."""

    if choice.codec is not None:
        _native(choice.codec.kind, encoded)
        return
    if choice.choice.selector or choice.token is str:
        _native("s", encoded)
        return
    if choice.token is bool:
        _exact(encoded, bool, "boolean Decision")
        _native("i", int(cast(bool, encoded)))
        return
    if choice.token is int:
        _native("i", encoded)
        return
    if choice.token is float:
        _native("f", encoded)
        return
    if isinstance(choice.token, type) and issubclass(choice.token, Enum):
        _native("s", encoded)
        return
    if not isinstance(encoded, (tuple, list)):
        raise DecodeError("a tuple Decision needs a list")
    item_type = type(encoded[0]) if encoded else int
    kinds = {int: "ints", float: "floats", str: "strings"}
    if item_type not in kinds or any(type(item) is not item_type for item in encoded):
        raise DecodeError("a tuple Decision needs a homogeneous native list or AttributeCodec")
    _native(kinds[item_type], encoded)


def choice_subset(
    schema: tuple[ChoiceAttribute, ...], paths: tuple[str, ...]
) -> tuple[ChoiceAttribute, ...]:
    """Resolve an ordered exact subset from one compiled choice schema."""

    if len(paths) != len(set(paths)):
        raise AuthoringError("selected choice_paths must be unique")
    by_path = {item.choice.path: item for item in schema}
    missing = tuple(path for path in paths if path not in by_path)
    if missing:
        raise AuthoringError(
            f"selected choice_paths are absent from the compiled schema: {missing}"
        )
    return tuple(by_path[path] for path in paths)


def resolve_choice_subset(
    operation: Any, schema: tuple[ChoiceAttribute, ...]
) -> tuple[tuple[ChoiceAttribute, Answer[object]], ...]:
    """Read the actual compiled selector/Decision answers for a live occurrence."""

    return tuple((item, occurrence_answer_at(operation, item.choice.reference)) for item in schema)


def choice_schema(operation: Any) -> tuple[ChoiceAttribute, ...]:
    return _choice_schema(type(operation), occurrence_persistable(operation))


def compiled_choice_schema(
    operation_type: type[DataflowOp], compiled: object
) -> tuple[ChoiceAttribute, ...]:
    """Operation specialization hook over the existing compiled Space walk."""

    from finn.dataflow.space.compiler import _CompiledSpace  # noqa: PLC0415
    from finn.dataflow.space.occurrence import _collect_persistable  # noqa: PLC0415

    if not isinstance(compiled, _CompiledSpace):
        raise AuthoringError("native choice attributes require a compiled Space")
    selectors: list[PersistableChoice] = []
    decisions: list[PersistableChoice] = []
    _collect_persistable(compiled, f"{compiled.namespace}.", selectors, decisions)
    return _choice_schema(operation_type, (*selectors, *decisions))


def operation_choice_schema(operation_type: type[DataflowOp]) -> tuple[ChoiceAttribute, ...]:
    """Compile one operation class into a detached, explicitly held schema."""

    from finn.dataflow.space.compiler import compile_space_model  # noqa: PLC0415

    namespace = operation_type.root_namespace
    model = compile_space_model(
        operation_type,
        namespace,
        problem_namespace=f"problem.{namespace}",
    )
    return compiled_choice_schema(operation_type, model._compiled_tree())


def _choice_schema(
    operation_type: type[DataflowOp], choices: tuple[PersistableChoice, ...]
) -> tuple[ChoiceAttribute, ...]:
    from finn.dataflow.ops.schema import Attribute, DatatypeAttribute, attribute_name as source_name  # noqa: PLC0415
    from finn.dataflow.ops.base import source_declarations  # noqa: PLC0415

    names = {name: "metadata" for name in RESERVED_ATTRIBUTES}
    names.update(
        {
            source_name(name, decl): "source"
            for name, decl in source_declarations(operation_type)
            if isinstance(decl, (Attribute, DatatypeAttribute))
        }
    )
    result = []
    for choice in choices:
        name = attribute_name(choice.path)
        if name in names:
            raise AuthoringError(
                f"native attribute collision: {choice.path!r} and {names[name]!r} "
                f"encode as {name!r}"
            )
        names[name] = choice.path
        codec = choice.codec
        token = choice.reference.semantics.type_token
        if codec is not None and not isinstance(codec, AttributeCodec):
            raise AuthoringError(
                f"{choice.path} requires canonical=AttributeCodec with a native kind"
            )
        is_enum = isinstance(token, type) and issubclass(token, Enum)
        if codec is None and isinstance(token, type) and issubclass(token, Enum):
            if any(type(item.value) not in (int, str) for item in token):
                raise AuthoringError(
                    f"{choice.path}: Enum values must be stable strings or integers"
                )
            if len({str(item.value) for item in token}) != len(token):
                raise AuthoringError(f"{choice.path}: Enum values collide as strings")
        if (
            codec is None
            and not choice.selector
            and token not in (bool, int, float, str, tuple)
            and not is_enum
        ):
            raise AuthoringError(
                f"{choice.path} requires canonical=AttributeCodec; no native encoding for {token}"
            )
        result.append(ChoiceAttribute(choice, name, codec, token))
    return tuple(result)


def read_attributes(node: Any) -> dict[str, NativeAttribute]:
    """Inspect ordinary ONNX attributes without a state-document parser."""

    from onnx import AttributeProto, helper  # noqa: PLC0415

    kinds = {
        AttributeProto.INT: "i",
        AttributeProto.FLOAT: "f",
        AttributeProto.STRING: "s",
        AttributeProto.INTS: "ints",
        AttributeProto.FLOATS: "floats",
        AttributeProto.STRINGS: "strings",
    }
    result = {}
    seen: set[str] = set()
    for item in node.attribute:
        if item.name in seen:
            raise DecodeError(f"duplicate node attribute {item.name!r}")
        seen.add(item.name)
        kind = kinds.get(item.type)
        if kind is None:
            # Other ONNX source attributes remain outside this choice schema.
            continue
        value = helper.get_attribute_value(item)
        if kind == "s":
            value = value.decode("utf-8")
        elif kind == "strings":
            value = tuple(part.decode("utf-8") for part in value)
        elif kind in {"ints", "floats"}:
            value = tuple(value)
        result[item.name] = NativeAttribute(kind, value)
    return result


def serialize_choices(operation: Any) -> dict[str, NativeAttribute]:
    return serialize_choice_values(capture_decided_choices(operation))


def capture_decided_choices(
    operation: Any,
) -> tuple[tuple[ChoiceAttribute, object], ...]:
    """Capture every currently reachable decided choice exactly once."""

    return tuple(
        (item, answer.value)
        for item in choice_schema(operation)
        if isinstance(answer := occurrence_answer_at(operation, item.choice.reference), Decided)
    )


def serialize_choice_values(
    values: tuple[tuple[ChoiceAttribute, object], ...],
) -> dict[str, NativeAttribute]:
    """Encode an already captured choice set without rereading an occurrence."""

    result = {}
    for item, value in values:
        try:
            encoded = item.encode(value)
            decoded = item.decode(encoded)
            if decoded != value or type(decoded) is not type(value):
                raise DecodeError("native codec does not round-trip the Decision value")
            result[item.name] = encoded
        except (ValueError, TypeError) as error:
            raise AuthoringError(f"cannot encode {item.choice.path}: {error}") from error
    return result


def captured_choice_mapping(
    values: tuple[tuple[ChoiceAttribute, object], ...],
) -> dict[str, object]:
    """Return one path-keyed semantic view of an already captured choice set."""

    result: dict[str, object] = {}
    for item, value in values:
        path = item.choice.path
        if path in result:
            raise AuthoringError(f"compiled choice path {path!r} is duplicated")
        result[path] = value
    return result


def hydrate(operation: Any) -> Any:
    from finn.dataflow.ops.base import DataflowOpError  # noqa: PLC0415

    try:
        written = read_attributes(operation.onnx_node)
        schema = choice_schema(operation)
        present_names = {item.name for item in operation.onnx_node.attribute}
        if "dataflow_state" in present_names:
            raise DecodeError(
                "legacy JSON dataflow_state is unsupported; reconstruct native state explicitly"
            )
        metadata = RESERVED_ATTRIBUTES - {SCOPE_ID_ATTRIBUTE}
        has_choices = any(item.name in present_names for item in schema)
        if has_choices or metadata & present_names:
            version = written.get(SCHEMA_VERSION_ATTRIBUTE)
            if version != NativeAttribute("i", operation.schema_version):
                raise DecodeError(
                    f"this build writes schema version {operation.schema_version}, found {version}"
                )
            fingerprint = written.get(FINGERPRINT_ATTRIBUTE)
            if fingerprint != NativeAttribute("s", operation.local_problem_fingerprint):
                raise DecodeError(
                    "node stores choices made against a different problem; reconstruct explicitly"
                )
        values: dict[QualifiedPath, object] = {}
        for item in schema:
            if item.name not in present_names:
                continue
            if item.name not in written:
                raise DecodeError(f"unsupported ONNX kind for {item.name}")
            try:
                values[item.choice.reference.path] = item.decode(written[item.name])
            except (ValueError, TypeError) as error:
                raise DecodeError(f"cannot decode {item.name}: {error}") from error
        # The declaration walk is an inventory, not an assignment order. The
        # engine orders this compatible batch by its dependency graph, including
        # selectors, applicability and domains that read child exports.
        try:
            operation = occurrence_commit_paths(operation, values)
        except RequestError as error:
            detail = "; ".join(
                f"{finding.path}: {finding.message} [{finding.code}]" for finding in error.findings
            )
            raise DecodeError(f"cannot replay recorded Decisions: {detail}") from error
        except (ValueError, TypeError) as error:
            raise DecodeError(f"cannot replay recorded Decisions: {error}") from error
        # Catch attributes on unreachable branches even if the engine accepted
        # an inapplicable assignment provisionally while a selector was open.
        reachable = serialize_choices(operation)
        if any(item.name in present_names and item.name not in reachable for item in schema):
            raise DecodeError("recorded Decision is not reachable in the selected point")
        return operation
    except (DecodeError, UnicodeError) as error:
        raise DataflowOpError(f"{operation.onnx_node.name}: {error}") from error


__all__ = [
    "AttributeCodec",
    "DecodeError",
    "NativeAttribute",
    "attribute_name",
    "capture_decided_choices",
    "captured_choice_mapping",
    "choice_subset",
    "choice_schema",
    "decode_choice_value",
    "encode_choice_value",
    "operation_choice_schema",
    "read_attributes",
    "resolve_choice_subset",
    "serialize_choice_values",
]
