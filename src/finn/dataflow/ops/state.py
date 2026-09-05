# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One canonical document holding everything a dataflow node's design state is.

The alternative was one ONNX attribute per Decision -- ``PE=2``, ``SIMD=4``,
``dataflow_design="dot_product"`` -- which is legible in Netron and wrong in
three ways.  It cannot express a Decision nested inside a helper Space or a
selector that has no flat name.  It has no ordering, so two equal points can
serialize to different bytes.  And it *overlays*: a write leaves behind every
attribute the new point does not mention, so a node that once chose ``supplied``
keeps its ``weight_supply`` after switching to ``dot_product``, and the next
reload tries to replay a choice with nowhere to go.

So the design state is one document, written whole::

    {
      "schema": "finn.dataflow.state/1",
      "family": "finn.dataflow.mvau",
      "family_version": "1",
      "problem_fingerprint": "...",
      "commitment_stage": "dataflow",
      "assignments": {
        "design.case": {"codec": "dataflow.selector@1", "value": "dot_product"},
        "design.dot_product.pe": {"codec": "dataflow.int@1", "value": 2}
      }
    }

The keys are **root-relative compiled paths**, not friendly operation-local
aliases.  ``"PE"`` cannot name a Decision nested three Subspaces down, cannot
distinguish two Decisions called ``tile`` in different subspaces, and requires
somebody to have remembered to list it.  A compiled path is generated, unique
by construction, and discovered from the model -- so a Decision added to a
Design is persisted without anyone touching the operation.

Every value carries the identity and version of the codec that wrote it, so a
changed encoding is a refusal on the next load rather than old bytes silently
reinterpreted under new rules.

**What stays outside it, and why.**  ``dataflow_scope_id`` is separate because
it is how the node is *found* before its state can be parsed or replaced.  The
operation's source-semantic attributes -- ``narrow_weights``, ``accDataType``,
``noActivation`` -- also stay separate, and they are not a duplicate authority:
they define the mathematical operation, while everything in the document
describes the implementation selected for it.  Family, version and fingerprint
move *into* the document, so those facts stop having two persisted homes.

**Readable, not opaque.**  Canonical UTF-8 JSON with sorted keys, so a human can
read it in Netron, a diff is meaningful, and two equal points produce equal
bytes.  There is no second flat copy of any value: "write-only compatibility"
is not compatibility, it is a duplicate representation that a user can edit and
reasonably expect to take effect.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from json import dumps, loads
from types import MappingProxyType
from typing import Any, cast

from finn.dataflow.model.declarations import (
    AuthoringError,
    CanonicalValue,
    PersistentCodec,
)

#: The one attribute the design state lives in.
STATE_ATTRIBUTE = "dataflow_state"

#: The envelope's own version, independent of any operation family version and
#: of any individual value codec version.  Three versions, because they change
#: for three unrelated reasons.
STATE_SCHEMA = "finn.dataflow.state/1"


class DecodeError(ValueError):
    """A node's design state cannot be read as this layer's document."""


#: The layer's name for a Decision's persistence codec.  It *is*
#: :class:`~finn.dataflow.model.declarations.PersistentCodec`, because a
#: Decision declares it and a Decision lives in ``model/`` -- two types would
#: mean a declaration could carry one the persistence layer did not accept.
DecisionCodec = PersistentCodec


#: The JSON kinds a structural value may take, spelled as exact Python types.
#: ``type(value) is`` throughout, never ``isinstance``: ``True`` is an instance
#: of ``int``, and a codec that accepted it would decode ``{"value": true}``
#: into ``1`` and call the round trip successful.
_JSON_KIND = {bool: "a JSON bool", int: "a JSON integer", str: "a JSON string"}


def _exact(kind: type, identity: str) -> PersistentCodec[Any]:
    """One codec that accepts exactly one JSON kind, in both directions.

    Symmetry is the whole point.  A decoder looser than its encoder means two
    different documents load as the same design, so a document a human edited
    -- or an older writer produced under different rules -- is reinterpreted
    instead of refused, and the difference surfaces as a build that does not
    match what anybody saved.
    """

    what = _JSON_KIND[kind]

    def encode(value: object) -> CanonicalValue:
        if type(value) is not kind:
            raise AuthoringError(
                f"the {identity} codec writes {kind.__name__}, not {type(value).__name__}"
            )
        return cast(CanonicalValue, value)

    def decode(value: CanonicalValue) -> object:
        if type(value) is not kind:
            raise DecodeError(f"the {identity} codec wrote {what}, not {value!r}")
        return value

    return DecisionCodec(identity, 1, encode, decode)


BOOL_CODEC: PersistentCodec[Any] = _exact(bool, "dataflow.bool")
INT_CODEC: PersistentCodec[Any] = _exact(int, "dataflow.int")
STRING_CODEC: PersistentCodec[Any] = _exact(str, "dataflow.string")


def _encode_float(value: object) -> CanonicalValue:
    from math import isfinite  # noqa: PLC0415

    if type(value) is not float:
        raise AuthoringError(f"the dataflow.float codec writes float, not {type(value).__name__}")
    if not isfinite(value):
        raise AuthoringError("a persisted float must be finite")
    return {"float_hex": value.hex()}


def _decode_float(value: CanonicalValue) -> object:
    """Exactly the tagged form, because a bare JSON number is not exact.

    ``1.1`` written as a decimal and read back is a different double on some
    round trips, and ``2.0`` is indistinguishable from the integer ``2``.  The
    hexadecimal form is both exact and unambiguous, so it is the only form this
    codec accepts.
    """

    if not isinstance(value, dict) or set(value) != {"float_hex"}:
        raise DecodeError(f"the dataflow.float codec wrote {{'float_hex': ...}}, not {value!r}")
    text = value["float_hex"]
    if not isinstance(text, str):
        raise DecodeError("a dataflow.float value is a hexadecimal string")
    return float.fromhex(text)


FLOAT_CODEC: PersistentCodec[Any] = DecisionCodec("dataflow.float", 1, _encode_float, _decode_float)

#: The value kinds whose encoding is not a judgement call.  Anything else -- a
#: tile shape, a scheduling record, a policy object -- declares its own codec,
#: because only its author knows what its identity is.
STRUCTURAL_CODECS: Mapping[type, PersistentCodec[Any]] = MappingProxyType(
    {bool: BOOL_CODEC, int: INT_CODEC, str: STRING_CODEC, float: FLOAT_CODEC}
)


def structural_codec(token: object) -> PersistentCodec[Any] | None:
    """The codec for one structural type token, or ``None`` if it has none."""

    if isinstance(token, type):
        return STRUCTURAL_CODECS.get(token)
    return None


#: A SubspaceChoice selector's alternative id.  Given its own codec even though the
#: values are stable strings today, so that a future change to how a selection
#: is written is a version bump rather than a silent reinterpretation.
SELECTOR_CODEC: PersistentCodec[Any] = _exact(str, "dataflow.selector")


def enum_codec(enum_type: type[Enum]) -> PersistentCodec[Any]:
    """The structural default for an Enum-valued Decision.

    Its *member value*, never its ordinal: reordering the members of an enum is
    an ordinary refactor and must not silently repoint every saved graph at a
    different alternative.  Decoding goes back through the enum type, so a value
    the enum no longer has is a refusal rather than a bare string that fails the
    Decision's domain check much later.

    The members' own value type is required to be single and structural, and it
    is checked on the way in as strictly as any other codec checks: an enum over
    strings must not accept ``0`` because ``Enum(0)`` happens to raise the same
    class of error later, and one over ints must not accept ``True``.
    """

    kinds = {type(member.value) for member in enum_type}
    if len(kinds) != 1 or not (kinds <= {bool, int, str}):
        named = sorted(kind.__name__ for kind in kinds)
        raise AuthoringError(
            f"{enum_type.__qualname__} has member values of {named}; an Enum-valued "
            "Decision persists through one structural kind, or declares "
            "canonical=PersistentCodec(...)"
        )
    (kind,) = kinds

    def encode(value: object) -> CanonicalValue:
        return cast(CanonicalValue, enum_type(value).value)

    def decode(value: CanonicalValue) -> object:
        if type(value) is not kind:
            raise DecodeError(
                f"{enum_type.__qualname__} persists as {_JSON_KIND[kind]}, not {value!r}"
            )
        return enum_type(value)

    return DecisionCodec(
        f"dataflow.enum.{enum_type.__module__}.{enum_type.__qualname__}", 1, encode, decode
    )


#: The commitment stages a document may name.  Checked on decode, so a
#: misspelled or future stage is a refusal rather than a value that compares
#: unequal to everything and quietly means "unchecked".
KNOWN_STAGES = frozenset({"dataflow", "physical"})

#: Exactly the members a document has.  Both directions are checked: a missing
#: one is incomplete, and an extra one means the writer knew something this
#: reader does not.
ENVELOPE_MEMBERS = frozenset(
    {"schema", "family", "family_version", "problem_fingerprint", "commitment_stage", "assignments"}
)


@dataclass(frozen=True, slots=True)
class Assignment:
    """One persisted choice: the codec that wrote it, and what it wrote."""

    codec: str
    value: CanonicalValue


def parse_codec_tag(tag: str) -> tuple[str, int]:
    """Split ``identity@version``, refusing anything that is not one.

    Checked on read rather than trusted, because the tag is the whole of the
    reinterpretation guard: an empty identity or a non-numeric version would
    compare unequal to every declared codec and turn a precise "this build
    declares a different encoding" into an unexplained mismatch.
    """

    identity, separator, version = tag.rpartition("@")
    if not separator or not identity:
        raise DecodeError(f"{tag!r} is not a codec identity@version tag")
    if not version.isdigit() or int(version) < 1:
        raise DecodeError(f"the codec tag {tag!r} does not name a positive version")
    return identity, int(version)


@dataclass(frozen=True, slots=True)
class DataflowState:
    """The decoded design state of one node."""

    family: str
    family_version: str
    problem_fingerprint: str
    commitment_stage: str
    assignments: Mapping[str, Assignment]


def encode_state(
    *,
    family: str,
    family_version: str,
    problem_fingerprint: str,
    commitment_stage: str,
    assignments: Mapping[str, Assignment],
) -> str:
    """Serialize one design state canonically.

    ``sort_keys`` and a fixed separator are the whole of "canonical": two equal
    points must produce equal bytes, or a content-addressed build cache and a
    graph diff both become noise.
    """

    payload = {
        "schema": STATE_SCHEMA,
        "family": family,
        "family_version": family_version,
        "problem_fingerprint": problem_fingerprint,
        "commitment_stage": commitment_stage,
        "assignments": {
            path: {"codec": item.codec, "value": item.value} for path, item in assignments.items()
        },
    }
    return dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def decode_state(raw: str | bytes) -> DataflowState:
    """Read one design state back, refusing anything this layer did not write.

    Strict in every direction.  A permissive reader turns a renamed
    declaration, a stale path or a changed codec into a silently dropped
    assignment -- and a silently dropped assignment is a design that reloads as
    something other than what was saved, which is the one failure persistence
    exists to prevent.
    """

    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    if not text:
        raise DecodeError("this node carries no dataflow state")
    try:
        payload = loads(text)
    except ValueError as error:
        raise DecodeError(f"this node's dataflow state is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise DecodeError("a dataflow state document is a JSON object")
    if payload.get("schema") != STATE_SCHEMA:
        raise DecodeError(
            f"this node's dataflow state is {payload.get('schema')!r}; this build writes "
            f"{STATE_SCHEMA!r}.  A schema this build does not know is not reinterpreted"
        )
    missing = sorted(ENVELOPE_MEMBERS - set(payload))
    if missing:
        raise DecodeError(f"this node's dataflow state is missing {missing}")
    extra = sorted(set(payload) - ENVELOPE_MEMBERS)
    if extra:
        raise DecodeError(
            f"this node's dataflow state carries {extra}, which this build does not know; "
            "it was written by a later writer and is not reinterpreted"
        )
    stage = payload["commitment_stage"]
    if stage not in KNOWN_STAGES:
        raise DecodeError(f"this node's dataflow state names commitment stage {stage!r}")
    raw_assignments = payload["assignments"]
    if not isinstance(raw_assignments, dict):
        raise DecodeError("a dataflow state document's assignments are a JSON object")
    assignments: dict[str, Assignment] = {}
    for path, entry in raw_assignments.items():
        if not isinstance(path, str) or not path or path != path.strip():
            raise DecodeError(f"{path!r} is not a canonical assignment path")
        if not isinstance(entry, dict) or set(entry) != {"codec", "value"}:
            raise DecodeError(
                f"the assignment at {path!r} is not a {{codec, value}} pair; every persisted "
                "value carries the identity and version of the codec that wrote it"
            )
        if not isinstance(entry["codec"], str):
            raise DecodeError(f"the assignment at {path!r} has no codec identity@version tag")
        parse_codec_tag(entry["codec"])
        assignments[path] = Assignment(entry["codec"], entry["value"])
    for name in ("family", "family_version", "problem_fingerprint"):
        value = payload[name]
        if not isinstance(value, str) or not value:
            raise DecodeError(f"a dataflow state document's {name} is a non-empty string")
    return DataflowState(
        payload["family"],
        payload["family_version"],
        payload["problem_fingerprint"],
        stage,
        MappingProxyType(assignments),
    )


def decode_dataflow_state(node: Any) -> DataflowState | None:
    """The design state of one live node, or ``None`` if it carries none."""

    for attribute in node.attribute:
        if attribute.name == STATE_ATTRIBUTE:
            return decode_state(attribute.s)
    return None


def format_dataflow_state(node: Any) -> str:
    """One node's design state, rendered for a human.

    The inspection facility that replaces reading ``PE`` off the node.  A
    printer is the right answer to "I want to see the folding"; a second copy
    of every value written into a flat attribute is not, because the moment it
    exists someone can edit it and reasonably expect it to take effect.
    """

    state = decode_dataflow_state(node)
    if state is None:
        return f"{node.name}: no dataflow state"
    lines = [
        f"{node.name}  [{state.family}@{state.family_version}]",
        f"  problem     {state.problem_fingerprint}",
        f"  checked to  {state.commitment_stage}",
        "  assignments:",
    ]
    if not state.assignments:
        lines.append("    (none)")
    for path in sorted(state.assignments):
        item = state.assignments[path]
        lines.append(f"    {path} = {item.value!r}   [{item.codec}]")
    return "\n".join(lines)


__all__ = [
    "BOOL_CODEC",
    "ENVELOPE_MEMBERS",
    "FLOAT_CODEC",
    "INT_CODEC",
    "KNOWN_STAGES",
    "SELECTOR_CODEC",
    "STATE_ATTRIBUTE",
    "STATE_SCHEMA",
    "STRING_CODEC",
    "STRUCTURAL_CODECS",
    "Assignment",
    "DataflowState",
    "DecisionCodec",
    "DecodeError",
    "decode_dataflow_state",
    "decode_state",
    "encode_state",
    "enum_codec",
    "format_dataflow_state",
    "parse_codec_tag",
    "structural_codec",
]
