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


def _encode_structural(value: object) -> CanonicalValue:
    if value is None or type(value) in (bool, int, str):
        return value  # type: ignore[return-value]
    if type(value) is float:
        from math import isfinite  # noqa: PLC0415

        if not isfinite(value):
            raise AuthoringError("a persisted float must be finite")
        return {"float_hex": value.hex()}
    raise AuthoringError(
        f"a Decision value of type {type(value).__name__} has no canonical encoding; "
        "give its Decision a canonical=DecisionCodec(...)"
    )


def _decode_structural(value: CanonicalValue) -> object:
    """Read back what ``_encode_structural`` wrote, and nothing else.

    A float and an Enum are tagged on the way out precisely so they can be
    recognized on the way in; an untagged mapping is not something this codec
    produced, so it is refused rather than handed back as a dict.
    """

    if isinstance(value, dict):
        if set(value) == {"float_hex"}:
            return float.fromhex(str(value["float_hex"]))
        raise DecodeError(f"the structural codec did not write {value!r}")
    return value


#: The structural default: the value kinds whose encoding is not a judgement
#: call.  Anything else -- a tile shape, a scheduling record, a policy object --
#: declares its own, because only its author knows what its identity is.
STRUCTURAL_DECISION_CODEC: PersistentCodec[Any] = DecisionCodec(
    "dataflow.structural", 1, _encode_structural, _decode_structural
)

#: A Variant selector's alternative id.  Given its own codec even though the
#: values are stable strings today, so that a future change to how a selection
#: is written is a version bump rather than a silent reinterpretation.
SELECTOR_CODEC: PersistentCodec[Any] = DecisionCodec("dataflow.selector", 1, str, str)


def enum_codec(enum_type: type[Enum]) -> PersistentCodec[Any]:
    """The structural default for an Enum-valued Decision.

    Its *member value*, never its ordinal: reordering the members of an enum is
    an ordinary refactor and must not silently repoint every saved graph at a
    different alternative.  Decoding goes back through the enum type, so a value
    the enum no longer has is a refusal rather than a bare string that fails the
    Decision's domain check much later.
    """

    return DecisionCodec(
        f"dataflow.enum.{enum_type.__module__}.{enum_type.__qualname__}",
        1,
        lambda value: cast(CanonicalValue, enum_type(value).value),
        lambda value: enum_type(value),
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
        if not isinstance(entry["codec"], str) or "@" not in entry["codec"]:
            raise DecodeError(f"the assignment at {path!r} has no codec identity@version tag")
        assignments[path] = Assignment(entry["codec"], entry["value"])
    for name in ("family", "family_version", "problem_fingerprint"):
        if not isinstance(payload[name], str):
            raise DecodeError(f"a dataflow state document's {name} is a string")
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
    "ENVELOPE_MEMBERS",
    "KNOWN_STAGES",
    "SELECTOR_CODEC",
    "Assignment",
    "STATE_ATTRIBUTE",
    "STATE_SCHEMA",
    "STRUCTURAL_DECISION_CODEC",
    "DataflowState",
    "DecisionCodec",
    "DecodeError",
    "decode_dataflow_state",
    "decode_state",
    "enum_codec",
    "encode_state",
    "format_dataflow_state",
]
