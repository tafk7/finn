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
        "design.$selection": "dot_product",
        "design.dot_product.pe": 2
      }
    }

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
from typing import Any

from finn.dataflow.model.declarations import AuthoringError, CanonicalValue

#: The one attribute the design state lives in.
STATE_ATTRIBUTE = "dataflow_state"

#: The envelope's own version, independent of any operation family version and
#: of any individual value codec version.  Three versions, because they change
#: for three unrelated reasons.
STATE_SCHEMA = "finn.dataflow.state/1"


class DecodeError(ValueError):
    """A node's design state cannot be read as this layer's document."""


@dataclass(frozen=True, slots=True)
class DecisionCodec:
    """How one Decision's values cross the persistence boundary.

    Declaration-owned and versioned, exactly as ``CanonicalValueCodec`` already
    is for a ``Problem``.  There is no ``repr()`` fallback: a value that cannot
    be written and read back to something the Decision's domain accepts must
    fail where the Decision is *written*, not where it is next loaded, and a
    repr is precisely the encoding that looks like it worked.
    """

    identity: str
    version: int
    encode: Any
    decode: Any

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity:
            raise AuthoringError("a DecisionCodec identity is a non-empty stable string")
        if type(self.version) is not int or self.version < 1:
            raise AuthoringError(f"DecisionCodec {self.identity!r} needs a positive version")

    @property
    def tag(self) -> str:
        return f"{self.identity}@{self.version}"


def _encode_structural(value: object) -> CanonicalValue:
    if value is None or type(value) in (bool, int, str):
        return value  # type: ignore[return-value]
    if type(value) is float:
        from math import isfinite  # noqa: PLC0415

        if not isfinite(value):
            raise AuthoringError("a persisted float must be finite")
        return {"float_hex": value.hex()}
    if isinstance(value, Enum):
        return {"enum": str(value.value)}
    raise AuthoringError(
        f"a Decision value of type {type(value).__name__} has no canonical encoding; "
        "give its Decision a canonical=DecisionCodec(...)"
    )


#: The structural default: the value kinds whose encoding is not a judgement
#: call.  Anything else -- a tile shape, a scheduling record, a policy object --
#: declares its own, because only its author knows what its identity is.
STRUCTURAL_DECISION_CODEC = DecisionCodec(
    "dataflow.structural", 1, _encode_structural, lambda value: value
)

#: A Variant selector's alternative id.  Given its own codec even though the
#: values are stable strings today, so that a future change to how a selection
#: is written is a version bump rather than a silent reinterpretation.
SELECTOR_CODEC = DecisionCodec("dataflow.selector", 1, str, str)


@dataclass(frozen=True, slots=True)
class DataflowState:
    """The decoded design state of one node."""

    family: str
    family_version: str
    problem_fingerprint: str
    commitment_stage: str
    assignments: Mapping[str, object]


def encode_state(
    *,
    family: str,
    family_version: str,
    problem_fingerprint: str,
    commitment_stage: str,
    assignments: Mapping[str, object],
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
        "assignments": dict(assignments),
    }
    return dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def decode_state(raw: str | bytes) -> DataflowState:
    """Read one design state back, refusing anything this layer did not write."""

    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    if not text:
        raise DecodeError("this node carries no dataflow state")
    try:
        payload = loads(text)
    except ValueError as error:
        raise DecodeError(f"this node's dataflow state is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise DecodeError("a dataflow state document is a JSON object")
    schema = payload.get("schema")
    if schema != STATE_SCHEMA:
        raise DecodeError(
            f"this node's dataflow state is {schema!r}; this build writes {STATE_SCHEMA!r}. "
            "A schema this build does not know is not reinterpreted"
        )
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict):
        raise DecodeError("a dataflow state document's assignments are a JSON object")
    return DataflowState(
        str(payload.get("family", "")),
        str(payload.get("family_version", "")),
        str(payload.get("problem_fingerprint", "")),
        str(payload.get("commitment_stage", "")),
        MappingProxyType(dict(assignments)),
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
        lines.append(f"    {path} = {state.assignments[path]!r}")
    return "\n".join(lines)


__all__ = [
    "SELECTOR_CODEC",
    "STATE_ATTRIBUTE",
    "STATE_SCHEMA",
    "STRUCTURAL_DECISION_CODEC",
    "DataflowState",
    "DecisionCodec",
    "DecodeError",
    "decode_dataflow_state",
    "decode_state",
    "encode_state",
    "format_dataflow_state",
]
