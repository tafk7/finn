# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Descriptor-only logical map queries with explicit native/export capabilities.

No handle or context crosses this interface. A certified descriptor can be used
by the existing route/factoring layer; a symbolic result without export cannot
be reported as a realizable hardware route. Finite word routes qualify one
source/frame/pass/set, preserving repeated first-word association inside that
scope. They do not infer selector values or cross-pass correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

from finn.core.reporting.findings import Refusal
from finn.dataflow_next._isl import _Service, _Unsupported
from finn.dataflow_next.logical import Available, Rejected, Result, Unsupported, _reject
from finn.dataflow_next.maps import PositionMap
from finn.dataflow_next.traversal import BeatSequence, Grouped, Selected, Traversal


@dataclass(frozen=True)
class MapExport:
    descriptor: PositionMap
    source_ids: tuple[str, ...]
    method: Literal["certified-descriptor", "bounded-legacy-extraction"]
    enumerated_positions: int

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, PositionMap):
            raise TypeError("a map export carries a PositionMap")
        ids = tuple(self.source_ids)
        if len(ids) != len(self.descriptor.sources) or not all(
            isinstance(name, str) and name for name in ids
        ):
            raise ValueError("map exports retain the ordered source-slot IDs")
        object.__setattr__(self, "source_ids", ids)
        if (
            type(self.enumerated_positions) is not int
            or (self.method == "certified-descriptor" and self.enumerated_positions != 0)
            or (
                self.method == "bounded-legacy-extraction"
                and not (self.enumerated_positions == prod(self.descriptor.shape) <= 65536)
            )
            or self.method not in ("certified-descriptor", "bounded-legacy-extraction")
        ):
            raise ValueError("export provenance must match its exact certified or bounded path")


@dataclass(frozen=True)
class WordScope:
    source: str
    frame: str
    pass_id: str
    set_id: str | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(v, str) and v for v in (self.source, self.frame, self.pass_id)):
            raise ValueError("source/frame/pass IDs are nonempty strings")
        if self.set_id is not None and (not isinstance(self.set_id, str) or not self.set_id):
            raise ValueError("a set ID is absent or a nonempty string")


@dataclass(frozen=True)
class BeatRoute:
    source: BeatSequence | Selected
    sink: BeatSequence | Selected
    scope: WordScope

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source, (BeatSequence, Selected))
            or not isinstance(self.sink, (BeatSequence, Selected))
            or not isinstance(self.scope, WordScope)
        ):
            raise TypeError("a beat route retains sequences and an explicit WordScope")


@dataclass(frozen=True)
class SourceWordBound:
    value: int
    scope: WordScope
    unit: Literal["source-words"] = "source-words"
    horizon: Literal["finite-pass-qualified-sequence"] = "finite-pass-qualified-sequence"

    def __post_init__(self) -> None:
        if type(self.value) is not int or self.value < 0 or not isinstance(self.scope, WordScope):
            raise ValueError("a word bound has a nonnegative integer and a WordScope")
        if self.unit != "source-words" or self.horizon != "finite-pass-qualified-sequence":
            raise ValueError("a word bound is finite order storage in source words")


def equal(
    left: PositionMap,
    right: PositionMap,
    *,
    left_ids: tuple[str, ...] | None = None,
    right_ids: tuple[str, ...] | None = None,
) -> Result[bool]:
    """Native semantic equality with reader/source metadata and explicit source identity."""
    service = _Service()
    try:
        return Available(
            service.equal(
                service.import_descriptor(left, left_ids),
                service.import_descriptor(right, right_ids),
            ),
            "native-equality",
        )
    except Refusal as error:
        return Rejected(tuple(error.findings), str(error))


def compose(
    outer: PositionMap,
    inner: PositionMap,
    source: int = 0,
    *,
    outer_ids: tuple[str, ...] | None = None,
    inner_ids: tuple[str, ...] | None = None,
    certificate: PositionMap | None = None,
    extraction_budget: int = 0,
) -> Result[MapExport]:
    """Native selected-source substitution plus certified or explicitly bounded export."""
    service = _Service()
    outer_ids = (
        outer_ids
        if outer_ids is not None
        else tuple(f"outer:{i}" for i in range(len(outer.sources)))
    )
    inner_ids = (
        inner_ids
        if inner_ids is not None
        else tuple(f"inner:{i}" for i in range(len(inner.sources)))
    )
    try:
        result = service.substitute(
            service.import_descriptor(outer, outer_ids),
            service.import_descriptor(inner, inner_ids),
            source,
        )
        exported = service.export_descriptor(result, certificate, budget=extraction_budget)
        return Available(
            MapExport(
                exported.descriptor,
                result.source_ids,
                exported.method,
                exported.enumerated_positions,
            ),
            exported.method,
        )
    except Refusal as error:
        return Rejected(tuple(error.findings), str(error))
    except _Unsupported as error:
        return Unsupported("descriptor-export", "compose", str(error))


def through(pmap: PositionMap, form: Traversal | Grouped) -> Result[Traversal | Grouped]:
    """Retained loop derivation with native final delivery equality; no offset arrays."""
    service = _Service()
    try:
        return Available(
            service.through(service.import_descriptor(pmap), form), "native-through-verification"
        )
    except Refusal as error:
        return Rejected(tuple(error.findings), str(error))
    except _Unsupported as error:
        return Unsupported("descriptor-export", "through", str(error))


def _forms(route: BeatRoute) -> tuple[Traversal | Grouped, Traversal | Grouped]:
    source, sink = route.source, route.sink
    if isinstance(source, Selected) or isinstance(sink, Selected):
        if (
            not isinstance(source, Selected)
            or not isinstance(sink, Selected)
            or not source.shares(sink)
        ):
            raise ValueError("selected routes require the same sets and selector stream")
        if route.scope.set_id is None:
            raise _Unsupported(
                "name the whole selected set's scope; selector values are not inferred"
            )
        source, sink = source.sequence, sink.sequence
    elif route.scope.set_id is not None:
        raise ValueError("unselected sequences have no selected-set scope")
    assert isinstance(source, BeatSequence) and isinstance(sink, BeatSequence)
    # This is a finite source sequence, not automatic cyclic association. Its
    # descriptor retains CYCLIC for the later adapter/executor to interpret.
    return source.form, sink.form


def beat_route(
    source: BeatSequence | Selected,
    sink: BeatSequence | Selected,
    source_scope: WordScope,
    sink_scope: WordScope,
) -> Result[BeatRoute]:
    if source_scope != sink_scope:
        return _reject("occurrence-mismatch", "beat_route", "source/frame/pass/set scopes differ")
    route = BeatRoute(source, sink, source_scope)
    service = _Service()
    try:
        first, last = _forms(route)
        service.beat_route(
            first,
            last,
            source_id=source_scope.source,
            frame_id=source_scope.frame,
            pass_id=source_scope.pass_id,
            set_id=source_scope.set_id,
        )
        return Available(route, "native-whole-word-association")
    except Refusal as error:
        return Rejected(tuple(error.findings), str(error))
    except _Unsupported as error:
        return Unsupported("word-scope", "beat_route", str(error))
    except ValueError as error:
        return _reject("selection-not-shared", "beat_route", str(error))


def word_at(route: BeatRoute, beat: int) -> Result[int]:
    """One requested source word; explicit query, never an implicit full extraction."""
    service = _Service()
    if type(beat) is not int or beat < 0:
        return _reject("beat-invalid", "word_at", "beat indices are nonnegative integers")
    try:
        first, last = _forms(route)
        native = service.beat_route(
            first,
            last,
            source_id=route.scope.source,
            frame_id=route.scope.frame,
            pass_id=route.scope.pass_id,
            set_id=route.scope.set_id,
        )
        return Available(service.word_at(native, beat), "native-word-query")
    except Refusal as error:
        return Rejected(tuple(error.findings), str(error))
    except _Unsupported as error:
        return Unsupported("word-scope", "word_at", str(error))
    except ValueError as error:
        return _reject("word-query-invalid", "word_at", str(error))


def order_bound(route: BeatRoute) -> Result[SourceWordBound]:
    service = _Service()
    try:
        first, last = _forms(route)
        native = service.beat_route(
            first,
            last,
            source_id=route.scope.source,
            frame_id=route.scope.frame,
            pass_id=route.scope.pass_id,
            set_id=route.scope.set_id,
        )
        return Available(
            SourceWordBound(service.order_bound(native), route.scope), "native-integer-maximum"
        )
    except Refusal as error:
        return Rejected(tuple(error.findings), str(error))
    except _Unsupported as error:
        return Unsupported("word-order", "order_bound", str(error))
    except ValueError as error:
        return _reject("selection-not-shared", "order_bound", str(error))
