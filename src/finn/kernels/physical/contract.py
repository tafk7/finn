# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Stream contracts: what one stream end carries, and whether two ends may connect.

A contract joins three levels, each owned elsewhere and checked here together:

- logical: element encoding, lanes, beat form, repetition and marker rules;
- physical: low-field-first packing of the element into the transport word;
- protocol: the ready/valid (or AXIS) pins, marker pins and clock/reset names.

``compatibility`` compares a producing and a consuming end. A logical mismatch
can only be repaired by an adapter kernel, which ``forms.classify`` names
(reorder or replay, width conversion, lane regroup). A pure lane permutation,
padding and reset polarity leave the sequence unchanged; they are properties of
the connection, which ``Composition.connect`` realizes as wires: a producer's
padding bits are left unconnected inside the composition, and a consumer's
padding is driven with zeros.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from finn.core.space import ValueSemantics, default_semantics
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.datatypes.scalar import ScalarEncoding
from finn.kernels.physical.forms import (
    Adaptation,
    Every,
    Repetition,
    Traversal,
    classify,
)
from finn.kernels.physical.stream import ReadyValidStream


class Level(Enum):
    LOGICAL = "logical"
    PHYSICAL = "physical"
    PROTOCOL = "protocol"


@dataclass(frozen=True)
class Mismatch:
    level: Level
    code: str
    message: str


@dataclass(frozen=True)
class StreamContract:
    """One stream end: its transport plus the logical sequence it carries.

    ``markers`` maps a transport marker signal to the rule it follows. For a
    producer these are guarantees; for a consumer, requirements. Transport
    markers without a rule can be neither required nor matched.
    """

    transport: ReadyValidStream
    element: ScalarEncoding
    form: Traversal
    repetition: Repetition = Repetition.ONCE
    markers: Mapping[str, Every] | tuple[tuple[str, Every], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.transport, ReadyValidStream):
            raise TypeError("a stream contract has a ReadyValidStream transport")
        if not isinstance(self.element, ScalarEncoding) or not isinstance(self.form, Traversal):
            raise TypeError("a stream contract has a ScalarEncoding element and a Traversal")
        if not isinstance(self.repetition, Repetition):
            raise TypeError("a stream contract has a Repetition")
        if self.payload_bits > self.transport.data_width:
            raise ValueError(
                f"{self.transport.name}: {self.form.lanes} x {self.element.bits}-bit lanes "
                f"exceed the {self.transport.data_width}-bit data word"
            )
        markers = dict(self.markers)
        widths = {marker.signal: marker.width for marker in self.transport.markers}
        for signal, rule in markers.items():
            if signal not in widths or widths[signal] != 1 or not isinstance(rule, Every):
                raise ValueError(f"{signal}: a rule needs a one-bit transport marker")
        object.__setattr__(self, "markers", tuple(sorted(markers.items())))

    @property
    def lanes(self) -> int:
        return self.form.lanes

    @property
    def payload_bits(self) -> int:
        return self.form.lanes * self.element.bits

    @property
    def rules(self) -> dict[str, Every]:
        return dict(self.markers)


STREAM_CONTRACT: ValueSemantics[StreamContract] = default_semantics(StreamContract)


def compatibility(
    source: StreamContract, sink: StreamContract, *, source_is_top: bool, sink_is_top: bool
) -> tuple[Mismatch, ...]:
    """Every reason ``source`` may not drive ``sink``; empty when they connect.

    ``*_is_top`` marks an end on the composed module's own boundary, whose
    endpoint direction is seen from outside (a top input is a source inside).
    """
    found: list[Mismatch] = []

    def refuse(level: Level, code: str, message: str) -> None:
        found.append(Mismatch(level, code, message))

    produces = Endpoint.TARGET if source_is_top else Endpoint.INITIATOR
    consumes = Endpoint.INITIATOR if sink_is_top else Endpoint.TARGET
    if source.transport.endpoint is not produces or sink.transport.endpoint is not consumes:
        refuse(Level.PROTOCOL, "stream-direction", "the source must produce and the sink consume")

    if source.element != sink.element:
        refuse(
            Level.LOGICAL,
            "stream-element",
            f"{source.element.datatype_name} cannot feed {sink.element.datatype_name}",
        )
    if sink.repetition is Repetition.CYCLIC and source.repetition is not Repetition.CYCLIC:
        refuse(Level.LOGICAL, "stream-repetition", "a single pass cannot feed a cyclic consumer")
    produced = _presented(source, sink)
    if produced is None:
        refuse(
            Level.LOGICAL,
            "stream-form",
            "the consumer's pass is not whole repetitions of the cyclic source",
        )
    else:
        verdict = classify(produced, sink.form)
        if verdict.adaptation not in (Adaptation.IDENTITY, Adaptation.LANE_PERMUTATION):
            detail = f": {verdict.reorder}" if verdict.reorder else ""
            refuse(
                Level.LOGICAL,
                "stream-form",
                f"needs a {verdict.adaptation.value} adapter ({verdict.detail}){detail}",
            )

    offered = source.rules
    for signal, rule in sink.rules.items():
        if rule not in offered.values():
            refuse(
                Level.LOGICAL,
                "stream-marker",
                f"{signal} requires a marker every {rule.period} beats; none is produced",
            )

    if source_is_top:
        consumed = {pair[0] for pair in marker_pairs(source, sink)}
        unused = [m.signal for m in source.transport.markers if m.signal not in consumed]
        if unused:
            refuse(
                Level.PROTOCOL,
                "stream-top-marker",
                f"top input markers {unused} would be left unconsumed",
            )
    if sink_is_top and sink.transport.markers:
        missing = [m.signal for m in sink.transport.markers if m.signal not in sink.rules]
        if missing:
            refuse(Level.PROTOCOL, "stream-top-marker", f"top output markers {missing} lack rules")
    return tuple(found)


def _presented(source: StreamContract, sink: StreamContract) -> Traversal | None:
    """The sequence a source presents over one consumer pass (None if it cannot align)."""
    if source.repetition is Repetition.ONCE or source.form.shape != sink.form.shape:
        return source.form
    if sink.form.beats % source.form.beats:
        return None
    count = sink.form.beats // source.form.beats
    return source.form if count == 1 else source.form.repeated(count)


def lane_permutation(source: StreamContract, sink: StreamContract) -> tuple[int, ...]:
    """Sink field -> source field; the identity unless the lanes are only reordered."""
    produced = _presented(source, sink)
    if produced is not None:
        verdict = classify(produced, sink.form)
        if verdict.adaptation is Adaptation.LANE_PERMUTATION:
            return verdict.lane_permutation
    return tuple(range(sink.lanes))


def marker_pairs(source: StreamContract, sink: StreamContract) -> tuple[tuple[str, str], ...]:
    """(source signal, sink signal) for each required sink marker."""
    offered = source.rules
    pairs: list[tuple[str, str]] = []
    used: set[str] = set()
    for signal, rule in sink.rules.items():
        for candidate, candidate_rule in offered.items():
            if candidate_rule == rule and candidate not in used:
                pairs.append((candidate, signal))
                used.add(candidate)
                break
    return tuple(pairs)


class StreamMismatch(ValueError):
    def __init__(self, source: str, sink: str, mismatches: tuple[Mismatch, ...]) -> None:
        self.mismatches = mismatches
        details = "; ".join(f"{item.code}: {item.message}" for item in mismatches)
        super().__init__(f"{source} -> {sink}: {details}")


__all__ = [
    "Level",
    "Mismatch",
    "STREAM_CONTRACT",
    "StreamContract",
    "StreamMismatch",
    "compatibility",
    "lane_permutation",
    "marker_pairs",
]
