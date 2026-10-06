# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Transport: native ready/valid pins, AXIS, and the contract of one channel end.

**Ready/valid.** A transfer occurs on the associated rising clock edge with
valid and ready, outside reset. Valid data and sidebands are held while
stalled. ``ReadyValidStream`` describes the native pins; ``AxisBeat`` a
homogeneous AXIS beat, lane zero lowest, with only the complete beat padded to a
byte boundary. Scalar encodings keep their QONNX widths. A kernel's
``AxiStreamPort`` (``finn.kernels.port``) builds one from its lanes.

**Contracts.** A ``StreamContract`` joins three levels, each owned elsewhere
and checked here together:

- logical: element encoding, lanes, beat form, repetition and marker rules;
- physical: packing of the lanes into the transport word, lane zero lowest;
- protocol: the ready/valid (or AXIS) pins, marker pins and clock/reset names.

``compatibility`` compares a producing and a consuming end: the producer's
values must fit the consumer's element (``ScalarEncoding.fits``). A logical
mismatch is the channel's to repair: its plan (``finn.dataflow.plan``) names the
steps (reorder or replay, width conversion, markers) and its adapter carries
them out (``finn.kernels.adapters``). A pure lane permutation, padding and
reset polarity leave the sequence unchanged; they are
properties of the connection, realized as wires (``lane_permutation``,
``marker_pairs``): a producer's padding bits are left unconnected, and a
consumer's padding is driven with zeros. A marker closing every beat
(``LevelEnd.constant``) is one too: tied high where the producer offers none.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from finn.core.space import ValueSemantics, default_semantics
from finn.dataflow.datatypes import (
    QONNXDataType,
    canonical_qonnx_datatype,
    qonnx_datatype_width,
)
from finn.dataflow.plan import Unrealizable, presented
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import (
    Adaptation,
    BeatSequence,
    LevelEnd,
    Repetition,
    Traversal,
    classify,
)
from finn.kernels.artifacts.abi import Bus, Direction, Endpoint, Member, Signal, StandardProtocol

# -- native ready/valid ------------------------------------------------------------------


class MarkerKind(Enum):
    LAST = "last"
    LOOP_END = "loop_end"


@dataclass(frozen=True)
class StreamMarker:
    signal: str
    kind: MarkerKind
    width: int = 1

    def __post_init__(self) -> None:
        if not self.signal or not isinstance(self.kind, MarkerKind):
            raise ValueError("a stream marker names a signal and its meaning")
        if type(self.width) is not int or self.width < 1:
            raise ValueError("a stream marker has positive width")
        if self.kind is not MarkerKind.LOOP_END and self.width != 1:
            raise ValueError("a last marker is one bit")


@dataclass(frozen=True)
class ReadyValidStream:
    """Unpadded native pins, optionally lowered to a byte-aligned AXI interface.

    LAST only identifies a frame boundary; the kernel supplies its cross-port
    meaning. LOOP_END carries the native nested-loop completion vector.
    """

    name: str
    data_width: int
    endpoint: Endpoint
    data: str
    valid: str
    ready: str
    clock: str | None = None
    reset: str | None = None
    markers: tuple[StreamMarker, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.endpoint, Endpoint):
            raise ValueError("a stream requires a name and endpoint")
        if type(self.data_width) is not int or self.data_width < 1:
            raise ValueError("stream data width must be positive")
        object.__setattr__(self, "markers", tuple(self.markers))
        names = (self.data, self.valid, self.ready, *(marker.signal for marker in self.markers))
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(
            names
        ):
            raise ValueError("stream pins require distinct nonempty names")

    def pins(self) -> tuple[Signal, ...]:
        """Expose the native pins, without asserting an AXI protocol or changing widths."""
        forward = Direction.OUT if self.endpoint is Endpoint.INITIATOR else Direction.IN
        backward = Direction.IN if self.endpoint is Endpoint.INITIATOR else Direction.OUT
        return (
            Signal(self.data, forward, self.data_width),
            Signal(self.valid, forward, 1),
            *(Signal(marker.signal, forward, marker.width) for marker in self.markers),
            Signal(self.ready, backward, 1),
        )

    def axis_bus(self) -> Bus:
        """Map compatible native signals onto AXI; adapters must supply any padding."""
        if self.data_width % 8:
            raise ValueError("AXI stream data width must be byte aligned")
        if len(self.markers) > 1 or any(
            marker.kind is not MarkerKind.LAST for marker in self.markers
        ):
            raise ValueError("AXI lowering supports only a single LAST marker")
        return Bus(
            self.name,
            StandardProtocol.AXIS,
            (
                Member("tdata", self.data, self.data_width),
                Member("tvalid", self.valid),
                Member("tready", self.ready),
                *(Member("tlast", marker.signal) for marker in self.markers),
            ),
            endpoint=self.endpoint,
            associated_clock=self.clock,
            associated_reset=self.reset,
        )


# -- AXIS ----------------------------------------------------------------------------------


@dataclass(frozen=True, init=False)
class AxisBeat:
    """A homogeneous beat with lane zero in the least-significant bits.

    ``last`` declares the pin. Its workload-dependent meaning is supplied when
    binding to a logical port. ``dtype`` is the datatype value itself (qonnx's
    are interned and frozen), not reduced to a bit width.
    """

    name: str
    dtype: QONNXDataType
    elements_per_beat: int
    endpoint: Endpoint
    last: bool

    def __init__(
        self,
        name: str,
        dtype: QONNXDataType,
        elements_per_beat: int,
        *,
        endpoint: Endpoint,
        last: bool = False,
    ) -> None:
        dtype = canonical_qonnx_datatype(dtype)
        if not isinstance(name, str) or not name:
            raise ValueError("an AXIS declaration requires a nonempty name")
        if type(elements_per_beat) is not int or elements_per_beat <= 0:
            raise ValueError("elements per beat must be a positive integer")
        if not isinstance(endpoint, Endpoint) or type(last) is not bool:
            raise ValueError("AXIS requires an Endpoint and a boolean last flag")
        if qonnx_datatype_width(dtype) <= 0:
            raise ValueError("AXIS scalar encodings must have positive width")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "elements_per_beat", elements_per_beat)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "last", last)

    @property
    def element_bits(self) -> int:
        return qonnx_datatype_width(self.dtype)

    @property
    def payload_bits(self) -> int:
        return self.element_bits * self.elements_per_beat

    @property
    def data_width(self) -> int:
        return self.carrier_bits

    @property
    def carrier_bits(self) -> int:
        return (self.payload_bits + 7) // 8 * 8

    def bus(self, *, clock: str | None = None, reset: str | None = None) -> Bus:
        """Lower to the existing, purely physical ABI representation."""
        return self.native(clock=clock, reset=reset).axis_bus()

    def native(self, *, clock: str | None = None, reset: str | None = None) -> ReadyValidStream:
        """The native transport underlying this typed, padded AXI profile."""
        return ReadyValidStream(
            self.name,
            self.data_width,
            self.endpoint,
            f"{self.name}_tdata",
            f"{self.name}_tvalid",
            f"{self.name}_tready",
            clock,
            reset,
            (StreamMarker(f"{self.name}_tlast", MarkerKind.LAST),) if self.last else (),
        )


# -- the contract of one channel end -------------------------------------------------------


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
    """One channel end: its transport plus the logical sequence it carries.

    ``markers`` maps a transport marker to the rule it follows: a one-bit
    marker by its signal (``olast``), or one bit of a wider loop-completion
    marker as ``signal[bit]`` (``olst[1]``). For a producer these are
    guarantees; for a consumer, requirements. Transport markers without a rule
    can be neither required nor matched.
    """

    transport: ReadyValidStream
    element: ScalarEncoding
    form: Traversal
    repetition: Repetition = Repetition.ONCE
    markers: Mapping[str, LevelEnd] | tuple[tuple[str, LevelEnd], ...] = ()

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
        for key, rule in markers.items():
            signal, bit = marker_bit(key)
            width = widths.get(signal, 0)
            if (width != 1 if bit is None else bit >= width) or not isinstance(rule, LevelEnd):
                raise ValueError(f"{key}: a rule needs a one-bit transport marker or marker bit")
            if not rule.aligned(self.form):
                raise ValueError(f"{key}: a marker every {rule.beats} beats closes no loop level")
        object.__setattr__(self, "markers", tuple(sorted(markers.items())))

    @property
    def lanes(self) -> int:
        return self.form.lanes

    @property
    def payload_bits(self) -> int:
        return self.form.lanes * self.element.bits

    @property
    def rules(self) -> dict[str, LevelEnd]:
        return dict(self.markers)

    @property
    def sequence(self) -> BeatSequence:
        """The logical sequence this end presents, without its transport."""
        return BeatSequence(self.form, self.repetition, tuple(self.rules.values()))


STREAM_CONTRACT: ValueSemantics[StreamContract] = default_semantics(StreamContract)


def marker_bit(key: str) -> tuple[str, int | None]:
    """The signal a marker rule key names, and the bit of it (None for a one-bit marker)."""
    match = re.fullmatch(r"(\w+)\[(\d+)\]", key)
    return (match[1], int(match[2])) if match else (key, None)


def compatibility(
    source: StreamContract, sink: StreamContract, *, source_is_top: bool, sink_is_top: bool
) -> tuple[Mismatch, ...]:
    """Every reason ``source`` may not drive ``sink``; empty when they connect.

    Elements are not compared: a channel's ``well_formed`` holds each end against
    its tensor's element, and every stage between carries that element.

    ``*_is_top`` marks an end on the composed module's own boundary, whose
    endpoint direction is seen from outside (a top input is a source inside).
    """
    found: list[Mismatch] = []

    def refuse(level: Level, code: str, message: str) -> None:
        found.append(Mismatch(level, code, message))

    produces = Endpoint.TARGET if source_is_top else Endpoint.INITIATOR
    consumes = Endpoint.INITIATOR if sink_is_top else Endpoint.TARGET
    if source.transport.endpoint is not produces or sink.transport.endpoint is not consumes:
        refuse(Level.PROTOCOL, "channel-direction", "the source must produce and the sink consume")

    if sink.repetition is Repetition.CYCLIC and source.repetition is not Repetition.CYCLIC:
        refuse(Level.LOGICAL, "channel-repetition", "a single pass cannot feed a cyclic consumer")
    produced = _presented(source, sink)
    if produced is None:
        refuse(
            Level.LOGICAL,
            "channel-form",
            "the consumer's pass is not whole repetitions of the cyclic source",
        )
    else:
        verdict = classify(produced, sink.form)
        if verdict.adaptation not in (Adaptation.IDENTITY, Adaptation.LANE_PERMUTATION):
            detail = f": {verdict.reorder}" if verdict.reorder else ""
            refuse(
                Level.LOGICAL,
                "channel-form",
                f"needs a {verdict.adaptation.value} adapter ({verdict.detail}){detail}",
            )

    offered = source.rules
    for signal, rule in sink.rules.items():
        if rule not in offered.values() and not rule.constant:
            refuse(
                Level.LOGICAL,
                "channel-marker",
                f"{signal} requires a marker every {rule.beats} beats; none is produced",
            )

    if source_is_top:
        consumed = {
            marker_bit(offered)[0]
            for offered, _ in marker_pairs(source, sink)
            if offered is not None
        }
        unused = [m.signal for m in source.transport.markers if m.signal not in consumed]
        if unused:
            refuse(
                Level.PROTOCOL,
                "channel-top-marker",
                f"top input markers {unused} would be left unconsumed",
            )
    if sink_is_top and sink.transport.markers:
        ruled = {marker_bit(key)[0] for key in sink.rules}
        missing = [m.signal for m in sink.transport.markers if m.signal not in ruled]
        if missing:
            refuse(Level.PROTOCOL, "channel-top-marker", f"top output markers {missing} lack rules")
    return tuple(found)


def _presented(source: StreamContract, sink: StreamContract) -> Traversal | None:
    """What a source presents over one consumer pass (``plan.presented``); None if it
    cannot align. A single pass is compared as it is."""
    if source.repetition is Repetition.ONCE:
        return source.form
    try:
        return presented(source.sequence, sink.sequence)
    except Unrealizable:
        return None


def lane_permutation(source: StreamContract, sink: StreamContract) -> tuple[int, ...]:
    """Sink lane -> source lane; the identity unless the lanes are only reordered."""
    produced = _presented(source, sink)
    if produced is not None:
        verdict = classify(produced, sink.form)
        if verdict.adaptation is Adaptation.LANE_PERMUTATION:
            return verdict.lane_permutation
    return tuple(range(sink.lanes))


def marker_pairs(
    source: StreamContract, sink: StreamContract
) -> tuple[tuple[str | None, str], ...]:
    """(source signal, sink signal) for each required sink marker; a source of None ties
    a constant marker (one closing every beat) the source does not offer high."""
    offered = source.rules
    pairs: list[tuple[str | None, str]] = []
    used: set[str] = set()
    for signal, rule in sink.rules.items():
        for candidate, candidate_rule in offered.items():
            if candidate_rule == rule and candidate not in used:
                pairs.append((candidate, signal))
                used.add(candidate)
                break
        else:
            if rule.constant:
                pairs.append((None, signal))
    return tuple(pairs)


__all__ = [
    "AxisBeat",
    "Level",
    "MarkerKind",
    "Mismatch",
    "ReadyValidStream",
    "STREAM_CONTRACT",
    "StreamContract",
    "StreamMarker",
    "compatibility",
    "lane_permutation",
    "marker_bit",
    "marker_pairs",
]
