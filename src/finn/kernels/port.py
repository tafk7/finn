# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Resolved physical port values and pure protocol calculations.

Actual authoring uses PortModel/SequencePort/IndexedPort/ScheduledPort real
children and explicitly authored parent PortSurface references. Native module
source reads model inputs serially, then calls these pure helpers. No flat
lowering, named callback tables, read lists or compatibility aliases remain.
port_surface/port_endpoint discover actual lexical Group paths and stop at
real Space boundaries; metadata never authorizes a model's reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from finn.core.space import (
    Group,
    Rejected,
    Space,
    reject,
    space_formals,
)
from finn.core.space._nodes import NodeDecl
from finn.core.space.declarations import Declaration, Required, declared_path
from finn.dataflow.schedule import Index, Pace, Schedule
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import LevelEnd, Marker, Traversal
from finn.kernels.artifacts.abi import Clocking, Direction, Endpoint, Pin
from finn.kernels.artifacts.module import Held
from finn.kernels.base import Kernel
from finn.kernels.laws import AXIS, AXIS_MARKERS, IDLE_LEVEL, Axis, Native, TransportKind
from finn.kernels.transport import (
    AxisBeat,
    MarkerKind,
    ReadyValidTransport,
    StreamContract,
    StreamMarker,
)

if TYPE_CHECKING:
    from finn.kernels.port_model import PortModel, PortSurface


def _applies(space: Space, guard: str | None) -> bool:
    """Whether a member guarded by the member ``guard`` of ``space`` applies: always where it
    has none; else where that member holds."""
    return guard is None or bool(getattr(space, guard))


@dataclass(frozen=True)
class LoopEnds:
    """A native loop-completion vector (L9's exception, FinnLib's ``olst``): the level each
    of its bits closes, ``levels[bit]``. Its pin is its transport's
    (``finn.kernels.laws.Native.loop_ends``)."""

    levels: tuple[LevelEnd, ...]


@dataclass(frozen=True)
class PhysicalPort:
    """Resolved port inputs for pure physical calculations, without authoring callbacks.

    A PortModel reads these literal fields in an ordinary method. This record is
    neither a Space nor a symbolic binding/allowed-read table. ``name`` is the port's
    name; its pins follow from its ``transport`` (``finn.kernels.laws``).
    """

    name: str
    endpoint: Endpoint
    transport: TransportKind = AXIS
    loop_ends: LoopEnds | None = None
    lanes: tuple[Index, ...] = ()
    schedule: Schedule | None = None
    reduces: tuple[Index, ...] = ()
    holds: tuple[Index, ...] = ()


Presenting = ScalarEncoding | Stream


def _given(port: PhysicalPort, name: str) -> Any:
    """Resolved physical input ``name``; declarations must be resolved by their owner first."""
    value = getattr(port, name)
    if isinstance(value, (Declaration, Required)):
        raise TypeError(
            f"{port.name}: its {name} is a member of its kernel; "
            "resolve physical inputs before calculation"
        )
    return value


def _element_of(presenting: Presenting) -> ScalarEncoding:
    return presenting.element if isinstance(presenting, Stream) else presenting


def stream_refusal(stream: Stream) -> Rejected | None:
    """The existing physical schemas cover loop nests and one-beat level-end markers.

    Grouped and held LevelDone remain useful logical values, not implicit new
    hardware capabilities. Refuse them before creating pins or a stored interface.
    """
    sequence = stream.sequence
    if not isinstance(sequence.form, Traversal):
        return reject("port-presentation-unsupported", "physical ports do not yet support Grouped")
    if any(not isinstance(rule, LevelEnd) for rule in sequence.markers):
        return reject(
            "port-marker-unsupported", "physical ports do not yet support held LevelDone markers"
        )
    return None


def _markers(presenting: Presenting) -> tuple[Marker, ...]:
    return presenting.sequence.markers if isinstance(presenting, Stream) else ()


def lane_count(port: PhysicalPort, presenting: Presenting) -> int | Rejected:
    """The lanes a beat carries: what it presents; idle, one."""
    if isinstance(presenting, Stream):
        return presenting.sequence.form.lanes
    lanes = _given(port, "lanes")
    if lanes:
        return reject(
            "port-idle",
            f"{_given(port, 'name')}: an idle port carries one lane, not lanes of {lanes}",
        )
    return 1


def axis(port: PhysicalPort, presenting: Presenting) -> AxisBeat | Rejected:
    """Its AXIS beat: on its bus, its element's lanes, with a ``TLAST`` when it presents a
    marker (L9)."""
    label = _given(port, "name")
    kind = _given(port, "transport")
    if not isinstance(kind, Axis):
        return reject("port-transport", f"{label}: a native port has no AXIS beat")
    if _given(port, "loop_ends") is not None:
        return reject(
            "port-markers",
            f"{label}: an AXIS port carries one TLAST, no loop-completion vector (L9)",
        )
    unsupported = stream_refusal(presenting) if isinstance(presenting, Stream) else None
    if unsupported is not None:
        return unsupported
    endpoint = _given(port, "endpoint")
    bus = kind.bus_name(label, endpoint)
    markers = _markers(presenting)
    if len(markers) > AXIS_MARKERS:
        return reject("port-markers", f"{bus} has one TLAST; the sequence needs more (L9)")
    lanes = lane_count(port, presenting)
    if isinstance(lanes, Rejected):
        return lanes
    try:
        return AxisBeat(
            bus,
            _element_of(presenting).dtype,
            lanes,
            endpoint=endpoint,
            last=bool(markers),
            spelling=kind.spelling,
        )
    except ValueError as error:
        return reject("port-lanes", f"{bus}: {error}")


def transport(
    port: PhysicalPort, presenting: Presenting, clocking: Clocking | None
) -> ReadyValidTransport | Rejected:
    """Its ready/valid pins, on its kernel's clock and reset: its AXIS beat's, or its native
    pins with its loop-completion vector, named by its transport (``finn.kernels.laws``)."""
    unsupported = stream_refusal(presenting) if isinstance(presenting, Stream) else None
    if unsupported is not None:
        return unsupported
    clock = None if clocking is None else clocking.clock
    reset = None if clocking is None else clocking.reset
    label, kind = _given(port, "name"), _given(port, "transport")
    if isinstance(kind, Axis):
        beat = axis(port, presenting)
        return beat if isinstance(beat, Rejected) else beat.native(clock=clock, reset=reset)
    assert isinstance(kind, Native)
    endpoint = _given(port, "endpoint")
    data, valid, ready = kind.pins(endpoint)
    markers: tuple[StreamMarker, ...] = ()
    ends: LoopEnds | None = _given(port, "loop_ends")
    presented = _markers(presenting)
    if ends is not None:
        signal = kind.loop_ends(endpoint)
        if signal is None:
            return reject(
                "port-markers", f"{label}: a native port named in full has no loop-completion pin"
            )
        unclosed = [rule.beats for rule in presented if rule not in ends.levels]
        if not ends.levels or unclosed:
            return reject(
                "port-markers",
                f"{label}: no bit of {signal} closes every {unclosed} beats"
                if unclosed
                else f"{label}: {signal} closes no level",
            )
        markers = (StreamMarker(signal, MarkerKind.LOOP_END, len(ends.levels)),)
    elif presented:
        return reject("port-markers", f"{label}: ready/valid pins carry no marker (L9)")
    lanes = lane_count(port, presenting)
    if isinstance(lanes, Rejected):
        return lanes
    bits = lanes * _element_of(presenting).bits
    try:
        return ReadyValidTransport(label, bits, endpoint, data, valid, ready, clock, reset, markers)
    except ValueError as error:
        return reject("port-signals", f"{label}: {error}")


def data_width(port: PhysicalPort, presenting: Presenting) -> int | Rejected:
    """The bits of its words: its lanes times its element's bits, padded to bytes for
    AXIS."""
    found = transport(port, presenting, None)
    return found if isinstance(found, Rejected) else found.data_width


def pins(
    port: PhysicalPort, presenting: Presenting, clocking: Clocking
) -> tuple[Pin, ...] | Rejected:
    """Its pins of the module: its native ready/valid signals, or its AXIS bus."""
    if isinstance(_given(port, "transport"), Native):
        found = transport(port, presenting, clocking)
        return found if isinstance(found, Rejected) else found.pins()
    beat = axis(port, presenting)
    if isinstance(beat, Rejected):
        return beat
    return (beat.bus(clock=clocking.clock, reset=clocking.reset),)


def held(port: PhysicalPort, presenting: Presenting, clocking: Clocking) -> Held | Rejected:
    """While idle (its element alone): its forward pins of a target and its ready of an
    initiator held low (L10); placed, none."""
    if isinstance(presenting, Stream):
        return Held()
    found = transport(port, presenting, clocking)
    if isinstance(found, Rejected):
        return found
    inputs: list[tuple[str, int]] = []
    unused: list[str] = []
    for signal in found.pins():
        if signal.direction is Direction.IN:
            inputs.append((signal.name, IDLE_LEVEL))
        else:
            unused.append(signal.name)
    return Held(tuple(inputs), tuple(unused))


def beats(stream: Stream) -> int:
    """The beats it carries a frame."""
    return stream.sequence.form.beats


def pace(port: PhysicalPort) -> Pace | None:
    """When it presents each beat: its schedule, with its ``reduces`` and ``holds``; None
    for a given sequence."""
    schedule: Schedule | None = _given(port, "schedule")
    if schedule is None:
        return None
    return Pace(schedule, _given(port, "reduces"), _given(port, "holds"))


def contract(
    port: PhysicalPort, stream: Stream, clocking: Clocking | None
) -> StreamContract | Rejected:
    """What it states to a channel: its transport, its stream (element, form, repetition,
    markers, each by the transport pin that carries it) and its pace."""
    sequence = stream.sequence
    unsupported = stream_refusal(stream)
    if unsupported is not None:
        return unsupported
    assert isinstance(sequence.form, Traversal)
    found = transport(port, stream, clocking)
    if isinstance(found, Rejected):
        return found
    ends: LoopEnds | None = _given(port, "loop_ends")
    markers: tuple[tuple[str, LevelEnd], ...]
    if ends is not None:
        # Each bit whose level it presents, by its bit (``olst[1]``).
        markers = tuple(
            (f"{found.markers[0].signal}[{bit}]", level)
            for bit, level in enumerate(ends.levels)
            if level in sequence.markers
        )
    elif found.markers:
        markers = ((found.markers[0].signal, cast(LevelEnd, sequence.markers[0])),)
    else:
        markers = ()
    return StreamContract(
        found, stream.element, sequence.form, sequence.repetition, markers, pace(port)
    )


# -- a port a kernel with children presents for a child -------------------------------------


def _lexical_port(namespace: type[Space] | Space, port: str) -> PortModel | PortSurface | None:
    """Find an actual child or authored surface, rebased through its lexical Groups."""
    from finn.kernels.port_model import (  # noqa: PLC0415 - breaks base/port cycle
        PortModel,
        PortSurface,
    )

    frame: type[Space] | Space | Group = namespace
    segments = port.split(".")
    for segment in segments[:-1]:
        group = getattr(frame, segment, None)
        if not isinstance(group, Group):
            return None
        frame = group
    child = getattr(frame, segments[-1], None)
    if isinstance(child, PortSurface):
        return child.at(frame)
    return child if isinstance(child, PortModel) else None


def port_endpoint(space_type: type[Space], port: str) -> Endpoint | None:
    """Known child binding/default or parent surface; unknown inputs remain unknown."""
    if not (isinstance(space_type, type) and issubclass(space_type, Kernel)):
        return None
    from finn.kernels.port_model import PortSurface  # noqa: PLC0415

    child = _lexical_port(space_type, port)
    if isinstance(child, PortSurface):
        return child.endpoint if isinstance(child.endpoint, Endpoint) else None
    if child is not None:
        path = declared_path(child)
        if not path or not isinstance(path[-1], NodeDecl):
            return None
        record = path[-1]
        formal = space_formals(record.space_type).get("endpoint")
        endpoint = record.bindings.get("endpoint", None if formal is None else formal.default)
        return endpoint if isinstance(endpoint, Endpoint) else None
    return None


def port_surface(namespace: type[Space] | Space, port: str) -> PortSurface:
    """Actual child or parent references, with lexical Group and real-Space boundaries."""
    from finn.kernels.port_model import PortSurface  # noqa: PLC0415

    space_type = namespace if isinstance(namespace, type) else type(namespace)
    if not issubclass(space_type, Space) or (
        not isinstance(namespace, type) and declared_path(namespace) is None
    ):
        raise TypeError("port_surface requires a Space class or node declaration")
    child = _lexical_port(namespace, port)
    if isinstance(child, PortSurface):
        return child
    if child is not None:
        return PortSurface.of(child)
    raise KeyError(port)


__all__ = [
    "LoopEnds",
    "axis",
    "beats",
    "Presenting",
    "PhysicalPort",
    "contract",
    "data_width",
    "held",
    "lane_count",
    "port_endpoint",
    "port_surface",
    "pace",
    "pins",
    "transport",
]
