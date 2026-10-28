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
from typing import TYPE_CHECKING, Any

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
from finn.dataflow.traversal import LevelEnd
from finn.kernels.artifacts.abi import Clocking, Direction, Endpoint, Pin
from finn.kernels.artifacts.module import Held
from finn.kernels.base import UntimedKernel
from finn.kernels.transport import (
    AxisBeat,
    AxisSpelling,
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
    """A native loop-completion vector on ready/valid pins (FinnLib's ``olst``): its
    ``signal``, and the level each of its bits closes, ``levels[bit]``."""

    signal: str
    levels: tuple[LevelEnd, ...]


@dataclass(frozen=True)
class PhysicalPort:
    """Resolved port inputs for pure physical calculations, without authoring callbacks.

    A PortModel reads these literal fields in an ordinary method. This record is
    neither a Space nor a symbolic binding/allowed-read table.
    """

    name: str
    endpoint: Endpoint
    signals: tuple[str, ...] = ()
    loop_ends: LoopEnds | None = None
    spelling: AxisSpelling = AxisSpelling.LOWER
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


def _markers(presenting: Presenting) -> tuple[LevelEnd, ...]:
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
    """Its AXIS beat: its element's lanes, with a ``TLAST`` when it presents a marker."""
    label = _given(port, "name")
    markers = _markers(presenting)
    if len(markers) > 1:
        return reject("port-markers", f"{label} has one TLAST; the sequence needs more")
    lanes = lane_count(port, presenting)
    if isinstance(lanes, Rejected):
        return lanes
    try:
        return AxisBeat(
            label,
            _element_of(presenting).dtype,
            lanes,
            endpoint=_given(port, "endpoint"),
            last=bool(markers),
            spelling=_given(port, "spelling"),
        )
    except ValueError as error:
        return reject("port-lanes", f"{label}: {error}")


def transport(
    port: PhysicalPort, presenting: Presenting, clocking: Clocking | None
) -> ReadyValidTransport | Rejected:
    """Its ready/valid pins, on its kernel's clock and reset: its AXIS beat's, or its
    ``signals`` with its loop-completion vector."""
    clock = None if clocking is None else clocking.clock
    reset = None if clocking is None else clocking.reset
    label, signals = _given(port, "name"), _given(port, "signals")
    if not signals:
        beat = axis(port, presenting)
        return beat if isinstance(beat, Rejected) else beat.native(clock=clock, reset=reset)
    if len(signals) != 3:
        return reject("port-signals", f"{label}: signals are data, valid and ready")
    data, valid, ready = signals
    markers: tuple[StreamMarker, ...] = ()
    ends: LoopEnds | None = _given(port, "loop_ends")
    presented = _markers(presenting)
    if ends is not None:
        unclosed = [rule.beats for rule in presented if rule not in ends.levels]
        if not ends.levels or unclosed:
            return reject(
                "port-markers",
                f"{label}: no bit of {ends.signal} closes every {unclosed} beats"
                if unclosed
                else f"{label}: {ends.signal} closes no level",
            )
        markers = (StreamMarker(ends.signal, MarkerKind.LOOP_END, len(ends.levels)),)
    elif presented:
        return reject("port-markers", f"{label}: ready/valid pins carry no marker")
    lanes = lane_count(port, presenting)
    if isinstance(lanes, Rejected):
        return lanes
    bits = lanes * _element_of(presenting).bits
    endpoint = _given(port, "endpoint")
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
    """Its pins of the module: its ready/valid signals, or its AXIS bus."""
    if _given(port, "signals"):
        found = transport(port, presenting, clocking)
        return found if isinstance(found, Rejected) else found.pins()
    beat = axis(port, presenting)
    if isinstance(beat, Rejected):
        return beat
    return (beat.bus(clock=clocking.clock, reset=clocking.reset),)


def held(port: PhysicalPort, presenting: Presenting, clocking: Clocking) -> Held | Rejected:
    """While idle (its element alone): its forward pins of a target and its ready of an
    initiator held low; placed, none."""
    if isinstance(presenting, Stream):
        return Held()
    found = transport(port, presenting, clocking)
    if isinstance(found, Rejected):
        return found
    inputs: list[tuple[str, int]] = []
    unused: list[str] = []
    for signal in found.pins():
        if signal.direction is Direction.IN:
            inputs.append((signal.name, 0))
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
    found = transport(port, stream, clocking)
    if isinstance(found, Rejected):
        return found
    ends: LoopEnds | None = _given(port, "loop_ends")
    markers: tuple[tuple[str, LevelEnd], ...]
    if ends is not None:
        # Each bit whose level it presents, by its bit (``olst[1]``).
        markers = tuple(
            (f"{ends.signal}[{bit}]", level)
            for bit, level in enumerate(ends.levels)
            if level in sequence.markers
        )
    elif found.markers:
        markers = ((found.markers[0].signal, sequence.markers[0]),)
    else:
        markers = ()
    return StreamContract(
        found, stream.element, sequence.form, sequence.repetition, markers, pace(port)
    )


# -- a port a kernel with children presents for a child -------------------------------------


def _lexical_port(namespace: type[Space] | Space, port: str) -> PortModel | None:
    """Find a real port through Group frames, stopping at other real Spaces."""
    from finn.kernels.port_model import PortModel  # noqa: PLC0415 - breaks base/port cycle

    frame: object = namespace
    segments = port.split(".")
    for segment in segments[:-1]:
        group = getattr(frame, segment, None)
        if not isinstance(group, Group):
            return None
        frame = group
    child = getattr(frame, segments[-1], None)
    return child if isinstance(child, PortModel) else None


def port_endpoint(space_type: type[Space], port: str) -> Endpoint | None:
    """Known child binding/default or parent surface; unknown inputs remain unknown."""
    if not (isinstance(space_type, type) and issubclass(space_type, UntimedKernel)):
        return None
    from finn.kernels.port_model import PortSurface  # noqa: PLC0415

    native = getattr(space_type, port, None)
    if isinstance(native, PortSurface):
        return native.endpoint if isinstance(native.endpoint, Endpoint) else None
    child = _lexical_port(space_type, port)
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
    native = getattr(namespace, port, None)
    if isinstance(native, PortSurface):
        return native.at(namespace)
    child = _lexical_port(namespace, port)
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
