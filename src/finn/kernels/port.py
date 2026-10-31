# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Resolved physical port values and pure protocol calculations.

Actual authoring uses StreamPort real children, a module's and a kernel with children's
alike. Native module source reads model inputs serially, then calls these pure helpers.
``stream_port`` and ``port_direction`` find a kernel's port by its name, through its lexical
Group paths, stopping at real Space boundaries but for a member's port it exposes by a node
alias: the port whose readouts a channel and its sizing read (``point.<port>.side``), and
its direction; ``exposed_side`` and ``exposed_pins`` give an exposed port's as its kernel
presents it. They never authorize a model's reads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

from finn.core.space import (
    Group,
    Rejected,
    Space,
    reject,
)
from finn.core.space._aliases import AliasEntry, alias_table
from finn.core.space.declarations import Declaration, Required, declared_path
from finn.dataflow.schedule import Index
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import LevelEnd, Marker, Traversal
from finn.kernels.artifacts.abi import Clocking, Direction, Endpoint, Pin
from finn.kernels.artifacts.module import Held
from finn.kernels.base import Kernel, PortPins
from finn.kernels.laws import AXIS, AXIS_MARKERS, IDLE_LEVEL, Axis, Native, TransportKind
from finn.kernels.transport import (
    AxisBeat,
    MarkerKind,
    ReadyValidTransport,
    StreamMarker,
    StreamPins,
)

if TYPE_CHECKING:
    from finn.kernels.port_model import StreamPort
    from finn.kernels.sides import Side


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


def stream_endpoint(direction: Direction) -> Endpoint:
    """A stream's bus role, from its port's direction: an input is a target (an AXIS
    slave, ``s_axis_*``), an output an initiator (a master, ``m_axis_*``). No stream is
    ``INOUT``: its port refuses it (``port-direction``)."""
    if direction is Direction.IN:
        return Endpoint.TARGET
    if direction is Direction.OUT:
        return Endpoint.INITIATOR
    raise ValueError(f"a stream is an input or an output, not {direction}")


@dataclass(frozen=True)
class PhysicalPort:
    """Resolved port inputs for pure physical calculations, without authoring callbacks.

    A StreamPort reads these literal fields in an ordinary method. This record is
    neither a Space nor a symbolic binding/allowed-read table. ``name`` is the port's
    name; ``endpoint`` its bus role, from its direction (``stream_endpoint``); its pins
    follow from its ``transport`` (``finn.kernels.laws``).
    """

    name: str
    endpoint: Endpoint
    transport: TransportKind = AXIS
    loop_ends: LoopEnds | None = None
    lanes: tuple[Index, ...] = ()


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


def stream_pins(
    port: PhysicalPort, stream: Stream, clocking: Clocking | None
) -> StreamPins | Rejected:
    """What a channel wires to: its stream on its transport's pins, each marker of it by the
    pin that carries it."""
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
    return StreamPins(found, stream, markers)


# -- a kernel's port, by its name ------------------------------------------------------------


def stream_port(namespace: type[Space] | Space, port: str) -> StreamPort:
    """Kernel ``namespace``'s stream port ``port`` (a dotted path through its Groups,
    ``group.port``), or the member's stream port it exposes by a node alias of that name
    (``x = compute.x``, ``finn.kernels.port_model.exposed_ports``), as a reference to read
    its readouts of (``namespace`` a Space class, or a node declaration for a reference
    below it): ``point.query(stream_port(type(point), "x").side)``. An exposed port's
    readouts are its member's port's, labelled as the member labels them: ``exposed_side``
    and ``exposed_pins`` give them as the kernel presents them. Stops at child Spaces: a
    member's port is its kernel's only where it exposes it. KeyError where ``port`` names
    no stream port of it."""
    from finn.kernels.port_model import StreamPort  # noqa: PLC0415 - base/port cycle

    space_type = namespace if isinstance(namespace, type) else type(namespace)
    if not issubclass(space_type, Space) or (
        not isinstance(namespace, type) and declared_path(namespace) is None
    ):
        raise TypeError("a port is read of a Space class or node declaration")
    if _exposed_entry(space_type, port) is not None:
        return cast("StreamPort", getattr(namespace, port))
    frame: type[Space] | Space | Group = namespace
    segments = port.split(".")
    for segment in segments[:-1]:
        group = getattr(frame, segment, None)
        if not isinstance(group, Group):
            raise KeyError(port)
        frame = group
    found = getattr(frame, segments[-1], None)
    if not isinstance(found, StreamPort):
        raise KeyError(port)
    return found


def _exposed_entry(space_type: type[Space], port: str) -> AliasEntry | None:
    """The node alias ``port`` of ``space_type`` naming a member's stream port, in every
    selection; None where ``port`` is no such alias."""
    from finn.kernels.port_model import StreamPort  # noqa: PLC0415 - base/port cycle

    entry = alias_table(cast(Any, space_type)).get(port)
    if entry is None or not entry.ends:
        return None
    if not all(issubclass(end.space_type, StreamPort) for end in entry.ends):
        return None
    return entry


def port_direction(space_type: type[Space], port: str) -> Direction | None:
    """Whether kernel ``space_type``'s port ``port`` is an input or an output, by its kind
    (``InPort``, ``OutPort``): an exposed port's is its member's port's, the one every node
    the alias may name is of. None where it is not known: no such port."""
    from finn.kernels.port_model import InPort, OutPort  # noqa: PLC0415 - base/port cycle

    if not (isinstance(space_type, type) and issubclass(space_type, Kernel)):
        return None
    entry = _exposed_entry(space_type, port)
    if entry is not None:
        kinds = [end.space_type for end in entry.ends]
    else:
        try:
            found = stream_port(space_type, port)
        except KeyError:
            return None
        kinds = [type(found)]
    if all(issubclass(kind, InPort) for kind in kinds):
        return Direction.IN
    if all(issubclass(kind, OutPort) for kind in kinds):
        return Direction.OUT
    return None


# -- a port as its kernel presents it -------------------------------------------------------
#
# A kernel presents a port of its own under the port's label, its pins on its own module; a
# member's port it exposes (``x = compute.x``) under the alias's name, its pins on the leaf
# whose module has them: the node the alias reaches, by its owner's path below the kernel,
# each member read where its holder's netlist places it (``Kernel.placed_at``) and a
# selector's case by its key (``compute.packed``). The port itself stays its owner's, and so
# do its answers: only the name and the instance change, as its kernel presents it.


def exposed_name(space_type: type[Space], port: str) -> str | None:
    """The name kernel class ``space_type`` presents its port ``port`` under where it is no
    port of its own: the node alias's (``x``); None for its own, labelled by its label."""
    return None if _exposed_entry(space_type, port) is None else port


def exposed_side(space_type: type[Space], port: str, side: Side) -> Side:
    """``side``, port ``port``'s read through ``stream_port``, as kernel class
    ``space_type`` presents it: an exposed port's under the alias's name."""
    name = exposed_name(space_type, port)
    return side if name is None else replace(side, label=name)


def exposed_pins(point: Space, port: str, found: PortPins) -> PortPins:
    """``found``, port ``port``'s stream pins read at kernel ``point`` through
    ``stream_port``, as ``point`` presents them: an exposed port's under the alias's name,
    on the leaf the alias reaches (``PortPins.instance``, above)."""
    name = exposed_name(type(point), port)
    if name is None:
        return found
    return PortPins(name, found.pins, _instance(point, getattr(point, port)))


def _instance(point: Space, node: Space) -> str:
    """The leaf holding port ``node``, below ``point``: the owner's path of the port's
    kernel, each member where its holder places it (``Kernel.placed_at``)."""
    from finn.core.space import inspection  # noqa: PLC0415

    walked = inspection.path(point)
    below = inspection.path(node)[len(walked) + 1 if walked else 0 :]
    segments = below.split(".")[:-1]
    parts: list[str] = []
    current: Any = point
    index = 0
    while index < len(segments):
        segment = segments[index]
        placed: Mapping[str, str] = getattr(type(current), "placed_at", {})
        parts.append(placed.get(segment, segment))
        current = getattr(current, segment)
        walked = f"{walked}.{segment}" if walked else segment
        index += 1
        if inspection.path(current) != walked:
            # A selector reads as its selected case, one name below: its key.
            parts.append(segments[index])
            walked = f"{walked}.{segments[index]}"
            index += 1
    return ".".join(parts)


__all__ = [
    "LoopEnds",
    "axis",
    "beats",
    "Presenting",
    "PhysicalPort",
    "data_width",
    "held",
    "lane_count",
    "port_direction",
    "pins",
    "stream_endpoint",
    "stream_pins",
    "exposed_name",
    "exposed_pins",
    "exposed_side",
    "stream_port",
    "transport",
]
