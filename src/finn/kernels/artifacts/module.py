# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A module to build: one FinnLib module (``Leaf``), or a flat netlist of them (``Composed``).

A ``Leaf`` is a FinnLib module bound to a configuration: its name, parameters
and pins, the files that provide it, the data it reads, and what it holds while
part of it is idle (``Held``). A ``Composed`` module is a ``Fragment`` with
pins: leaf instances, the ``Link`` of each stream hop between their pins and
the control buses it presents (``BusExport``). Every instance is a leaf: the
netlist is flat, and grouping it into modules is a later decision of the flow.

A fragment names its instances by labels relative to its owner (``compute.packed``);
its parent places it with ``under(node)`` and joins its children's with
``merge``. A link end names its instance by such a label, ``None`` for the
root's own pins, or ``^label`` for a node beside the owner (a channel's user,
beside the channel). Constructors check only that a value is well formed; what
a configuration may be is decided in the Spaces that derive it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import Union

from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    ClockAlignment,
    Direction,
    Pin,
    PinInfo,
    Reset,
    abi_pins,
    validate_pins,
)
from finn.kernels.artifacts.contributions import CopiedSource, GeneratedData
from finn.kernels.artifacts.projection import digest

#: A module parameter.
Scalar = Union[bool, int, float, str, Enum]
ScalarTable = tuple[tuple[str, Scalar], ...]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_LABEL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


class BuildError(Exception):
    """A module is not well formed, or cannot be emitted."""


def _table(values: ScalarTable, *, label: str) -> ScalarTable:
    items = tuple(values)
    names = tuple(name for name, _ in items)
    if len(names) != len(set(names)):
        raise BuildError(f"{label} names one value twice")
    if any(not isinstance(name, str) or not name for name in names):
        raise BuildError(f"{label} uses non-empty string names")
    for name, value in items:
        if not isinstance(value, (bool, int, float, str, Enum)):
            raise BuildError(f"{label} value {name!r} is not a scalar")
    return tuple(sorted(items, key=lambda item: item[0]))


def _rtl_scalar(value: Scalar) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, Enum):
        raw = value.value
        if isinstance(raw, bool):
            return str(int(raw))
        if isinstance(raw, (int, float, str)):
            return str(raw)
        raise BuildError(f"enum parameter {value!r} has no canonical RTL scalar spelling")
    return str(value)


def typed_canonical(value: object) -> object:
    """``value`` with each enum and dataclass tagged by its type, for a digest."""
    if isinstance(value, Enum):
        return (f"enum:{type(value).__module__}.{type(value).__qualname__}", value.name)
    if is_dataclass(value) and not isinstance(value, type):
        tag = f"dataclass:{type(value).__module__}.{type(value).__qualname__}"
        return (
            tag,
            tuple(
                (item.name, typed_canonical(getattr(value, item.name))) for item in fields(value)
            ),
        )
    if isinstance(value, Mapping):
        return tuple((name, typed_canonical(value[name])) for name in sorted(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(typed_canonical(item) for item in value)
    return value


def sanitize_stem(stem: str) -> str:
    """``stem`` as an RTL identifier of at most 40 characters."""
    sanitized = "".join(
        character if (character.isascii() and character.isalnum()) or character == "_" else "_"
        for character in stem
    )
    if not sanitized or sanitized[0].isdigit():
        sanitized = "_" + sanitized
    if len(sanitized) > 40 or not _IDENTIFIER.fullmatch(sanitized):
        raise BuildError(f"{stem!r} is not a stem of at most 40 identifier characters")
    return sanitized


@dataclass(frozen=True)
class ProducerIdentity:
    """Who derived a composed module, and the version of what it derives."""

    producer_id: str
    contract_version: str

    def __post_init__(self) -> None:
        if not self.producer_id or not self.contract_version:
            raise BuildError("a producer needs an id and a version")


# -- one module's pins and what it holds ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pins:
    """A module's pins in declared order, its parameters as RTL spells them, and its
    aligned clocks.

    ``ports`` keeps its name although it holds ``Pin`` values: a composed module's name
    carries the digest of its typed fields (``fingerprint``), field names included, so
    renaming the field would rename every composed module."""

    ports: tuple[Pin, ...]
    parameters: tuple[tuple[str, str], ...] = ()
    clock_alignments: tuple[ClockAlignment, ...] = ()

    def __post_init__(self) -> None:
        parameters = tuple(sorted(self.parameters, key=lambda item: item[0]))
        if len(parameters) != len({name for name, _ in parameters}):
            raise BuildError("a module's pins name one parameter twice")
        pins, alignments = tuple(self.ports), tuple(sorted(self.clock_alignments))
        validate_pins(pins, alignments)
        object.__setattr__(self, "ports", pins)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "clock_alignments", alignments)


@dataclass(frozen=True)
class Held:
    """Pins a module leaves out of the netlist: ``inputs`` held constant, as (pin,
    value), and ``unused`` outputs left unconnected."""

    inputs: tuple[tuple[str, int], ...] = ()
    unused: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple((pin, value) for pin, value in self.inputs))
        object.__setattr__(self, "unused", tuple(self.unused))


@dataclass(frozen=True)
class Leaf:
    """One FinnLib module bound to a configuration: everything needed to instantiate and
    build it, and nothing about the Space that derived it."""

    implementation_id: str
    implementation_version: str
    name: str
    parameters: ScalarTable
    pins: Pins
    sources: tuple[CopiedSource, ...] = ()
    data: tuple[GeneratedData, ...] = ()
    held: Held = Held()

    def __post_init__(self) -> None:
        if not self.implementation_id or not self.implementation_version:
            raise BuildError("a module needs an implementation id and version")
        if not _IDENTIFIER.fullmatch(self.name):
            raise BuildError(f"{self.name!r} is not an RTL module identifier")
        parameters = _table(self.parameters, label="module parameter table")
        spelled = tuple((name, _rtl_scalar(value)) for name, value in parameters)
        if self.pins.parameters != spelled:
            raise BuildError(
                "the pins' parameter strings must be the canonical RTL spellings of the "
                "typed module parameter table"
            )
        sources, data = tuple(self.sources), tuple(self.data)
        if any(not isinstance(item, CopiedSource) for item in sources):
            raise BuildError("a module's sources are copied sources")
        if any(not isinstance(item, GeneratedData) for item in data):
            raise BuildError("a module's data files are generated data")
        pins = abi_pins(self.pins.ports)
        for pin, value in self.held.inputs:
            info = pins.get(pin)
            if info is None or info.direction is not Direction.IN:
                raise BuildError(f"{self.name} holds {pin!r}, which is none of its inputs")
            if not 0 <= value < 1 << info.width:
                raise BuildError(f"{self.name}.{pin} cannot hold {value} in {info.width} bits")
        for pin in self.held.unused:
            info = pins.get(pin)
            if info is None or info.direction is not Direction.OUT:
                raise BuildError(f"{self.name} leaves {pin!r} unused, which is none of its outputs")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "data", data)


# -- a flat netlist ------------------------------------------------------------------------


def _beside(prefix: str, label: str) -> str:
    """``label`` under ``prefix``: ``prefix.label``; ``^label``, a node beside the
    fragment's owner, is placed beside ``prefix``'s last node."""
    if label.startswith("^"):
        parent, _, _ = prefix.rpartition(".")
        return f"{parent}.{label[1:]}" if parent else label[1:]
    return f"{prefix}.{label}" if label else prefix


@dataclass(frozen=True)
class LinkEnd:
    """One end of a link: an instance's ready/valid pins, or the root's own when
    ``instance`` is None. ``data_bits`` is the data pin's width."""

    instance: str | None
    data: str
    data_bits: int
    valid: str
    ready: str

    def __post_init__(self) -> None:
        if not all((self.data, self.valid, self.ready)):
            raise BuildError("a link end names its data, valid and ready pins")
        if type(self.data_bits) is not int or self.data_bits < 1:
            raise BuildError(f"{self.data}: a data pin is at least one bit")

    def under(self, prefix: str) -> LinkEnd:
        return (
            self
            if self.instance is None
            else replace(self, instance=_beside(prefix, self.instance))
        )


Marker = tuple[str, Union[int, None], str, Union[int, None]]
"""(source pin, bit, sink pin, bit); a bit of None is a one-bit marker pin."""


@dataclass(frozen=True)
class Link:
    """One stream hop: sink lane ``i`` takes source lane ``lanes[i]``, lane zero least
    significant, each ``lane_bits`` wide; valid forward, ready back, and each marker pair
    from source to sink."""

    source: LinkEnd
    sink: LinkEnd
    lane_bits: int
    lanes: tuple[int, ...]
    markers: tuple[Marker, ...] = ()

    def __post_init__(self) -> None:
        lanes = tuple(self.lanes)
        if type(self.lane_bits) is not int or self.lane_bits < 1 or not lanes:
            raise BuildError("a link carries at least one lane of at least one bit")
        if min(lanes) < 0 or (max(lanes) + 1) * self.lane_bits > self.source.data_bits:
            raise BuildError(f"{self.source.data}: a lane lies outside the source word")
        if len(lanes) * self.lane_bits > self.sink.data_bits:
            raise BuildError(f"{self.sink.data}: the lanes exceed the sink word")
        object.__setattr__(self, "lanes", lanes)
        object.__setattr__(self, "markers", tuple(tuple(pair) for pair in self.markers))

    @property
    def payload_bits(self) -> int:
        return len(self.lanes) * self.lane_bits

    def under(self, prefix: str) -> Link:
        return replace(self, source=self.source.under(prefix), sink=self.sink.under(prefix))


@dataclass(frozen=True)
class BusExport:
    """An instance's bus, presented as the root's target port ``port`` (its members
    ``<port>_<MEMBER>``)."""

    instance: str
    bus: Bus
    port: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.port):
            raise BuildError(f"{self.port!r} is not a port identifier")


@dataclass(frozen=True)
class Fragment:
    """Leaf instances by label, the links between their pins, and the buses they present."""

    instances: tuple[tuple[str, Leaf], ...] = ()
    links: tuple[Link, ...] = ()
    exports: tuple[BusExport, ...] = ()

    def __post_init__(self) -> None:
        instances, links, exports = tuple(self.instances), tuple(self.links), tuple(self.exports)
        labels = [label for label, _ in instances]
        if len(labels) != len(set(labels)):
            twice = sorted({label for label in labels if labels.count(label) > 1})
            raise BuildError(f"a netlist places {twice} twice")
        ports = [item.port for item in exports]
        if len(ports) != len(set(ports)):
            twice = sorted({port for port in ports if ports.count(port) > 1})
            raise BuildError(f"a netlist presents the ports {twice} twice")
        if any(not isinstance(leaf, Leaf) for _, leaf in instances):
            raise BuildError("a netlist instantiates leaves only")
        object.__setattr__(self, "instances", instances)
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "exports", exports)

    def under(self, prefix: str) -> Fragment:
        """This fragment placed at node ``prefix``: labels and link ends below it, each
        presented port prefixed ``<prefix>_`` (``first_s_axilite``)."""
        port = prefix.replace(".", "_")
        return Fragment(
            tuple((_beside(prefix, label), leaf) for label, leaf in self.instances),
            tuple(link.under(prefix) for link in self.links),
            tuple(
                replace(item, instance=_beside(prefix, item.instance), port=f"{port}_{item.port}")
                for item in self.exports
            ),
        )


def merge(*fragments: Fragment) -> Fragment:
    """The fragments side by side: every instance, link and presented bus of each."""
    return Fragment(
        tuple(item for fragment in fragments for item in fragment.instances),
        tuple(item for fragment in fragments for item in fragment.links),
        tuple(item for fragment in fragments for item in fragment.exports),
    )


@dataclass(frozen=True)
class Composed:
    """A flat netlist with pins: one generated module. Well formed when its links name
    pins that exist and every instance input and root output has exactly one driver."""

    implementation_id: str
    implementation_version: str
    stem: str
    pins: Pins
    fragment: Fragment

    def __post_init__(self) -> None:
        if not self.implementation_id or not self.implementation_version:
            raise BuildError("a module needs an implementation id and version")
        sanitize_stem(self.stem)
        root = abi_pins(self.pins.ports)
        leaves = dict(self.fragment.instances)
        for label in leaves:
            if not _LABEL.fullmatch(label):
                raise BuildError(f"{label!r} is not an instance label of the root")

        def pins_of(instance: str | None, what: str) -> Mapping[str, PinInfo]:
            if instance is None:
                return root
            leaf = leaves.get(instance)
            if leaf is None:
                raise BuildError(f"{what} names {instance!r}, which the netlist does not place")
            return abi_pins(leaf.pins.ports)

        for link in self.fragment.links:
            for end in (link.source, link.sink):
                pins = pins_of(end.instance, "a link")
                where = end.instance or "the root"
                missing = [pin for pin in (end.data, end.valid, end.ready) if pin not in pins]
                if missing:
                    raise BuildError(f"{where} has no pins {missing}")
                if pins[end.data].width != end.data_bits:
                    raise BuildError(f"{where}.{end.data} is not {end.data_bits} bits")
            for source, _, sink, _ in link.markers:
                for instance, pin in ((link.source.instance, source), (link.sink.instance, sink)):
                    if pin not in pins_of(instance, "a marker"):
                        raise BuildError(f"{instance or 'the root'} has no marker pin {pin!r}")
        for item in self.fragment.exports:
            pins = pins_of(item.instance, "a presented bus")
            absent = [member.physical for member in item.bus.signals if member.physical not in pins]
            if absent:
                raise BuildError(f"{item.instance} has no pins {absent} of {item.bus.name}")
            outer = [f"{item.port}_{member.logical.upper()}" for member in item.bus.signals]
            if any(pin not in root for pin in outer):
                raise BuildError(f"the root has no pins {outer} to present {item.bus.name} at")
        self._driven_once(
            root, {label: abi_pins(leaf.pins.ports) for label, leaf in leaves.items()}
        )

    def _driven_once(
        self, root: Mapping[str, PinInfo], children: Mapping[str, Mapping[str, PinInfo]]
    ) -> None:
        """Every instance input and every root output bit has exactly one driver: a link,
        a held value, a clock or reset by role, or a presented bus."""
        drivers: dict[tuple[str | None, str, int], int] = {}

        def drive(instance: str | None, pin: str, bits: range | None = None) -> None:
            info = (root if instance is None else children[instance])[pin]
            for bit in bits if bits is not None else range(info.width):
                key = (instance, pin, bit)
                drivers[key] = drivers.get(key, 0) + 1

        for link in self.fragment.links:
            drive(link.sink.instance, link.sink.data)
            drive(link.sink.instance, link.sink.valid)
            drive(link.source.instance, link.source.ready)
            for _, _, pin, bit in link.markers:
                drive(link.sink.instance, pin, None if bit is None else range(bit, bit + 1))
        for label, leaf in self.fragment.instances:
            held = {pin for pin, _ in leaf.held.inputs}
            for pin in held:
                drive(label, pin)
            for pin, info in children[label].items():
                by_role = isinstance(info.role, (Clock, Reset)) and info.bus is None
                if info.direction is Direction.IN and by_role and pin not in held:
                    drive(label, pin)
        for item in self.fragment.exports:
            directions = dict(item.bus.member_directions())
            for member in item.bus.signals:
                if directions[member.physical] is Direction.IN:
                    drive(item.instance, member.physical)
                else:
                    drive(None, f"{item.port}_{member.logical.upper()}")
        wanted = [
            (None, pin, info) for pin, info in root.items() if info.direction is Direction.OUT
        ] + [
            (label, pin, info)
            for label, pins in children.items()
            for pin, info in pins.items()
            if info.direction is Direction.IN
        ]
        for instance, pin, info in wanted:
            counts = {drivers.get((instance, pin, bit), 0) for bit in range(info.width)}
            if counts != {1}:
                problem = "nothing drives" if 0 in counts else "more than one driver drives"
                where = "the root" if instance is None else instance
                raise BuildError(f"{problem} {where}.{pin}")


Module = Union[Leaf, Composed]


def fingerprint(module: Module) -> str:
    """The digest of everything the module says; equal modules share it."""
    return digest(("module-v1", typed_canonical(module)))


def module_name(module: Module) -> str:
    """A leaf's own name; a composed module's ``<stem>__<fingerprint[:16]>``."""
    if isinstance(module, Leaf):
        return module.name
    return f"{sanitize_stem(module.stem)}__{fingerprint(module)[:16]}"


__all__ = [
    "BuildError",
    "BusExport",
    "Composed",
    "Fragment",
    "Held",
    "Leaf",
    "Link",
    "LinkEnd",
    "Marker",
    "Module",
    "Pins",
    "ProducerIdentity",
    "Scalar",
    "ScalarTable",
    "fingerprint",
    "merge",
    "module_name",
    "sanitize_stem",
    "typed_canonical",
]
