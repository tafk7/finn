# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Emitting a module: its sources and data files, written to a directory in compile order.

The one step that touches the filesystem, in two parts:

- **Codegen.** Every leaf's copied sources are read from named roots
  (``finnlib``), so a module never carries a checkout path. Each path is
  staged once and providers come first (``sources.ordered``); two different
  files claiming one path or one symbol are refused. Each data file is written
  once.
- **Netlist.** A ``Composed`` module adds one SystemVerilog module, named
  ``<stem>__<fingerprint>`` (``module.module_name``) and written here from its
  value: its ports from its pins; one net per instance input and per output
  something reads; per link, the data lanes (lane zero least significant), the
  sink's padding driven zero (a root output carries the source's own padding),
  valid forward, ready back and each marker bit (a constant one tied high);
  each leaf's held inputs tied and its other outputs left open; each instance
  clock and reset pin driven by its role (a free clock from the root's, a clock
  at twice it from the root's doubled clock, a reset from the root's, inverted
  when the polarities differ);
  each presented bus wired member by member to ``<port>_<MEMBER>``.

Nothing here decides or checks what a configuration may be: a module arrives
valid from the Space that derived it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    Derived,
    Direction,
    Free,
    Reset,
    Signal,
    abi_pins,
)
from finn.kernels.artifacts.module import (
    BuildError,
    Composed,
    Leaf,
    Link,
    LinkEnd,
    Module,
    module_name,
)
from finn.kernels.artifacts.projection import content_digest
from finn.kernels.artifacts.sources import SourceError, SourceFile, ordered


@dataclass(frozen=True)
class EmittedModule:
    """A module written to ``directory``: its name, its sources in compile order and
    its data files, each relative to ``directory``."""

    entry_point: str
    directory: Path
    sources: tuple[str, ...]
    data: tuple[str, ...] = ()


def _read(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise BuildError(f"{label} {path} cannot be read") from error


def _leaves(module: Module) -> tuple[Leaf, ...]:
    if isinstance(module, Leaf):
        return (module,)
    return tuple(leaf for _, leaf in module.fragment.instances)


def emit_module(module: Module, directory: Path, *, roots: Mapping[str, Path]) -> EmittedModule:
    """Write the module's sources and data files into ``directory``; ``roots`` resolves
    each copied source's root."""

    drafts: list[tuple[SourceFile, bytes]] = []
    data: dict[str, bytes] = {}
    read: dict[tuple[str, str], bytes] = {}
    for leaf in _leaves(module):
        for item in leaf.sources:
            key = (item.root, item.path)
            if key not in read:
                root = roots.get(item.root)
                if root is None:
                    raise BuildError(f"no source root resolves {item.root!r}")
                read[key] = _read(Path(root) / item.path, "copied source")
            content = read[key]
            file = SourceFile(item.path, content_digest(content), item.provides, item.requires)
            drafts.append((file, content))
        for datum in leaf.data:
            if data.setdefault(datum.path, datum.data) != datum.data:
                raise BuildError(f"two different data files are named {datum.path}")
    name = module_name(module)
    if isinstance(module, Composed):
        text = netlist(module, name).encode()
        drafts.append((SourceFile(f"{name}.sv", content_digest(text), (f"module:{name}",)), text))
    if not drafts:
        raise BuildError("a module has at least one source")
    try:
        order = ordered([file for file, _ in drafts])
    except SourceError as error:
        raise BuildError(str(error)) from error
    contents = dict(drafts)

    directory.mkdir(parents=True, exist_ok=True)
    for file in order:
        target = directory / file.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents[file])
    written = tuple(file.path for file in order)
    for path, content in data.items():
        if path in written:
            raise BuildError(f"a data file and a source are both named {path}")
        (directory / path).write_bytes(content)
    return EmittedModule(name, directory, written, tuple(data))


# -- the netlist ---------------------------------------------------------------------------


def _width(width: int) -> str:
    return "" if width == 1 else f" [{width - 1}:0]"


def _bits(net: str, width: int, offset: int = 0, bits: int | None = None) -> str:
    """``bits`` of ``net`` (``width`` wide) from ``offset``; the whole net by default."""
    bits = width if bits is None else bits
    if bits == width and offset == 0:
        return net
    if bits == 1:
        return f"{net}[{offset}]"
    return f"{net}[{offset + bits - 1}:{offset}]"


def _constant(width: int, value: int) -> str:
    return f"{width}'h{value:x}"


def instance_name(label: str) -> str:
    """The netlist's instance of the module's instance ``label``: what a synthesis
    report names it by."""
    return "u_" + label.replace(".", "_")


def instance_net(label: str, pin: str) -> str:
    """The net of instance ``label``'s ``pin`` in the netlist: what a testbench reads, by
    hierarchical name, to observe a link at that end. Present for every input pin and for
    every output pin something reads."""
    return f"n__{instance_name(label)}__{pin}"


class _Netlist:
    """The text of one composed module, gathered section by section."""

    def __init__(self, module: Composed) -> None:
        self.module = module
        self.root = abi_pins(module.abi.pins)
        self.pins = {label: abi_pins(leaf.abi.pins) for label, leaf in module.fragment.instances}
        self.read: set[tuple[str, str]] = set()
        self.assigns: list[str] = []

    def net(self, instance: str | None, pin: str) -> str:
        if instance is None:
            return pin
        if self.pins[instance][pin].direction is Direction.OUT:
            self.read.add((instance, pin))
        return instance_net(instance, pin)

    def width(self, instance: str | None, pin: str) -> int:
        return (self.root if instance is None else self.pins[instance])[pin].width

    def assign(self, destination: str, source: str) -> None:
        self.assigns.append(f"    assign {destination} = {source};")

    def link(self, link: Link) -> None:
        source, sink, bits = link.source, link.sink, link.lane_bits
        out, into = self.net(source.instance, source.data), self.net(sink.instance, sink.data)
        lanes = link.lanes
        lane = 0
        while lane < len(lanes):
            # A run of consecutive source lanes is one assignment.
            run = 1
            while lane + run < len(lanes) and lanes[lane + run] == lanes[lane] + run:
                run += 1
            self.assign(
                _bits(into, sink.data_bits, lane * bits, run * bits),
                _bits(out, source.data_bits, lanes[lane] * bits, run * bits),
            )
            lane += run
        payload = link.payload_bits
        if sink.data_bits > payload:
            # A root output carries the source's own padding; an instance gets zeros.
            carried = (
                min(sink.data_bits, source.data_bits) - payload if sink.instance is None else 0
            )
            if carried:
                self.assign(
                    _bits(into, sink.data_bits, payload, carried),
                    _bits(out, source.data_bits, payload, carried),
                )
            zeros = sink.data_bits - payload - carried
            if zeros:
                self.assign(
                    _bits(into, sink.data_bits, payload + carried, zeros), _constant(zeros, 0)
                )
        self.assign(self.net(sink.instance, sink.valid), self.net(source.instance, source.valid))
        self.assign(self.net(source.instance, source.ready), self.net(sink.instance, sink.ready))
        for produced, produced_bit, consumed, consumed_bit in link.markers:
            self.assign(
                self._marker(sink, consumed, consumed_bit),
                _constant(1, 1)
                if produced is None
                else self._marker(source, produced, produced_bit),
            )

    def _marker(self, end: LinkEnd, pin: str, bit: int | None) -> str:
        width = self.width(end.instance, pin)
        return _bits(self.net(end.instance, pin), width, bit or 0, 1)

    def roles(self) -> dict[str, tuple[str, bool]]:
        """The root's clock, doubled clock and reset pins by role, and whether its reset
        is active low."""
        found: dict[str, tuple[str, bool]] = {}
        for pin in self.module.abi.pins:
            if not isinstance(pin, Signal) or pin.direction is not Direction.IN:
                continue
            role = pin.role
            if isinstance(role, Clock):
                key = "clock" if isinstance(role.rate, Free) else "doubled"
                found.setdefault(key, (pin.name, True))
            elif isinstance(role, Reset):
                found.setdefault("reset", (pin.name, role.active_low))
        return found

    def drive(self, label: str, leaf: Leaf) -> None:
        """Hold what the leaf holds; drive each clock and reset pin by its role."""
        held = dict(leaf.held.inputs)
        roles = self.roles()
        for name, info in self.pins[label].items():
            if info.direction is not Direction.IN:
                continue
            net = self.net(label, name)
            if name in held:
                self.assign(net, _constant(info.width, held[name]))
                continue
            if info.bus is not None:
                continue
            role = info.role
            if isinstance(role, Clock):
                rate = role.rate
                key = "doubled" if isinstance(rate, Derived) else "clock"
                if isinstance(rate, Derived) and rate.ratio != 2:
                    raise BuildError(
                        f"{label}.{name}: only a clock at twice the root's is supplied"
                    )
                if key not in roles:
                    raise BuildError(f"{label}.{name}: the root has no {key} clock")
                self.assign(net, roles[key][0])
            elif isinstance(role, Reset):
                if "reset" not in roles:
                    raise BuildError(f"{label}.{name}: the root has no reset")
                reset, active_low = roles["reset"]
                self.assign(net, reset if active_low == role.active_low else f"!{reset}")

    def present(self, instance: str, bus: Bus, port: str) -> None:
        directions = dict(bus.member_directions())
        for member in bus.signals:
            inner, outer = self.net(instance, member.physical), f"{port}_{member.logical.upper()}"
            if directions[member.physical] is Direction.IN:
                self.assign(inner, outer)
            else:
                self.assign(outer, inner)

    def text(self, name: str) -> str:
        fragment = self.module.fragment
        for item in fragment.links:
            self.link(item)
        for label, leaf in fragment.instances:
            self.drive(label, leaf)
        for export in fragment.exports:
            self.present(export.instance, export.bus, export.port)
        ports = ",\n".join(
            f"    {info.direction.value} logic{_width(info.width)} {pin}"
            for pin, info in self.root.items()
        )
        nets: list[str] = []
        blocks: list[str] = []
        for label, leaf in fragment.instances:
            connections = []
            for pin, info in self.pins[label].items():
                used = info.direction is Direction.IN or (label, pin) in self.read
                if used:
                    nets.append(f"    logic{_width(info.width)} {instance_net(label, pin)};")
                connections.append(f"        .{pin}({instance_net(label, pin) if used else ''})")
            parameters = ""
            if leaf.abi.parameters:
                parameters = (
                    " #(\n"
                    + ",\n".join(f"        .{key}({value})" for key, value in leaf.abi.parameters)
                    + "\n    )"
                )
            blocks.append(
                f"    {leaf.name}{parameters} {instance_name(label)} (\n"
                + ",\n".join(connections)
                + "\n    );"
            )
        sections = ["\n".join(nets), "\n".join(self.assigns), "\n\n".join(blocks)]
        body = "\n\n".join(section for section in sections if section)
        return (
            "// Generated by finn.kernels.artifacts.build -- do not edit.\n"
            "// A flat netlist of FinnLib modules.\n"
            f"module {name} (\n{ports}\n);\n{body}\nendmodule\n"
        )


def netlist(module: Composed, name: str) -> str:
    """The SystemVerilog text of ``module`` under ``name``."""
    return _Netlist(module).text(name)


__all__ = ["EmittedModule", "emit_module", "instance_name", "instance_net", "netlist"]
