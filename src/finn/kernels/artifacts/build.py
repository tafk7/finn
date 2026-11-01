# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Emitting a module: its sources and data files, written to a directory in compile order.

The one step that touches the filesystem, in two parts:

- **Codegen.** Every leaf's copied sources are read from named roots
  (``finnlib``), so a module never carries a checkout path, and an HLS leaf's
  request is taken as its product: the Verilog its synthesis exported, from the
  directory ``built`` names for the request's top (``finn.toolchain.hls``, through
  ``finn.transformation.kernels.hls.built_hls``), staged below a directory named
  by the top, its top file providing the module, and the memory images beside it
  (``*.dat``) written as data files. Each file is read once, into one snapshot
  (``capture``), which the module is named from and its files are written from.
  Each path is staged once and providers come first (``sources.ordered``); two
  different files claiming one path or one symbol are refused. Each data file is
  written once. Before anything is written, two staged names one filesystem may take for
  one file (alike but for case, or one a directory of the other) are refused, and so is a
  destination where a staged name passes through a link: one staged file is one file
  written (``_refuse_aliases``).
- **Netlist.** A ``Composed`` module adds one SystemVerilog module, named
  ``<stem>_<fingerprint>`` (``module.module_name``, of the snapshot) and written here from its
  value: its ports from its pins; one net per instance input and per output
  something reads; per link, the data lanes (lane zero least significant), the
  sink's padding driven zero (a root output carries the source's own padding),
  valid forward, ready back and each marker bit (a constant one tied high);
  each leaf's held inputs tied and its other outputs left open; each instance
  clock and reset pin driven by its role (a free clock from the root's, a clock
  at twice it from the root's doubled clock, a reset from the root's, inverted
  when the polarities differ);
  each presented bus wired member by member to ``<port>_<MEMBER>``.

A module is a detached description, carrying no producer-validity certificate.
The owning build operation checks its source design before calling this emitter;
local source, ABI and compilation checks still belong here.
"""

from __future__ import annotations

import stat
from collections.abc import Mapping, Sequence
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
from finn.kernels.artifacts.contributions import ContributionError, HlsSource, relative_name
from finn.kernels.artifacts.module import (
    BuildError,
    Composed,
    Leaf,
    Link,
    LinkEnd,
    Module,
    Snapshot,
    instance_name,
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


#: The files of an HLS product that are sources, and those that are its memories' images.
HLS_SOURCE_SUFFIXES = (".v",)
HLS_DATA_SUFFIXES = (".dat",)


def _hls_product(request: HlsSource, built: Mapping[str, Path]) -> tuple[tuple[str, bytes], ...]:
    """An HLS request's product, each file's bytes by its name: its exported Verilog, its
    top ``<top>.v`` among it, and the images its memories read (``*.dat``)."""
    function = request.function
    product = built.get(function)
    if product is None:
        raise BuildError(f"no HLS product is given for {function}: synthesize it first")
    names = sorted(path.name for path in Path(product).iterdir() if path.is_file())
    for name in names:
        try:
            relative_name(name, f"the HLS product {product}")
        except ContributionError as error:
            raise BuildError(str(error)) from error
    if f"{function}.v" not in names:
        raise BuildError(f"the HLS product {product} has no top {function}.v")
    unknown = [
        name for name in names if not name.endswith((*HLS_SOURCE_SUFFIXES, *HLS_DATA_SUFFIXES))
    ]
    if unknown:
        raise BuildError(
            f"the HLS product {product} holds files emission does not stage: {unknown}"
        )
    return tuple((name, _read(Path(product) / name, "HLS product file")) for name in names)


def capture(
    module: Module, *, roots: Mapping[str, Path], built: Mapping[str, Path] | None = None
) -> Snapshot:
    """Read every file the module's leaves stage from outside it, each once: each copied
    source from its root, ``roots`` resolving it by name, and each HLS request's product from
    the directory ``built`` names for its top. The module's name (``module_name``) and its
    emitted files are both this snapshot's. Before anything is read, a ``BuildError`` where two
    copied sources would be staged as one path (from two roots, or alike but for case), or one
    under an HLS product's directory (its top's name, whatever the case): one staged file is
    one file read."""
    staged: dict[str, tuple[str, str]] = {}
    tops = {
        item.function.casefold()
        for leaf in _leaves(module)
        for item in leaf.sources
        if isinstance(item, HlsSource)
    }
    for leaf in _leaves(module):
        for item in leaf.sources:
            if isinstance(item, HlsSource):
                continue
            other = staged.setdefault(item.path.casefold(), (item.root, item.path))
            if other[1] != item.path:
                raise BuildError(
                    f"two sources are staged as one file where case is not told apart: "
                    f"{other[1]} of {other[0]!r} and {item.path} of {item.root!r}"
                )
            if other != (item.root, item.path):
                raise BuildError(
                    f"two sources are staged as {item.path}: {other[0]!r}'s and {item.root!r}'s"
                )
            if item.path.split("/")[0].casefold() in tops:
                raise BuildError(
                    f"{item.path} of {item.root!r} would be staged among the HLS product of "
                    f"{item.path.split('/')[0]}"
                )
    copied: dict[tuple[str, str], bytes] = {}
    products: dict[str, tuple[tuple[str, bytes], ...]] = {}
    for leaf in _leaves(module):
        for item in leaf.sources:
            if isinstance(item, HlsSource):
                if item.function not in products:
                    products[item.function] = _hls_product(item, built or {})
                continue
            key = (item.root, item.path)
            if key not in copied:
                root = roots.get(item.root)
                if root is None:
                    raise BuildError(f"no source root resolves {item.root!r}")
                copied[key] = _read(Path(root) / item.path, "copied source")
    return Snapshot(
        tuple((root, path, content) for (root, path), content in copied.items()),
        tuple(products.items()),
    )


def emit_module(
    module: Module,
    directory: Path,
    *,
    roots: Mapping[str, Path],
    built: Mapping[str, Path] | None = None,
) -> EmittedModule:
    """Write the module's sources and data files into ``directory``, from one snapshot of
    them (``capture``: ``roots`` resolves each copied source's root, ``built`` each HLS
    request's product by its top's name), the module named from it."""
    snapshot = capture(module, roots=roots, built=built)
    copied = {(root, path): content for root, path, content in snapshot.copied}
    products = dict(snapshot.products)
    drafts: list[tuple[SourceFile, bytes]] = []
    data: dict[str, bytes] = {}

    def add_data(path: str, content: bytes) -> None:
        if data.setdefault(path, content) != content:
            raise BuildError(f"two different data files are named {path}")

    for leaf in _leaves(module):
        for item in leaf.sources:
            if isinstance(item, HlsSource):
                function = item.function
                for name, content in products[function]:
                    if name.endswith(HLS_DATA_SUFFIXES):
                        add_data(name, content)
                        continue
                    provides = (f"module:{function}",) if name == f"{function}.v" else ()
                    file = SourceFile(f"{function}/{name}", content_digest(content), provides)
                    drafts.append((file, content))
                continue
            content = copied[(item.root, item.path)]
            file = SourceFile(item.path, content_digest(content), item.provides, item.requires)
            drafts.append((file, content))
        for datum in leaf.data:
            add_data(datum.path, datum.data)
    name = module_name(module, snapshot)
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
    written = tuple(file.path for file in order)
    _refuse_aliases(directory, written, tuple(data))

    directory.mkdir(parents=True, exist_ok=True)
    for file in order:
        _place(directory / file.path, contents[file])
    for path, content in data.items():
        _place(directory / path, content)
    return EmittedModule(name, directory, written, tuple(data))


def _refuse_aliases(directory: Path, sources: Sequence[str], data: Sequence[str]) -> None:
    """Refuse, before anything is written, staged names that would reach one file twice: a
    data file named as a source; two names alike but for case, which a filesystem that does not
    tell case apart takes for one; a file's name that another's names as its directory; a
    ``directory`` that is, or lies below, a link or a file (the path to it, from the root, is
    checked as it is); and, in ``directory`` as it is, a staged name that passes through a link
    or a file, or ends at a directory (a link there would lead a staged name to another staged
    file's place, or out of the module's directory)."""
    for path in data:
        if path in sources:
            raise BuildError(f"a data file and a source are both named {path}")
    above_directory = directory.absolute()
    for here in (*reversed(above_directory.parents), above_directory):
        try:
            mode = here.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            kind = "a link" if stat.S_ISLNK(mode) else "a file"
            raise BuildError(
                f"{here} is {kind}, which the module's directory {directory} would be written "
                "through: emit into a directory without links"
            )
    names: dict[str, str] = {}
    for path in (*sources, *data):
        other = names.setdefault(path.casefold(), path)
        if other != path:
            raise BuildError(f"{other} and {path} are one file where case is not told apart")
    for path in (*sources, *data):
        parts = path.split("/")
        for depth in range(1, len(parts)):
            above = "/".join(parts[:depth]).casefold()
            if above in names:
                raise BuildError(f"{names[above]} is a file, and a directory of {path}")
        here = directory
        for depth, part in enumerate(parts, start=1):
            here = here / part
            try:
                mode = here.lstat().st_mode
            except FileNotFoundError:
                break
            if depth == len(parts):
                if stat.S_ISDIR(mode):
                    raise BuildError(f"{here} is a directory, where {path} would be written")
            elif stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                kind = "a link" if stat.S_ISLNK(mode) else "a file"
                raise BuildError(
                    f"{here} is {kind}, which {path} would be written through: emit into a "
                    "directory without links"
                )


def _place(target: Path, content: bytes) -> None:
    """Write ``content`` as a new file at ``target``, its directories made: what was there, a
    link or a file linked elsewhere among it, is replaced, not written through."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    target.write_bytes(content)


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


__all__ = [
    "EmittedModule",
    "capture",
    "emit_module",
    "instance_net",
    "netlist",
]
