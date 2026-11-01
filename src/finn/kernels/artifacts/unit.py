# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The core unit: a core and the grouper of its control buses, one module behind one
control port.

A shell that places a core presenting AXI-Lite buses places it as a unit, reached by one
control port, ``s_axi_control`` (``CONTROL``), its address the bits of the unit's window
(``finn.kernels.grouper``: ``map``, each bus's offset in the unit, and ``window``). The
unit's module (``CoreUnit``) holds two instances: the core's module, by its name (``core``),
and, for two buses or more, FinnLib's AXI-Lite decoder (``DECODER_SOURCE``), each of its
slaves wired member by member to one of the core's buses at its offset, its address
truncated to the bus's own bits (each window is aligned to itself, so its low bits are the
offset inside it). One bus is the unit's port, renamed, its address truncated likewise: no
decoder, so an address outside the bus's window is the bus's to answer. Every other pin of
the core is the unit's, of the same name.

A bus the unit wires is an AXI-Lite target of 32-bit data with every member but the
protections (``unsupported``): the decoder routes each, and a protection the bus does not
take is left open.

The unit is named as its content (``name``): ``finn_coreunit_<digest>``, the digest of its
text with its name left out and of the decoder's bytes, read once from FinnLib
(``emit_unit``), so one content has one name. The core's module is not among its files: the
unit's IP holds the core's as a subcore.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from finn.kernels.artifacts.abi import (
    CLOCK,
    RESET,
    Bus,
    Endpoint,
    Member,
    Pin,
    StandardProtocol,
    abi_pins,
    bus_window,
)
from finn.kernels.artifacts.build import EmittedModule
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.module import Abi, BuildError
from finn.kernels.artifacts.projection import content_digest, digest
from finn.kernels.artifacts.top import top_bus

CONTROL = "s_axi_control"
"""A unit's one control port."""

STEM = "finn_coreunit"
"""The stem of a unit's module name."""

CORE = "u_core"
"""The core's instance in its unit."""

GROUPER = "u_grouper"
"""The decoder's instance in its unit, where it has one (two buses or more)."""

DECODER_SOURCE = CopiedSource(
    "finnlib", "rtl/infra/axilite_decoder.sv", provides=("module:axilite_decoder",)
)
"""FinnLib's AXI-Lite decoder: one master to ``N`` slaves at given windows."""

#: The members of an AXI-Lite bus the unit wires, with their widths where fixed; the
#: protections (``PROTECTIONS``) where the bus takes them.
MEMBERS: Mapping[str, int | None] = {
    "awaddr": None,
    "awvalid": 1,
    "awready": 1,
    "wdata": 32,
    "wstrb": 4,
    "wvalid": 1,
    "wready": 1,
    "bresp": 2,
    "bvalid": 1,
    "bready": 1,
    "araddr": None,
    "arvalid": 1,
    "arready": 1,
    "rdata": 32,
    "rresp": 2,
    "rvalid": 1,
    "rready": 1,
}
PROTECTIONS = ("awprot", "arprot")

#: The decoder's master members its slaves share (``m_awaddr``); each other is routed, one
#: bit or one word a slave (``m_awvalid[i]``).
_SHARED = ("awaddr", "wdata", "wstrb", "araddr", *PROTECTIONS)


def _widths(address_bits: int) -> dict[str, int]:
    """Each member's width, the protections among them, its addresses ``address_bits``."""
    found = {name: width or address_bits for name, width in MEMBERS.items()}
    return {**found, "awprot": 3, "arprot": 3}


def unsupported(bus: Bus) -> str | None:
    """Why the unit cannot wire ``bus`` behind its port: not an AXI-Lite target, a member
    missing, or a member of another width than the unit's (``MEMBERS``: 32-bit data, its
    two addresses one width, a protection three bits); None where it can."""
    if bus.protocol is not StandardProtocol.AXILITE or bus.endpoint is not Endpoint.TARGET:
        return f"{bus.name} is no AXI-Lite target"
    widths = {member.logical: member.width for member in bus.signals}
    missing = [name for name in MEMBERS if name not in widths]
    if missing:
        return f"{bus.name} has no {', '.join(missing)}"
    wrong = [
        f"{name} of {widths[name]} bits, not {width}"
        for name, width in MEMBERS.items()
        if width is not None and widths[name] != width
    ]
    wrong += [
        f"{name} of {widths[name]} bits, not 3"
        for name in PROTECTIONS
        if name in widths and widths[name] != 3
    ]
    if widths["awaddr"] != widths["araddr"]:
        wrong.append(f"araddr of {widths['araddr']} bits, not awaddr's {widths['awaddr']}")
    return f"{bus.name}: {'; '.join(wrong)}" if wrong else None


@dataclass(frozen=True)
class CoreUnit:
    """The unit of the core whose module is named ``core``, of the ABI ``core_abi``: its
    AXI-Lite targets, in order, at the offsets ``map`` in the unit's ``window`` (the
    grouper's). A ``BuildError`` where it cannot be made: no bus, a map of another length, a
    bus the unit cannot wire (``unsupported``), or one not aligned to its window or outside
    the unit's."""

    core: str
    core_abi: Abi
    map: tuple[int, ...]
    window: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "map", tuple(self.map))
        buses = self.buses
        if not buses:
            raise BuildError(f"{self.core} presents no AXI-Lite bus: its unit is the core")
        if len(self.map) != len(buses):
            raise BuildError(
                f"{self.core}: {len(self.map)} offsets for its {len(buses)} AXI-Lite buses"
            )
        if self.window < 1 or self.window & (self.window - 1):
            raise BuildError(f"a unit's window is a power of two, not {self.window}")
        for bus, offset in zip(buses, self.map, strict=True):
            reason = unsupported(bus)
            if reason is not None:
                raise BuildError(f"{self.core}: {reason}")
            size = bus_window(bus)
            assert size is not None  # it has an awaddr
            if offset % size or offset + size > self.window:
                raise BuildError(
                    f"{self.core}.{bus.name}: {size} bytes at {offset:#x} is not aligned to "
                    f"itself within the unit's {self.window}"
                )

    @property
    def pins(self) -> tuple[Pin, ...]:
        """The core's pins."""
        return self.core_abi.pins

    @property
    def buses(self) -> tuple[Bus, ...]:
        """The core's AXI-Lite targets, in its pins' order: the buses behind the port."""
        return tuple(
            pin
            for pin in self.pins
            if isinstance(pin, Bus)
            and pin.protocol is StandardProtocol.AXILITE
            and pin.endpoint is Endpoint.TARGET
        )

    @property
    def address_bits(self) -> int:
        """The bits of the unit's address: its window's."""
        return self.window.bit_length() - 1

    @property
    def port(self) -> Bus:
        """The unit's control port, ``CONTROL``: the one bus's members, or the decoder's
        (every member, the protections among them), its addresses the unit's bits."""
        if len(self.buses) == 1:
            names = [member.logical for member in self.buses[0].signals]
        else:
            names = [*MEMBERS, *PROTECTIONS]
        widths = _widths(self.address_bits)
        members = (Member(name, name, widths[name]) for name in names)
        return top_bus(Bus(CONTROL, StandardProtocol.AXILITE, members), CONTROL, CLOCK, RESET)

    @property
    def abi(self) -> Abi:
        """The unit's pins: the core's, its AXI-Lite targets replaced by its port, where
        the first of them was; its clocks aligned as the core's."""
        buses = self.buses
        first = self.pins.index(buses[0])
        kept = [pin for pin in self.pins if pin not in buses]
        return Abi((*kept[:first], self.port, *kept[first:]), (), self.core_abi.clock_alignments)

    def text(self, name: str) -> str:
        """The unit's module, named ``name``."""
        abi = self.abi
        ports = ",\n".join(
            f"    {info.direction.value} logic{_width(info.width)} {pin}"
            for pin, info in abi_pins(abi.pins).items()
        )
        buses = self.buses
        port = {member.logical: member.physical for member in self.port.signals}
        connections = [
            f"        .{pin}({pin})"
            for pin in abi_pins(self.pins)
            if pin not in {member.physical for bus in buses for member in bus.signals}
        ]
        sections: list[str] = []
        if len(buses) == 1:
            for member in buses[0].signals:
                outer = port[member.logical]
                if member.logical in ("awaddr", "araddr"):
                    outer = f"{outer}[{member.width - 1}:0]"
                connections.append(f"        .{member.physical}({outer})")
        else:
            sections += [_decoder_nets(len(buses), self.address_bits), self._decoder(port)]
            for index, bus in enumerate(buses):
                for member in bus.signals:
                    net = f"m_{member.logical}"
                    if member.logical in ("awaddr", "araddr"):
                        net = f"{net}[{member.width - 1}:0]"
                    elif member.logical not in _SHARED:
                        net = f"{net}[{index}]"
                    connections.append(f"        .{member.physical}({net})")
        sections.append(f"    {self.core} {CORE} (\n" + ",\n".join(connections) + "\n    );")
        body = "\n\n".join(sections)
        return (
            "// Generated by finn.kernels.artifacts.unit -- do not edit.\n"
            "// A core unit: the core and its AXI-Lite buses behind one control port.\n"
            f"module {name} (\n{ports}\n);\n{body}\nendmodule\n"
        )

    def _decoder(self, port: Mapping[str, str]) -> str:
        """The decoder's instance: its slave port the unit's, its master side the nets."""
        bits = self.address_bits
        buses = self.buses
        sizes = [bus_window(bus) for bus in buses]
        base = ", ".join(f"{bits}'h{offset:x}" for offset in reversed(self.map))
        size = ", ".join(f"8'd{(each or 1).bit_length() - 1}" for each in reversed(sizes))
        parameters = [
            f".N({len(buses)})",
            f".ADDR_WIDTH({bits})",
            f".BASE({{{base}}})",
            f".SIZE({{{size}}})",
        ]
        pins = [".aclk(ap_clk)", ".aresetn(ap_rst_n)"]
        pins += [f".s_{name}({port[name]})" for name in (*MEMBERS, *PROTECTIONS)]
        pins += [f".m_{name}(m_{name})" for name in (*MEMBERS, *PROTECTIONS)]
        return (
            "    axilite_decoder #(\n"
            + ",\n".join(f"        {each}" for each in parameters)
            + f"\n    ) {GROUPER} (\n"
            + ",\n".join(f"        {each}" for each in pins)
            + "\n    );"
        )

    def name(self, decoder: bytes | None) -> str:
        """The unit's module name: ``finn_coreunit_<digest>``, of its text with its name left
        out and of the decoder's bytes (``decoder``; None for one bus, which takes none)."""
        found = digest(
            ("coreunit-v1", self.text(STEM), None if decoder is None else content_digest(decoder))
        )
        return f"{STEM}_{found[:16]}"


def _width(width: int) -> str:
    return "" if width == 1 else f" [{width - 1}:0]"


def _decoder_nets(count: int, bits: int) -> str:
    """The decoder's master side as nets: a shared member once, a routed one a bit or a word
    a slave."""
    lines = []
    for name, width in _widths(bits).items():
        if name in _SHARED:
            lines.append(f"    wire{_width(width)} m_{name};")
        else:
            lines.append(f"    wire [{count - 1}:0]{_width(width).lstrip()} m_{name};")
    return "\n".join(lines)


def emit_unit(unit: CoreUnit, directory: Path, *, roots: Mapping[str, Path]) -> EmittedModule:
    """Write the unit's files into ``directory``: the decoder, read once from its root
    (``roots`` resolving ``DECODER_SOURCE``'s by name), for two buses or more, and the
    unit's module, named from them (``CoreUnit.name``). The core's files are not among
    them."""
    files: list[tuple[str, bytes]] = []
    decoder = None
    if len(unit.buses) > 1:
        root = roots.get(DECODER_SOURCE.root)
        if root is None:
            raise BuildError(f"no source root resolves {DECODER_SOURCE.root!r}")
        try:
            decoder = (Path(root) / DECODER_SOURCE.path).read_bytes()
        except OSError as error:
            raise BuildError(f"the decoder {DECODER_SOURCE.path} cannot be read") from error
        files.append((DECODER_SOURCE.path, decoder))
    name = unit.name(decoder)
    files.append((f"{name}.sv", unit.text(name).encode()))
    for path, content in files:
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return EmittedModule(name, directory, tuple(path for path, _ in files))


__all__ = [
    "CONTROL",
    "CORE",
    "DECODER_SOURCE",
    "GROUPER",
    "MEMBERS",
    "PROTECTIONS",
    "STEM",
    "CoreUnit",
    "emit_unit",
    "unsupported",
]
