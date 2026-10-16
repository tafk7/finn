# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The IP-XACT interface script FINN emits, read as records.

``finn.kernels.artifacts.ipxact.interface_tcl`` declares a packaged IP's bus interfaces
as Tcl statements. ``interfaces`` reads those statements back into one ``BusInterface``
per declared interface (its mode, type, port maps, bus parameters and memory map), so a
test states what the interface is and not how the script spells it: a reformatting of
the script does not fail it, and a wrong value in a statement the test does not quote
cannot pass.

Only the statements ``interface_tcl`` emits are read. Any other statement is refused
(``ScriptError``), so a new kind of statement is noticed, and not skipped.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field


class ScriptError(ValueError):
    """A statement the interface script reader does not know, or one out of place."""


@dataclass
class AddressBlock:
    """One ``ipx::add_address_block`` of a memory map.

    The script sets no base address: a block starts at its map's origin, ``0``.
    """

    name: str
    base: int = 0
    range: int | None = None
    width: int | None = None
    usage: str | None = None


@dataclass
class MemoryMap:
    """One ``ipx::add_memory_map`` and its address blocks."""

    name: str
    blocks: list[AddressBlock] = field(default_factory=list)


@dataclass
class BusInterface:
    """One ``ipx::add_bus_interface`` and what the script sets on it."""

    name: str
    mode: str | None = None
    bus_type: str | None = None
    abstraction_type: str | None = None
    #: Logical port -> physical port (``TDATA`` -> ``s_axis_0_tdata``).
    port_maps: dict[str, str] = field(default_factory=dict)
    #: Bus parameter name -> its value, as the script writes it (a string).
    parameters: dict[str, str] = field(default_factory=dict)
    memory_map: MemoryMap | None = None
    #: The memory map this interface refers to (``slave_memory_map_ref``).
    memory_map_ref: str | None = None


_NAME = r"([A-Za-z_][\w.:]*)"
_VALUE = r"(\S+)"

_ADD_BUS = re.compile(rf"set bus \[ipx::add_bus_interface {_NAME} \$core\]")
_BUS_PROPERTY = re.compile(
    rf"set_property (abstraction_type_vlnv|bus_type_vlnv|interface_mode|slave_memory_map_ref)"
    rf" {_VALUE} \$bus"
)
_PORT_MAP = re.compile(rf"set_property physical_name {_VALUE} \[ipx::add_port_map {_NAME} \$bus\]")
_PARAMETER = re.compile(rf"set_property value {_VALUE} \[ipx::add_bus_parameter {_NAME} \$bus\]")
_ADD_MAP = re.compile(rf"set map \[ipx::add_memory_map {_NAME} \$core\]")
_ADD_BLOCK = re.compile(rf"set block \[ipx::add_address_block {_NAME} \$map\]")
_BLOCK_PROPERTY = re.compile(rf"set_property (range|width|usage) {_VALUE} \$block")

#: The statements that stand outside every bus interface's block: before the first, the
#: removal of what ``ipx::package_project`` inferred; after the last, the user's
#: resolution of the parameters.
_PREAMBLE = (
    re.compile(r"set core \[ipx::current_core\]"),
    re.compile(r"foreach item \[ipx::get_(?:bus_interfaces|memory_maps) -of_objects \$core\] \{"),
    re.compile(r" +ipx::remove_(?:bus_interface|memory_map) \[get_property NAME \$item\] \$core"),
    re.compile(r"\}"),
)
_RESOLVE = re.compile(r"set_property value_resolve_type user \[.*\]")


def _refusal(number: int, line: str, why: str) -> ScriptError:
    return ScriptError(f"line {number}, {why}: {line!r}")


def interfaces(tcl: Iterable[str]) -> dict[str, BusInterface]:
    """The bus interfaces the script declares, by name, in the order declared.

    Refuses a statement it does not know, a property set outside the block it belongs
    to, and a name or a parameter declared twice.
    """
    found: dict[str, BusInterface] = {}
    bus: BusInterface | None = None
    block: AddressBlock | None = None
    for number, line in enumerate(tcl, 1):
        if (match := _ADD_BUS.fullmatch(line)) is not None:
            if match[1] in found:
                raise _refusal(number, line, "an interface declared twice")
            bus = found[match[1]] = BusInterface(match[1])
            block = None
        elif (match := _RESOLVE.fullmatch(line)) is not None:
            continue
        elif any(pattern.fullmatch(line) for pattern in _PREAMBLE):
            if bus is not None:
                raise _refusal(number, line, "an inferred interface removed after a declaration")
        elif bus is None:
            raise _refusal(number, line, "a statement outside any interface it does not know")
        elif (match := _BUS_PROPERTY.fullmatch(line)) is not None:
            key, value = match[1], match[2]
            if key == "interface_mode":
                bus.mode = value
            elif key == "bus_type_vlnv":
                bus.bus_type = value
            elif key == "abstraction_type_vlnv":
                bus.abstraction_type = value
            else:
                bus.memory_map_ref = value
        elif (match := _PORT_MAP.fullmatch(line)) is not None:
            if match[2] in bus.port_maps:
                raise _refusal(number, line, f"port {match[2]} mapped twice")
            bus.port_maps[match[2]] = match[1]
        elif (match := _PARAMETER.fullmatch(line)) is not None:
            if match[2] in bus.parameters:
                raise _refusal(number, line, f"parameter {match[2]} set twice")
            bus.parameters[match[2]] = match[1]
        elif (match := _ADD_MAP.fullmatch(line)) is not None:
            bus.memory_map = MemoryMap(match[1])
            block = None
        elif (match := _ADD_BLOCK.fullmatch(line)) is not None:
            if bus.memory_map is None:
                raise _refusal(number, line, "an address block with no memory map")
            block = AddressBlock(match[1])
            bus.memory_map.blocks.append(block)
        elif (match := _BLOCK_PROPERTY.fullmatch(line)) is not None:
            if block is None:
                raise _refusal(number, line, "a block property with no address block")
            key, value = match[1], match[2]
            if key == "usage":
                block.usage = value
            else:
                setattr(block, key, int(value))
        else:
            raise _refusal(number, line, "a statement it does not know")
    return found


__all__ = ["AddressBlock", "BusInterface", "MemoryMap", "ScriptError", "interfaces"]
