# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``pynq`` shell's linker: the Zynq block design of an integration description
(``finn.shells.integration``), the ``vivado-block-design`` linker's code.

It takes every name the platform gives and places every control bus:

- the cells: each unit's is its name (a core's its node's, a mover's ``in<i>`` or
  ``out<j>``), its pins the description's, which are its IP's own; the fixed shell's are
  the template's (``STATIC_INSTANCES``);
- the wires (``wiring``): each stream from its source pin to its sink pin; each memory port
  to the memory interconnect's next slave port (``S<nn>_AXI``); each unit's control port
  (``s_axi_control``), in the description's order (the input movers, the core, the output
  movers), from the control tree's next master port (``M<nn>_AXI``): one a unit with a
  window, a core unit's buses behind its grouper; every cell's clock and reset from the
  memory interconnect's;
- the control tree (``control_tree``): SmartConnects (``LOW_AREA``, one slave port), each
  giving the fixed shell's ``control_fanout`` ports at most
  (``finn.kernels.shell_rules.interconnect_tree``, by which the static region is costed),
  named ``control_<level>_<index>``, the root's slave port the processor's control master
  (``CONTROL_MASTERS``), each level's ports the next level's slave ports, filled in order,
  the last level's the units'; each on the row's main clock and the template's reset
  (``rst_ps``, wired by hand: SmartConnect takes no clock automation);
- the placement (``placement``): each unit's window from the fixed shell's
  ``control_base``, in that order, aligned to itself
  (``finn.kernels.shell_rules.placed_windows``, by which the shell's control window rule
  admitted them); the block design assigns each unit's port its address from it, and it
  returns each unit's base (``unit_bases``, ``Linked.placement``), the one Vivado was
  given;
- the board (``project_script``): its board part, the board's declared fact
  (``board_part``), and its processor's family, the fixed shell's processor's
  (``PROCESSORS``).

It wires one AXI-Lite bus and one memory port a mover: a mover whose contract states other
counts is refused, by name (``wire_one_bus_each``), as is a control window below the row's
least window (``range-below-least``), and a control interconnect other than SmartConnect
(``control-interconnect-unsupported``). The control tree takes any number of units: the
shell's rules bound them by the processor's window. Its template wires the memory
interconnect into the processor's ``HP0`` port and runs every cell on the row's main clock:
a memory port bound to another target, a unit on another clock and a description with no
memory port are refused, each by name (``wiring``), as is a processor or board it cannot
select (``project_script``).

``BlockDesignLinker`` is the declared code. Its ``check`` refuses what the block design does
not wire, and a board or processor the project cannot select, before any tool runs.
``link`` writes the block design (``block_design``) into
``templates.custom_zynq_shell_template`` (``project_script``) in a new project under the
directory it is given, runs Vivado on it and collects what it made (``collect``): the
bitfile, its hardware handoff beside it, the main clock as the routed design delivers it
and the design's timing checks there, read from the routed timing report, an image that
misses setup, hold or pulse width refused (``admitted``: Vivado writes the bitfile of a
design that misses timing), and the reports, each by its path under a build's
``report/``: the routed timing report (``TIMING``), the template's ``impl_1``
hierarchical utilization report (``PLACED``), each IP's out-of-context synthesis report
(under ``OUT_OF_CONTEXT``), and what it named and placed (``BLOCK_DESIGN``, ``report``).
Every unit's IP is packaged before it links.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finn import resources
from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.kernels.artifacts.unit import CONTROL
from finn.kernels.movers import MoverContract
from finn.kernels.shell_rules import interconnect_tree, placed_windows
from finn.platform.shells import VIVADO_BLOCK_DESIGN, ShellRow, shell_row
from finn.shells.integration import MAIN, Integration, IntegrationError, Linked, Timing, Unit
from finn.shells.pynq import templates
from finn.toolchain.launcher import StrPath, Toolchain, ToolFailed
from finn.toolchain.vivado import PL_CLOCKS, routed_timing, tcl_quote, vivado_jobs

_LOG = logging.getLogger(__name__)

#: The name FINN declares this linker under (``[linkers.vivado-block-design]``).
LINKER = "vivado-block-design"

#: The project the template creates (``create_project finn_zynq_link``).
PROJECT = "finn_zynq_link"

#: The linker's reports, by their paths under a build's ``report/`` (``Linked.reports``):
#: the routed timing summary, the template's ``impl_1`` hierarchical utilization, the
#: directory of each IP's out-of-context synthesis utilization, and what it named and
#: placed (``report``).
TIMING = "post_route_timing.rpt"
PLACED = "placed_utilization.xml"
OUT_OF_CONTEXT = "out_of_context"
BLOCK_DESIGN = "block_design.json"

#: The block design's instance of each part of the fixed shell, by its role
#: (``FixedShell.resources``); the control interconnect's is its tree's instances' prefix
#: (``control_<level>_<index>``).
STATIC_INSTANCES = {
    "processor": "zynq_ps",
    "reset": "rst_ps",
    "memory_interconnect": "smartconnect_0",
    "control_interconnect": "control_",
}

CONTROL_IP = "smartconnect"
"""The control tree's IP: the template's ``$smartconnect_vlnv``."""

MEMORY_TARGET = "HP0"
"""The memory target the template wires the memory interconnect into: the processor's
``HP0`` port."""

PROCESSORS = {"zynq_ultra_ps_e": "zynq_us+", "processing_system7": "zynq_7000"}
"""The template's processor family (its ``ZYNQ_TYPE``) by the fixed shell's processor IP."""

CONTROL_MASTERS = {"zynq_us+": "M_AXI_HPM0_FPD", "zynq_7000": "M_AXI_GP0"}
"""The processor's control master, which the control tree's root takes, by its family."""


@dataclass(frozen=True)
class Wire:
    """One wire of the block design: its ``kind`` (``axis``, ``aximm``, ``axilite``,
    ``clock`` or ``reset``), from ``source`` to ``sink``, each ``<cell>/<pin>``."""

    kind: str
    source: str
    sink: str


@dataclass(frozen=True)
class Address:
    """A control bus's place: its ``interface`` (``<cell>/<pin>``), its ``offset`` and
    ``range`` (bytes) in the processor's space."""

    interface: str
    offset: int
    range: int


class LinkerRefused(Refusal):
    """A description the block design does not link: one rejection, owned by the linker
    (``vivado-block-design``), its message the finding's line."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__((Finding(FindingKind.REJECTION, code, VIVADO_BLOCK_DESIGN, message),))


def wire_one_bus_each(contract: MoverContract, where: str) -> None:
    """Refuse a mover (at ``where``, for the message) the block design cannot wire: it
    connects one AXI-Lite bus and one memory port a mover, while the shell's rules and
    costing read the counts the mover's contract states."""
    if (contract.control_buses, contract.memory_ports) != (1, 1):
        raise IntegrationError(
            "mover-buses-unwired",
            VIVADO_BLOCK_DESIGN,
            f"{where}: its {contract.kind} mover states {contract.control_buses} AXI-Lite "
            f"buses and {contract.memory_ports} memory ports; the {VIVADO_BLOCK_DESIGN!r} "
            "linker wires one of each a mover",
        )


def linked_row(integration: Integration) -> ShellRow:
    """The row ``integration`` was made for, refused unless it states the fixed shell and
    the control window the block design places its units in."""
    row = shell_row(*integration.row)
    if row.fixed is None or row.control is None:
        raise LinkerRefused(
            "integration-unexported",
            f"the {row.shell!r} shell states no fixed shell or no control window, which "
            "the block design places its units in",
        )
    return row


def family(row: ShellRow) -> str:
    """The template's processor family of ``row``'s fixed shell (``PROCESSORS``), refused
    where the block design instantiates no such processor (``processor-unsupported``)."""
    assert row.fixed is not None
    found = PROCESSORS.get(row.fixed.processor)
    if found is None:
        raise LinkerRefused(
            "processor-unsupported",
            f"the processor {row.fixed.processor!r} is none the block design instantiates "
            f"({', '.join(PROCESSORS)})",
        )
    return found


def control_instance(level: int, index: int) -> str:
    """The control tree's instance ``index`` of its level ``level``, the root's 0."""
    return f"control_{level}_{index}"


def control_tree(integration: Integration, row: ShellRow) -> tuple[tuple[int, ...], ...]:
    """The control tree of ``integration``'s units with a window (``placed_units``): each
    level's instances' ports, from the root (``interconnect_tree``, at the fixed shell's
    ``control_fanout``)."""
    assert row.fixed is not None
    return interconnect_tree(len(placed_units(integration)), row.fixed.control_fanout)


def wiring(integration: Integration, row: ShellRow) -> tuple[Wire, ...]:
    """The block design's wires of ``integration`` in ``row`` (see the module docstring),
    each mover first checked to be one it wires (``wire_one_bus_each``), and each unit's
    window, the range it assigns, to be the row's least window at least
    (``range-below-least``), which its linker assigns at least (its ``least_range``, by the
    row's declaration). Every endpoint is in a unit's window (``Integration``)."""
    assert row.fixed is not None and row.control is not None
    for edge in integration.edges:
        wire_one_bus_each(edge.contract, f"{edge.tensor} ({edge.port})")
    least = row.control.least_window
    for name, size in ((unit.name, unit.window) for unit in placed_units(integration)):
        assert size is not None
        if size < least:
            raise LinkerRefused(
                "range-below-least",
                f"{name}: {size} bytes; the row assigns a control window {least} at least",
            )
    if not integration.bindings:
        raise LinkerRefused(
            "memory-unbound",
            "the description binds no memory port: the block design's memory interconnect "
            "takes one at least",
        )
    for binding in integration.bindings:
        if binding.target != MEMORY_TARGET:
            raise LinkerRefused(
                "memory-target-unsupported",
                f"{binding.port}: bound to {binding.target!r}; the block design wires "
                f"memory into {MEMORY_TARGET!r} alone",
            )
    if row.fixed.control_interconnect != CONTROL_IP:
        raise LinkerRefused(
            "control-interconnect-unsupported",
            f"the control interconnect {row.fixed.control_interconnect!r}; the block design "
            f"builds its control tree of {CONTROL_IP!r}",
        )
    master = CONTROL_MASTERS[family(row)]
    for clock in integration.clocks:
        if clock.clock != MAIN:
            raise LinkerRefused(
                "clock-unsupported",
                f"{clock.unit}/{clock.pin}: on the clock {clock.clock!r}; the block design "
                f"runs every cell on the row's {MAIN!r} clock",
            )
    memory = STATIC_INSTANCES["memory_interconnect"]
    processor, reset = STATIC_INSTANCES["processor"], STATIC_INSTANCES["reset"]
    wires = [Wire("axis", each.source, each.sink) for each in integration.connections]
    wires += [
        Wire("aximm", each.port, f"{memory}/S{index:02d}_AXI")
        for index, each in enumerate(integration.bindings)
    ]
    tree = control_tree(integration, row)
    instances = [
        control_instance(depth, index)
        for depth, level in enumerate(tree)
        for index, _ in enumerate(level)
    ]
    if tree:
        wires.append(Wire("axilite", f"{processor}/{master}", f"{control_instance(0, 0)}/S00_AXI"))
    for depth, level in enumerate(tree):
        below = (
            [
                f"{control_instance(depth + 1, index)}/S00_AXI"
                for index in range(len(tree[depth + 1]))
            ]
            if depth + 1 < len(tree)
            else [f"{unit.name}/{CONTROL}" for unit in placed_units(integration)]
        )
        ports = [
            f"{control_instance(depth, index)}/M{port:02d}_AXI"
            for index, count in enumerate(level)
            for port in range(count)
        ]
        wires += [Wire("axilite", port, sink) for port, sink in zip(ports, below, strict=True)]
    for instance in instances:
        wires.append(Wire("clock", f"{processor}/{row.clocks.main}", f"{instance}/aclk"))
        wires.append(Wire("reset", f"{reset}/peripheral_aresetn", f"{instance}/aresetn"))
    for clock in integration.clocks:
        wires.append(Wire("clock", f"{memory}/aclk", f"{clock.unit}/{clock.pin}"))
        wires.append(Wire("reset", f"{memory}/aresetn", f"{clock.unit}/{clock.reset}"))
    return tuple(wires)


def placed_units(integration: Integration) -> tuple[Unit, ...]:
    """The units of ``integration`` with a control window, in its order: one control port
    each."""
    return tuple(unit for unit in integration.units if unit.window is not None)


def placement(integration: Integration, row: ShellRow) -> tuple[Address, ...]:
    """Each control port of ``integration``'s units with a window (``placed_units``), in
    its order, from the fixed shell's ``control_base``, aligned to its window
    (``placed_windows``)."""
    assert row.fixed is not None
    units = placed_units(integration)
    windows = [unit.window for unit in units if unit.window is not None]
    bases = placed_windows(windows, row.fixed.control_base)
    return tuple(
        Address(f"{unit.name}/{CONTROL}", base, window)
        for unit, base, window in zip(units, bases, windows, strict=True)
    )


def unit_bases(integration: Integration, row: ShellRow) -> dict[str, int]:
    """Each unit of ``integration`` with a control window at its base address
    (``placement``); an endpoint is at its unit's base and its offset in the unit."""
    return {
        address.interface.split("/")[0]: address.offset for address in placement(integration, row)
    }


def report(integration: Integration) -> dict[str, Any]:
    """What the linker names and places of ``integration``, as JSON: each unit's IP, the
    wires and the addresses."""
    row = linked_row(integration)
    return {
        "units": [
            {
                "name": unit.name,
                "ip": None
                if unit.ip is None
                else {"vlnv": unit.ip.vlnv, "repositories": list(unit.ip.repositories)},
            }
            for unit in integration.units
        ],
        "wires": [
            {"kind": each.kind, "source": each.source, "sink": each.sink}
            for each in wiring(integration, row)
        ],
        "addresses": [
            {"interface": each.interface, "offset": hex(each.offset), "range": each.range}
            for each in placement(integration, row)
        ],
    }


def _owner(pin: str) -> str:
    return pin.split("/", 1)[0]


def _interface(own: str, peer: str) -> str:
    return f"connect_bd_intf_net [get_bd_intf_pins {own}] [get_bd_intf_pins {peer}]"


def debug_lines(wires: tuple[Wire, ...], row: ShellRow) -> list[str]:
    """Integrated logic analyzers on every stream between a mover and the core (each net
    named by its master pin) and on the memory interconnect's port to the processor, as
    the template's debug block sets them on the HWCustomOp flow's, each on the processor's
    clock the row runs the cells on (its ``main``)."""
    assert row.fixed is not None
    streams = [wire.source.replace("/", "_") for wire in wires if wire.kind == "axis"]
    port = f"{row.fixed.memory_interconnect}_0_M00_AXI"
    source = f"/{STATIC_INSTANCES['processor']}/{row.clocks.main}"
    clock = f'CLK_SRC "{source}" SYSTEM_ILA "Auto" APC_EN "0"'
    axi = " ".join(
        f'{channel} "Data and Trigger"'
        for channel in ("AXI_R_ADDRESS", "AXI_R_DATA", "AXI_W_ADDRESS", "AXI_W_DATA")
    )
    probes = [f'[get_bd_intf_nets {port}] {{{axi} AXI_W_RESPONSE "Data and Trigger" {clock} }}']
    probes += [
        f'[get_bd_intf_nets {net}] {{AXIS_SIGNALS "Data and Trigger" {clock} }}' for net in streams
    ]
    return [
        *(f"set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {{{net}}}]" for net in streams),
        f"set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {{{port}}}]",
        "apply_bd_automation -rule xilinx.com:bd_rule:debug -dict [list " + " ".join(probes) + "]",
    ]


def block_design(integration: Integration, row: ShellRow, hw_debug: bool = False) -> str:
    """The block design's cells and wires, as the template's custom section takes them: the
    control tree's instances (``control_tree``), their ports and their clocks and resets;
    then for each unit in the description's order, its IP repositories and cell, then each wire of
    one of its pins, its own pin first: its memory port to the memory interconnect, its
    control port from the control interconnect with its address (``placement``'s, its
    offset and range), its clock and reset, and each stream it takes. Each unit's IP is
    packaged first (``unit-unpackaged`` otherwise); with ``hw_debug``, ``debug_lines``
    follow."""
    wires = wiring(integration, row)
    addresses = {address.interface: address for address in placement(integration, row)}
    lines = []
    tree = control_tree(integration, row)
    instances = {
        control_instance(depth, index): count
        for depth, level in enumerate(tree)
        for index, count in enumerate(level)
    }
    for instance, count in instances.items():
        lines.append(f"create_bd_cell -type ip -vlnv $smartconnect_vlnv {instance}")
        lines.append(
            f"set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI {count} "
            f"CONFIG.STRATEGY LOW_AREA] [get_bd_cells {instance}]"
        )
    for wire in wires:
        if _owner(wire.sink) in instances:
            if wire.kind == "axilite":
                lines.append(_interface(wire.sink, wire.source))
            else:
                lines.append(
                    f"connect_bd_net [get_bd_pins {wire.sink}] [get_bd_pins {wire.source}]"
                )
    for unit in integration.units:
        if unit.ip is None or not unit.ip.repositories:
            raise LinkerRefused(
                "unit-unpackaged", f"{unit.name}: its IP is not packaged; package it first"
            )
        instance = unit.name
        repositories = " ".join(tcl_quote(path) for path in unit.ip.repositories)
        lines.append(
            "set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] "
            f"[list {repositories}]] [current_project]"
        )
        lines.append("update_ip_catalog -rebuild -scan_changes")
        lines.append(f"create_bd_cell -type ip -vlnv {unit.ip.vlnv} {instance}")
        mine = [wire for wire in wires if instance in (_owner(wire.source), _owner(wire.sink))]
        for wire in mine:
            if wire.kind == "aximm":
                lines.append(_interface(wire.source, wire.sink))
        for wire in mine:
            if wire.kind == "axilite":
                address = addresses[wire.sink]
                lines.append(_interface(wire.sink, wire.source))
                lines.append(
                    f"assign_bd_address [get_bd_addr_segs {wire.sink}/Reg*] "
                    f"-offset {address.offset:#010x} -range {address.range}"
                )
        for kind in ("clock", "reset"):
            for wire in mine:
                if wire.kind == kind:
                    lines.append(
                        f"connect_bd_net [get_bd_pins {wire.sink}] [get_bd_pins {wire.source}]"
                    )
        for wire in mine:
            if wire.kind == "axis" and _owner(wire.sink) == instance:
                lines.append(_interface(wire.sink, wire.source))
    if hw_debug:
        lines += debug_lines(wires, row)
    return "\n".join(lines) + "\n"


def project_script(
    integration: Integration,
    design: str,
    board: Mapping[str, str],
    row: ShellRow,
    jobs: int | None = None,
) -> str:
    """The project's Tcl (``ip_config.tcl``): ``templates.custom_zynq_shell_template``
    filled with the description's clock (the period asked, in whole MHz as the template
    takes it), its memory ports, board and
    part, the board's part (``board``'s ``board_part``) and the processor's family
    (``PROCESSORS``, by the fixed shell's processor), the block design ``design``, and the
    runs Vivado launches at once (``jobs``, by default the machine's cores, capped:
    ``finn.toolchain.vivado.vivado_jobs``).
    A board with no board part (``board-part-unstated``) or a processor the template does
    not instantiate (``processor-unsupported``) is refused. The template's debug block
    stays off: its nets are the HWCustomOp flow's (``debug_lines``)."""
    board_part = selected_board(integration, board)
    processor = family(row)
    boards = " ".join(tcl_quote(path) for path in resources.paths("vivado-boards"))
    # Every board repository; the template is %-formatted next.
    template = templates.custom_zynq_shell_template.replace(
        "$BOARD_FILES$", boards.replace("%", "%%")
    )
    return template % (
        int(1 / (integration.period_ns * 0.001)),
        len(integration.bindings),
        integration.row[1],
        integration.part,
        board_part,
        processor,
        design,
        0,
        vivado_jobs(jobs),
    )


def selected_board(integration: Integration, board: Mapping[str, str]) -> str:
    """The board part the project selects, ``board``'s ``board_part``; a board with none is
    refused (``board-part-unstated``)."""
    board_part = board.get("board_part")
    if not board_part:
        raise LinkerRefused(
            "board-part-unstated",
            f"the board {integration.row[1]!r} states no board_part, which the project selects",
        )
    return board_part


def collect(
    project: str, integration: Integration, row: ShellRow, transcript: StrPath | None = None
) -> Linked:
    """What Vivado made in ``project`` (the template's project) of ``integration``: the
    bitfile (``image``) and its hardware handoff beside it, each unit's base
    (``unit_bases``), the row's main clock as the routed design delivers it and its timing
    there, admitted (``admitted``: an image that misses timing is refused), and the
    reports by their paths under a build's ``report/`` (the module docstring),
    ``BLOCK_DESIGN`` written into the project. A bitfile or hardware handoff not there is
    Vivado's failure (``ToolFailed``, its errors read from the run's ``transcript``,
    pointing at the project's logs): it exits cleanly when its implementation fails."""
    root = Path(project)
    runs = root / f"{PROJECT}.runs"
    bitfile = runs / "impl_1" / "top_wrapper.bit"
    handoffs = [
        root / f"{PROJECT}.{tree}" / "sources_1" / "bd" / "top" / "hw_handoff" / "top.hwh"
        for tree in ("gen", "srcs")
    ]
    hwh = next((path for path in handoffs if path.is_file()), None)
    if not bitfile.is_file() or hwh is None:
        raise ToolFailed.of(
            "vivado",
            transcript=transcript,
            what=f"it made no bitfile or hardware handoff: check the logs under {project}",
        )
    designed = root / BLOCK_DESIGN
    designed.write_text(json.dumps(report(integration), indent=2))
    reports = {
        TIMING: runs / "impl_1" / "top_wrapper_timing_summary_routed.rpt",
        PLACED: root / "synth_report.xml",
        BLOCK_DESIGN: designed,
    }
    reports.update(
        {
            f"{OUT_OF_CONTEXT}/{path.name}": path
            for path in sorted(runs.glob("*_synth_1/*_utilization_synth.rpt"))
        }
    )
    mhz, timing = admitted(reports[TIMING], integration.period_ns)
    return Linked(
        image=bitfile,
        placement=unit_bases(integration, row),
        clocks_mhz={MAIN: mhz},
        reports=reports,
        beside=(hwh,),
        timing=timing,
    )


#: Each timing check an image must meet (``finn.toolchain.vivado.CHECKS``), by its
#: refusal's code, its name in the message and the remedy: a longer period meets setup
#: and a primitive's minimum period or pulse width, never hold.
MISSED = {
    "setup": ("timing-unmet", "setup", "ask a longer period (the target's period_ns)"),
    "hold": (
        "timing-hold-unmet",
        "hold",
        "a hold time is missed at any period, so a longer one does not meet it; the "
        "report's timing details name the failing paths",
    ),
    "pulse_width": (
        "timing-pulse-width-unmet",
        "pulse width",
        "a clock's period or pulse width is below a primitive's minimum; ask a longer "
        "period (the target's period_ns)",
    ),
}


def admitted(report: Path, period_ns: float) -> tuple[float, Timing]:
    """The main clock the routed design delivers, in MHz, and its timing (``Timing``),
    read once from its routed timing report ``report`` (``TIMING``,
    ``finn.toolchain.vivado.routed_timing``): the PL clock's period in the report's clock
    summary, which Vivado gives as near the ``period_ns`` asked as the processor's clocks
    reach, and the design's setup, hold and pulse-width checks at it, each met only with
    no failing endpoint and no negative slack (``Check.met``: a violation rounded to
    ``-0.000`` fails). Vivado writes the bitfile of a design that misses timing: a check
    missed is refused, each with its remedy (``MISSED``: ``timing-unmet``, setup;
    ``timing-hold-unmet``, hold, missed at any period; ``timing-pulse-width-unmet``), as
    is no report (``timing-unreported``), and one that cannot be read, is cut short, or
    names no PL clock (``timing-unread``), each naming the report and the cause."""

    def unread(cause: str) -> LinkerRefused:
        return LinkerRefused(
            "timing-unread",
            f"{report}: {cause}: whether the image meets its clock is unknown",
        )

    try:
        text = report.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise LinkerRefused(
            "timing-unreported",
            f"no routed timing report ({report}): whether the image meets its clock is unknown",
        ) from None
    except (OSError, UnicodeDecodeError) as error:
        raise unread(f"not read ({type(error).__name__}: {error})") from error
    try:
        routed = routed_timing(text)
    except ValueError as error:
        raise unread(f"not a whole timing summary ({error})") from error
    name = next((clock for clock in PL_CLOCKS if clock in routed.clocks), None)
    if name is None:
        raise unread(
            f"names no PL clock ({', '.join(PL_CLOCKS)}; it lists "
            f"{', '.join(sorted(routed.clocks)) or 'none'})"
        )
    period, mhz = routed.clocks[name]["period_ns"], routed.clocks[name]["mhz"]
    wns = routed.design["setup"].worst_ns
    if not math.isfinite(wns):
        raise unread(f"its worst setup slack is {wns}")
    for check, (code, label, remedy) in MISSED.items():
        found = routed.design[check]
        if not found.met:
            raise LinkerRefused(
                code,
                f"{report}: the routed design's worst {label} slack is {found.worst_ns} ns, "
                f"{found.failing_endpoints} of {found.endpoints} endpoints failing, at its "
                f"{name} clock's {period} ns ({mhz} MHz; {period_ns} ns asked): it misses "
                f"timing; {remedy}",
            )
    return mhz, Timing(period, dict(routed.design), TIMING)


class BlockDesignLinker:
    """The ``vivado-block-design`` linker (see the module docstring): ``debug`` inserts
    integrated logic analyzers on the movers' streams and on the memory interconnect's
    port to the processor; Vivado launches as many runs at once as the toolchain's
    selection says (``vivado_jobs``)."""

    def __init__(self, *, debug: bool = False) -> None:
        self.debug = debug

    def check(self, integration: Integration, board: Mapping[str, str]) -> None:
        """Refuse what the block design does not wire (``wiring``), a board with no board
        part and a processor the template does not instantiate, before any tool runs."""
        row = linked_row(integration)
        wiring(integration, row)
        selected_board(integration, board)
        family(row)

    def link(
        self,
        integration: Integration,
        board: Mapping[str, str],
        directory: Path,
        tools: Toolchain,
    ) -> Linked:
        """Link ``integration``, every unit's IP packaged, on the board whose facts are
        ``board``, in a new project under ``directory``: the block design written into the
        template, Vivado run on it through ``tools``, and what it made collected. What the
        linker refuses is refused before Vivado runs."""
        row = linked_row(integration)
        design = block_design(integration, row, self.debug)
        text = project_script(integration, design, board, row, tools.selection.vivado_jobs)
        Path(directory).mkdir(parents=True, exist_ok=True)
        project = tempfile.mkdtemp(prefix="vivado_zynq_proj_", dir=directory)
        os.chmod(project, 0o755)
        script = project + "/ip_config.tcl"
        with open(script, "w") as f:
            f.write(text)
        ran = tools.run(
            "vivado",
            ["-mode", "batch", "-source", script],
            cwd=project,
            replay=project + "/synth_project.sh",
        )
        _LOG.info(f"Vivado's project: {project}")
        return collect(project, integration, row, getattr(ran, "transcript", None))


__all__ = [
    "BLOCK_DESIGN",
    "LINKER",
    "OUT_OF_CONTEXT",
    "PLACED",
    "PROJECT",
    "STATIC_INSTANCES",
    "TIMING",
    "Address",
    "BlockDesignLinker",
    "CONTROL_IP",
    "CONTROL_MASTERS",
    "LinkerRefused",
    "MEMORY_TARGET",
    "MISSED",
    "PROCESSORS",
    "Wire",
    "admitted",
    "block_design",
    "collect",
    "control_instance",
    "control_tree",
    "debug_lines",
    "family",
    "linked_row",
    "placed_units",
    "placement",
    "project_script",
    "report",
    "selected_board",
    "unit_bases",
    "wire_one_bus_each",
    "wiring",
]
