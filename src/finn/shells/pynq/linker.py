# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``pynq`` shell's linker: the Zynq block design of an integration description
(``finn.shells.integration``), the ``vivado-block-design`` linker's code.

It takes every name the platform gives and places every control bus:

- the cells: each unit's is its name (a core's its node's, a mover's ``in<i>`` or
  ``out<j>``), its pins the description's, which are its IP's own; the fixed shell's are
  the template's (``STATIC_INSTANCES``);
- the wires (``wiring``): each stream from its source pin to its sink pin; each memory port
  to the memory interconnect's next slave port (``S<nn>_AXI``); each control bus, in the
  description's order (the input movers, the core, the output movers), from the control
  interconnect's next master port (``M<nn>_AXI``); every cell's clock and reset from the
  memory interconnect's;
- the placement (``placement``): each control bus from the fixed shell's ``control_base``,
  in that order, aligned to its range (``finn.kernels.shell_rules.placed_windows``, by which
  the shell's control window rule admitted them); the block design assigns each its address
  from it, so the placement it returns is the one Vivado was given;
- the board (``project_script``): its board part, the board's declared fact
  (``board_part``), and its processor's family, the fixed shell's processor's
  (``PROCESSORS``).

It wires one AXI-Lite bus and one memory port a mover: a mover whose contract states other
counts is refused, by name (``wire_one_bus_each``). Its template's control interconnect
takes ``CONTROL_SLAVES`` AXI-Lite buses: a description with more is refused before any tool
runs (``control-slaves-exceeded``), where the shell's rules no longer bound them. Its
template wires the memory
interconnect into the processor's ``HP0`` port and runs every cell on the row's main clock:
a memory port bound to another target, a unit on another clock and a description with no
memory port are refused, each by name (``wiring``), as is a processor or board it cannot
select (``project_script``).

``BlockDesignLinker.link`` writes the block design (``block_design``) into
``templates.custom_zynq_shell_template`` (``project_script``) in a new project under the
directory it is given, runs Vivado on it and collects what it made (``collect``): the
bitfile, its hardware handoff, the routed timing report, the template's ``impl_1``
hierarchical utilization report and each IP's out-of-context synthesis report. Every
unit's IP is packaged before it links.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finn import resources
from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.kernels.movers import MoverContract
from finn.kernels.shell_rules import placed_windows
from finn.platform.shells import VIVADO_BLOCK_DESIGN, ShellRow, shell_row
from finn.shells.integration import MAIN, Integration, IntegrationError, Linked
from finn.shells.pynq import templates
from finn.toolchain.launcher import StrPath, Toolchain, ToolFailed
from finn.toolchain.vivado import tcl_quote, vivado_jobs

#: The project the template creates (``create_project finn_zynq_link``).
PROJECT = "finn_zynq_link"

#: The template's instance of each part of the fixed shell (``FixedShell``'s fields); the
#: reset's is a prefix, as Vivado's clock automation names it after the processor's clock
#: (``rst_zynq_ps_187M``).
STATIC_INSTANCES = {
    "processor": "zynq_ps",
    "reset": "rst_zynq_ps_",
    "memory_interconnect": "smartconnect_0",
    "control_interconnect": "axi_interconnect_0",
}

MEMORY_TARGET = "HP0"
"""The memory target the template wires the memory interconnect into: the processor's
``HP0`` port."""

CONTROL_SLAVES = 9
"""The AXI-Lite buses the template's control interconnect takes (its ``NUM_AXILITE``)."""

PROCESSORS = {"zynq_ultra_ps_e": "zynq_us+", "processing_system7": "zynq_7000"}
"""The template's processor family (its ``ZYNQ_TYPE``) by the fixed shell's processor IP."""


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


def wiring(integration: Integration, row: ShellRow) -> tuple[Wire, ...]:
    """The block design's wires of ``integration`` in ``row`` (see the module docstring),
    each mover first checked to be one it wires (``wire_one_bus_each``)."""
    assert row.fixed is not None
    for edge in integration.edges:
        wire_one_bus_each(edge.contract, f"{edge.tensor} ({edge.port})")
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
    if len(integration.endpoints) > CONTROL_SLAVES:
        raise LinkerRefused(
            "control-slaves-exceeded",
            f"the description places {len(integration.endpoints)} AXI-Lite buses; the "
            f"block design's control interconnect takes {CONTROL_SLAVES}",
        )
    for clock in integration.clocks:
        if clock.clock != MAIN:
            raise LinkerRefused(
                "clock-unsupported",
                f"{clock.unit}/{clock.pin}: on the clock {clock.clock!r}; the block design "
                f"runs every cell on the row's {MAIN!r} clock",
            )
    memory = f"{row.fixed.memory_interconnect}_0"
    control = f"{row.fixed.control_interconnect}_0"
    wires = [Wire("axis", each.source, each.sink) for each in integration.connections]
    wires += [
        Wire("aximm", each.port, f"{memory}/S{index:02d}_AXI")
        for index, each in enumerate(integration.bindings)
    ]
    wires += [
        Wire("axilite", f"{control}/M{index:02d}_AXI", f"{each.unit}/{each.bus}")
        for index, each in enumerate(integration.endpoints)
    ]
    for clock in integration.clocks:
        wires.append(Wire("clock", f"{memory}/aclk", f"{clock.unit}/{clock.pin}"))
        wires.append(Wire("reset", f"{memory}/aresetn", f"{clock.unit}/{clock.reset}"))
    return tuple(wires)


def placement(integration: Integration, row: ShellRow) -> tuple[Address, ...]:
    """Each control bus of ``integration``, in its order, from the fixed shell's
    ``control_base``, aligned to its range (``placed_windows``), at its offset in its
    unit."""
    assert row.fixed is not None
    endpoints = integration.endpoints
    offsets = placed_windows([each.range for each in endpoints], row.fixed.control_base)
    return tuple(
        Address(f"{each.unit}/{each.bus}", offset + each.offset, each.range)
        for each, offset in zip(endpoints, offsets, strict=True)
    )


def report(integration: Integration) -> dict[str, Any]:
    """What the linker names and places of ``integration``, as JSON: the wires and the
    addresses."""
    row = linked_row(integration)
    return {
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
    """The block design's cells and wires, as the template's custom section takes them: for
    each unit in the description's order, its IP repositories and cell, then each wire of
    one of its pins, its own pin first: its memory port to the memory interconnect, each
    control bus from the control interconnect with its address (``placement``'s, its
    offset and range), its clock and reset, and each stream it takes. Each unit's IP is
    packaged first (``unit-unpackaged`` otherwise); with ``hw_debug``, ``debug_lines``
    follow."""
    wires = wiring(integration, row)
    addresses = {address.interface: address for address in placement(integration, row)}
    lines = []
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
    takes it), its control buses and memory ports, board and part, the board's part
    (``board``'s ``board_part``) and the processor's family (``PROCESSORS``, by the fixed
    shell's processor), the block design ``design``, and the runs Vivado launches at once
    (``jobs``, by default the machine's cores, capped: ``finn.toolchain.vivado.vivado_jobs``).
    A board with no board part (``board-part-unstated``) or a processor the template does
    not instantiate (``processor-unsupported``) is refused. The template's debug block
    stays off: its nets are the HWCustomOp flow's (``debug_lines``)."""
    assert row.fixed is not None
    board_part = board.get("board_part")
    if not board_part:
        raise LinkerRefused(
            "board-part-unstated",
            f"the board {integration.row[1]!r} states no board_part, which the project selects",
        )
    family = PROCESSORS.get(row.fixed.processor)
    if family is None:
        raise LinkerRefused(
            "processor-unsupported",
            f"the processor {row.fixed.processor!r} is none the block design instantiates "
            f"({', '.join(PROCESSORS)})",
        )
    boards = " ".join(tcl_quote(path) for path in resources.paths("vivado-boards"))
    # Every board repository; the template is %-formatted next.
    template = templates.custom_zynq_shell_template.replace(
        "$BOARD_FILES$", boards.replace("%", "%%")
    )
    return template % (
        int(1 / (integration.period_ns * 0.001)),
        len(integration.endpoints),
        len(integration.bindings),
        integration.row[1],
        integration.part,
        board_part,
        family,
        design,
        0,
        vivado_jobs(jobs),
    )


def collect(
    project: str, integration: Integration, row: ShellRow, transcript: StrPath | None = None
) -> Linked:
    """What Vivado made in ``project`` (the template's project) of ``integration``: the
    bitfile (``image``), the placement of each control bus (``<unit>/<bus>``), and the
    reports: the hardware handoff (``hwh``), the routed timing summary (``timing``), the
    template's ``impl_1`` hierarchical utilization (``placed``), the project itself
    (``project``) and each IP's out-of-context synthesis utilization by its run
    (``out_of_context/<run>``). A bitfile or hardware handoff not there is Vivado's failure
    (``ToolFailed``, its errors read from the run's ``transcript``, pointing at the
    project's logs): it exits cleanly when its implementation fails."""
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
    reports = {
        "hwh": hwh,
        "timing": runs / "impl_1" / "top_wrapper_timing_summary_routed.rpt",
        "placed": root / "synth_report.xml",
        "project": root,
    }
    reports.update(
        {
            f"out_of_context/{path.name[: -len('_utilization_synth.rpt')]}": path
            for path in sorted(runs.glob("*_synth_1/*_utilization_synth.rpt"))
        }
    )
    places = placement(integration, row)
    return Linked(
        image=bitfile,
        placement={
            f"{each.unit}/{each.bus}": address.offset
            for each, address in zip(integration.endpoints, places)
        },
        reports=reports,
    )


class BlockDesignLinker:
    """The ``vivado-block-design`` linker (see the module docstring): Vivado launches
    ``jobs`` runs at once (``vivado_jobs``); ``hw_debug`` inserts integrated logic analyzers
    on the movers' streams and on the memory interconnect's port to the processor."""

    def __init__(self, *, jobs: int | None = None, hw_debug: bool = False) -> None:
        self.jobs = jobs
        self.hw_debug = hw_debug

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
        design = block_design(integration, row, self.hw_debug)
        text = project_script(integration, design, board, row, self.jobs)
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
        return collect(project, integration, row, getattr(ran, "transcript", None))


__all__ = [
    "PROJECT",
    "STATIC_INSTANCES",
    "Address",
    "BlockDesignLinker",
    "LinkerRefused",
    "MEMORY_TARGET",
    "PROCESSORS",
    "Wire",
    "block_design",
    "collect",
    "debug_lines",
    "linked_row",
    "placement",
    "project_script",
    "report",
    "wire_one_bus_each",
    "wiring",
]
