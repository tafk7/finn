# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""What FINN knows of Vivado: the installation a toolchain selects (``installation``,
``release``, ``glbl``), how many runs it launches at once, how to read what its tools
print (``messages``: ``SEVERITY: [Tool N-M] text``, the form Vivado, xvlog, xelab, xsim
and the HLS frontends share; whether no licence was found), and how to read its reports
(the routed clocks; utilization, as plain counts by resource: ``lut``, ``ff``,
``bram36``, ``bram18``, ``uram``, ``dsp``).

Every reader of "which Vivado" reads it here, through a toolchain, never from this
process's environment; each raises ``finn.toolchain.launcher.ToolUnavailable`` when
the toolchain selects no Vivado.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# A module, not names: the launcher reads this module in turn (``vivado_jobs``).
from finn.toolchain import launcher

if TYPE_CHECKING:
    from finn.toolchain.launcher import Toolchain


def installation(toolchain: Toolchain) -> Path:
    """The Vivado installation ``toolchain`` selects: the directory its environment's
    ``XILINX_VIVADO`` names. ``ToolUnavailable`` when it names none, or no directory."""
    named = toolchain.environment.get("XILINX_VIVADO")
    if not named:
        raise launcher.ToolUnavailable(
            "vivado", "this toolchain names no Vivado installation (XILINX_VIVADO)"
        )
    if not Path(named).is_dir():
        raise launcher.ToolUnavailable("vivado", f"XILINX_VIVADO={named} is no directory")
    return Path(named)


def release(toolchain: Toolchain) -> tuple[int, int]:
    """The release, (year, minor), of the Vivado ``toolchain`` runs: its answer to
    ``vivado -version`` through the toolchain's route (``Toolchain.probe``), asked once
    per selection in a process. ``ToolUnavailable`` when no Vivado answers."""
    year, minor = toolchain.probe("vivado")[:2]
    return year, minor


def glbl(toolchain: Toolchain) -> Path:
    """The global module (``glbl.v``) of the Vivado installation ``toolchain`` selects,
    which every simulation of Xilinx primitives elaborates beside its top.
    ``ToolUnavailable`` when there is no installation, or it holds no ``glbl.v``."""
    path = installation(toolchain) / "data" / "verilog" / "src" / "glbl.v"
    if not path.is_file():
        raise launcher.ToolUnavailable("vivado", f"the installation holds no {path}")
    return path


#: A message line of Vivado's tools: its severity, its ID (``Common 17-345``, ``VRFC
#: 10-3180``; some messages have none) and its text.
_MESSAGE = re.compile(
    r"^\s*(FATAL_ERROR|FATAL|ERROR|CRITICAL WARNING|WARNING|INFO)\s*:\s*(?:\[([^\]]+)\]\s*)?(.*)$"
)
#: The severities that end a run, before the critical warnings that may not.
ERROR_SEVERITIES = ("FATAL_ERROR", "FATAL", "ERROR")
#: What a tool prints when it found no licence: Vivado's ``[Common 17-345]`` and its
#: words, and FlexLM's.
_NO_LICENCE = re.compile(
    r"\[Common 17-345\]|valid licen[cs]e was not found|licen[cs]e checkout failed|"
    r"failed to get a licen[cs]e|no licen[cs]e (is )?available|FLEX(net|lm) licensing error",
    re.IGNORECASE,
)
#: The code of a run that found no licence (``finn.toolchain.launcher.ToolFailed``).
LICENCE_UNAVAILABLE = "licence-unavailable"


@dataclass(frozen=True)
class Messages:
    """What a run of a Vivado tool printed, read by ``messages``: ``counts`` by
    (severity, ID; ``""`` for a message without one), the ``first`` line of each,
    the ``errors`` (every error line, then every critical warning, each in the order
    printed), and whether no licence was found (``licence_unavailable``)."""

    counts: Mapping[tuple[str, str], int]
    first: Mapping[tuple[str, str], str]
    errors: tuple[str, ...]
    licence_unavailable: bool

    @property
    def code(self) -> str:
        """``LICENCE_UNAVAILABLE`` when no licence was found, else ``tool-error``."""
        return LICENCE_UNAVAILABLE if self.licence_unavailable else "tool-error"


def messages(lines: str | Iterable[str]) -> Messages:
    """The messages of what a Vivado tool printed (a transcript's text, or its lines):
    each ``SEVERITY: [Tool N-M] text`` line counted by severity and ID, its errors and
    critical warnings, and whether it says no licence was found. A suppressed or
    benign ID is the reader's policy: every message is counted."""
    if isinstance(lines, str):
        lines = lines.splitlines()
    counts: dict[tuple[str, str], int] = {}
    first: dict[tuple[str, str], str] = {}
    errors: list[str] = []
    critical: list[str] = []
    licence = False
    for line in lines:
        line = line.rstrip()
        licence = licence or _NO_LICENCE.search(line) is not None
        found = _MESSAGE.match(line)
        if found is None:
            continue
        severity, ident = found.group(1), found.group(2) or ""
        key = (severity, ident)
        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, line.strip())
        if severity in ERROR_SEVERITIES:
            errors.append(line.strip())
        elif severity == "CRITICAL WARNING":
            critical.append(line.strip())
    return Messages(counts, first, (*errors, *critical), licence)


#: The most Vivado runs a build launches at once by default (``launch_runs -jobs``):
#: a Zynq shell's block design synthesizes about ten IPs out of context, one run
#: each, and each run takes a few GB of memory.
VIVADO_JOBS_CAP = 16


def vivado_jobs(requested: int | None = None) -> int:
    """The number of runs Vivado launches at once (``launch_runs -jobs``): the
    ``requested`` number, or by default the machine's cores, at most VIVADO_JOBS_CAP.
    The one check of a number of jobs: a positive integer (``finn.toolchain.launcher.
    Selection.vivado_jobs`` states one through it)."""
    if requested is None:
        return max(1, min(os.cpu_count() or 1, VIVADO_JOBS_CAP))
    if type(requested) is not int or requested < 1:
        raise ValueError(f"Vivado's jobs must be a positive number, not {requested!r}")
    return requested


def parse_clock_summary(report_path):
    """The clocks a Vivado timing summary report's "Clock Summary" table lists, by
    name: each one's period (ns) and frequency (MHz), as routed. A generated clock's
    indented name is read without its indent."""
    with open(report_path) as f:
        lines = f.read().splitlines()
    clocks = {}
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "| Clock Summary")
    except StopIteration:
        return clocks
    header = next(i for i in range(start, len(lines)) if lines[i].startswith("Clock "))
    for line in lines[header + 2 :]:
        if not line.strip():
            break
        match = re.match(r"\s*(\S+)\s+\{[^}]*\}\s+(-?[\d.]+)\s+(-?[\d.]+)", line)
        if match:
            clocks[match.group(1)] = {
                "period_ns": float(match.group(2)),
                "mhz": float(match.group(3)),
            }
    return clocks


#: The clock the Zynq shell's PS drives the accelerator with: Zynq UltraScale+'s
#: and Zynq-7000's name for it.
PL_CLOCKS = ("clk_pl_0", "clk_fpga_0")


def delivered_clock(
    timing_report: str,
    period_ns: float,
    cycles: int | None = None,
    objective_fps: float | None = None,
) -> dict[str, Any]:
    """The clock the routed design delivers (a PL clock of the timing report's clock
    summary) beside the period asked, and, given the partition's bottleneck
    ``cycles`` a frame, the frames a second at each; ``objective_fps`` is the
    throughput asked. A delivered period other than the one asked is a ``warning``
    with both numbers (the shell's PS gives its nearest clock to the request)."""
    clocks = parse_clock_summary(timing_report)
    name = next((clock for clock in PL_CLOCKS if clock in clocks), None)
    if name is None:
        return {
            "target_period_ns": period_ns,
            "warning": f"no PL clock ({', '.join(PL_CLOCKS)}) in {timing_report}: "
            f"its clock summary lists {sorted(clocks)}",
        }
    delivered, mhz = clocks[name]["period_ns"], clocks[name]["mhz"]
    report: dict[str, Any] = {
        "clock": name,
        "target_period_ns": period_ns,
        "delivered_period_ns": delivered,
        "delivered_mhz": mhz,
    }
    if cycles is not None:
        report["bottleneck_cycles"] = cycles
        report["fps_at_target"] = round(1e9 / (period_ns * cycles), 1)
        report["fps_at_delivered"] = round(mhz * 1e6 / cycles, 1)
    if objective_fps is not None:
        report["objective_fps"] = objective_fps
    # The report states periods to the picosecond.
    if abs(delivered - period_ns) >= 0.0005:
        warning = (
            f"the shell delivers {name} at {delivered} ns ({mhz} MHz), "
            f"not the {period_ns} ns asked"
        )
        if cycles is not None:
            warning += (
                f": {cycles} cycles a frame give {report['fps_at_delivered']:,.0f} fps at it, "
                f"{report['fps_at_target']:,.0f} at the clock asked"
            )
        if objective_fps is not None:
            warning += f"; the objective is {objective_fps:,.0f} fps"
        report["warning"] = warning
    return report


#: A hierarchical utilization report's column, by the resource it counts; a DSP column
#: is any whose header starts with ``DSP`` (``DSP Blocks``, ``DSP48 Blocks``).
REPORT_COLUMNS = {
    "Total LUTs": "lut",
    "FFs": "ff",
    "RAMB36": "bram36",
    "RAMB18": "bram18",
    "URAM": "uram",
}

#: A flat utilization report's rows by the resource they count; the first row of each
#: name counts (later sections repeat primitive names).
_SITE_TYPES = {
    "CLB LUTs": "lut",
    "Slice LUTs": "lut",
    "CLB Registers": "ff",
    "Slice Registers": "ff",
    "RAMB36/FIFO": "bram36",
    "RAMB18": "bram18",
    "URAM": "uram",
    "DSPs": "dsp",
}


def _counted(header: list[str], cells: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column, cell in zip(header, cells, strict=True):
        key = "dsp" if column.startswith("DSP") else REPORT_COLUMNS.get(column)
        if key is not None:
            counts[key] = counts.get(key, 0) + int(cell)
    return counts


def hierarchical_utilization(text: str) -> list[tuple[int, str, dict[str, int]]]:
    """The rows of the text ``report_utilization -hierarchical``, in order: each
    instance's depth (the top 0), its name and its counts. ``ToolFailed`` when the
    report holds no such table."""
    rows: list[tuple[int, str, dict[str, int]]] = []
    header: list[str] | None = None
    indent = 0
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = line.split("|")[1:-1]
        names = [cell.strip() for cell in cells]
        if header is None:
            if names[:2] == ["Instance", "Module"]:
                header = names
            continue
        name = cells[0]
        depth = len(name) - len(name.lstrip())
        if not rows:
            indent = depth
        rows.append(((depth - indent) // 2, names[0], _counted(header, names)))
    if not rows:
        raise launcher.ToolFailed("vivado", what="no hierarchical utilization table in the report")
    return rows


def placed_hierarchy(text: str) -> list[tuple[int, str, dict[str, int]]]:
    """The rows of ``report_utilization -hierarchical -format xml``, in order: each
    instance's depth (the top 0), its name and its counts. ``ValueError`` when the
    report holds no table, or no row."""
    table = ET.fromstring(text).find(".//table")
    if table is None:
        raise ValueError("no hierarchical utilization table in the report")
    header: list[str] = []
    rows: list[tuple[int, str, dict[str, int]]] = []
    for row in table.findall("tablerow"):
        heads = [cell.get("contents", "") for cell in row.findall("tableheader")]
        if heads:
            header = heads
            continue
        cells = [cell.get("contents", "") for cell in row.findall("tablecell")]
        if not cells:
            continue
        name = cells[0]
        depth = (len(name) - len(name.lstrip(" "))) // 2
        rows.append((depth, name.strip(), _counted(header, cells)))
    if not rows:
        raise ValueError("no rows in the hierarchical utilization table")
    return rows


def utilization_synth(text: str) -> dict[str, int]:
    """The counts of the flat ``report_utilization`` of one synthesized IP (a
    ``<run>_utilization_synth.rpt``): its LUTs, registers, block RAM tiles, URAMs and
    DSPs, each the first row of its site type. ``ValueError`` without a LUT row."""
    counts: dict[str, int] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) < 2:
            continue
        site = cells[0].rstrip("*").strip()
        key = _SITE_TYPES.get(site) or ("dsp" if site.startswith("DSP48") else None)
        if key is not None and key not in counts and cells[1].isdigit():
            counts[key] = int(cells[1])
    if "lut" not in counts:
        raise ValueError("no LUT row in the utilization report")
    return counts


def tcl_quote(value: object) -> str:
    """Quote a literal Tcl word, including whitespace and substitution characters."""
    value = str(value)
    for char in ("\\", '"', "$", "[", "]"):
        value = value.replace(char, "\\" + char)
    return '"' + value.replace("\n", "\\n").replace("\r", "\\r") + '"'
