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

import os
import re
from typing import Any

#: The most Vivado runs a build launches at once by default (``launch_runs -jobs``):
#: a Zynq shell's block design synthesizes about ten IPs out of context, one run
#: each, and each run takes a few GB of memory.
VIVADO_JOBS_CAP = 16


def vivado_jobs(requested: "int | None" = None) -> int:
    """The number of runs Vivado launches at once (``launch_runs -jobs``): the
    ``requested`` number, or by default the machine's cores, at most VIVADO_JOBS_CAP.
    The one check of a number of jobs: a positive integer (``finn.util.toolchain.
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
