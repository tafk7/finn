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


def _parse_vivado_utilization_report(report_path):
    """Parse a Vivado utilization report file to extract resource counts.

    The utilization report has a table format with | delimiters. We look for
    specific resource rows and extract the "Used" column value.

    Args:
        report_path: Path to the utilization report file (ooc_utilization.rpt)

    Returns:
        Dictionary with resource counts (LUT, FF, DSP, BRAM, BRAM_18K, BRAM_36K, URAM, SRL)
    """
    if not os.path.exists(report_path):
        return {}

    with open(report_path, "r") as f:
        content = f.read()

    ret = {}

    # Define patterns to search for in the utilization report
    # Format is typically: | Resource Name | Used | Fixed | Prohibited | Available | Util% |
    # We want to extract the "Used" column (index 2 after splitting by |, since index 0 is empty)
    #
    # Example BRAM section from Vivado report:
    # | Block RAM Tile    |    2 |     0 |          0 |       312 |  0.64 |
    # |   RAMB36/FIFO*    |    0 |     0 |          0 |       312 |  0.00 |
    # |   RAMB18          |    4 |     0 |          0 |       624 |  0.64 |
    #
    # Block RAM Tile = combined BRAM metric (already calculated by Vivado)
    # RAMB36/FIFO* = BRAM_36K count
    # RAMB18 = BRAM_18K count
    resource_patterns = {
        # LUT patterns - try multiple variations across Vivado versions
        "LUT": [r"^\|\s*CLB LUTs\s*\|", r"^\|\s*Slice LUTs\s*\|", r"^\|\s*LUT as Logic\s*\|"],
        # FF patterns
        "FF": [
            r"^\|\s*CLB Registers\s*\|",
            r"^\|\s*Slice Registers\s*\|",
            r"^\|\s*Register as Flip Flop\s*\|",
        ],
        # DSP patterns - generic to match DSP48E1, DSP48E2, DSP58, etc.
        "DSP": [r"^\|\s*DSPs\s*\|", r"^\|\s*DSP48E\d*\s*\|", r"^\|\s*DSP58\s*\|"],
        # BRAM patterns - provide raw counts, users can calculate combined BRAM if needed
        # RAMB36/FIFO* is the 36K count
        # RAMB18 is the 18K count
        "BRAM_36K": [r"^\|\s*RAMB36/FIFO\*\s*\|", r"^\|\s*RAMB36E\d*\s*\|"],
        "BRAM_18K": [r"^\|\s*RAMB18\s*\|", r"^\|\s*RAMB18E\d*\s*\|"],
        # URAM patterns
        "URAM": [r"^\|\s*URAM\s*\|", r"^\|\s*URAM288\s*\|"],
        # SRL patterns
        "SRL": [r"^\|\s*LUT as Shift Register\s*\|", r"^\|\s*SRLC\d+E\s*\|"],
    }

    lines = content.split("\n")
    for resource_name, patterns in resource_patterns.items():
        for pattern in patterns:
            for line in lines:
                if re.match(pattern, line, re.IGNORECASE):
                    # Split by | and get the "Used" column (typically index 2)
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 3:
                        try:
                            ret[resource_name] = int(parts[2])
                            break
                        except ValueError:
                            continue
            if resource_name in ret:
                break

    return ret


def _parse_vivado_timing_report(report_path):
    """Parse a Vivado timing summary report to extract WNS.

    Args:
        report_path: Path to the timing report file (ooc_timing.rpt)

    Returns:
        Dictionary with timing metrics (WNS)
    """
    if not os.path.exists(report_path):
        return {}

    with open(report_path, "r") as f:
        content = f.read()

    ret = {}

    # Look for WNS in the timing summary
    # Format varies but typically: "WNS(ns)      TNS(ns)  ..." with values on next line
    # Or: "| WNS | TNS | ..." table format
    wns_patterns = [
        r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*(-?[\d.]+)",  # Header followed by value
        r"^\s*(-?[\d.]+)\s+(-?[\d.]+)\s+\d+\s+\d+\s+(-?[\d.]+)",  # Table row with WNS first
        r"Worst Negative Slack:\s*(-?[\d.]+)\s*ns",  # Explicit label
    ]

    for pattern in wns_patterns:
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            try:
                ret["WNS"] = float(match.group(1))
                break
            except (ValueError, IndexError):
                continue

    return ret


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


def _parse_vivado_power_report(report_path):
    """Parse a Vivado power report to extract total on-chip power.

    Args:
        report_path: Path to the power report file (ooc_power.rpt)

    Returns:
        Dictionary with power metrics (total_power_W)
    """
    if not os.path.exists(report_path):
        return {}

    with open(report_path, "r") as f:
        content = f.read()

    ret = {}

    # Look for total on-chip power
    # Format: "| Total On-Chip Power (W) | X.XXX |" or similar
    power_patterns = [
        r"Total On-Chip Power \(W\)\s*\|\s*([\d.]+)",
        r"Total On-Chip Power\s*:\s*([\d.]+)\s*W",
    ]

    for pattern in power_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                ret["total_power_W"] = float(match.group(1))
                break
            except (ValueError, IndexError):
                continue

    return ret


def _parse_ooc_metadata(metadata_path):
    """Parse the OOC metadata file for clock period and Vivado version.

    Args:
        metadata_path: Path to the metadata file (ooc_metadata.txt)

    Returns:
        Dictionary with metadata (clk_period_ns, vivado_version)
    """
    if not os.path.exists(metadata_path):
        return {}

    ret = {}
    with open(metadata_path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                if key == "clk_period_ns":
                    ret[key] = float(value)
                else:
                    ret[key] = value

    return ret


def parse_ooc_synth_results(report_dir):
    """Parse OOC synthesis results from Vivado report files.

    This function parses the utilization, timing, and power reports generated
    by CreateStitchedIP with run_pnr=True. The reports are in Vivado's native
    text format, which is parsed on the Python side for better maintainability.

    Args:
        report_dir: Path to the directory containing the OOC report files
                    (ooc_utilization.rpt, ooc_timing.rpt, ooc_power.rpt, ooc_metadata.txt)

    Returns:
        Dictionary with resource utilization, timing, and power metrics
        including calculated fmax_mhz, or None if the report files don't exist.
    """
    util_path = os.path.join(report_dir, "ooc_utilization.rpt")
    timing_path = os.path.join(report_dir, "ooc_timing.rpt")
    power_path = os.path.join(report_dir, "ooc_power.rpt")
    metadata_path = os.path.join(report_dir, "ooc_metadata.txt")

    # Check if at least the utilization report exists
    if not os.path.exists(util_path):
        return None

    # Parse all reports
    ret = {}
    ret.update(_parse_vivado_utilization_report(util_path))
    ret.update(_parse_vivado_timing_report(timing_path))
    ret.update(_parse_vivado_power_report(power_path))
    ret.update(_parse_ooc_metadata(metadata_path))

    # Calculate fmax from WNS
    wns = float(ret.get("WNS", 0))
    clk_ns = float(ret.get("clk_period_ns", 5.0))
    if clk_ns - wns > 0:
        ret["fmax_mhz"] = 1000.0 / (clk_ns - wns)
    else:
        ret["fmax_mhz"] = 0

    return ret
