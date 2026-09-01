############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""D6a OOC synthesis, IP-XACT packaging, and catalog-stitch gate.

Runs inside the FINN container and requires Vivado.  The generated artifact is
the same production DotProduct plus FINN RTL memstream composition exercised by
``supplied_dot_product_numeric.py``; this fixture carries it through the three
tool-backed stages that numerical simulation does not reach.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np  # type: ignore[import-not-found]

from dataflow.rtlsim.composed_mvau_equiv import record_identity
from dataflow.rtlsim.composed_mvau_numeric import MVAUDspBlock, _weights
from dataflow.rtlsim.supplied_dot_product_numeric import CASE, requirements_for
from finn.dataflow.hardware import DEFAULT_BUILDER, BuilderIdentity, TargetIdentity
from finn.dataflow.ops.mvau.hardware.binding import finnlib_root
from finn.dataflow.ops.mvau.hardware.composition import (
    complete_decomposed_synthesis,
    complete_ip_package,
    package_decomposed_artifact,
    prepare_decomposed_synthesis,
    prepare_ip_package,
)
from finn.util.basic import get_vivado_version

PASS, FAIL = 0, 1

# A readily licensed UltraScale+ part is enough to prove all D6a stages.  The
# numerical gate separately exercises the DSP58 realization.
HARDWARE_CASE = replace(CASE, label="dot_product_memstream_hardware", target=MVAUDspBlock.DSP48E2)

_UTILIZATION = re.compile(r"^\|\s*(DSP\w*)\s*\|\s*(\d+)\s*\|", re.MULTILINE)
_SUMMARY_ROW = "DSPs"
_BRAM_UTILIZATION = re.compile(
    r"^\|\s*(Block RAM Tile|RAMB(?:18|36)(?:E\d)?(?:/FIFO)?\*?)\s*\|\s*([0-9.]+)\s*\|",
    re.MULTILINE,
)
_ADDRESS_WIDTH_WARNING = re.compile(
    r"actual bit length .* differs from formal bit length .* "
    r"for port 's_axilite_(?:AWADDR|ARADDR)'"
)


def _builder() -> BuilderIdentity:
    version = get_vivado_version()
    if version is None:
        return DEFAULT_BUILDER
    return BuilderIdentity("vivado", f"{version[0]}.{version[1]}")


def _run_vivado(directory: Path, script: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["vivado", "-mode", "batch", "-nojournal", "-nolog", "-notrace", "-source", str(script)],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )


def _errors(completed: subprocess.CompletedProcess[str]) -> list[str]:
    return [
        line
        for line in (*completed.stdout.splitlines(), *completed.stderr.splitlines())
        if line.startswith("ERROR:")
    ]


def _show_failure(completed: subprocess.CompletedProcess[str]) -> None:
    print(completed.stdout[-8000:])
    print(completed.stderr[-3000:], file=sys.stderr)


def _stitch_script(commands: tuple[str, ...], part: str, report: Path) -> str:
    setup = "\n".join(command for command in commands if not command.startswith("create_bd_cell"))
    cells = "\n".join(command for command in commands if command.startswith("create_bd_cell"))
    return f"""
create_project -in_memory -part {part}
{setup}
create_bd_design "stitch"
{cells}
create_bd_port -dir I -type clk ap_clk
set_property CONFIG.FREQ_HZ 250000000 [get_bd_ports ap_clk]
create_bd_port -dir I -type clk ap_clk2x
set_property CONFIG.FREQ_HZ 500000000 [get_bd_ports ap_clk2x]
create_bd_port -dir I -type rst ap_rst_n
set_property CONFIG.POLARITY ACTIVE_LOW [get_bd_ports ap_rst_n]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins supplied/ap_clk]
connect_bd_net [get_bd_ports ap_clk2x] [get_bd_pins supplied/ap_clk2x]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins supplied/ap_rst_n]
foreach pin [get_bd_intf_pins -quiet supplied/*] {{
    if {{[llength [get_bd_intf_nets -quiet -of_objects $pin]] == 0}} {{
        make_bd_intf_pins_external $pin
    }}
}}
foreach pin [get_bd_pins -quiet supplied/*] {{
    if {{[llength [get_bd_nets -quiet -of_objects $pin]] == 0}} {{
        make_bd_pins_external $pin
    }}
}}
validate_bd_design
save_bd_design
set wrapper [file tail [make_wrapper -files [get_files stitch.bd] -top]]
set out [open {{{report}}} w]
puts $out "cell=[get_property NAME [get_bd_cells supplied]]"
puts $out "interfaces=[join [lsort [get_property NAME [get_bd_intf_pins supplied/*]]] ,]"
puts $out "wrapper=$wrapper"
close $out
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-synthesis",
        action="store_true",
        help="reuse an earlier OOC verdict while debugging packaging or stitching",
    )
    arguments = parser.parse_args(argv)

    root = Path(os.environ["FINN_ROOT"])
    library_root = Path(finnlib_root(root))
    if not (library_root / "rtl").is_dir():
        print(f"FinnLib RTL not found under {library_root}")
        return FAIL
    record_identity(str(root), str(library_root))

    weights = _weights(HARDWARE_CASE, np.random.RandomState(0xD6A))
    requirements = requirements_for(weights, root, HARDWARE_CASE)
    target = TargetIdentity(requirements.target_fpga_part, requirements.clock_period_ns)
    builder = _builder()

    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        packaged = package_decomposed_artifact(requirements, scratch / "packages")
        print(f"unit: {Path(packaged.directory).name}")
        print(f"build: {builder.backend_id} {builder.tool_version}")

        if arguments.skip_synthesis:
            print("OOC synthesis: NOT RERUN (earlier verdict retained)")
        else:
            synthesis = prepare_decomposed_synthesis(
                packaged, target, scratch / "synthesis", builder=builder
            )
            synthesized = _run_vivado(Path(synthesis.directory), Path(synthesis.script_path))
            synth_errors = _errors(synthesized)
            if (
                synthesized.returncode != 0
                or synth_errors
                or not Path(synthesis.report_path).is_file()
            ):
                _show_failure(synthesized)
                print("OOC synthesis: FAIL")
                return FAIL
            if _ADDRESS_WIDTH_WARNING.search(synthesized.stdout + synthesized.stderr):
                _show_failure(synthesized)
                print("OOC synthesis: FAIL (top-level AXI-Lite address widths disagree)")
                return FAIL
            completed_synthesis = complete_decomposed_synthesis(synthesis)
            report = Path(completed_synthesis.report_path).read_text()
            cells = {name: int(count) for name, count in _UTILIZATION.findall(report)}
            inferred = sum(count for name, count in cells.items() if name != _SUMMARY_ROW)
            if inferred <= 0:
                print(f"OOC synthesis: FAIL (no DSP primitive in {cells or 'report'})")
                return FAIL
            memories = {name: float(count) for name, count in _BRAM_UTILIZATION.findall(report)}
            if not memories or max(memories.values()) <= 0:
                print(f"OOC synthesis: FAIL (no BRAM resource in {memories or 'report'})")
                return FAIL
            print(f"OOC synthesis: PASS ({inferred} DSP primitives, BRAM resources {memories})")

        prepared_ip = prepare_ip_package(
            packaged, target.fpga_part, scratch / "ip", builder=builder
        )
        packaging = _run_vivado(Path(prepared_ip.directory), Path(prepared_ip.script_path))
        package_errors = _errors(packaging)
        if packaging.returncode != 0 or package_errors:
            _show_failure(packaging)
            print("IP-XACT packaging: FAIL")
            return FAIL
        component = complete_ip_package(prepared_ip)
        print(f"IP-XACT packaging: PASS ({component.vlnv})")

        stitch_directory = scratch / "stitch"
        stitch_directory.mkdir()
        stitch_report = stitch_directory / "stitch.txt"
        stitch_script = stitch_directory / "stitch.tcl"
        stitch_script.write_text(
            _stitch_script(
                component.instantiation_commands("supplied"), target.fpga_part, stitch_report
            )
        )
        stitching = _run_vivado(stitch_directory, stitch_script)
        stitch_errors = _errors(stitching)
        if stitching.returncode != 0 or stitch_errors or not stitch_report.is_file():
            _show_failure(stitching)
            print("catalog stitch: FAIL")
            return FAIL
        observed = dict(
            line.split("=", 1) for line in stitch_report.read_text().splitlines() if "=" in line
        )
        interfaces = set(filter(None, observed.get("interfaces", "").split(",")))
        required = {"in0_V", "out0_V", "s_axis_0", "s_axilite"}
        missing = required - interfaces
        cell = observed.get("cell", "").lstrip("/")
        if cell != "supplied" or not observed.get("wrapper") or missing:
            print(f"catalog stitch: FAIL (cell={observed.get('cell')}, missing={sorted(missing)})")
            return FAIL
        print(f"catalog stitch: PASS (interfaces {sorted(interfaces)})")

    print("RESULT: D6A HARDWARE PASS")
    return PASS


if __name__ == "__main__":
    sys.exit(main())
