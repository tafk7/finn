############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Fixture 7: the packaged unit really is stitchable.

Runs INSIDE the FINN Docker container; needs Vivado.

Fixture 6 synthesizes the staged sources directly, so it proves the RTL builds
and never executes the instantiation command or touches a reported pin name.
That left the two claims packaging actually makes unvalidated -- and writing
this fixture found both of them wrong.

- **The generated ``.sv`` top could not be placed in a block design at all.**
  ``create_bd_cell -type module -reference`` refuses a SystemVerilog top file
  (``[filemgmt 56-195] ... not allowed as the top file in the reference``), and
  ``-type hier -reference`` is not the alternative it appears to be: Vivado
  reports ``[BD 41-1695] Specified '-reference' ... will be ignored while
  creating 'hier' cell`` and silently makes an empty hierarchy.  A plain-Verilog
  shim over the same ports is what works, which is why baseline FINN stages
  ``{gen_top_module}_wrapper.v``.  The packaged unit now emits one.
- **The reported signal names did not exist on the cell.**  The model said
  ``in0_V_TDATA`` where the RTL declares ``in0_V_tdata``.  A unit test can
  compare the model against the generated text -- and now does -- but only
  Vivado can say the *cell* has that pin.

Neither was reachable by reading code, which is the argument for the fixture.

So this builds a real block design, runs the unit's own
``instantiation_commands`` verbatim, and then asks the tool to list the cell's
pins and compare them with what the unit reported.  Nothing here re-derives a
command or a name: anything this fixture had to reconstruct would be a claim
about the fixture rather than about the packaged unit.

It also instantiates the unit **twice**, because "one artifact, many
placements" is the whole of Phase 5 and two cells over one staged directory is
what that means in a block design.

Usage, from ``finn/``::

    bash run-docker.sh bash tests/dataflow/rtlsim/run_composed_stitch.sh
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from dataflow.rtlsim.composed_mvau_equiv import (
    CONFIGS_BY_LABEL,
    Config,
    decomposed_requirements,
    record_identity,
)
from finn.dataflow.ops.mvau.binding import finnlib_root
from finn.dataflow.ops.mvau.artifacts._implementation import package_decomposed_artifact

#: One configuration is enough.  The claim is structural -- the command form and
#: the pin names -- and neither varies with folding or DSP generation.  A Zynq
#: part because block-design creation needs no licensed feature there.
DEFAULT_LABEL = "softvec"

#: The two cells, so reuse is exercised rather than asserted.
INSTANCES = ("mvau_first", "mvau_second")

PASS, FAIL, SKIP = 0, 1, 2

_UNLICENSED = "A valid license was not found"


def _tcl(
    adds: list[str], cells: list[str], part: str, instances: tuple[str, ...], report: Path
) -> str:
    """Create a block design, run the unit's own commands, then report the pins.

    ``create_project`` in memory and ``create_bd_design`` are the minimum that
    makes ``create_bd_cell`` meaningful; there is no synthesis here, because
    fixture 6 already owns that question.

    The unit's commands are split into its ``add_files`` and its
    ``create_bd_cell`` lines only so that ``update_compile_order`` can go
    between them.  Nothing is rewritten -- each line is emitted verbatim and in
    order, because a command this fixture reconstructed would be a claim about
    the fixture rather than about the packaged unit.

    ``source_mgmt_mode All`` is likewise a property of how this fixture opens a
    project and not of the unit.  An in-memory project defaults to *manual*
    compile order, and in that mode Vivado ignores module references entirely:
    ``create_bd_cell -type module`` fails with "Failed to resolve reference",
    which reads exactly like the module not existing.

    The pin list is written as JSON so the comparison happens in Python against
    the packaged unit, not in Tcl against a list this file typed out.
    """

    # Both kinds.  Given the ``_V_t*`` naming Vivado infers ``in0_V``, ``in1_V``
    # and ``out0_V`` as AXI-Stream bus interfaces, and their members then stop
    # being plain pins -- so a check that only looked at ``get_bd_pins`` would
    # report the streams missing exactly when stitching had worked best.
    reports = "\n".join(
        f"dict set pins {instance} [lsort [get_property NAME [get_bd_pins {instance}/*]]]\n"
        f"dict set intf {instance} "
        f"[lsort [get_property NAME [get_bd_intf_pins {instance}/*]]]"
        for instance in instances
    )
    return f"""
create_project -in_memory -part {part}
set_property source_mgmt_mode All [current_project]
{chr(10).join(adds)}
update_compile_order -fileset sources_1
create_bd_design "stitch"
{chr(10).join(cells)}
set pins [dict create]
set intf [dict create]
{reports}
proc emit {{d}} {{
    return [json::write object {{*}}[concat {{*}}[lmap k [dict keys $d] {{
        list $k [json::write array {{*}}[lmap p [dict get $d $k] {{json::write string $p}}]]
    }}]]]
}}
set out [open {{{report}}} w]
puts $out [json::write object pins [emit $pins] intf [emit $intf]]
close $out
"""


def _run_vivado(directory: Path, script: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["vivado", "-mode", "batch", "-nojournal", "-nolog", "-notrace", "-source", str(script)],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )


def run_one(config: Config, finn_root: str) -> int:
    print(f"\n========== fixture 7: {config.label} ==========")
    requirements = decomposed_requirements(config)
    with tempfile.TemporaryDirectory() as scratch:
        packaged = package_decomposed_artifact(requirements, scratch)
        directory = Path(packaged.directory)
        report = directory / "pins.json"

        # Every command the block design runs comes from the unit itself.  Both
        # placements ask for the same ``add_files`` -- one artifact, two cells --
        # so those are deduplicated while keeping first-seen order, and the two
        # ``create_bd_cell`` lines are kept.
        adds: list[str] = []
        cells: list[str] = []
        for instance in INSTANCES:
            for command in packaged.instantiation_commands(instance):
                target = adds if command.startswith("add_files") else cells
                if command not in target:
                    target.append(command)

        script = directory / "stitch.tcl"
        script.write_text(
            "package require json::write\n"
            + _tcl(adds, cells, requirements.target_fpga_part, INSTANCES, report)
        )
        print(f"  unit: {directory.name}")
        for command in cells:
            print(f"  cell: {command}")

        completed = _run_vivado(directory, script)
        errors = [line for line in completed.stdout.splitlines() if line.startswith("ERROR:")]
        if any(_UNLICENSED in line for line in errors):
            print(f"  {config.label.upper()}: SKIPPED (no license for this device)")
            return SKIP
        if completed.returncode != 0 or not report.is_file():
            print(completed.stdout[-6000:])
            print(completed.stderr[-2000:], file=sys.stderr)
            print(f"  {config.label.upper()}: FAIL (the block design did not build)")
            return FAIL
        if errors:
            print("\n".join(errors))
            print(f"  {config.label.upper()}: FAIL (Vivado reported errors)")
            return FAIL

        actual = json.loads(report.read_text())
        reported = {
            signal
            for interface in packaged.stream_interfaces
            for signal in (
                interface.data_signal,
                interface.valid_signal,
                interface.ready_signal,
            )
        } | {item.signal for item in packaged.control_interfaces}

        ok = True
        for instance in INSTANCES:
            pins = set(actual["pins"].get(instance, ()))
            interfaces = set(actual["intf"].get(instance, ()))
            if not pins and not interfaces:
                print(f"  {instance}: nothing reported -- the cell was not created")
                ok = False
                continue
            # A reported name is satisfied either as a plain pin or as a member
            # of an inferred bus interface, whose pins Vivado groups under the
            # interface name.
            missing = {
                name
                for name in reported
                if name not in pins and not any(name.startswith(f"{item}_") for item in interfaces)
            }
            print(f"  {instance}: {len(pins)} pins, interfaces {sorted(interfaces)}")
            if missing:
                # The failure this fixture exists for: a name the unit publishes
                # that the instantiated cell does not have.
                print(f"    reported but absent from the cell: {sorted(missing)}")
                print(f"    the cell actually has: {sorted(pins)}")
                ok = False

        if not ok:
            print(f"  {config.label.upper()}: FAIL (reported interface does not match the cell)")
            return FAIL
        print(f"  {config.label.upper()}: PASS (two cells, every reported pin present)")
        return PASS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_LABEL, choices=sorted(CONFIGS_BY_LABEL))
    arguments = parser.parse_args()

    root = str(Path(__file__).resolve().parents[3])
    # The resolved FinnLib checkout, not the FINN root again: the header exists
    # so a passing run says which revisions it passed against, and reporting
    # FINN's hash under both names says nothing about FinnLib at all.
    record_identity(root, str(finnlib_root(root)))

    outcome = run_one(CONFIGS_BY_LABEL[arguments.config], root)
    counts = {PASS: "1 passed", FAIL: "1 failed", SKIP: "1 skipped"}
    print(f"\n{counts[outcome]}")
    print("RESULT:", "FIXTURE 7 PASS" if outcome == PASS else "FIXTURE 7 FAIL")
    # A skip is not a pass: it says the question was never asked.
    return 0 if outcome == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
