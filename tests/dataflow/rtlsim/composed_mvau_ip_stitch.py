############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Fixture 9: the unit enters an IP repository and stitches beside another layer.

Runs INSIDE the FINN Docker container; needs Vivado.

Fixture 7 places the packaged unit with ``add_files`` and ``create_bd_cell
-type module``.  That is a block design assembled by hand.  FINN's own build
path does something different for every layer: it points ``ip_repo_paths`` at a
repository and resolves a VLNV, and it connects layers *by interface* rather
than pin by pin.  Item 7 of the migration plan names ``ip_repo_paths``
specifically, and this is what it asks for.

Three claims, and each one can fail on its own:

- **The component resolves.**  ``ipx::package_project`` produced something an
  IP catalog will index, and ``create_bd_cell -type ip -vlnv`` finds it.
- **Its interfaces are in the component, not only in the eye of a block
  design.**  Fixture 7 saw ``in0_V`` inferred because ``-type module`` re-runs
  inference over the source every time.  A packaged IP has to *carry* the
  inference, or a stitcher that connects interfaces finds loose pins.  That is
  why the unit is connected to real AXI-Stream IP here -- as a source on one
  cell and as a sink on the other, since the two directions are inferred
  separately and can be wrong separately.
- **Two placements resolve one component.**  One artifact, many placements is
  the whole of Phase 5; through a repository it means two cells and one VLNV.

Scope, per the Phase 6 plan §6.3 and §16.4 before it: stitch and validate.  No
bitstream, no deployment, no driver.  Numerical behaviour is fixture 8's, on
the same RTL these cells contain.

Usage, from ``finn/``::

    bash run-docker.sh bash tests/dataflow/rtlsim/run_composed_ip_stitch.sh
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from dataflow.rtlsim.composed_mvau_equiv import (
    CONFIGS_BY_LABEL,
    Config,
    decomposed_requirements,
    record_identity,
)
from finn.dataflow.hardware import DEFAULT_BUILDER, BuilderIdentity
from finn.dataflow.ops.mvau.hardware.binding import finnlib_root
from finn.dataflow.ops.mvau.hardware.composition import (
    PackagedDecomposedArtifact,
    complete_ip_package,
    find_ip_package,
    package_decomposed_artifact,
    prepare_ip_package,
)
from finn.util.basic import get_vivado_version

#: One configuration is enough.  The claim is structural -- does a component
#: resolve, does it carry its interfaces, do two cells share it -- and none of
#: that varies with folding or DSP generation.  An UltraScale+ part, because
#: block-design work needs no licensed feature there.
DEFAULT_LABEL = "softvec"

#: The two cells, so reuse through a repository is exercised rather than
#: asserted.
INSTANCES = ("mvau_first", "mvau_second")

#: The other layer.  A stock AXI-Stream FIFO, chosen because it is *not* ours:
#: connecting our inferred interface to a Xilinx one is what says the inference
#: produced a real ``axis_rtl`` interface rather than a plausible-looking name.
FIFO_VLNV = "xilinx.com:ip:axis_data_fifo:2.0"

#: Which of our streams each FIFO is attached to, and in which direction.  The
#: output of one cell feeds a FIFO; another FIFO feeds the input of the other.
#: Both directions, because a source interface and a sink interface are
#: inferred separately and can be wrong separately.
#:
#: Named by the unit's own *role* -- ``activation``, ``output`` -- rather than
#: by a bus name typed here.  The bus name is Vivado's, derived from the signal
#: prefix, so :func:`bus_name` reads it off the unit's reported signals.  A
#: fixture that hard-coded ``out0_V`` would keep passing over a unit that had
#: renamed the port, which is the class of defect fixture 7 found.
FIFO_LINKS = (
    ("out_fifo", INSTANCES[0], "output", "S_AXIS"),
    ("in_fifo", INSTANCES[1], "activation", "M_AXIS"),
)


def beat_bytes(packaged: PackagedDecomposedArtifact, role: str) -> int:
    """How wide one beat of a stream is, in bytes, as the unit reports it.

    The other layer has to be told: ``axis_data_fifo`` defaults to one byte and
    a block design refuses to connect a two-byte stream to it.  Read off the
    unit rather than typed here, so the enclosing design is sized by what is
    actually being stitched.
    """

    for item in packaged.stream_interfaces:
        if item.id.rsplit(".", 1)[-1] == role:
            return item.physical_width_bits // 8
    raise AssertionError(f"the packaged unit publishes no {role!r} stream")


def bus_name(packaged: PackagedDecomposedArtifact, role: str) -> str:
    """The bus interface Vivado infers for one of the unit's streams.

    Inference groups ``<prefix>_tdata``/``_tvalid``/``_tready`` under
    ``<prefix>``, so the name follows from the signals the unit publishes.
    Reading it off them is what keeps this fixture describing the unit rather
    than describing itself.
    """

    for item in packaged.stream_interfaces:
        if item.id.rsplit(".", 1)[-1] == role:
            return item.data_signal.rsplit("_", 1)[0]
    raise AssertionError(f"the packaged unit publishes no {role!r} stream")


PASS, FAIL, SKIP = 0, 1, 2

_UNLICENSED = "A valid license was not found"

#: The clock the block design runs at.  It is the fixture's, not the unit's --
#: nothing in a packaged component names an enclosing design's clock.
CLOCK_HZ = 250000000


def _builder() -> BuilderIdentity:
    """The tool this run will actually use, read by the caller.

    Identity construction stays a pure function of its arguments; a caller that
    knows its tool version passes it, and this one is about to invoke exactly
    this Vivado.  Same argument as fixture 6.
    """

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


def _stitch_tcl(
    cells: list[str],
    part: str,
    report: Path,
    buses: Mapping[str, str],
    widths: Mapping[str, int],
) -> str:
    """Build the design, connect by interface, validate, and report.

    Every command that places our unit comes from the packaged component
    itself; nothing here reconstructs a VLNV or a repository path, because a
    command this fixture wrote would be a claim about this fixture.

    What *is* written here is the enclosing design: the clock, the reset, and
    the stock FIFOs.  Those are the caller's by definition -- a packaged
    component cannot name the design it will be dropped into.
    """

    fifos = "\n".join(
        f"create_bd_cell -type ip -vlnv {FIFO_VLNV} {name}\n"
        f"set_property CONFIG.TDATA_NUM_BYTES {widths[role]} [get_bd_cells {name}]"
        for name, _, role, _ in FIFO_LINKS
    )
    links = "\n".join(
        f"connect_bd_intf_net [get_bd_intf_pins {instance}/{buses[role]}] "
        f"[get_bd_intf_pins {name}/{theirs}]"
        for name, instance, role, theirs in FIFO_LINKS
    )
    clocked = [*INSTANCES, *(name for name, _, _, _ in FIFO_LINKS)]
    nets = "\n".join(
        f"connect_bd_net [get_bd_ports ap_clk] [get_bd_pins {instance}/ap_clk]\n"
        f"connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins {instance}/ap_rst_n]"
        if instance in INSTANCES
        else f"connect_bd_net [get_bd_ports ap_clk] [get_bd_pins {instance}/s_axis_aclk]\n"
        f"connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins {instance}/s_axis_aresetn]"
        for instance in clocked
    )
    doubled = "\n".join(
        f"connect_bd_net [get_bd_ports ap_clk2x] [get_bd_pins {instance}/ap_clk2x]"
        for instance in INSTANCES
    )
    reports = "\n".join(
        f"dict set intf {instance} [lsort [get_property NAME [get_bd_intf_pins {instance}/*]]]"
        for instance in INSTANCES
    )
    return f"""
create_project -in_memory -part {part}
create_bd_design "stitch"
{chr(10).join(cells)}
{fifos}
create_bd_port -dir I -type clk ap_clk
set_property CONFIG.FREQ_HZ {CLOCK_HZ} [get_bd_ports ap_clk]
create_bd_port -dir I -type clk ap_clk2x
set_property CONFIG.FREQ_HZ {2 * CLOCK_HZ} [get_bd_ports ap_clk2x]
create_bd_port -dir I -type rst ap_rst_n
set_property CONFIG.POLARITY ACTIVE_LOW [get_bd_ports ap_rst_n]
{nets}
{doubled}
{links}
# Everything not wired above becomes an external port, so the design is
# complete rather than merely syntactically valid.
foreach pin [get_bd_intf_pins -quiet -of_objects [get_bd_cells]] {{
    if {{[llength [get_bd_intf_nets -quiet -of_objects $pin]] == 0}} {{
        make_bd_intf_pins_external $pin
    }}
}}
validate_bd_design
save_bd_design
# ``make_wrapper`` returns the file it generated.  Reading its result is the
# only reliable way to name it: an in-memory project never adds the generated
# wrapper to a fileset, so ``get_files`` finds nothing and the design looks
# like it failed to elaborate when it did.
set wrapper [file tail [make_wrapper -files [get_files stitch.bd] -top]]
set intf [dict create]
{reports}
proc emit {{d}} {{
    return [json::write object {{*}}[concat {{*}}[lmap k [dict keys $d] {{
        list $k [json::write array {{*}}[lmap p [dict get $d $k] {{json::write string $p}}]]
    }}]]]
}}
set out [open {{{report}}} w]
puts $out [json::write object intf [emit $intf] \\
    cells [json::write array {{*}}[lmap c [get_bd_cells] {{json::write string $c}}]] \\
    wrapper [json::write string $wrapper]]
close $out
"""


def run_one(config: Config) -> int:
    print(f"\n========== fixture 9: {config.label} ==========")
    requirements = decomposed_requirements(config)
    builder = _builder()
    with tempfile.TemporaryDirectory() as scratch:
        packaged = package_decomposed_artifact(requirements, scratch)
        part = requirements.target_fpga_part

        done = find_ip_package(packaged, part, builder=builder)
        if done is not None:  # pragma: no cover - no store is configured here
            print(f"  reuse: {Path(done.directory).name}")
            component = done
        else:
            prepared = prepare_ip_package(packaged, part, scratch, builder=builder)
            print(f"  unit:  {Path(packaged.directory).name}")
            print(f"  ip:    {Path(prepared.directory).name}")
            print(f"  vlnv:  {prepared.vlnv}")
            print(f"  key:   {prepared.key}")
            print(f"  build: {builder.backend_id} {builder.tool_version}")
            # The packaged unit is read and not written, which is the whole
            # point of this stage having a place of its own.
            before = sorted(item.name for item in Path(packaged.directory).iterdir())
            packaging = _run_vivado(Path(prepared.directory), Path(prepared.script_path))
            errors = [line for line in packaging.stdout.splitlines() if line.startswith("ERROR:")]
            if any(_UNLICENSED in line for line in errors):
                print(f"  {config.label.upper()}: SKIPPED (no license for this device)")
                return SKIP
            if packaging.returncode != 0 or errors:
                print(packaging.stdout[-6000:])
                print(packaging.stderr[-2000:], file=sys.stderr)
                print(f"  {config.label.upper()}: FAIL (packaging did not complete)")
                return FAIL
            after = sorted(item.name for item in Path(packaged.directory).iterdir())
            assert after == before, "packaging wrote into the packaged unit"
            # The caller invoked the tool, so the caller says the run is done --
            # and this refuses unless the component description exists.
            component = complete_ip_package(prepared)
            print(f"  component: {Path(component.component_path).name}")

        # Every command placing our unit comes from the component itself.  Both
        # cells ask for the same repository, so those are deduplicated while
        # keeping first-seen order; the two create lines are kept.
        cells: list[str] = []
        for instance in INSTANCES:
            for command in component.instantiation_commands(instance):
                if command.startswith("create_bd_cell") or command not in cells:
                    cells.append(command)
        for command in cells:
            if command.startswith("create_bd_cell"):
                print(f"  cell:  {command}")

        stitch_dir = Path(scratch) / "stitch"
        stitch_dir.mkdir()
        report = stitch_dir / "stitch.json"
        script = stitch_dir / "stitch.tcl"
        buses = {role: bus_name(packaged, role) for _, _, role, _ in FIFO_LINKS}
        widths = {role: beat_bytes(packaged, role) for _, _, role, _ in FIFO_LINKS}
        script.write_text(
            "package require json::write\n" + _stitch_tcl(cells, part, report, buses, widths)
        )
        completed = _run_vivado(stitch_dir, script)
        errors = [line for line in completed.stdout.splitlines() if line.startswith("ERROR:")]
        if any(_UNLICENSED in line for line in errors):
            print(f"  {config.label.upper()}: SKIPPED (no license for this device)")
            return SKIP
        if completed.returncode != 0 or not report.is_file():
            print(completed.stdout[-8000:])
            print(completed.stderr[-2000:], file=sys.stderr)
            print(f"  {config.label.upper()}: FAIL (the block design did not build)")
            return FAIL
        if errors:
            print("\n".join(errors))
            print(f"  {config.label.upper()}: FAIL (Vivado reported errors)")
            return FAIL

        actual = json.loads(report.read_text())
        ok = True
        for instance in INSTANCES:
            interfaces = set(actual["intf"].get(instance, ()))
            print(f"  {instance}: interfaces {sorted(interfaces)}")
            missing = {name for name in ("in0_V", "in1_V", "out0_V") if name not in interfaces}
            if missing:
                # The failure this fixture exists for: the component did not
                # carry an interface the unit publishes, so a stitcher sees
                # loose pins where it expects a stream.
                print(f"    absent from the component: {sorted(missing)}")
                ok = False
        # ``get_bd_cells`` yields path strings -- ``/in_fifo`` -- and the leading
        # separator is the design root rather than part of the name.
        placed = {str(name).lstrip("/") for name in actual["cells"]}
        for name, _instance, _role, _theirs in FIFO_LINKS:
            if name not in placed:
                print(f"    the other layer {name} was not placed")
                ok = False
        if not actual.get("wrapper"):
            print("    no wrapper was generated, so the design did not elaborate")
            ok = False
        else:
            print(f"  wrapper: {actual['wrapper']}")
    print(f"  {config.label.upper()}: {'PASS' if ok else 'FAIL'}")
    return PASS if ok else FAIL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(CONFIGS_BY_LABEL), default=DEFAULT_LABEL)
    arguments = parser.parse_args(argv)

    finn_root = os.environ["FINN_ROOT"]
    library_root = str(finnlib_root(finn_root))
    if not os.path.isdir(os.path.join(library_root, "rtl")):
        print(f"FinnLib RTL not found under {library_root}; set FINNLIB_ROOT or fetch-repos.sh")
        return FAIL
    record_identity(finn_root, library_root)

    outcome = run_one(CONFIGS_BY_LABEL[arguments.config])
    # A skip is not a pass: it says the question was never asked.
    print(
        "RESULT:",
        {PASS: "FIXTURE 9 PASS", FAIL: "FIXTURE 9 FAIL"}.get(outcome, "FIXTURE 9 SKIPPED"),
    )
    return outcome if outcome != SKIP else FAIL


if __name__ == "__main__":
    sys.exit(main())
