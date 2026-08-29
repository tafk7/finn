############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Fixture 6: the decomposed MVAU synthesizes, with real DSPs inferred.

Runs INSIDE the FINN Docker container; needs Vivado.

Fixture 5 compares behaviour, and it does so with ``FORCE_BEHAVIORAL`` at the
value the design point declares -- which is 0, so it is already the synthesis
description being simulated.  That still leaves the question this answers: does
the composed structure actually *synthesize*, and does the tool infer DSP
primitives rather than falling back to fabric?

Falling back would not fail a simulation.  It would silently produce something
with the right numbers and the wrong area and timing, which is precisely the
failure a behavioural comparison cannot see.

Usage, from ``finn/``::

    bash run-docker.sh bash tests/dataflow/rtlsim/run_composed_synth.sh
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from dataflow.rtlsim.composed_mvau_equiv import (
    CONFIGS,
    CONFIGS_BY_LABEL,
    Config,
    decomposed_requirements,
    record_identity,
)
from finn.dataflow.mvau.hardware.binding import finnlib_root
from finn.dataflow.mvau.hardware.composition import write_decomposed_artifact

#: Synthesis is slow and mostly repeats itself, so only one configuration per
#: DSP generation runs by default -- the generation is what changes the
#: primitive the tool infers, and that is the thing under test.
#:
#: ``packed`` targets Versal, whose synthesis needs a licensed feature that a
#: plain development machine does not have.  It stays in the list because the
#: DSP58 path deserves the check wherever the license exists; a missing licence
#: is reported as SKIPPED, not as a pass and not as an RTL failure.
DEFAULT_LABELS = ("softvec", "packed")

#: ``report_utilization`` prints a summary row (``DSPs``) and a detail row per
#: primitive (``DSP48E2``, ``DSP58``, ...).  Summing them double-counts, so the
#: detail rows are the ones that matter and the summary is only a cross-check.
_SUMMARY_ROW = "DSPs"

_UTILIZATION = re.compile(r"^\|\s*(DSP\w*)\s*\|\s*(\d+)\s*\|", re.MULTILINE)

#: How Vivado reports a device or feature this installation is not licensed for.
_UNLICENSED = "A valid license was not found"

#: Set on the per-configuration workers, which share the dispatcher's log and
#: would otherwise repeat its header once per configuration.
_IDENTITY_RECORDED = "FIXTURE6_IDENTITY_RECORDED"


def _tcl(top: str, part: str, sources: list[str], report: Path) -> str:
    reads = "\n".join(f"read_verilog -sv {{{path}}}" for path in sources)
    return f"""
{reads}
synth_design -top {top} -part {part} -mode out_of_context
report_utilization -file {{{report}}}
"""


def _run_vivado(directory: Path, script: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "vivado",
            "-mode",
            "batch",
            "-nojournal",
            "-nolog",
            "-notrace",
            "-source",
            str(script),
        ],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )


#: Exit codes the dispatcher distinguishes.  A skip is not a pass: it says the
#: question was never asked, which is what the summary has to report.
PASS, FAIL, SKIP = 0, 1, 2


def run_one(config: Config) -> int:
    print(f"\n========== fixture 6: {config.label} ==========")
    requirements = decomposed_requirements(config)
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        sources = list(write_decomposed_artifact(requirements, directory))
        report = directory / "utilization.rpt"
        script = directory / "synth.tcl"
        script.write_text(
            _tcl(
                requirements.top_module_name,
                requirements.target_fpga_part,
                sources,
                report,
            )
        )
        print(f"  top:  {requirements.top_module_name} on {requirements.target_fpga_part}")
        completed = _run_vivado(directory, script)
        errors = [line for line in completed.stdout.splitlines() if line.startswith("ERROR:")]
        if any(_UNLICENSED in line for line in errors):
            print(f"  {config.label.upper()}: SKIPPED (no license for this device)")
            return SKIP
        if completed.returncode != 0 or not report.is_file():
            print(completed.stdout[-4000:])
            print(completed.stderr[-2000:], file=sys.stderr)
            print(f"  {config.label.upper()}: FAIL (synthesis did not complete)")
            return FAIL
        if errors:
            print("\n".join(errors))
            print(f"  {config.label.upper()}: FAIL (synthesis reported errors)")
            return FAIL
        cells = {name: int(count) for name, count in _UTILIZATION.findall(report.read_text())}
        # Count the per-primitive rows only.  The report also carries a summary
        # row that totals them, and adding it in reports twice the DSPs there
        # actually are.
        inferred = sum(count for name, count in cells.items() if name != _SUMMARY_ROW)
        print(f"  cells: {cells or '(none reported)'}")
        if inferred <= 0:
            print(f"  {config.label.upper()}: FAIL (no DSP primitives inferred)")
            return FAIL
    print(f"  {config.label.upper()}: PASS ({inferred} DSP primitives)")
    return PASS


def _run_each_in_its_own_process(labels: list[str]) -> tuple[int, int, int]:
    passed = failed = skipped = 0
    environment = {**os.environ, _IDENTITY_RECORDED: "1"}
    for label in labels:
        completed = subprocess.run(
            [sys.executable, __file__, "--config", label], check=False, env=environment
        )
        if completed.returncode == PASS:
            passed += 1
        elif completed.returncode == SKIP:
            skipped += 1
        else:
            print(f"  {label.upper()}: FAIL (exit {completed.returncode})")
            failed += 1
    return passed, failed, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(CONFIGS_BY_LABEL))
    parser.add_argument(
        "--all",
        action="store_true",
        help="synthesize every configuration, not just one per DSP generation",
    )
    arguments = parser.parse_args(argv)

    finn_root = os.environ["FINN_ROOT"]
    library_root = str(finnlib_root(finn_root))
    if not os.path.isdir(os.path.join(library_root, "rtl")):
        print(f"FinnLib RTL not found under {library_root}; set FINNLIB_ROOT or fetch-repos.sh")
        return FAIL

    # A synthesis result that does not say which revisions it synthesized is
    # not a result.  Fixture 5 prints this and fixture 6 did not, so its log
    # could not be attributed to a checkout at all.
    if not os.environ.get(_IDENTITY_RECORDED):
        record_identity(finn_root, library_root)

    if arguments.config is not None:
        return run_one(CONFIGS_BY_LABEL[arguments.config])

    labels = [config.label for config in CONFIGS] if arguments.all else list(DEFAULT_LABELS)
    passed, failed, skipped = _run_each_in_its_own_process(labels)
    # Everything skipped is not a pass: nothing was actually synthesized, and
    # saying PASS there would be the fixture lying about its own coverage.
    ok = failed == 0 and passed > 0
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    print("RESULT:", "FIXTURE 6 PASS" if ok else "FIXTURE 6 FAIL")
    return PASS if ok else FAIL


if __name__ == "__main__":
    sys.exit(main())
