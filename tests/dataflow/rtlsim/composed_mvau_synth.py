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
from finn.dataflow.artifacts import DEFAULT_BUILDER, BuilderIdentity, TargetIdentity
from finn.dataflow.ops.mvau.binding import finnlib_root
from finn.dataflow.ops.mvau.artifacts._implementation import (
    complete_decomposed_synthesis,
    find_decomposed_synthesis,
    package_decomposed_artifact,
    prepare_decomposed_synthesis,
)
from finn.util.basic import get_vivado_version

#: Synthesis is slow and mostly repeats itself, so only one configuration per
#: DSP generation runs by default -- the generation is what changes the
#: primitive the tool infers, and that is the thing under test.
#:
#: ``packed`` targets Versal, whose synthesis needs a licensed feature that a
#: plain development machine does not have.  It stays in the list because the
#: DSP58 path deserves the check wherever the license exists; a missing licence
#: is reported as SKIPPED, not as a pass and not as an RTL failure.
#:
#: ``dsp48e1`` joined in Phase 6e.  It is a third generation and therefore a
#: third primitive to infer, and it is the one with no prior evidence at all:
#: it was in the part table from the first fixture and in no configuration, so
#: nothing had ever synthesized it.  ``xc7z020clg400-1`` is 7-series and needs
#: no licensed feature.
DEFAULT_LABELS = ("softvec", "packed", "dsp48e1")

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


def _builder() -> BuilderIdentity:
    """The tool this run will actually use, read by the *caller*.

    Identity construction stays a pure function of its arguments -- that is why
    ``BuilderIdentity`` is a caller-supplied label rather than a probe.  A
    caller that knows its tool version is expected to pass it, and this fixture
    is one: it is about to invoke exactly this Vivado.  Reading it here is what
    closes the gap between the label and the run, rather than leaving the
    default ``unspecified`` to stand for a tool that was in fact identified.
    """

    version = get_vivado_version()
    if version is None:
        return DEFAULT_BUILDER
    return BuilderIdentity("vivado", f"{version[0]}.{version[1]}")


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
        # Synthesize the *packaged* unit, not a directory chosen here: the OOC
        # stage takes the packaged unit as its input, and a fixture that staged
        # its own copy would be synthesizing something adjacent to what a
        # consumer builds rather than the thing itself.
        packaged = package_decomposed_artifact(requirements, scratch)
        target = TargetIdentity(requirements.target_fpga_part, requirements.clock_period_ns)
        # The synthesis stage materializes into its *own* directory, named from
        # its own key.  Writing a script, constraints and a report into the
        # packaged unit would mutate an immutable artifact, and two runs of one
        # package -- another part, another clock, another tool -- would
        # overwrite each other.
        builder = _builder()
        done = find_decomposed_synthesis(packaged, target, builder=builder)
        if done is not None:  # pragma: no cover - no store is configured here
            print(f"  reuse: {Path(done.directory).name}")
            return PASS
        synthesis = prepare_decomposed_synthesis(packaged, target, scratch, builder=builder)
        directory = Path(synthesis.directory)
        report = Path(synthesis.report_path)
        print(f"  top:   {packaged.top_module_name} on {target.fpga_part}")
        print(f"  unit:  {Path(packaged.directory).name}")
        print(f"  clock: {target.clock_period_ns} ns")
        print(f"  build: {builder.backend_id} {builder.tool_version}")
        print(f"  synth: {directory.name}")
        print(f"  key:   {synthesis.key}")
        # The packaged unit is untouched by this run, which is the whole point
        # of the stage having a place of its own.
        assert directory != Path(packaged.directory)
        assert sorted(item.name for item in Path(packaged.directory).iterdir()) == sorted(
            packaged.identity.layout
        ), "synthesis wrote into the packaged unit"
        completed = _run_vivado(directory, Path(synthesis.script_path))
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
        # The run is only complete once its declared outputs exist, and saying
        # so is the caller's job because the caller is what invoked the tool.
        completed = complete_decomposed_synthesis(synthesis)
        assert completed.report_path == str(report)
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
