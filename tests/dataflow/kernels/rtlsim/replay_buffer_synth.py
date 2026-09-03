# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel-only packaging and out-of-context synthesis evidence for ReplayBuffer.

The buffer is memory, not arithmetic, so the resource expectation is a storage
one: a depth-``LEN`` word buffer at width ``W`` must show up as registers or as
block RAM, and a run that reports neither has synthesized something other than
what the Kernel declared.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import cast

from finn.dataflow.artifacts.derivation import (
    ArtifactRef,
    ContentRef,
    Derivation,
    OutputLayout,
    ProducerIdentity,
    RequestSchema,
    ToolRequirement,
    build_key,
)
from finn.dataflow.artifacts.formats import RtlModuleDirectory
from finn.dataflow.artifacts.formats.rtl_module import RtlModuleOptions
from finn.dataflow.artifacts.packaging import Target, plan_package
from finn.dataflow.artifacts.projection import content_digest
from finn.dataflow.artifacts.request import (
    LogicalMount,
    PreparedToolRun,
    ResourceRequirements,
    ToolchainIdentity,
)
from finn.dataflow.kernels.artifacts import (
    kernel_source_derivation,
    portable_kernel_component,
    resolve_kernel_contributions,
)
from finn.dataflow.kernels.replay_buffer import FINNLIB_ROOT
from finn.util.basic import get_vivado_version

from dataflow.kernels.rtlsim.replay_buffer_numeric import (
    CASES_BY_LABEL,
    Case,
    _configure,
    record_identity,
)

PASS, FAIL, SKIP = 0, 1, 2
DEFAULT_LABELS = ("identity", "three_folds", "wide_lanes")
PART = "xczu3eg-sbva484-1-e"
CLOCK_PERIOD_NS = 4.0
UTILIZATION = re.compile(r"^\|\s*([A-Za-z][\w ]*?)\s*\|\s*(\d+)\s*\|", re.MULTILINE)
#: Categories that count as a real ``LEN``-deep store, as opposed to the
#: handshake registers every stream endpoint has.
MEMORY = ("Block RAM Tile", "LUT as Memory")
REGISTERS = ("Register as Flip Flop", "Register as Latch")
UNLICENSED = "A valid license was not found"


class _Contents:
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self.blobs = blobs

    def get_blob(self, reference: ContentRef) -> bytes:
        return self.blobs[reference.digest]


def _toolchain() -> ToolchainIdentity:
    version = get_vivado_version()
    text = "unknown" if version is None else f"{version[0]}.{version[1]}"
    return ToolchainIdentity("vivado", text, install_id="local-vivado")


def _package(case: Case, directory: Path):
    kernel = _configure(case)
    finnlib = Path(os.environ["FINNLIB_ROOT"])
    resolved = resolve_kernel_contributions(kernel, roots={FINNLIB_ROOT: finnlib})
    source_derivation = kernel_source_derivation(kernel, resolved)
    source_ref = ArtifactRef(source_derivation.kind, build_key(source_derivation))
    component = portable_kernel_component(kernel, source_ref, resolved)
    blobs = {
        source.content.digest: (finnlib / source.path).read_bytes()
        for source in resolved.definition.files
    }
    package = plan_package(
        RtlModuleDirectory(),
        component,
        Target(PART),
        RtlModuleOptions(),
        _Contents(blobs),
    )
    package_dir = directory / "package"
    package_dir.mkdir()
    for name, data in package.contents:
        target = package_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return kernel, package, package_dir


def _synthesis_derivation(
    package_kind: str,
    package_key: str,
    constraints: bytes,
    toolchain: ToolchainIdentity,
) -> Derivation:
    return Derivation(
        kind="ooc-synthesis",
        schema_version="replay-kernel-ooc-v1",
        producer=ProducerIdentity("finn.kernel.replay-buffer-ooc", "1"),
        inputs=(
            ("package", ArtifactRef(package_kind, package_key)),
            ("constraints", ContentRef(content_digest(constraints))),
        ),
        request=RequestSchema("vivado -mode batch -source {script}"),
        tool=ToolRequirement("vivado", toolchain.version),
        options=(("part", PART), ("clock_period_ns", CLOCK_PERIOD_NS)),
        outputs=OutputLayout(("utilization.rpt",)),
    )


def run_one(case: Case) -> int:
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        kernel, package, package_dir = _package(case, scratch)
        constraints = (
            f"create_clock -period {CLOCK_PERIOD_NS} -name clk [get_ports clk]\n"
        ).encode()
        constraints_path = scratch / "clock.xdc"
        constraints_path.write_bytes(constraints)
        report = scratch / "utilization.rpt"
        script = scratch / "synth.tcl"
        sources = tuple(
            package_dir / name
            for name, _data in package.contents
            if Path(name).suffix in {".v", ".sv"}
        )
        generic = " ".join(f"{name}={value}" for name, value in kernel.parameters.items())
        script.write_text(
            "\n".join(
                (
                    *(f"read_verilog -sv {{{source}}}" for source in sources),
                    f"read_xdc {{{constraints_path}}}",
                    f"synth_design -top replay_buffer -part {PART} "
                    f"-mode out_of_context -generic {{{generic}}}",
                    f"report_utilization -file {{{report}}}",
                )
            )
            + "\n"
        )
        toolchain = _toolchain()
        synthesis = _synthesis_derivation(
            package.derivation.kind,
            build_key(package.derivation),
            constraints,
            toolchain,
        )
        request = PreparedToolRun(
            synthesis.kind,
            build_key(synthesis),
            toolchain,
            synthesis.request,
            mounts=(
                LogicalMount(
                    "package",
                    ArtifactRef(package.derivation.kind, build_key(package.derivation)),
                ),
            ),
            substitutions=(("script", "synth.tcl"),),
            environment_allowlist=("XILINX_VIVADO",),
            expected_outputs=("utilization.rpt",),
            resources=ResourceRequirements(timeout_seconds=900, cpus=4, licences=("Synthesis",)),
        )
        print(
            f"{case.label}: {PART}, vivado {toolchain.version}, "
            f"LEN={kernel.parameters['LEN']} REP={kernel.parameters['REP']} "
            f"W={kernel.parameters['W']}, request {request.build_key[:16]}"
        )
        completed = subprocess.run(
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
            cwd=scratch,
            capture_output=True,
            text=True,
            check=False,
        )
        errors = [line for line in completed.stdout.splitlines() if line.startswith("ERROR:")]
        if any(UNLICENSED in line for line in errors):
            print(f"{case.label}: SKIP (no licence)")
            return SKIP
        if completed.returncode != 0 or errors or not report.is_file():
            print(completed.stdout[-5000:])
            print(completed.stderr[-2000:])
            print(f"{case.label}: FAIL")
            return FAIL
        cells = {name: int(count) for name, count in UTILIZATION.findall(report.read_text())}
        memory = {name: cells.get(name, 0) for name in MEMORY}
        registers = {name: cells.get(name, 0) for name in REGISTERS}
        # `REP == 1` is the identity: nothing is replayed, so a correct buffer
        # optimizes its store away and only handshake state survives.  Above one
        # neuron fold a store must actually appear -- accepting "any nonzero
        # storage category" would let a buffer that kept nothing but its
        # handshake registers pass, which is precisely the broken case.
        replays = int(cast(int, kernel.parameters["REP"])) > 1
        if replays and sum(memory.values()) <= 0:
            print(f"{case.label}: FAIL (a replaying buffer synthesized no memory; {cells})")
            return FAIL
        if not replays and sum(memory.values()) > 0:
            print(f"{case.label}: FAIL (an identity buffer synthesized memory; {cells})")
            return FAIL
        if sum(registers.values()) <= 0:
            print(f"{case.label}: FAIL (no sequential state at all; {cells})")
            return FAIL
        print(f"{case.label}: PASS (memory={memory}, registers={registers})")
        return PASS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES_BY_LABEL))
    arguments = parser.parse_args()
    record_identity()
    labels = DEFAULT_LABELS if arguments.case is None else (arguments.case,)
    results = tuple(run_one(CASES_BY_LABEL[label]) for label in labels)
    passed = results.count(PASS)
    failed = results.count(FAIL)
    skipped = results.count(SKIP)
    ok = failed == 0 and passed > 0
    print(
        f"RESULT: {'PASS' if ok else 'FAIL'} ({passed} passed, {failed} failed, {skipped} skipped)"
    )
    return PASS if ok else FAIL


if __name__ == "__main__":
    raise SystemExit(main())
