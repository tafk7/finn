# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel-only packaging and out-of-context synthesis evidence for DotpAxi."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]

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
from finn.dataflow.model.dotp_axi import DspBlock
from finn.dataflow.model.kernel_artifacts import (
    kernel_source_derivation,
    portable_kernel_component,
    resolve_kernel_contributions,
)
from finn.util.basic import get_vivado_version

from dataflow.model.rtlsim.dotp_axi_numeric import (
    CASES_BY_LABEL,
    Case,
    _configure,
    record_identity,
)

PASS, FAIL, SKIP = 0, 1, 2
DEFAULT_LABELS = ("dsp48e1", "signed_random_softvec", "identity")
PARTS = {
    DspBlock.DSP48E1: "xc7z020clg400-1",
    DspBlock.DSP48E2: "xczu3eg-sbva484-1-e",
    DspBlock.DSP58: "xcvc1902-vsva2197-2MP-e-S",
}
EXPECTED_PRIMITIVES = {
    DspBlock.DSP48E1: "DSP48E1",
    DspBlock.DSP48E2: "DSP48E2",
    DspBlock.DSP58: "DSP58",
}
UTILIZATION = re.compile(r"^\|\s*(DSP\w*)\s*\|\s*(\d+)\s*\|", re.MULTILINE)
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
    weights = np.zeros((case.matrix_width, case.matrix_height), dtype=np.float32)
    kernel = _configure(case, weights)
    finnlib = Path(os.environ["FINNLIB_ROOT"])
    resolved = resolve_kernel_contributions(kernel, roots={"finnlib": finnlib})
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
        Target(PARTS[case.target]),
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
    case: Case,
    constraints: bytes,
    toolchain: ToolchainIdentity,
) -> Derivation:
    return Derivation(
        kind="ooc-synthesis",
        schema_version="dotp-kernel-ooc-v1",
        producer=ProducerIdentity("finn.kernel.dotp-axi-ooc", "1"),
        inputs=(
            ("package", ArtifactRef(package_kind, package_key)),
            ("constraints", ContentRef(content_digest(constraints))),
        ),
        request=RequestSchema("vivado -mode batch -source {script}"),
        tool=ToolRequirement("vivado", toolchain.version),
        options=(
            ("part", PARTS[case.target]),
            ("clock_period_ns", 4.0),
        ),
        outputs=OutputLayout(("utilization.rpt",)),
    )


def run_one(case: Case) -> int:
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        kernel, package, package_dir = _package(case, scratch)
        constraints = (
            "create_clock -period 4.0 -name ap_clk [get_ports ap_clk]\n"
            "create_clock -period 2.0 -name ap_clk2x [get_ports ap_clk2x]\n"
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
        generic = " ".join(
            f"{name}={int(value) if isinstance(value, bool) else value}"
            for name, value in kernel.parameters.items()
        )
        script.write_text(
            "\n".join(
                (
                    *(f"read_verilog -sv {{{source}}}" for source in sources),
                    f"read_xdc {{{constraints_path}}}",
                    f"synth_design -top dotp_axi -part {PARTS[case.target]} "
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
            case,
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
            f"{case.label}: {PARTS[case.target]}, vivado {toolchain.version}, "
            f"request {request.build_key[:16]}"
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
        primitive = EXPECTED_PRIMITIVES[case.target]
        if cells.get(primitive, 0) <= 0:
            print(f"{case.label}: FAIL (no {primitive}; cells={cells})")
            return FAIL
        print(f"{case.label}: PASS ({primitive}={cells[primitive]})")
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
