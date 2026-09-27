"""Read-only contract probes; run with this checkout's src on PYTHONPATH.

Optional --rtl runs the small FIFO experiment against the supplied FinnLib.
Tool work products go into a temporary directory, outside the checkout.
"""

import argparse
import ast
import json
from pathlib import Path
import subprocess
import tempfile

from finn.core.space import compile_space, inspection
from finn.kernels import DotpAxiKernel, DspBlock, IntToFp32Kernel, MVAU
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.datatypes.values import resolve_qonnx_datatype_name as dtype


def describe(result):
    return {
        "state": type(result).__name__,
        "findings": [{"code": f.code, "message": f.message} for f in result.findings],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finnlib", type=Path, required=True)
    parser.add_argument("--rtl", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = {}

    base = ThresholdingAxiKernel(
        input_dtype=dtype("BIPOLAR"), threshold_dtype=dtype("INT5"),
        thresholds=(((-2, 0, 3),),), bias=-1, pe=1,
        depth_trigger_bram=0, depth_trigger_uram=0,
    )
    condition = ThresholdingAxiKernel.types_supported
    report["threshold_invalid_type_before_axilite_choice"] = describe(base.query(condition))
    report["threshold_invalid_type_after_axilite_choice"] = describe(
        base.with_choices(use_axilite=False).query(condition)
    )

    for name in ("INT0", "UINT0"):
        try:
            result = IntToFp32Kernel(input_dtype=dtype(name)).build_requirements.inspect()
            report[name] = describe(result.accepted_result)
        except Exception as error:
            report[name] = {"exception": type(error).__name__, "message": str(error)}

    dotp = DotpAxiKernel({
        DotpAxiKernel.pe: 2**32, DotpAxiKernel.simd: 1,
        DotpAxiKernel.target_dsp: DspBlock.DSP58,
        DotpAxiKernel.segment_length: 0,
        DotpAxiKernel.activation.dtype: dtype("INT3"),
        DotpAxiKernel.weights.dtype: dtype("INT3"),
        DotpAxiKernel.result.dtype: dtype("INT9"),
    }).with_choices(compute_pumping=False)
    report["dotp_pe_overflow"] = describe(dotp.build_requirements.query())
    report["mvau_views"] = [
        item.key for item in inspection.members(compile_space(MVAU)) if item.kind == "view"
    ]

    paths = set()
    for source in (root / "src/finn/kernels").glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if (
                isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value.startswith(("rtl/", "hls/"))
                and node.value.endswith((".sv", ".hpp"))
            ):
                paths.add(node.value)
    report["declared_finnlib_paths"] = sorted(paths)
    report["missing_paths"] = sorted(p for p in paths if not (args.finnlib / p).is_file())
    report["also_missing_at_flat_path"] = sorted(
        p for p in paths if not (args.finnlib / p.split("/")[0] / Path(p).name).is_file()
    )

    if args.rtl:
        tb = Path(__file__).with_name("fifo_capacity_tb.sv")
        commands = (
            ["xvlog", "--sv", str(args.finnlib.resolve() / "rtl/fifo.sv"), str(tb)],
            ["xelab", "fifo_capacity_tb", "-s", "fifo_probe", "-timescale", "1ns/1ps"],
            ["xsim", "fifo_probe", "-runall"],
        )
        with tempfile.TemporaryDirectory(prefix="kernel-fifo-probe-") as directory:
            for command in commands:
                completed = subprocess.run(
                    command, cwd=directory, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=120, check=True,
                )
            report["fifo_rtl"] = [
                line for line in completed.stdout.splitlines() if "FIFO_PROBE" in line
            ]
            if not report["fifo_rtl"]:
                raise RuntimeError("FIFO simulation did not reach its result")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
