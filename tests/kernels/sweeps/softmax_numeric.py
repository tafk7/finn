# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Numeric XSim conformance of FinnLib's HLS ``SoftMax``: rows at the function's edges.

Each case places ``SoftmaxKernel`` between two boundary channels (``s_axis_0`` its
input, ``m_axis_0`` its output) at one SIMD and streams one tensor through it, free and
stalled, taking the output's words as they arrive (``finn.core.rtlsim_bench.stream_out``;
each simulation is its own simulator process). The cases cover the rows its hardware
treats apart, lean, not a matrix:

- FLOAT32 rows: random, constant, one value far above the rest, nearly equal values,
  and binary32's greatest and least finite values in one row;
- FLOAT32 rows with infinities: one ``+inf`` (it takes all), two and all (they share),
  ``+inf`` beside ``-inf``, ``-inf`` among finite values (zero), and one finite value
  among ``-inf``;
- integer rows: INT8 at its extremes; INT4, whose lanes fill no whole byte (the top's
  glue reads them from a padded beat); INT32 near its extremes, which binary32 rounds;
- SIMD 1, a divisor and the whole row.

The expected values are the spec's test-side reference (``kernels.specs.softmax``'s
``softmax``, in float64; no KernelOp reaches the kernel). Each result is compared within
``TOLERANCE`` absolute, provisional (an approximating kernel's oracle is not decided);
each run prints the largest error it saw. Run with Vivado and an HLS frontend selected
(FinnLib is the ``finnlib`` resource): each case's top is synthesized first, or taken
from the HLS cache.
"""

from __future__ import annotations

import argparse
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from qonnx.core.datatype import DataType

from finn.core.rtlsim_bench import Receive, stream_out
from finn.core.rtlsim_pacing import FREE, STALLED
from finn.core.space import Available, design_space
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import Traversal, offsets, pack
from finn.harness.toolchain import print_identity
from finn.kernels.channels import Channel
from finn.kernels.network import Interface, Network, wire
from finn.kernels.softmax import SoftmaxKernel
from finn.toolchain.launcher import host_toolchain
from kernels.helpers import FULL_DSP48E2, built, choose, cycles_of, with_direct_transports
from kernels.specs.softmax import TOLERANCE, softmax

PORTS = {"input": "s_axis_0", "output": "m_axis_0"}
FLOAT32 = "FLOAT32"
GREATEST = float(np.finfo(np.float32).max)


@dataclass(frozen=True)
class Case:
    label: str
    dtype: str
    shape: tuple[int, ...]
    simd: int


CASES = (
    Case("float_rows", FLOAT32, (5, 16), simd=4),
    Case("float_infinities", FLOAT32, (6, 8), simd=2),
    Case("int8_extremes", "INT8", (4, 12), simd=12),
    Case("int4_partial_bytes", "INT4", (3, 6), simd=3),
    Case("int32_rounded", "INT32", (3, 4), simd=1),
)


def rows(case: Case, rng: np.random.Generator) -> npt.NDArray[Any]:
    """The case's input: random rows, and the rows its label is about first."""
    count, n = case.shape
    found: npt.NDArray[Any]
    if case.dtype == FLOAT32:
        inf = np.inf
        if case.label == "float_infinities":
            special = [
                [inf, 1, 2, 3, -1, 0, 5, 4],
                [inf, 1, inf, 3, -1, 0, 5, 4],
                [inf] * n,
                [inf, -inf, 0, 7, -inf, inf, 1, inf],
                [-inf, 1, 2, -inf, 0, 3, -inf, 4],
                [-inf] * (n - 1) + [0.0],
            ]
        else:
            special = [
                list(np.full(n, 1.5)),
                [50.0, *rng.standard_normal(n - 1)],
                list(rng.standard_normal(n) * 0.01),
                [GREATEST, -GREATEST, *(rng.standard_normal(n - 2) * 1e3)],
            ]
        found = (rng.standard_normal((count, n)) * 4).astype(np.float32)
        found[: len(special)] = np.array(special, dtype=np.float32)
        return found
    low, high = ordinary_integer_bounds(DataType[case.dtype])
    found = rng.integers(low, high + 1, size=case.shape, dtype=np.int64)
    found[0] = low
    found[1] = [high if column % 2 else low for column in range(n)]
    if case.dtype == "INT32":
        found[2] = [high, high - 1, high - 1000, low + 1][:n]
    return found


def placed(case: Case) -> Any:
    """The kernel on boundary channels ``s_axis_0`` and ``m_axis_0``, at SIMD ``case.simd``."""
    tensor = Tensor(case.shape, ScalarEncoding(DataType[case.dtype]))
    alone = SoftmaxKernel(platform=FULL_DSP48E2, input_tensor=tensor)
    tensors = {"input": tensor, "output": design_space(alone).output_tensor}
    members: dict[str, Any] = {}
    for name, given in tensors.items():
        end = Interface("softmax", f"{name}_end")
        side: dict[str, Any] = {"producer" if name == "output" else "consumer": end}
        members[name] = wire(tensor=given, port=PORTS[name], platform=FULL_DSP48E2, **side)
    members["softmax"] = SoftmaxKernel(platform=FULL_DSP48E2, input_tensor=tensor)
    return with_direct_transports(choose(Network(members), {"softmax.simd": case.simd}))


def stem(case: Case) -> str:
    """The stem of a case's module: ``finn_softmax_<label>``."""
    return f"softmax_{case.label}".lower()


def lanes(values: npt.NDArray[Any], dtype: str) -> npt.NDArray[np.int64]:
    """Each element as the integer its lane carries: an integer itself, a float its bits."""
    if dtype == FLOAT32:
        return np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.int64)
    return np.asarray(values).astype(np.int64)


def received(form: Traversal, words: tuple[int, ...]) -> npt.NDArray[np.float64]:
    """The binary32 results ``words`` carry, lanes low first, in their tensor's shape."""
    found = np.full(int(np.prod(form.shape)), np.nan)
    bits = np.array(
        [[(word >> (32 * lane)) & 0xFFFFFFFF for lane in range(form.lanes)] for word in words],
        dtype=np.uint32,
    )
    found[offsets(form).reshape(form.beats, form.lanes)] = bits.view(np.float32)
    return found.reshape(form.shape)


def run(case: Case, evidence: Path) -> None:
    point = placed(case)
    forms: dict[str, Traversal] = {}
    for name in PORTS:
        ends = point.member(name).query(Channel.endpoints).value
        assert ends.source.form == ends.sink.form, (case.label, name, ends)
        forms[name] = ends.source.form
    assert forms["output"].lanes == case.simd, (case.label, forms["output"])
    bits = {"input": DataType[case.dtype].bitwidth(), "output": 32}
    rng = np.random.default_rng(zlib.crc32(case.label.encode()))
    x = rows(case, rng)
    expected = softmax(x)
    assert not np.isnan(expected).any(), (case.label, "a row without a value")
    stimulus = list(pack(forms["input"], lanes(x, case.dtype).ravel().tolist(), bits["input"]))
    module = built(point, stem(case))
    cycles = cycles_of(point)
    assert isinstance(cycles, Available), (case.label, cycles)
    output = PORTS["output"]
    for mode, pacing in (("free", FREE), ("stalled", STALLED)):
        directory = evidence / case.label / mode
        directory.mkdir(parents=True, exist_ok=True)
        words = stream_out(
            module,
            directory,
            inputs={PORTS["input"]: (stimulus, forms["input"].lanes * bits["input"])},
            outputs={output: Receive(forms["output"].beats, forms["output"].lanes * 32)},
            pacing=pacing,
            cycles=cycles.value,
            toolchain=host_toolchain(),
        )[output]
        found = received(forms["output"], words)
        close = np.isclose(found, expected, rtol=0.0, atol=TOLERANCE, equal_nan=False)
        with np.errstate(invalid="ignore"):
            error = float(np.nanmax(np.abs(found - expected)))
        if not close.all():
            where = tuple(int(i) for i in np.argwhere(~close)[0])
            raise AssertionError(
                f"{case.label} {mode}: {int((~close).sum())} results beyond {TOLERANCE:g}; "
                f"at {where}: {float(found[where])!r}, expected {float(expected[where])!r}"
            )
        print(f"PASS {case.label} {mode}: largest error {error:.3g}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case.label for case in CASES])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.output or Path(tempfile.mkdtemp(prefix="softmax-evidence-"))
    print_identity()
    print(f"Evidence: {directory}", flush=True)
    for case in CASES:
        if args.case in (None, case.label):
            run(case, directory)


if __name__ == "__main__":
    main()
