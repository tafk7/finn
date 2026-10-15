# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Numeric XSim conformance of FinnLib's HLS ``layernorm``: rows where its statistics
round, and rows where epsilon decides the result.

Each case places ``LayerNormKernel`` between two boundary channels (``s_axis_0`` its
input, ``m_axis_0`` its output) at one SIMD and one epsilon, and streams one FLOAT32
tensor through it, free and stalled, taking the output's words as they arrive
(``finn.core.rtlsim_bench.stream_out``; each simulation is its own simulator process).
The cases cover the rows its hardware treats apart, lean, not a matrix:

- random rows at several scales, and rows offset far from zero against their spread,
  where the mean's rounding reaches every result;
- near-constant rows, whose variance is below epsilon or near it, so that epsilon
  decides the result; constant rows (centred to zero, or to a few rounding steps);
- a whole row a beat (one beat a row, a larger epsilon), and a beat's lanes three, no
  power of two (the lanes' tree is not balanced).

The expected values are the spec's test-side reference (``kernels.specs.layernorm``'s
``layernorm``, in float64; no KernelOp reaches the kernel). Each result is compared
within its row's bound (``bounds``: ``TOLERANCE`` times the row's condition), provisional
(an approximating kernel's oracle is not decided); each run prints the largest error it
saw, and the largest over its row's condition. Run with Vivado and an HLS frontend
selected (FinnLib is the ``finnlib`` resource): each case's top is synthesized first, or
taken from the HLS cache.
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
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import Traversal, offsets, pack
from finn.harness.toolchain import print_identity
from finn.kernels.channels import Channel
from finn.kernels.layernorm import LayerNormKernel
from finn.kernels.network import Interface, Network, wire
from finn.toolchain.launcher import host_toolchain
from kernels.helpers import FULL_DSP48E2, built, choose, cycles_of, with_direct_transports
from kernels.specs.layernorm import EPSILON, bounds, condition, layernorm

PORTS = {"input": "s_axis_0", "output": "m_axis_0"}


@dataclass(frozen=True)
class Case:
    label: str
    shape: tuple[int, int]
    simd: int
    epsilon: float = EPSILON


CASES = (
    Case("scaled_rows", (8, 64), simd=8),
    Case("near_constant", (8, 16), simd=2),
    Case("whole_row", (4, 8), simd=8, epsilon=1e-2),
    Case("odd_lanes", (4, 12), simd=3),
)


def rows(case: Case, rng: np.random.Generator) -> npt.NDArray[np.float32]:
    """The case's input: the rows its label is about first, random rows after."""
    count, n = case.shape
    normal = rng.standard_normal
    special: list[Any]
    if case.label == "scaled_rows":
        special = [
            normal(n) * 4,
            10 + normal(n),
            100 + normal(n),
            -1e3 + normal(n) * 30,
            normal(n) * 1e6,
            normal(n) * 1e-6,
        ]
    elif case.label == "near_constant":
        special = [
            np.full(n, 1.5),  # its sums exact: centred to zero
            np.full(n, 0.1),  # its sums round: centred to a few rounding steps
            np.zeros(n),
            normal(n) * 1e-4,  # variance 1e-8, far below epsilon
            1 + normal(n) * 1e-4,  # the same, offset: the mean's rounding reaches it
            normal(n) * 3e-3,  # variance about epsilon
            np.eye(1, n, 3)[0] * 1e-3,  # one value apart, the rest zero
        ]
    elif case.label == "whole_row":
        special = [normal(n) * 0.05, np.full(n, -2.0)]  # variance a quarter of epsilon; none
    else:
        special = [5 + normal(n) * 0.5]
    found = (normal((count, n)) * 2).astype(np.float32)
    found[: len(special)] = np.array(special, dtype=np.float32)
    return found


def placed(case: Case) -> Any:
    """The kernel on boundary channels ``s_axis_0`` and ``m_axis_0``, at SIMD ``case.simd``."""
    tensor = Tensor(case.shape, ScalarEncoding(DataType["FLOAT32"]))

    def kernel() -> LayerNormKernel:
        return LayerNormKernel(platform=FULL_DSP48E2, input_tensor=tensor, epsilon=case.epsilon)

    tensors = {"input": tensor, "output": design_space(kernel()).output_tensor}
    members: dict[str, Any] = {}
    for name, given in tensors.items():
        end = Interface("layernorm", f"{name}_end")
        side: dict[str, Any] = {"producer" if name == "output" else "consumer": end}
        members[name] = wire(tensor=given, port=PORTS[name], platform=FULL_DSP48E2, **side)
    members["layernorm"] = kernel()
    return with_direct_transports(choose(Network(members), {"layernorm.simd": case.simd}))


def stem(case: Case) -> str:
    """The stem of a case's module: ``finn_layernorm_<label>``."""
    return f"layernorm_{case.label}".lower()


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
    rng = np.random.default_rng(zlib.crc32(case.label.encode()))
    x = rows(case, rng)
    expected = layernorm(x, case.epsilon)
    allowed = bounds(x, case.epsilon)
    assert np.isfinite(expected).all(), (case.label, "a row without a finite value")
    bits = x.view(np.uint32).astype(np.int64).ravel().tolist()
    stimulus = list(pack(forms["input"], bits, 32))
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
            inputs={PORTS["input"]: (stimulus, forms["input"].lanes * 32)},
            outputs={output: Receive(forms["output"].beats, forms["output"].lanes * 32)},
            pacing=pacing,
            cycles=cycles.value,
            toolchain=host_toolchain(),
        )[output]
        found = received(forms["output"], words)
        with np.errstate(invalid="ignore"):
            error = np.abs(found - expected)
        close = error <= allowed  # a NaN is never close
        if not close.all():
            where = tuple(int(i) for i in np.argwhere(~close)[0])
            raise AssertionError(
                f"{case.label} {mode}: {int((~close).sum())} results beyond their bounds; "
                f"at {where}: {float(found[where])!r}, expected {float(expected[where])!r}, "
                f"bound {float(allowed[where]):.3g}"
            )
        relative = float((error / condition(x, case.epsilon)).max())
        print(
            f"PASS {case.label} {mode}: largest error {float(error.max()):.3g}, "
            f"over its row's condition {relative:.3g}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case.label for case in CASES])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.output or Path(tempfile.mkdtemp(prefix="layernorm-evidence-"))
    print_identity()
    print(f"Evidence: {directory}", flush=True)
    for case in CASES:
        if args.case in (None, case.label):
            run(case, directory)


if __name__ == "__main__":
    main()
