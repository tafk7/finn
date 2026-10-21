# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Numeric XSim conformance of FinnLib's ``pwpolyf``: each function at its segments'
edges and its clamps.

Each case places ``PwPolyKernel`` between two boundary channels
(``s_axis_0`` its input, ``m_axis_0`` its output) at one PE and streams one tensor
through it, free and stalled, taking the output's words as they arrive
(``finn.core.rtlsim_bench.stream_out``; each simulation is its own simulator process).
Each function is one case, its tensor the same values (``values``), lean, not a matrix:

- where the hardware changes polynomial, both signs: each segment's least magnitude
  (``0.25 * 2**o * (1 + s/8)``, octave ``o`` from 0 to 4, ``s`` from 0 to 7) and the
  binary32 just below it, and the clamp, 8, and the binary32 just below it;
- at and beyond the clamps: 16, 1000, binary32's greatest finite value and infinity,
  both signs;
- near zero: both zeros, the least subnormal and 1e-30, both signs;
- the rest random in [-10, 10];
- PE 1, a divisor and the whole innermost axis.

The expected values are the spec's test-side reference (``support.kernel_specs.pwpolyf``'s
``activation``, the function in float64; no KernelOp reaches the kernel), each result
within ``TOLERANCE[function]`` absolute, provisional (an approximating kernel's oracle
is not decided); each run prints the largest error it saw. At the clamps the results are
also compared exactly with the values the RTL states it gives there (``clamped``). NaN
is not streamed: the hardware reads it as a clamped magnitude. Run with Vivado selected
(FinnLib is the ``finnlib`` resource).
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
from support.helpers import FULL_DSP58, built, choose, cycles_of, with_direct_transports
from support.kernel_specs.pwpolyf import CLAMP, TOLERANCE, activation, clamped

from finn.core.rtlsim_bench import Receive, stream_out
from finn.core.rtlsim_pacing import FREE, STALLED
from finn.core.space import Available, design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import Traversal, offsets, pack
from finn.harness.toolchain import print_identity
from finn.kernels.channels import Channel
from finn.kernels.network import Interface, Network, wire
from finn.kernels.pwpolyf import PwPolyKernel
from finn.toolchain.launcher import host_toolchain

PORTS = {"input": "s_axis_0", "output": "m_axis_0"}
SHAPE = (16, 16)
GREATEST = float(np.finfo(np.float32).max)
LEAST_SUBNORMAL = float(np.finfo(np.float32).smallest_subnormal)


@dataclass(frozen=True)
class Case:
    function: str
    pe: int


CASES = (
    Case("gelu", pe=4),
    Case("silu", pe=16),
    Case("sigmoid", pe=1),
    Case("tanh", pe=2),
)


def edges() -> npt.NDArray[np.float32]:
    """Each positive magnitude where the hardware changes polynomial, and the clamp."""
    found = [0.25 * 2.0**octave * (1 + step / 8) for octave in range(5) for step in range(8)]
    return np.array([*found, CLAMP], dtype=np.float32)


def values(function: str) -> npt.NDArray[np.float32]:
    """The case's input, in ``SHAPE``: the edges and the binary32 below each, the clamps'
    far side and zero's neighbourhood, both signs, then random values in [-10, 10]."""
    at = edges()
    below = np.nextafter(at, np.float32(0))
    special = np.array(
        [16.0, 1000.0, GREATEST, np.inf, 0.0, LEAST_SUBNORMAL, 1e-30], dtype=np.float32
    )
    magnitudes = np.concatenate([at, below, special])
    chosen = np.concatenate([magnitudes, -magnitudes])
    rng = np.random.default_rng(zlib.crc32(function.encode()))
    size = int(np.prod(SHAPE))
    assert chosen.size <= size, (chosen.size, size)
    found = rng.uniform(-10.0, 10.0, size=size).astype(np.float32)
    found[: chosen.size] = chosen
    return found.reshape(SHAPE)


def placed(case: Case) -> Any:
    """The kernel on boundary channels ``s_axis_0`` and ``m_axis_0``, at PE ``case.pe``."""
    tensor = Tensor(SHAPE, ScalarEncoding(DataType["FLOAT32"]))
    facts: dict[str, Any] = dict(platform=FULL_DSP58, function=case.function)
    alone = PwPolyKernel(**facts, input_tensor=tensor)
    tensors = {"input": tensor, "output": design_space(alone).output_tensor}
    members: dict[str, Any] = {}
    for name, given in tensors.items():
        end = Interface("activation", f"{name}_contract")
        side: dict[str, Any] = {"producer" if name == "output" else "consumer": end}
        members[name] = wire(tensor=given, port=PORTS[name], platform=FULL_DSP58, **side)
    members["activation"] = PwPolyKernel(**facts, input_tensor=tensor)
    return with_direct_transports(choose(Network(members), {"activation.pe": case.pe}))


def stem(case: Case) -> str:
    """The stem of a case's module: ``finn_pwpolyf_<function>``."""
    return f"pwpolyf_{case.function}"


def received(form: Traversal, words: tuple[int, ...]) -> npt.NDArray[np.float64]:
    """The binary32 results ``words`` carry, lanes low first, in their tensor's shape."""
    found = np.full(int(np.prod(form.shape)), np.nan)
    bits = np.array(
        [[(word >> (32 * lane)) & 0xFFFFFFFF for lane in range(form.lanes)] for word in words],
        dtype=np.uint32,
    )
    found[offsets(form).reshape(form.beats, form.lanes)] = bits.view(np.float32)
    return found.reshape(form.shape)


def checked(case: Case, x: npt.NDArray[Any], found: npt.NDArray[np.float64]) -> float:
    """The largest error of ``found`` against the function of ``x``; an ``AssertionError``
    naming the first result beyond the tolerance, or at a clamp not the RTL's value."""
    expected = activation(case.function, x)
    tolerance = TOLERANCE[case.function]
    close = np.isclose(found, expected, rtol=0.0, atol=tolerance, equal_nan=False)
    if not close.all():
        where = tuple(int(i) for i in np.argwhere(~close)[0])
        raise AssertionError(
            f"{case.function}: {int((~close).sum())} results beyond {tolerance:g}; at {where} "
            f"(x = {float(x[where])!r}): {float(found[where])!r}, "
            f"expected {float(expected[where])!r}"
        )
    stated = clamped(case.function, x)
    at = ~np.isnan(stated)
    if not (found[at] == stated[at]).all():
        where = tuple(int(i) for i in np.argwhere(at & (found != stated))[0])
        raise AssertionError(
            f"{case.function}: at x = {float(x[where])!r} the clamp gives "
            f"{float(found[where])!r}, not {float(stated[where])!r}"
        )
    with np.errstate(invalid="ignore"):
        return float(np.nanmax(np.abs(found - expected)))


def run(case: Case, evidence: Path) -> None:
    point = placed(case)
    forms: dict[str, Traversal] = {}
    for name in PORTS:
        ends = point.member(name).query(Channel.sides).value
        assert ends.source.form == ends.sink.form, (case.function, name, ends)
        forms[name] = ends.source.form
    assert forms["output"].lanes == case.pe, (case.function, forms["output"])
    x = values(case.function)
    lanes = x.view(np.uint32).astype(np.int64)
    stimulus = list(pack(forms["input"], lanes.ravel().tolist(), 32))
    module = built(point, stem(case))
    cycles = cycles_of(point)
    assert isinstance(cycles, Available), (case.function, cycles)
    output = PORTS["output"]
    for mode, pacing in (("free", FREE), ("stalled", STALLED)):
        directory = evidence / case.function / mode
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
        error = checked(case, x, received(forms["output"], words))
        print(f"PASS {case.function} {mode}: largest error {error:.3g}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case.function for case in CASES])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.output or Path(tempfile.mkdtemp(prefix="pwpolyf-evidence-"))
    print_identity()
    print(f"Evidence: {directory}", flush=True)
    for case in CASES:
        if args.case in (None, case.function):
            run(case, directory)


if __name__ == "__main__":
    main()
