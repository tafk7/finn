# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Numeric XSI conformance of FinnLib's ``eltwise``: each operation once, at its types' extremes.

Each case places ``EltwiseKernel`` between three boundary channels (``s_axis_0`` its
``lhs``, ``s_axis_1`` its ``rhs``, ``m_axis_0`` its result) and streams one tensor of
each operand through it, free and stalled. The cases cover what the kernel's design
space offers, lean (decision KT10), not a matrix:

- each operation (ADD, SUB, SBR, MUL) on integers, signed and unsigned (an unsigned
  difference is signed), from 4 to 16 bits;
- the broadcast the kernel declares: an ``rhs`` of ``lhs``'s trailing axes, presented
  replayed at the boundary, over one leading axis and over two;
- float arithmetic (DSP58 only): an integer ``lhs`` converted, against a FLOAT32
  ``rhs`` scaled by ``b_scale``, and a FLOAT32 product;
- PE 1, a divisor and the whole innermost extent.

Each operand's first row along its leading axis is its type's least value and its
second its greatest (a FLOAT32 operand: a bound the case states, so that no result
overflows); ``rhs``'s first two columns are its least and greatest, so every pair of
extremes meets. The rest is random, seeded by the case. The expected words come from
the spec's test-side reference (``kernels.specs.eltwise.computed``; no KernelOp
reaches eltwise, decision KT12 A1), packed in each port's order; words that differ are
decoded as an order (``finn.harness.orders``). A float is compared bit for bit. Run
with Vivado selected (FinnLib is the ``finnlib`` resource); each simulation runs in a
fresh process.
"""

from __future__ import annotations

import argparse
import tempfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from qonnx.core.datatype import DataType

from finn.core.executors.xsim.pacing import FREE, STALLED
from finn.core.executors.xsim.rtl import materialize
from finn.core.space import design_space
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import Traversal, pack
from finn.harness.orders import Stream, decode
from finn.harness.toolchain import print_identity
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.eltwise import EltwiseKernel
from kernels.helpers import FULL_DSP58, Root, with_direct_transports
from kernels.specs.eltwise import computed
from kernels.sweeps.rtl_transport import drive_observed

PORTS = {"lhs": "s_axis_0", "rhs": "s_axis_1", "result": "m_axis_0"}
FLOAT32 = "FLOAT32"


@dataclass(frozen=True)
class Case:
    label: str
    operation: str
    lhs: str  # datatype names
    rhs: str
    lhs_shape: tuple[int, ...]
    rhs_shape: tuple[int, ...]  # lhs's trailing axes: a broadcast when fewer
    pe: int
    scale: float = 1.0
    # A FLOAT32 operand's extremes, -bound and bound: the largest finite by default.
    lhs_bound: float = float(np.finfo(np.float32).max)
    rhs_bound: float = float(np.finfo(np.float32).max)


CASES = (
    Case("add_int8_rows", "ADD", "INT8", "INT8", (4, 8), (8,), pe=4),
    Case("sub_uint8", "SUB", "UINT8", "UINT8", (4, 8), (4, 8), pe=1),
    Case("sbr_int4_planes", "SBR", "INT4", "INT4", (2, 3, 4), (3, 4), pe=2),
    Case("mul_int16", "MUL", "INT16", "INT16", (4, 8), (4, 8), pe=8),
    Case("sub_int8_float_scaled", "SUB", "INT8", FLOAT32, (4, 6), (6,), pe=2, scale=0.5),
    # 2**64 * 2**63 = 2**127: the greatest product near binary32's top, none beyond it.
    Case(
        "mul_float",
        "MUL",
        FLOAT32,
        FLOAT32,
        (4, 4),
        (4, 4),
        pe=4,
        lhs_bound=2.0**64,
        rhs_bound=2.0**63,
    ),
)


def placed(case: Case) -> Any:
    """The kernel on boundary channels ``s_axis_0``, ``s_axis_1`` and ``m_axis_0``, at PE
    ``case.pe``."""
    lhs, rhs = DataType[case.lhs], DataType[case.rhs]
    facts: dict[str, Any] = dict(
        operation=case.operation,
        lhs_dtype=lhs,
        rhs_dtype=rhs,
        b_scale=case.scale,
        platform=FULL_DSP58,
    )
    result = design_space(EltwiseKernel(**facts)).result_dtype
    tensors = {
        "lhs": Tensor(case.lhs_shape, ScalarEncoding(lhs)),
        "rhs": Tensor(case.rhs_shape, ScalarEncoding(rhs)),
        "result": Tensor(case.lhs_shape, ScalarEncoding(result)),
    }
    namespace: dict[str, Any] = {
        name: Channel(tensor=tensor, port=PORTS[name], platform=FULL_DSP58)
        for name, tensor in tensors.items()
    }
    namespace["arithmetic"] = EltwiseKernel(
        **facts,
        lhs_channel=namespace["lhs"],
        rhs_channel=namespace["rhs"],
        result_channel=namespace["result"],
    )
    space = type(f"Eltwise_{case.label}", (Root,), namespace)
    return with_direct_transports(commit(design_space(space()), {"arithmetic.pe": case.pe}))


def operand(case: Case, name: str, rng: np.random.Generator) -> npt.NDArray[Any]:
    """Operand ``name``'s values: random, its least and greatest at the head of its leading
    axis, and (``rhs``) of its innermost."""
    shape = getattr(case, f"{name}_shape")
    dtype = getattr(case, name)
    values: npt.NDArray[Any]
    if dtype == FLOAT32:
        bound = getattr(case, f"{name}_bound")
        values = (rng.standard_normal(shape) * 100).astype(np.float32)
        least, greatest = -bound, bound
    else:
        low, high = ordinary_integer_bounds(DataType[dtype])
        values = rng.integers(low, high + 1, size=shape, dtype=np.int64)
        least, greatest = low, high
    values[0], values[1] = least, greatest
    if name == "rhs":
        values[..., 0], values[..., 1] = least, greatest
    return values


def words(values: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
    """Each element as the integer its lanes carry: an integer itself, a float its bits."""
    if values.dtype == np.float32:
        return values.view(np.uint32).astype(np.int64)
    return values.astype(np.int64)


def as_values(found: npt.NDArray[np.int64], dtype: str) -> npt.NDArray[Any]:
    """The elements a stream's integers carry (``words``' inverse)."""
    return found.astype(np.uint32).view(np.float32) if dtype == FLOAT32 else found


def run(case: Case, evidence: Path) -> None:
    point = placed(case)
    kernel = point.arithmetic
    bits = {
        "lhs": kernel.lhs_dtype.bitwidth(),
        "rhs": kernel.rhs_dtype.bitwidth(),
        "result": kernel.result_dtype.bitwidth(),
    }
    # Each boundary presents what the kernel's port reads: rhs replayed, once per lhs row.
    forms: dict[str, Traversal] = {}
    for name in PORTS:
        ends = getattr(point, name).query(Channel.endpoints).value
        assert ends.source.form == ends.sink.form, (case.label, name, ends)
        forms[name] = ends.source.form
    assert forms["result"].lanes == case.pe, (case.label, forms["result"])
    rng = np.random.default_rng(zlib.crc32(case.label.encode()))
    values = {name: operand(case, name, rng) for name in ("lhs", "rhs")}
    inputs = {name: words(values[name]) for name in values}

    def reference(read: Mapping[str, npt.NDArray[np.int64]]) -> dict[str, Any]:
        lhs, rhs = (as_values(read[name], getattr(case, name)) for name in ("lhs", "rhs"))
        return {"result": words(computed(case.operation, lhs, rhs, case.scale))}

    expected = reference(inputs)["result"]
    if kernel.result_dtype.name != FLOAT32:
        low, high = ordinary_integer_bounds(kernel.result_dtype)
        assert low <= expected.min() and expected.max() <= high, (case.label, "result range")
    else:
        assert np.isfinite(as_values(expected, FLOAT32)).all(), (case.label, "result overflows")

    def packed(name: str, found: npt.NDArray[np.int64]) -> list[int]:
        return list(pack(forms[name], found.ravel().tolist(), bits[name]))

    stimulus = {PORTS[name]: packed(name, inputs[name]) for name in inputs}
    wanted = packed("result", expected)
    top, sources, data = materialize(point.module, evidence / case.label)
    mask = (1 << (case.pe * bits["result"])) - 1
    for stalled in (False, True):
        measured = drive_observed(
            top,
            sources,
            stimulus,
            {PORTS["result"]: len(wanted)},
            {},
            pacing=STALLED if stalled else FREE,
            directory=evidence / case.label / ("stalled" if stalled else "free"),
            data_files=data,
        )
        actual = [word & mask for word in measured["outputs"][PORTS["result"]]]
        if actual != wanted:
            decoded = decode(
                {"result": Stream(forms["result"], bits["result"])},
                {"result": actual},
                inputs={name: Stream(forms[name], bits[name]) for name in inputs},
                values=inputs,
                reference=reference,
            )
            found = decoded.message if decoded else f"{actual} != {wanted}"
            raise AssertionError(f"{case.label} stalled={stalled}: {found}")
        print(f"PASS {case.label} stalled={stalled}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case.label for case in CASES])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.output or Path(tempfile.mkdtemp(prefix="eltwise-evidence-"))
    print_identity()
    print(f"Evidence: {directory}", flush=True)
    for case in CASES:
        if args.case in (None, case.label):
            run(case, directory)


if __name__ == "__main__":
    main()
