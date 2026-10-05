# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit XSI conformance for the stream adapters, each in a root's module.

Every adapter kind a stream's plan takes (``vpc``, ``input_gen``, and chains of
both) runs between a cyclic producer presenting a tensor in some order and
thresholding reading it PE channels a beat. The thresholds map each INT4 value
v to the level v + 8, so every output word identifies the elements that
reached it; the expected words are the tensor's values in row-major order,
packed here independently of the kernels' forms. ``inner_shuffle`` (not a
stream candidate) runs placed between two boundary streams. Run with
Vivado selected (FinnLib is the ``finnlib`` resource); each simulation
runs in a fresh process.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]

from finn.dataflow.traversal import Traversal, vector_major
from kernels.helpers import print_identity
from kernels.rtlsim.rtl_transport import drive
from kernels.test_adapters import ELEMENT, adapted, columns_first, transposed, values
from kernels.xsim import materialize

BITS = ELEMENT.bits


def _pack(values, bits=BITS):
    mask = (1 << bits) - 1
    return sum((int(value) & mask) << (index * bits) for index, value in enumerate(values))


def _padded(words, bits):
    """Set every carrier padding bit, which the module must ignore."""
    carrier = (bits + 7) // 8 * 8
    padding = ((1 << carrier) - 1) ^ ((1 << bits) - 1)
    return [word | padding for word in words]


def _build(point, directory):
    return materialize(point.module, directory)


ROWS, CHANNELS = 3, 12
ROWS_AS_LANES = Traversal.over((ROWS, CHANNELS), ((1, CHANNELS, 1),), ((0, ROWS, 1),))
# (label, the producer's order, thresholding's PE, the adapter's modules)
ADAPTED = (
    ("vpc_2_3", vector_major((ROWS, CHANNELS), 2), 3, ("vpc",)),
    ("vpc_4_2", vector_major((ROWS, CHANNELS), 4), 2, ("vpc",)),
    ("vpc_3_12", vector_major((ROWS, CHANNELS), 3), 12, ("vpc",)),
    ("vpc_6_4", vector_major((ROWS, CHANNELS), 6), 4, ("vpc",)),
    ("input_gen_4", columns_first(ROWS, CHANNELS, 4), 4, ("input_gen",)),
    ("input_gen_3", columns_first(ROWS, CHANNELS, 3), 3, ("input_gen",)),
    ("vpc_input_gen", columns_first(ROWS, CHANNELS, 4), 2, ("vpc", "input_gen")),
    ("input_gen_vpc", columns_first(ROWS, CHANNELS, 1), 4, ("input_gen", "vpc")),
    ("vpc_input_gen_vpc", ROWS_AS_LANES, 2, ("vpc", "input_gen", "vpc")),
)
# SIMD 4 at 4x4, 8x4 and 4x8 stalled needs FinnLib's inner_shuffle page-guard
# fix (finn.kernels.transpose); the pinned d03f2fc fails them.
TRANSPOSES = ((4, 6, 2), (6, 6, 3), (4, 4, 2), (6, 9, 3), (4, 4, 4), (8, 4, 4), (4, 8, 4))


def run_adapted(label, source, pe, modules, evidence):
    point = adapted(source, pe)
    kinds = tuple(
        "vpc" if stage.label.rsplit(".", 1)[-1].startswith("vpc") else "input_gen"
        for stage in point.x.stages
    )
    assert kinds == modules, (label, kinds)
    levels = [value + 8 for row in values(*source.shape) for value in row]
    expected = [_pack(levels[start : start + pe], 4) for start in range(0, len(levels), pe)]
    top, sources, data = _build(point, evidence / f"adapted_{label}")
    mask = (1 << (4 * pe)) - 1
    for stalls in (False, True):
        out = drive(top, sources, {}, len(expected), stalls=stalls, data_files=data)
        assert [word & mask for word in out] == expected, (label, stalls)
        print(f"PASS {label} ({' -> '.join(modules)}) stalled={stalls}", flush=True)


def run_transpose(rows, cols, simd, evidence):
    rng = np.random.RandomState(7)
    batches = 2
    values = rng.randint(-8, 8, (batches, rows, cols))
    stimulus = [
        _pack(values[b, i, start : start + simd])
        for b in range(batches)
        for i in range(rows)
        for start in range(0, cols, simd)
    ]
    expected = [
        _pack(values[b, start : start + simd, j])
        for b in range(batches)
        for j in range(cols)
        for start in range(0, rows, simd)
    ]
    top, sources, data = _build(
        transposed(rows, cols, simd), evidence / f"shuffle_{rows}_{cols}_{simd}"
    )
    mask = (1 << (simd * BITS)) - 1
    for stalls in (False, True):
        out = drive(
            top,
            sources,
            {"in0": _padded(stimulus, simd * BITS)},
            len(expected),
            stalls=stalls,
            data_files=data,
        )
        assert [word & mask for word in out] == expected, (rows, cols, simd, stalls)
        print(f"PASS inner_shuffle {rows}x{cols}/{simd} stalled={stalls}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.output or Path(tempfile.mkdtemp(prefix="adapter-evidence-"))
    print_identity()
    print(f"Evidence: {directory}", flush=True)
    for label, source, pe, modules in ADAPTED:
        run_adapted(label, source, pe, modules, directory)
    for rows, cols, simd in TRANSPOSES:
        run_transpose(rows, cols, simd, directory)


if __name__ == "__main__":
    main()
