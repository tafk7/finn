# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit XSI conformance for the stream adapters, each in a composed module.

Every adapter kind a stream's plan takes (``vpc``, ``input_gen``, and chains of
both) runs between a cyclic producer presenting a tensor in some order and
thresholding reading it PE channels a beat. The thresholds map each INT4 value
v to the level v + 8, so every output word identifies the elements that
reached it; the expected words are the tensor's values in row-major order,
packed here independently of the kernels' forms. ``inner_shuffle`` (not a
stream candidate) runs placed between two boundary streams. Run with
FINN_ROOT, FINNLIB_ROOT and the XSI library path configured; each simulation
runs in a fresh process.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]

from kernels.rtlsim.rtl_transport import drive
from finn.dataflow.traversal import Traversal, vector_major
from kernels.test_adapters import ELEMENT, adapted, columns_first, transposed, values
from finn.kernels.artifacts.build import materialize_module_sources, prepare_module_build
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.resources import resource_root, template_root

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
    store = ArtifactStore(directory / "store")
    prepared = prepare_module_build(
        point.structure.requirements,
        roots={"kernels": resource_root(), "finnlib": Path(os.environ["FINNLIB_ROOT"])},
        template_roots=(template_root(),),
        blobs=store,
    )
    materialized = materialize_module_sources(prepared, store)
    return prepared.abi.entry_point, [
        str(Path(materialized.directory) / path) for path in materialized.files
    ]


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
TRANSPOSES = ((4, 6, 2), (6, 6, 3), (4, 4, 2), (6, 9, 3))
# FinnLib inner_shuffle (b9262df) emits undefined lanes for these when its input
# arrives in bursts with idle cycles between them and its output never stalls;
# FinnLib's own testbench fails the same way with that input timing (C6 record).
KNOWN_DEFECTS = ((4, 4, 4), (8, 4, 4), (4, 8, 4))


def run_adapted(label, source, pe, modules, evidence):
    point = adapted(source, pe)
    kinds = tuple(
        "vpc" if stage.name.startswith("vpc") else "input_gen"
        for stage in point.x.connection.stages
    )
    assert kinds == modules, (label, kinds)
    levels = [value + 8 for row in values(*source.shape) for value in row]
    expected = [_pack(levels[start : start + pe], 4) for start in range(0, len(levels), pe)]
    top, sources = _build(point, evidence / f"adapted_{label}")
    mask = (1 << (4 * pe)) - 1
    for stalls in (False, True):
        out = drive(top, sources, {}, len(expected), stalls=stalls)
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
    top, sources = _build(transposed(rows, cols, simd), evidence / f"shuffle_{rows}_{cols}_{simd}")
    mask = (1 << (simd * BITS)) - 1
    for stalls in (False, True):
        out = drive(
            top, sources, {"in0": _padded(stimulus, simd * BITS)}, len(expected), stalls=stalls
        )
        assert [word & mask for word in out] == expected, (rows, cols, simd, stalls)
        print(f"PASS inner_shuffle {rows}x{cols}/{simd} stalled={stalls}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--known-defects", action="store_true", help="run the inner_shuffle defect cases"
    )
    args = parser.parse_args()
    directory = args.output or Path(tempfile.mkdtemp(prefix="adapter-evidence-"))
    print(f"Evidence: {directory}", flush=True)
    for label, source, pe, modules in ADAPTED:
        run_adapted(label, source, pe, modules, directory)
    for rows, cols, simd in KNOWN_DEFECTS if args.known_defects else TRANSPOSES:
        run_transpose(rows, cols, simd, directory)


if __name__ == "__main__":
    main()
