# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit XSI conformance for the stream adapters, each in a composed module.

A module presents one stream at ``in0_V`` and the adapted one at ``out0_V``;
the expected words are packed here from the element values, independently of
the kernels' forms. Run with FINN_ROOT, FINNLIB_ROOT and the XSI library path
configured; each simulation runs in a fresh process.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]

from kernels.rtlsim.rtl_transport import drive
from kernels.test_adapters import ELEMENT, transposed, widths
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


WIDTHS = ((2, 3), (4, 2), (3, 12), (6, 4))
TRANSPOSES = ((4, 6, 2), (6, 6, 3), (4, 4, 4))


def run_width(before, after, evidence):
    rng = np.random.RandomState(5)
    values = rng.randint(-8, 8, (3, 12))
    flat = values.reshape(-1)
    stimulus = [_pack(flat[start : start + before]) for start in range(0, flat.size, before)]
    expected = [_pack(flat[start : start + after]) for start in range(0, flat.size, after)]
    top, sources = _build(widths(before, after), evidence / f"vpc_{before}_{after}")
    mask = (1 << (after * BITS)) - 1
    for stalls in (False, True):
        out = drive(
            top,
            sources,
            {"in0": _padded(stimulus, before * BITS)},
            len(expected),
            stalls=stalls,
        )
        assert [word & mask for word in out] == expected, (before, after, stalls)
        print(f"PASS vpc {before}->{after} stalled={stalls}", flush=True)


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
    args = parser.parse_args()
    directory = args.output or Path(tempfile.mkdtemp(prefix="adapter-evidence-"))
    print(f"Evidence: {directory}", flush=True)
    for before, after in WIDTHS:
        run_width(before, after, directory)
    for rows, cols, simd in TRANSPOSES:
        run_transpose(rows, cols, simd, directory)


if __name__ == "__main__":
    main()
