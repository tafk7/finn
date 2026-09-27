# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from onnx import helper

from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU


def _mvau(*, interleave: int) -> MVAU:
    node = helper.make_node(
        "MVAU",
        ["x", "w"],
        ["y"],
        domain="finn.custom_op.fpgadataflow",
        MW=32,
        MH=18,
        SIMD=4,
        PE=6,
        TH=interleave,
        numInputVectors=[1, 3, 3],
    )
    return MVAU(node)


def test_batch_interleaving_does_not_add_compute_iterations() -> None:
    standard = _mvau(interleave=1)
    interleaved = _mvau(interleave=3)

    assert standard.get_exp_cycles() == 3 * 8 * 9
    assert interleaved.get_exp_cycles() == standard.get_exp_cycles()
