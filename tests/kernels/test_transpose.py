# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The transpose kernel (FinnLib ``inner_shuffle``): its domain, its refusals and its stream
contract, on its spec's case (``kernels.specs.transpose``: a sequence of two 6 x 4
matrices, SIMD 2, 3 and 6)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from finn.core.space import Rejected, Space, inspection
from finn.dataflow.plan import Step, plan
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.harness.points import rejected
from finn.kernels.artifacts.abi import Signal
from finn.kernels.configure import commit
from finn.kernels.transpose import TransposeKernel
from kernels.conformance import KERNEL, unchosen
from kernels.helpers import FULL_DSP48E2
from kernels.specs.base import placed
from kernels.specs.thresholding import tensor
from kernels.specs.transpose import MATRICES, transpose

SIMDS = [factors["simd"] for factors in transpose()["factors"]]
BATCHES, ROWS, COLS = MATRICES


def _transpose(simd: int, **facts: object) -> Any:
    return placed(transpose(), {"simd": simd, "ram_style": "auto"}, **facts)


def _beats(form: Any) -> list[list[int]]:
    """Each beat's elements, as flat row-major offsets into its tensor."""
    return [
        [int(np.ravel_multi_index(position, form.shape)) for position in beat]
        for beat in form.positions()
    ]


def test_its_simd_is_any_divisor_of_the_rows_and_its_pages_any_memory() -> None:
    """SIMD divides I alone, the RTL's one constraint; J (4) is free: SIMD 3 does not divide
    it, and 4, which divides J and not I, is not a case. The pages' explicit styles come first,
    ordered by the bits they hold, then ``auto``."""
    base = unchosen(**transpose())
    viable = {item.key: item.cases for item in inspection.viable(base)}
    assert viable == {
        f"{KERNEL}.simd": (1, 2, 3, 6),
        f"{KERNEL}.ram_style": ("distributed", "block", "ultra", "auto"),
    }
    with pytest.raises(ValueError, match="domain-membership"):
        commit(base, {f"{KERNEL}.simd": 4})


@pytest.mark.parametrize("simd", SIMDS)
def test_its_input_is_each_matrix_flat_and_its_output_each_column(simd: int) -> None:
    """The input presents each matrix row-major, SIMD consecutive elements a beat (a beat
    spans two rows where SIMD does not divide J); the output presents each matrix's
    columns in turn, SIMD rows of a column a beat. The RTL's parameters name that geometry."""
    kernel = _transpose(simd)
    flat = np.arange(BATCHES * ROWS * COLS).reshape(MATRICES)
    rows_in = flat.reshape(-1, simd).tolist()
    columns = flat.transpose(0, 2, 1).reshape(-1, simd).tolist()
    assert _beats(kernel.input.presented.form) == rows_in
    assert _beats(kernel.output.presented.form) == columns
    parameters = dict(kernel.module.parameters)
    assert parameters == {"BITS": 4, "I": ROWS, "J": COLS, "RAM_STYLE": '"auto"', "SIMD": simd}
    widths = {pin.name: pin.width for pin in kernel.module.abi.pins if isinstance(pin, Signal)}
    assert widths["idat"] == widths["odat"] == 4 * simd


def test_a_producer_of_rows_at_another_lane_count_is_only_a_width_conversion_away() -> None:
    """Its input is the matrices' flat row-major order, so rows at one lane differ from it
    in width only, wherever its beats end."""
    produced = BeatSequence(vector_major(MATRICES, 1))
    for simd in SIMDS:
        assert plan(produced, _transpose(simd).input.presented).steps == (Step.WIDTH,)


def test_its_output_must_read_the_inputs_matrices() -> None:
    """The input reads through a view, so the kernel gives its extents; an output of
    another shape is refused, naming the axis."""
    transposed = (BATCHES, COLS, ROWS)
    case = {**transpose(), "outputs": {"output_channel": transposed}}
    refused = unchosen(**case).kernel.query(TransposeKernel.extents)
    assert isinstance(refused, Rejected)
    assert [(f.code, f.message) for f in refused.findings] == [
        ("kernel-extents", f"i is {ROWS} (given) and {COLS} (output axis 1)")
    ]


def test_a_tensor_of_one_axis_is_no_matrix() -> None:
    case = {
        **transpose(),
        "inputs": {"input_channel": tensor((ROWS,), "INT4")},
        "outputs": {"output_channel": (ROWS,)},
    }
    kernel: Space = unchosen(**case).kernel
    assert rejected(kernel) == {"kernel-extents"}


def test_it_admits_the_two_pages_its_rtl_counts_and_no_more() -> None:
    """``2 I J`` elements in 32 bits: 2**15 x 2**16 is one element too many."""
    assert not rejected(_transpose(2))
    case = {
        **transpose(),
        "inputs": {"input_channel": tensor((1 << 15, 1 << 16), "INT4")},
        "outputs": {"output_channel": (1 << 15, 1 << 16)},
    }
    assert rejected(unchosen(**case).kernel) == {"transpose-depth"}


def test_its_ultra_pages_need_ultraram_but_no_initial_contents() -> None:
    """Refused without UltraRAM (the spec's probe, ``uram-absent``); the pages start empty,
    so UltraRAM that takes no initial contents is enough."""
    platform = replace(FULL_DSP48E2, uram_init=False)
    kernel: Any = placed(transpose(), {"simd": 2, "ram_style": "ultra"}, platform=platform)
    assert dict(kernel.module.parameters)["RAM_STYLE"] == '"ultra"'
