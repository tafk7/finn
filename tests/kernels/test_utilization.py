# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The resource export: memory primitives, the counts read off the RTL, and refusals.

The measured numbers are out-of-context synthesis on xczu3eg-sbva484-1-i (Vivado
2025.2): what a correct rewrite of the statements must still state. The fitted LUT
and FF coefficients are not pinned here; the counts the RTL fixes are.
"""

from __future__ import annotations

from typing import Any

import pytest

from finn.core.space import Available, Rejected
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.kernels.base import RESOURCES
from finn.kernels.dotp import (
    Int8Dsp58DotpKernel,
    PackedDotpKernel,
    int8_dsp58_dotp_resources,
    packed_dotp_resources,
)
from finn.kernels.fifo import fifo_resources
from finn.kernels.target import DspBlock
from finn.kernels.thresholding import stage_style, thresholding_resources
from finn.kernels.utilization import Fit, Resources, bram18, lutram, memory, total
from kernels.helpers import FULL_DSP48E2, FULL_DSP58, eltwise, placed_dotp, threshold_base

INT2, INT4, INT8 = (resolve_qonnx_datatype_name(name) for name in ("INT2", "INT4", "INT8"))


def stated(point: Any) -> Resources:
    answer = point.query(type(point).exports[RESOURCES])
    assert isinstance(answer, Available), answer
    used: Resources = answer.value
    return used


@pytest.mark.parametrize(
    ("words", "bits", "measured"),
    [
        (196, 512, 15),  # TFC MatMul_0's weights: 7 RAMB36 + 1 RAMB18
        (128, 64, 2),  # TFC MatMul_1's weights: 1 RAMB36
        (4096, 128, 29),  # 14 RAMB36 + 1 RAMB18: the word split over two aspects
        (256, 2048, 57),  # 28 RAMB36 + 1 RAMB18
        (512, 16, 1),
    ],
)
def test_block_ram_is_counted_as_synthesis_maps_it(words: int, bits: int, measured: int) -> None:
    assert bram18(words, bits) == measured
    assert memory(words, bits, "block") == Resources(bram18=measured)


def test_lutram_follows_the_port_mode() -> None:
    # One port (memstream's table): 64 x 1 a LUT; TFC MatMul_3's 160 x 8 weights.
    assert lutram(160, 8, single_port=True) == 24
    # Simple dual port (input_gen's buffer): 32 x 14 in eight LUTs; measured 76 and 296.
    assert lutram(16, 128) == 74
    assert lutram(4, 512) == 293
    # Deeper: eight LUTs a RAM64M8 of 64 x 7, and a remainder's bits a LUT each beside
    # their write address; a whole number of RAM64M8 has no remainder (no phantom LUT).
    assert lutram(64, 7) == 8 and lutram(64, 14) == 16 and lutram(128, 7) == 16
    assert lutram(64, 8) == 8 + 2


def test_auto_places_by_size_and_depth_as_measured() -> None:
    assert memory(128, 32, "auto").bram18 == 0  # 4096 bits: LUTRAM
    assert memory(128, 64, "auto").bram18 == 2  # 8192 bits: block RAM
    assert memory(64, 128, "auto").bram18 == 0  # 8192 bits, 64 deep: LUTRAM
    # TFC MatMul_0 at 3e6 fps: block RAM; measured 21 RAMB36 with the last 56 bits elsewhere.
    assert memory(64, 1568, "auto").bram18 == 44
    # A read-only memory: block RAM from 128 words (conservative), LUT logic below.
    assert memory(2048, 8, "auto", rom=True).bram18 == 1
    assert memory(8, 8, "auto", rom=True) == Resources(lut=4)
    assert memory(0, 8, "block") == Resources()
    assert memory(4096, 72, "ultra") == Resources(uram=1)


@pytest.mark.parametrize(
    ("pe", "simd", "bits", "narrow", "dsp"),
    [
        (16, 16, 2, True, 32),  # TFC MatMul_0 at 1e6 fps: nine lanes a slice
        (1, 32, 2, True, 32),
        (16, 16, 4, False, 64),  # four lanes a slice
        (8, 32, 8, False, 128),  # two lanes a slice
        (1, 256, 8, False, 256),
    ],
)
def test_a_packed_dot_product_s_dsp_slices_are_the_rtl_s(
    pe: int, simd: int, bits: int, narrow: bool, dsp: int
) -> None:
    used = packed_dotp_resources(
        pe=pe,
        simd=simd,
        weight_width=bits,
        activation_width=bits,
        narrow_weights=narrow,
        dsp=DspBlock.DSP48E2,
    )
    assert used.dsp == dsp


def packed_2bit(*, pe: int, simd: int, pumped: bool) -> PackedDotpKernel:
    """A packed core on narrow 2-bit weights and 2-bit activations, as TFC's."""
    return placed_dotp(
        PackedDotpKernel,
        activation_dtype=INT2,
        weights_dtype=INT2,
        result_dtype=resolve_qonnx_datatype_name("INT16"),
        weights_range=(-1, 1),
        pe=pe,
        simd=simd,
        compute_pumping=pumped,
        platform=FULL_DSP48E2,
    )


def test_a_pumped_core_is_built_at_half_the_simd() -> None:
    """dotp_axi builds its core at DSP_SIMD, ceil(SIMD / 2) under pumped compute: half
    the slices of TFC MatMul_0's 32 at PE 16, SIMD 16."""
    assert stated(packed_2bit(pe=16, simd=16, pumped=False)).dsp == 32
    assert stated(packed_2bit(pe=16, simd=16, pumped=True)).dsp == 16
    # SIMD 5 padded to 6, three a cycle, in each of two pipes.
    assert stated(packed_2bit(pe=16, simd=5, pumped=True)).dsp == 2 * 3


def test_the_int8_core_states_a_dsp58_chain_a_pe_lane() -> None:
    assert int8_dsp58_dotp_resources(pe=4, simd=7) == Resources(dsp=4 * 3)
    point = placed_dotp(
        Int8Dsp58DotpKernel,
        activation_dtype=INT8,
        weights_dtype=INT8,
        result_dtype=resolve_qonnx_datatype_name("INT32"),
        pe=2,
        simd=9,
        platform=FULL_DSP58,
    )
    assert stated(point) == Resources(dsp=2 * 3)


def test_the_compressor_reducer_is_refused_by_name() -> None:
    point = placed_dotp(
        PackedDotpKernel,
        activation_dtype=INT4,
        weights_dtype=INT4,
        result_dtype=resolve_qonnx_datatype_name("INT16"),
        pe=2,
        simd=4,
        reducer="compressor",
        platform=FULL_DSP48E2,
    )
    answer = point.query(type(point).exports[RESOURCES])
    assert isinstance(answer, Rejected)
    assert [finding.code for finding in answer.findings] == ["dotp-resources"]


def test_a_block_fifo_states_its_two_memory_spaces() -> None:
    """DEPTH 1100: the RTL keeps 1024 words in block RAM and its 128-word hi space
    ``auto`` (shallower than a RAMB18), which a 1 kbit memory leaves in LUTRAM."""
    used = fifo_resources(1100, 8, "block")
    assert used.bram18 == 1
    assert used.lut > fifo_resources(1025, 8, "block").lut  # the hi space's LUTRAM
    # Shallow FIFOs are shift registers whatever the style: a LUT a bit for 32 words.
    assert fifo_resources(33, 8, "block").bram18 == 0
    assert fifo_resources(1024, 32, "ultra").uram == 1


def test_thresholds_read_only_and_shared_cost_no_memory() -> None:
    """Synthesis folds a constant table where every row is the same (TFC
    MultiThreshold_0: no RAM measured); a runtime-writable one is a memory."""

    def table(writable: bool) -> Resources:
        return thresholding_resources(
            pe=4,
            wi=8,
            wt=8,
            stage_depths=(512, 1024),
            depth_trigger_bram=1,
            depth_trigger_uram=0,
            use_axilite=writable,
            shared_row=True,
        )

    shared, writable = table(False), table(True)
    assert shared.bram18 == 0 and writable.bram18 == 4 * (1 + 1)
    # The RTL's assignment: UltraRAM first, then block RAM from its trigger.
    assert [stage_style(depth, 4, 16) for depth in (2, 4, 16)] == ["distributed", "block", "ultra"]
    assert stage_style(2, 0, 0) == "auto"


def test_a_thresholding_kernel_reads_its_table_for_shared_rows() -> None:
    choices = {"pe": 1, "use_axilite": False, "deep_pipeline": False, "ultra_stages": 0}
    distinct = threshold_base().with_choices(**choices, ram_style="auto")
    same = threshold_base(table=(((-2, 0, 3), (-2, 0, 3)),)).with_choices(
        **choices, ram_style="auto"
    )
    # Two rows of three thresholds: two stages, of depth 2 and 4, as LUT logic.
    assert stated(distinct).lut > stated(same).lut
    assert stated(distinct).bram18 == stated(same).bram18 == 0


def test_resources_add_and_a_fit_rounds() -> None:
    one = Resources(lut=1, ff=2, bram18=3, uram=4, dsp=5)
    assert one + one == one.times(2) == total([one, one])
    assert total([]) == Resources()
    assert Fit(1.4, (0.5, 2.0)).at(3, 1) == 5
    with pytest.raises(ValueError, match="features"):
        Fit(1.0, (1.0,)).at(1, 2)


def test_a_leaf_that_states_no_model_is_refused_by_name() -> None:
    point = eltwise()
    answer = point.query(type(point).exports[RESOURCES])
    assert isinstance(answer, Rejected)
    assert [finding.code for finding in answer.findings] == ["kernel-resources"]
