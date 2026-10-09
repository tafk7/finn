# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The resource export: memory primitives per fabric, the counts read off the RTL, and
refusals.

The measured numbers are Vivado 2025.2 out-of-context synthesis on xczu3eg-sbva484-1-i,
xczu7ev-ffvc1156-2-e, xc7z020clg400-1 and xcvc1902-vsva2197-2MP-e-S, each named where
it is pinned: what a correct rewrite of the statements must still state, and what a
change to the primitive table (``PRIMITIVES``) must change here with its numbers. The
fitted LUT and FF coefficients are not pinned here; the counts the RTL fixes are.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from finn.core.space import Available, Rejected, design_space
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.tensor import Tensor
from finn.dataflow.traversal import vector_major
from finn.kernels.base import RESOURCES
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.dotp import (
    Int8Dsp58DotpKernel,
    PackedDotpKernel,
    int8_dsp58_dotp_resources,
    packed_dotp_resources,
)
from finn.kernels.fifo import fifo_resources
from finn.kernels.input_generator import InputGeneratorKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.target import DspBlock
from finn.kernels.thresholding import (
    ThresholdingAxiKernel,
    stage_style,
    thresholding_resources,
)
from finn.kernels.transpose import TransposeKernel
from finn.kernels.utilization import (
    BLOCK_FIRST_BITS,
    PRIMITIVES,
    Fabric,
    Fit,
    Resources,
    bram18,
    lutram,
    lutram_storage,
    memory,
    read_mux,
    total,
    uram,
)
from kernels.adapted import ELEMENT, transposed
from kernels.helpers import (
    FULL_DSP48E2,
    FULL_DSP58,
    Root,
    eltwise,
    placed_dotp,
    threshold_base,
    with_direct_transports,
)

INT2, INT4, INT8 = (resolve_qonnx_datatype_name(name) for name in ("INT2", "INT4", "INT8"))


def stated(point: Any) -> Resources:
    answer = point.query(type(point).exports[RESOURCES])
    assert isinstance(answer, Available), answer
    used: Resources = answer.value
    return used


S7, US, V = Fabric.SERIES7, Fabric.ULTRASCALE, Fabric.VERSAL


@pytest.mark.parametrize(
    ("fabric", "words", "bits", "measured"),
    [
        (US, 196, 512, 15),  # TFC MatMul_0's weights on xczu3eg: 7 RAMB36 + 1 RAMB18
        (US, 128, 64, 2),  # TFC MatMul_1's weights on xczu3eg: 1 RAMB36
        (US, 4096, 128, 29),  # 14 RAMB36 + 1 RAMB18: the word split over two aspects
        (US, 256, 2048, 57),  # 28 RAMB36 + 1 RAMB18
        (US, 512, 16, 1),
        (US, 4096, 64, 15),  # a memstream on xczu7ev and on xcku040: 4- and 1-bit aspects
        (S7, 4096, 64, 15),  # the same memstream on xc7z020: the same aspects
        (S7, 256, 1024, 29),
        (V, 4096, 64, 16),  # on xcvc1902: RAMB18E5 has nothing narrower than 9 bits
        (V, 8192, 32, 16),
        (V, 1024, 256, 15),
        (V, 256, 64, 2),
    ],
)
def test_block_ram_is_counted_as_synthesis_maps_it_on_each_fabric(
    fabric: Fabric, words: int, bits: int, measured: int
) -> None:
    assert bram18(words, bits, fabric=fabric) == measured
    assert memory(words, bits, "block", fabric=fabric) == Resources(bram18=measured)


def test_ultraram_is_counted_in_the_fabric_s_aspects() -> None:
    # xczu7ev: one aspect, 72 x 4096; xcvc1902: four, a word never split across them
    # (8192 x 8 is one URAM288E5 at 9 x 32768, a transpose's 32 banks of 2048 x 8 eight).
    assert uram(4096, 64, fabric=US) == 1 and uram(8192, 8, fabric=US) == 2
    assert uram(8192, 8, fabric=V) == 1 and uram(4096, 64, fabric=V) == 1
    assert uram(1024, 256, fabric=V) == 4
    assert memory(2048, 8, "ultra", fabric=V).times(32) == Resources(uram=32)
    assert memory(2048, 8, "ultra", fabric=US).times(32) == Resources(uram=32)
    with pytest.raises(ValueError, match="no UltraRAM"):
        uram(4096, 72, fabric=S7)


@pytest.mark.parametrize(
    ("fabric", "words", "bits", "measured"),
    [
        # xc7z020, the LUTRAMs column: RAM64M (64 x 3 in four) and RAM64X1D (64 x 1 in
        # two) a bank, RAM32M (32 x 6 in four) to 32 rows.
        (S7, 1024, 256, 5472),  # a FIFO's buffer: 16 banks of 85 RAM64M and a RAM64X1D
        (S7, 256, 256, 1368),
        (S7, 4, 64, 44),  # input_gen's buffers
        (S7, 4, 32, 24),
        (S7, 32, 16, 12),
        (S7, 256, 2, 16),
        # xczu3eg: RAM64M8 (64 x 7 in eight) a bank, the remainder in RAM64X1D (1 bit),
        # RAM64M (2 bits) or RAM64M8 (4 bits); RAM32M16 (32 x 14) to 32 rows, the
        # remainder in RAM32M (2 and 4 bits) or RAM32M16 (8 and 12 bits).
        (US, 1024, 256, 4736),
        (US, 64, 8, 10),
        (US, 256, 2, 16),
        (US, 128, 4, 16),
        (US, 4, 256, 148),
        (US, 4, 64, 40),
        (US, 16, 128, 76),
        (US, 4, 512, 296),
        (US, 4, 96, 56),
        (US, 32, 196, 112),
        (US, 1024, 32, 640),  # xczu7ev, E1(c): 40 LUTs a bank at 32 bits, not 37
        (V, 1024, 32, 640),  # xcvc1902: the same LUTRAMs
        (V, 4096, 64, 4736),
    ],
)
def test_lutram_storage_is_the_fabric_s_primitives(
    fabric: Fabric, words: int, bits: int, measured: int
) -> None:
    assert lutram_storage(words, bits, fabric=fabric) == measured


@pytest.mark.parametrize(
    ("fabric", "words", "bits", "measured"),
    [
        # E1(c), the whole memory's LUTs: storage, a LUT a bank for the write decode, and
        # the read multiplexer. xczu7ev: MUXF7 and MUXF8 combine four LUT6s; the first
        # four banks are not free.
        (US, 128, 8, 26),
        (US, 256, 32, 196),
        (US, 512, 64, 728),
        (US, 1024, 32, 784),
        (US, 2048, 64, 2976),  # the 2:1 between two MUXF8s a LUT
        (US, 4096, 64, 5893 - 5),  # five LUTs under at 64 banks
        # xcvc1902: no wide multiplexer, LUT6s alone; one port the same as two.
        (V, 128, 64, 182),
        (V, 512, 8, 108),
        (V, 1024, 32, 816),
        (V, 2048, 32, 1648),
        (V, 4096, 32, 3300 - 4),  # four LUTs under at 64 banks
    ],
)
def test_a_lutram_s_decode_and_read_multiplexer_are_the_fabric_s(
    fabric: Fabric, words: int, bits: int, measured: int
) -> None:
    assert lutram(words, bits, fabric=fabric) == measured
    assert memory(words, bits, "distributed", fabric=fabric) == Resources(lut=measured)


def test_the_read_multiplexer_differs_by_fabric_from_eight_banks() -> None:
    # 2 and 4 banks: half a LUT and a LUT a bit on both; from 8, F7/F8 save on UltraScale.
    assert [read_mux(banks, 1, fabric=US) for banks in (2, 4, 8, 16, 32, 64)] == [
        1,
        1,
        2,
        4,
        9,
        17,
    ]
    assert [read_mux(banks, 2, fabric=V) for banks in (2, 4, 8, 16, 32, 64)] == [
        1,
        2,
        5,
        10,
        21,
        42,
    ]
    assert read_mux(1, 64, fabric=V) == 0
    # A one-port bank is a slice's primitive with its multiplexer inside: 512 rows on
    # UltraScale (RAM512X1S, MUXF9), 256 on the 7 series, 64 on Versal.
    assert lutram(1024, 32, fabric=V, single_port=True) == 688  # E1(c), written
    assert lutram(4096, 64, fabric=US, single_port=True, written=False) == 4096 + 128


def test_a_one_port_table_s_lutram_and_multiplexer_as_measured() -> None:
    """memstream's 4,096 x 64 table, never written, measured as the whole leaf (its
    output stream besides): xczu7ev 4,096 LUTRAM and 138 logic LUTs (4,096 RAMS64E1, 512
    MUXF9; two LUT6s a bit over eight 512-row banks); xcvc1902 4,096 and 1,363 (21 LUT6s
    a bit over 64 banks); xc7z020 4,096 and 266 (four a bit over sixteen 256-row
    banks)."""
    table = {
        fabric: memory(4096, 64, "distributed", fabric=fabric, single_port=True, written=False)
        for fabric in (S7, US, V)
    }
    assert table == {
        S7: Resources(lut=4096 + 256),
        US: Resources(lut=4096 + 128),
        V: Resources(lut=4096 + 1344),
    }
    # TFC MatMul_3's 160 x 8 weights: three LUTs a bit, in one bank.
    assert memory(160, 8, "distributed", fabric=US, single_port=True) == Resources(lut=24)


def test_a_memory_is_stated_in_an_explicit_style_only() -> None:
    # TFC MatMul_0 at 3e6 fps in block RAM: measured 21 RAMB36 with the last 56 bits
    # elsewhere (Vivado's packing, two RAMB18 over the aspects' count).
    assert memory(64, 1568, "block", fabric=US).bram18 == 44
    assert memory(0, 8, "block", fabric=US) == Resources()
    assert memory(4096, 72, "ultra", fabric=US) == Resources(uram=1)
    # ``auto`` is Vivado's placement, which states nothing: no statement of it is made.
    with pytest.raises(ValueError, match="'auto' is no explicit style"):
        memory(128, 64, "auto", fabric=US)


LUTRAM_FIRST = ("distributed", "block", "ultra", "auto")
BLOCK_FIRST = ("block", "distributed", "ultra", "auto")


def table(elements: int) -> Any:
    """A memstream holding ``elements`` INT4 values, one a word, its style open."""
    values = tuple(index % 8 for index in range(elements))
    return design_space(
        MemStreamKernel(
            dtype=INT4,
            form=vector_major((elements,), 1),
            contents=values,
            platform=FULL_DSP48E2,
        )
    )


def buffer(frame_words: int) -> Any:
    """An input generator streaming a frame of 32-bit words once, its style open."""
    return design_space(
        InputGeneratorKernel(
            word_bits=32,
            frame_words=frame_words,
            dims=(frame_words,),
            strides=(1,),
            platform=FULL_DSP48E2,
        )
    )


def pages(rows: int, cols: int) -> Any:
    """A transpose of two ``rows x cols`` INT4 matrices, SIMD 4, its pages' style open."""
    shape = (2, rows, cols)

    class Transposed(Root):
        a = Channel(tensor=Tensor(shape, ELEMENT), port="in0_V", platform=FULL_DSP48E2)
        b = Channel(tensor=Tensor(shape, ELEMENT), port="out0_V", platform=FULL_DSP48E2)
        shuffle = TransposeKernel(input_channel=a, output_channel=b, platform=FULL_DSP48E2)

    return commit(with_direct_transports(design_space(Transposed())), {"shuffle.simd": 4})


def cases(point: Any, decision: Any) -> tuple[str, ...]:
    found = point.field(decision).candidates()
    assert isinstance(found, Available), found
    return tuple(found.value)


def test_each_memory_offers_its_explicit_styles_by_size_then_auto() -> None:
    """Block RAM first from ``BLOCK_FIRST_BITS`` held, LUTRAM first below, UltraRAM after
    both, and ``auto`` last: a memstream's table (its contents, whatever the folding),
    an input generator's buffer (a frame, the most it holds) and a transpose's pages."""
    assert BLOCK_FIRST_BITS == 8192 == 2048 * 4 == 256 * 32 == 2 * 32 * 32 * 4
    assert cases(table(2047), MemStreamKernel.ram_style) == LUTRAM_FIRST
    assert cases(table(2048), MemStreamKernel.ram_style) == BLOCK_FIRST
    assert cases(buffer(255), InputGeneratorKernel.ram_style) == LUTRAM_FIRST
    assert cases(buffer(256), InputGeneratorKernel.ram_style) == BLOCK_FIRST
    assert cases(pages(32, 32).shuffle, TransposeKernel.ram_style) == BLOCK_FIRST
    assert cases(pages(16, 32).shuffle, TransposeKernel.ram_style) == LUTRAM_FIRST
    # The thresholds' stages left above the UltraRAM ones: distributed first.
    assert threshold_base().with_choices(ultra_stages=0).field(
        ThresholdingAxiKernel.ram_style
    ).candidates() == Available(("distributed", "auto"))


@pytest.mark.parametrize(
    "memory_point",
    [
        lambda: table(16).with_choices(ram_style="auto", pumped_memory=False),
        lambda: buffer(16).with_choices(ram_style="auto"),
        lambda: transposed(16, 32, 4, ram_style="auto"),
    ],
    ids=("memstream", "input_gen", "transpose"),
)
def test_a_memory_in_auto_states_no_resources(memory_point: Any) -> None:
    """``auto`` is Vivado's placement: the memory names why it states none, so a total
    over it is a lower bound naming it."""
    point = memory_point()
    answer = point.query(type(point).exports[RESOURCES])
    assert isinstance(answer, Rejected)
    assert [finding.code for finding in answer.findings] == ["memory-auto"]
    assert answer.findings[0].message.endswith("ram_style auto, placed by Vivado, not stated")


def test_every_fabric_s_primitives_state_their_evidence() -> None:
    assert set(PRIMITIVES) == set(Fabric)
    for primitives in PRIMITIVES.values():
        assert all(sentence.endswith(".") for _, sentence in primitives.evidence)
    with pytest.raises(ValueError, match="evidence for each"):
        replace(PRIMITIVES[US], evidence=(("bram18_sdp", "measured."),))


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
    ``auto`` (shallower than a RAMB18), which the FIFO's model of Vivado's placement
    leaves in LUTRAM at 1 kbit."""
    used = fifo_resources(1100, 8, "block", fabric=US)
    assert used.bram18 == 1
    assert used.lut > fifo_resources(1025, 8, "block", fabric=US).lut  # the hi's LUTRAM
    # Shallow FIFOs are shift registers whatever the style: a LUT a bit for 32 words.
    assert fifo_resources(33, 8, "block", fabric=US).bram18 == 0
    assert fifo_resources(1024, 32, "ultra", fabric=US).uram == 1
    # DEPTH 4096 x 64 in block RAM: 15 RAMB18 on xczu7ev, 16 on xcvc1902 (RAMB18E5).
    assert fifo_resources(4096, 64, "block", fabric=US).bram18 == 15
    assert fifo_resources(4096, 64, "block", fabric=V).bram18 == 16


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
            fabric=US,
        )

    shared, writable = table(False), table(True)
    assert shared.bram18 == 0 and writable.bram18 == 4 * (1 + 1)
    # The RTL's assignment: UltraRAM first, then block RAM from its trigger.
    assert [stage_style(depth, 4, 16) for depth in (2, 4, 16)] == ["distributed", "block", "ultra"]
    assert stage_style(2, 0, 0) == "auto"


def test_a_thresholding_kernel_reads_its_table_for_shared_rows() -> None:
    choices = {"pe": 1, "use_axilite": False, "deep_pipeline": False, "ultra_stages": 0}
    left = {"ram_style": "distributed", "block_stages": 0}
    distinct = threshold_base().with_choices(**choices, **left)
    same = threshold_base(table=(((-2, 0, 3), (-2, 0, 3)),)).with_choices(**choices, **left)
    # Two rows of three thresholds: two stages, of depth 2 and 4, as LUT logic.
    assert stated(distinct).lut > stated(same).lut
    assert stated(distinct).bram18 == stated(same).bram18 == 0
    # Left to ``auto``, Vivado places the stages: unstated, unless the table folds to
    # nothing (shared rows, never written), which places no memory.
    placed = threshold_base().with_choices(**choices, ram_style="auto")
    answer = placed.query(type(placed).exports[RESOURCES])
    assert isinstance(answer, Rejected)
    assert [finding.code for finding in answer.findings] == ["memory-auto"]
    assert "placed by Vivado, not stated" in answer.findings[0].message
    folded = threshold_base(table=(((-2, 0, 3), (-2, 0, 3)),)).with_choices(
        **choices, ram_style="auto"
    )
    assert stated(folded) == stated(same)


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
