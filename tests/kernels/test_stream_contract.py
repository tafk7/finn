# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Traversals, stream contracts, checked composition, and a reusable memory.

The stress cases come from baseline FINN: the tiled MVU's two internal
``input_gen`` adapters and the Shuffle op's inner/outer decomposition. Their
parameters are derived here from traversals alone and compared with the
hard-coded values in ``finn-rtllib`` and ``transpose_decomposition``.
"""

from pathlib import Path
import random

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Rejected, Unresolved, design_space
from finn.kernels.composite import Design
from finn.kernels.configure import commit
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.streams import Stream
from finn.kernels.target import DspBlock
from finn.kernels.artifacts.abi import Clock, Direction, Endpoint, Reset, Signal
from finn.kernels.artifacts.build import (
    GeneratedModuleName,
    ModuleABIRequirements,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.fifo import FifoKernel
from finn.kernels.physical.axi_stream import AxiStream
from finn.kernels.physical.composition import Composition, StreamEnd
from finn.kernels.physical.contract import StreamContract, compatibility
from finn.dataflow.traversal import (
    Adaptation,
    LevelEnd,
    Loop,
    Reorder,
    Repetition,
    Traversal,
    classify,
    pack,
    tile,
    vector_major,
)
from finn.kernels.physical.stream import MarkerKind, ReadyValidStream, StreamMarker
from finn.transformation.fpgadataflow.transpose_decomposition import (
    shuffle_perfect_loopnest_coeffs,
)
from kernels.xsim import pack as xsim_pack, requires_xsim, stream_through

ROOT = Path(__file__).resolve().parents[2]
INT3 = ScalarEncoding(DataType["INT3"])
INT4 = ScalarEncoding(DataType["INT4"])


def native(name, bits, endpoint, clock="clk", reset="rst", markers=()):
    return ReadyValidStream(
        name, bits, endpoint, f"{name}_d", f"{name}_v", f"{name}_r", clock, reset, markers
    )


def contract(form, endpoint, *, element=INT3, repetition=Repetition.ONCE, width=None, **kw):
    bits = width or form.lanes * element.bits
    return StreamContract(native("s", bits, endpoint), element, form, repetition, **kw)


def codes(mismatches):
    return {item.code for item in mismatches}


# -- traversals --------------------------------------------------------------------------


def test_tile_is_the_mvau_weight_order_and_packs_the_known_image():
    weights = tile(4, 4, 2, 2)
    assert (weights.lanes, weights.beats, weights.shape) == (4, 4, (4, 4))
    first, second = list(weights.positions())[:2]
    assert first == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert second == ((0, 2), (0, 3), (1, 2), (1, 3))
    matrix = ((-4, -3, -2, -1), (0, 1, 2, 3), (3, 2, 1, 0), (-1, -2, -3, -4))
    # Hand-packed INT3 fields: p0/s0, p0/s1, p1/s0, p1/s1, low first.
    assert pack(weights, matrix, 3) == (0x22C, 0x6BE, 0xDD3, 0x941)


def test_repetition_and_replay_are_stride_zero_loops():
    vector = vector_major((4,), 2)
    assert list(vector.repeated(2).positions()) == [((0,), (1,)), ((2,), (3,))] * 2
    rows = vector_major((2, 4), 2)
    replayed = rows.replayed(3, inner_beats=2)
    assert replayed.beats == 12
    assert [beat[0] for beat in replayed.positions()][:6] == [(0, 0), (0, 2)] * 3
    with pytest.raises(ValueError, match="divide"):
        vector_major((5,), 2)
    with pytest.raises(ValueError, match="shape"):
        pack(vector, (1, 2, 3), 4)
    assert LevelEnd(3).asserted(2) and not LevelEnd(3).asserted(3)


def _random_traversal(rng):
    while True:
        loops = [
            Loop(rng.choice((1, 2, 3)), rng.choice((0, 1, 2, 3, 6, 12)))
            for _ in range(rng.randint(1, 4))
        ]
        split = rng.randint(0, len(loops))
        try:
            return Traversal((4, 6), loops[:split], loops[split:])
        except ValueError:
            continue


def test_canonical_equality_is_equality_of_presented_sequences():
    rng = random.Random(1)
    for _ in range(4000):
        first, second = _random_traversal(rng), _random_traversal(rng)
        same = list(first.positions()) == list(second.positions())
        assert same is (first == second), (first, second)


# -- stress cases: tiled MVU and Shuffle ---------------------------------------------------

R, MW, MH, PE_T, SIMD_T, T = 6, 8, 6, 3, 2, 3
SF, NF = MW // SIMD_T, MH // PE_T


def test_tiled_mvu_input_adapter_is_derived_from_the_two_orders():
    boundary = vector_major((R, MW), SIMD_T)
    core = Traversal.over(
        (R, MW), ((0, R // T, T), (None, NF, 0), (1, SF, SIMD_T), (0, T, 1)), ((1, SIMD_T, 1),)
    )
    verdict = classify(boundary, core)
    # mvu_tiled_axi.sv: input_gen FM_SIZE=SF*TH, DIMS='{NF,SF,TH}, COEFS='{0,1,SF}.
    assert verdict.adaptation is Adaptation.REORDER
    assert verdict.reorder == Reorder(SF * T, (NF, SF, T), (0, 1, SF))


def test_tiled_mvu_output_reorder_is_derived_from_the_two_orders():
    core = Traversal.over((R, MH), ((0, R // T, T), (1, NF, PE_T), (0, T, 1)), ((1, PE_T, 1),))
    verdict = classify(core, vector_major((R, MH), PE_T))
    # mvu_tiled_axi.sv genReorder: FM_SIZE=NF*TH, DIMS='{TH,NF}, COEFS='{1,TH}.
    assert verdict.adaptation is Adaptation.REORDER
    assert verdict.reorder == Reorder(NF * T, (T, NF), (1, T))


def test_tiled_mvu_weight_chunks_are_a_width_conversion_a_delivery_can_avoid():
    chunked = Traversal.over(
        (MH, MW),
        ((0, NF, PE_T), (1, SF, SIMD_T), (0, T, 1)),
        ((0, PE_T // T, 1), (1, SIMD_T, 1)),
    )
    assert chunked.lanes == PE_T * SIMD_T // T
    assert classify(tile(MH, MW, PE_T, SIMD_T), chunked).adaptation is Adaptation.WIDTH_CONVERSION
    # A memory can simply produce the chunked order: no adapter at all.
    values = tuple(tuple((row * MW + col) % 7 - 3 for col in range(MW)) for row in range(MH))
    source = design_space(MemStreamKernel(dtype=DataType["INT3"], form=chunked, contents=values))
    sink = contract(chunked.repeated(R // T), Endpoint.TARGET)
    produced = source.output.contract
    assert compatibility(produced, sink, source_is_top=False, sink_is_top=False) == ()


def test_outer_shuffle_coefficients_match_finn():
    shape, perm, simd = (2, 3, 4), (1, 0, 2), 2
    moved = Traversal.over(shape, ((1, 3, 1), (0, 2, 1), (2, 2, simd)), ((2, simd, 1),))
    verdict = classify(vector_major(shape, simd), moved)
    finn = [1 if x == 1 else x // simd for x in shuffle_perfect_loopnest_coeffs(shape, perm)]
    assert verdict.adaptation is Adaptation.REORDER
    assert verdict.reorder.coefs == tuple(finn)
    assert verdict.reorder.dims == (3, 2, 2)


def test_inner_shuffle_moves_the_lane_axis():
    transposed = Traversal.over((4, 6), ((1, 6, 1), (0, 2, 2)), ((0, 2, 1),))
    verdict = classify(vector_major((4, 6), 2), transposed)
    assert verdict.adaptation is Adaptation.LANE_REGROUP


def test_different_operands_and_missing_positions_are_incompatible():
    assert classify(vector_major((4,), 2), vector_major((8,), 2)).adaptation is (
        Adaptation.INCOMPATIBLE
    )
    half = Traversal.over((4,), ((0, 1, 2),), ((0, 2, 1),))
    assert classify(vector_major((4,), 2), half).adaptation is Adaptation.INCOMPATIBLE


# -- compatibility -----------------------------------------------------------------------


def test_equal_contracts_connect_and_cyclic_sources_repeat_into_consumer_passes():
    weights = tile(4, 4, 2, 2)
    source = contract(weights, Endpoint.INITIATOR, repetition=Repetition.CYCLIC)
    for sink_form in (weights, weights.repeated(5)):
        sink = contract(sink_form, Endpoint.TARGET)
        assert compatibility(source, sink, source_is_top=False, sink_is_top=False) == ()


def test_mismatches_name_the_adapter_that_would_repair_them():
    found = compatibility(
        contract(tile(4, 4, 1, 4), Endpoint.INITIATOR),
        contract(tile(4, 4, 4, 1), Endpoint.TARGET),
        source_is_top=False,
        sink_is_top=False,
    )
    # Equal lanes and widths, different positions per beat.
    assert codes(found) == {"stream-form"}
    assert "lane_regroup" in next(iter(found)).message


def test_element_repetition_direction_and_marker_rules_are_checked():
    fold = vector_major((4,), 2)
    ok = contract(fold, Endpoint.INITIATOR)
    wider = contract(fold, Endpoint.TARGET, element=INT4, width=8)
    assert "stream-element" in codes(
        compatibility(ok, wider, source_is_top=False, sink_is_top=False)
    )
    cyclic_sink = contract(fold, Endpoint.TARGET, repetition=Repetition.CYCLIC)
    assert "stream-repetition" in codes(
        compatibility(ok, cyclic_sink, source_is_top=False, sink_is_top=False)
    )
    backwards = contract(fold, Endpoint.INITIATOR)
    assert "stream-direction" in codes(
        compatibility(ok, backwards, source_is_top=False, sink_is_top=False)
    )
    last = (StreamMarker("s_m", MarkerKind.LAST),)
    produced = StreamContract(
        native("s", 6, Endpoint.INITIATOR, markers=last), INT3, fold, markers={"s_m": LevelEnd(2)}
    )
    required = StreamContract(
        native("s", 6, Endpoint.TARGET, markers=last), INT3, fold, markers={"s_m": LevelEnd(1)}
    )
    assert "stream-marker" in codes(
        compatibility(produced, required, source_is_top=False, sink_is_top=False)
    )


def test_contracts_reject_lanes_wider_than_the_word_and_unknown_marker_rules():
    with pytest.raises(ValueError, match="exceed"):
        contract(vector_major((4,), 4), Endpoint.TARGET, width=8)
    with pytest.raises(ValueError, match="marker"):
        contract(vector_major((4,), 2), Endpoint.TARGET, markers={"missing": LevelEnd(2)})
    # A marker closes a loop level of its form: three beats close none of two.
    last = (StreamMarker("s_m", MarkerKind.LAST),)
    with pytest.raises(ValueError, match="closes no loop level"):
        StreamContract(
            native("s", 6, Endpoint.TARGET, markers=last),
            INT3,
            vector_major((4,), 2),
            markers={"s_m": LevelEnd(3)},
        )


# -- the delivery kernel -----------------------------------------------------------------


def hold(composition, instance, kernel):
    """Tie off what an unplaced memory holds, but the output this test connects itself."""
    connected = {pin.name for pin in kernel.output.contract.transport.pins()}
    tieoffs = kernel.tieoffs
    for pin, value in tieoffs.inputs:
        if pin not in connected:
            composition.tie(instance, pin, value)
    for pin in tieoffs.unused:
        if pin not in connected:
            composition.dispose(instance, pin, "tied off")


def delivery(form=None, values=(1, -2, 7, -8), **choices):
    form = vector_major((4,), 2) if form is None else form
    base = design_space(MemStreamKernel(dtype=DataType["INT4"], form=form, contents=values))
    return base.with_choices(**choices) if choices else base


def test_delivery_publishes_a_cyclic_contract_and_waits_only_for_its_own_choice():
    base = delivery()
    output = base.output.contract
    assert output.repetition is Repetition.CYCLIC and output.form == vector_major((4,), 2)
    assert output.payload_bits == output.transport.data_width == 8
    assert base.image == (0xE1, 0x87)
    assert isinstance(base.query(MemStreamKernel.build_requirements), Unresolved)
    requirements = delivery(ram_style="block", pumped_memory=False).build_requirements
    assert dict(requirements.parameters)["RAM_STYLE"] == '"block"'


@pytest.mark.parametrize(
    "values,dtype,message",
    [
        ((1, 2, 3), "INT4", "shape"),
        ((1, 2, 3, 8), "INT4", "admitted"),
        ((1, 2, 3, 4), "FLOAT32", None),
    ],
)
def test_delivery_refuses_values_outside_the_operand_contract(values, dtype, message):
    point = design_space(
        MemStreamKernel(dtype=DataType[dtype], form=vector_major((4,), 2), contents=values)
    )
    answer = point.with_choices(ram_style="auto", pumped_memory=False).query(
        MemStreamKernel.build_requirements
    )
    assert isinstance(answer, Rejected)
    if message:
        assert any(message in finding.message for finding in answer.findings)


# -- a second consumer: cyclic channel parameters into eltwise ---------------------------

CLOCKING = (
    Signal("ap_clk", Direction.IN, 1, Clock()),
    Signal(
        "ap_rst_n",
        Direction.IN,
        1,
        Reset(active_low=True, synchronous=True, synchronous_to=("ap_clk",)),
    ),
)
CHANNELS, PE, PIXELS = 4, 2, 3
PARAMETERS = (1, -2, 7, -8)


def eltwise_with_constant(form=None):
    """Eltwise ADD whose rhs is a memory's cyclic channel vector, joined directly."""
    form = vector_major((CHANNELS,), PE) if form is None else form
    int4, int5 = DataType["INT4"], DataType["INT5"]

    class Constant(Design):
        x = Stream(tensor=Tensor((PIXELS, CHANNELS), INT4), port="in0_V")
        c = Stream(tensor=Tensor((CHANNELS,), INT4), adaptable=False)
        y = Stream(tensor=Tensor((PIXELS, CHANNELS), ScalarEncoding(int5)), port="out0_V")
        rhs = MemStreamKernel(dtype=int4, form=form, contents=PARAMETERS, output_stream=c)
        add = EltwiseKernel(
            operation="ADD",
            pe=PE,
            lhs_dtype=int4,
            rhs_dtype=int4,
            b_scale=1.0,
            target_dsp=DspBlock.DSP58,
            lhs_stream=x,
            rhs_stream=c,
            result_stream=y,
        )

    return commit(
        design_space(Constant()), {"rhs.ram_style": "distributed", "rhs.pumped_memory": False}
    )


def test_the_delivery_kernel_serves_a_second_consumer_through_the_same_contract():
    structure = eltwise_with_constant().structure.structure
    assert {
        (wire.destination.bit_offset, wire.source.bit_offset)
        for wire in structure.wires
        if wire.destination.pin.signal_id == "bdat"
    } == {(0, 0), (4, 4)}
    # A delivery whose lanes carry other positions (0,2),(1,3) needs a lane regroup.
    strided = Traversal.over((CHANNELS,), ((0, 2, 1),), ((0, 2, 2),))
    refused = eltwise_with_constant(strided).c.query(Stream.connection)
    assert isinstance(refused, Rejected)
    (plan,) = [finding for finding in refused.findings if finding.code == "stream-plan"]
    assert "another lane axis" in plan.message


def test_a_pure_lane_permutation_is_realized_as_free_wiring():
    # Beat i carries x[i] as a 2x2 block; the consumer wants its lanes transposed.
    shape = (2, 2, 2)
    produced = Traversal.over(shape, ((0, 2, 1),), ((1, 2, 1), (2, 2, 1)))
    wanted = Traversal.over(shape, ((0, 2, 1),), ((2, 2, 1), (1, 2, 1)))
    verdict = classify(produced, wanted)
    assert verdict.adaptation is Adaptation.LANE_PERMUTATION
    assert verdict.lane_permutation == (0, 2, 1, 3)
    values = (((1, 2), (3, 4)), ((5, 6), (7, -8)))
    source = design_space(MemStreamKernel(dtype=DataType["INT4"], form=produced, contents=values))
    fifo = design_space(FifoKernel(word_bits=16, depth=2)).with_choices(ram_style="auto")
    fifo_in, fifo_out = fifo.input.transport, fifo.output.transport
    out = AxiStream("out0_V", DataType["INT4"], 4, endpoint=Endpoint.INITIATOR)
    top = StreamContract(
        out.native(clock="ap_clk", reset="ap_rst_n"), INT4, wanted, Repetition.CYCLIC
    )
    composition = Composition(
        ModuleABIRequirements(
            GeneratedModuleName("t"), (*CLOCKING, out.bus(clock="ap_clk", reset="ap_rst_n")), ()
        )
    )
    source = source.with_choices(ram_style="auto", pumped_memory=False)
    composition.add("u_source", source.build_requirements)
    composition.add("u_fifo", fifo.build_requirements)
    for owner in ("u_source", "u_fifo"):
        composition.drive(owner, "clk", "ap_clk")
        composition.drive(owner, "rst", "ap_rst_n")
    hold(composition, "u_source", source)
    composition.connect(
        StreamEnd("u_source", source.output.contract),
        StreamEnd("u_fifo", StreamContract(fifo_in, INT4, wanted)),
    )
    composition.connect(
        StreamEnd("u_fifo", StreamContract(fifo_out, INT4, wanted, Repetition.CYCLIC)),
        StreamEnd(None, top),
    )
    structure = composition.finish()
    crossed = {
        (wire.destination.bit_offset, wire.source.bit_offset)
        for wire in structure.wires
        if wire.destination.pin.signal_id == "idat"
    }
    assert crossed == {(0, 0), (4, 8), (8, 4), (12, 12)}


def test_clock_domains_must_be_attached_and_equal():
    stream = native("o", 8, Endpoint.INITIATOR, clock="clk")
    sink = native("i", 8, Endpoint.TARGET, clock="clk")
    top_abi = ModuleABIRequirements(GeneratedModuleName("t"), CLOCKING, ())
    composition = Composition(top_abi)
    composition.add("a", delivery(ram_style="auto", pumped_memory=False).build_requirements)
    composition.add("b", delivery(ram_style="auto", pumped_memory=False).build_requirements)
    found = composition.check(
        StreamEnd("a", StreamContract(stream, INT4, vector_major((4,), 2))),
        StreamEnd("b", StreamContract(sink, INT4, vector_major((4,), 2))),
    )
    assert "stream-clock" in codes(found)


@requires_xsim
def test_eltwise_with_cyclic_constant_computes_the_broadcast_sum(tmp_path):
    inputs = [(-8 + 3 * index) % 16 - 8 for index in range(CHANNELS * PIXELS)]
    expected = [value + PARAMETERS[index % CHANNELS] for index, value in enumerate(inputs)]
    words_in = [xsim_pack(inputs[i : i + PE], 4) for i in range(0, len(inputs), PE)]
    words_out = [xsim_pack(expected[i : i + PE], 5) for i in range(0, len(expected), PE)]
    stream_through(
        eltwise_with_constant().structure.requirements,
        tmp_path,
        inputs={"in0_V": (words_in, 4 * PE)},
        outputs={"out0_V": (words_out, 5 * PE)},
    )
