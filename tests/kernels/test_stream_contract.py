# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Traversals, channel-end contracts, checked composition, and a reusable memory.

The stress cases come from baseline FINN: the tiled MVU's two internal
``input_gen`` adapters and the Shuffle op's inner/outer decomposition. Their
parameters are derived here from traversals alone and compared with the
hard-coded values in ``finn-rtllib`` and ``transpose_decomposition``.
"""

import operator
import random
from math import prod

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.core.executors.xsim.rtl import pack_lanes, stream_through
from finn.core.space import Rejected, Unresolved, design_space
from finn.dataflow import traversal
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import (
    Adaptation,
    LevelEnd,
    Loop,
    Reorder,
    Repetition,
    Traversal,
    axis_strides,
    classify,
    offsets,
    pack,
    tile,
    unpack,
    vector_major,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.channels import Channel, wired
from finn.kernels.configure import commit
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.fifo import FifoKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.transport import (
    MarkerKind,
    ReadyValidStream,
    StreamContract,
    StreamMarker,
    compatibility,
    marker_pairs,
)
from finn.transformation.fpgadataflow.transpose_decomposition import (
    shuffle_perfect_loopnest_coeffs,
)
from kernels.helpers import FULL_DSP48E2, FULL_DSP58, Root, with_direct_transports
from kernels.xsim import requires_xsim

INT3 = ScalarEncoding(DataType["INT3"])
INT4 = ScalarEncoding(DataType["INT4"])


def native(name, bits, endpoint, clock="clk", reset="rst", markers=()):
    return ReadyValidStream(
        name, bits, endpoint, f"{name}_d", f"{name}_v", f"{name}_r", clock, reset, markers
    )


def contract(form, endpoint, *, element=INT3, repetition=Repetition.ONCE, width=None, **kw):
    bits = width or form.lanes * element.bits
    return StreamContract(native("s", bits, endpoint), element, form, repetition, **kw)


def mismatch_codes(mismatches):
    return {item.code for item in mismatches}


# -- traversals --------------------------------------------------------------------------


def test_tile_is_the_mvau_weight_order_and_packs_the_known_image():
    weights = tile(4, 4, 2, 2)
    assert (weights.lanes, weights.beats, weights.shape) == (4, 4, (4, 4))
    first, second = list(weights.positions())[:2]
    assert first == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert second == ((0, 2), (0, 3), (1, 2), (1, 3))
    matrix = ((-4, -3, -2, -1), (0, 1, 2, 3), (3, 2, 1, 0), (-1, -2, -3, -4))
    # Hand-packed INT3 lanes: p0/s0, p0/s1, p1/s0, p1/s1, low first.
    assert pack(weights, sum(matrix, ()), 3) == (0x22C, 0x6BE, 0xDD3, 0x941)


def test_repetition_and_replay_are_stride_zero_loops():
    vector = vector_major((4,), 2)
    assert list(vector.repeated(2).positions()) == [((0,), (1,)), ((2,), (3,))] * 2
    rows = vector_major((2, 4), 2)
    replayed = rows.replayed(3, inner_beats=2)
    assert replayed.beats == 12
    assert [beat[0] for beat in replayed.positions()][:6] == [(0, 0), (0, 2)] * 3
    with pytest.raises(ValueError, match="divide"):
        vector_major((5,), 2)
    with pytest.raises(ValueError, match="has 4 integers, not 3"):
        pack(vector, (1, 2, 3), 4)
    assert LevelEnd(3).asserted(2) and not LevelEnd(3).asserted(3)


def _packed_by_position(form, integers, bits):
    """``pack`` by its definition: each beat's positions, raveled row-major, low bits first."""
    strides, mask = axis_strides(form.shape), (1 << bits) - 1
    return tuple(
        sum(
            (integers[sum(map(operator.mul, position, strides))] & mask) << (lane * bits)
            for lane, position in enumerate(beat)
        )
        for beat in form.positions()
    )


def _random_form(rng):
    """A traversal of a random operand: vector-major, tiled, or loops of random extents
    along random axes, replay loops (stride 0) among them."""
    shape = tuple(rng.randint(1, 6) for _ in range(rng.randint(1, 3)))
    kind = rng.choice(("vector", "tile", "loops", "loops"))
    if kind == "vector":
        return vector_major(
            shape, rng.choice([d for d in range(1, shape[-1] + 1) if not shape[-1] % d])
        )
    if kind == "tile" and len(shape) == 2:
        rows, cols = shape
        pe = rng.choice([d for d in range(1, rows + 1) if not rows % d])
        return tile(rows, cols, pe, rng.choice([d for d in range(1, cols + 1) if not cols % d]))
    strides = (0, *axis_strides(shape))
    while True:
        loops = [Loop(rng.randint(1, 4), rng.choice(strides)) for _ in range(rng.randint(1, 5))]
        split = rng.randint(0, len(loops))
        try:
            return Traversal(shape, loops[:split], loops[split:])
        except ValueError:
            continue


def test_pack_is_its_definition_at_every_width_signedness_and_lane_count():
    """The words are each beat's lanes at their positions, the low ``bits`` of each
    integer's two's complement, lane zero lowest: for lanes 1..8 bits wide and wider
    than an int64, integers narrower or wider than their lanes, signed or not, and as a
    tuple or an int64 array (packed with numpy) or beyond 64 bits (one at a time)."""
    rng = random.Random(7)
    for _ in range(600):
        form = _random_form(rng)
        bits = rng.choice((*range(1, 18), 31, 32, 33, 63, 64, 65, 100, 128))
        width = rng.choice((1, 2, 3, 4, 7, 8, 9, 16, 32, 62, 63, 64, 65, 80))
        signed = rng.random() < 0.5
        low, high = (
            (-(1 << (width - 1)), (1 << (width - 1)) - 1) if signed else (0, (1 << width) - 1)
        )
        integers = tuple(rng.randint(low, high) for _ in range(prod(form.shape)))
        expected = _packed_by_position(form, integers, bits)
        assert len(expected) == form.beats
        assert pack(form, integers, bits) == expected, (form, bits, width, signed)
        if -(2**63) <= low and high < 2**63:
            array = np.array(integers, dtype=np.int64)
            assert pack(form, array, bits) == expected, (form, bits, width, signed)


@pytest.mark.parametrize("bits", (1, 63, 64, 65, 128))
def test_pack_uses_numpy_exactly_up_to_int64_and_python_ints_beyond(bits, monkeypatch):
    """At the int64 bounds pack stays with numpy (an exact-path call fails the test);
    one integer beyond them takes the exact path. Both give the words by definition."""
    form = vector_major((2, 4), 2)

    def refused(*args):
        raise AssertionError("integers within int64 took the exact path")

    edges = (2**63 - 1, -(2**63), -1, 0, 1, 2**62, -(2**62) - 1, 5)
    with monkeypatch.context() as patched:
        patched.setattr(traversal, "_pack_exact", refused)
        assert pack(form, edges, bits) == _packed_by_position(form, edges, bits)
    taken = []
    exact = traversal._pack_exact
    monkeypatch.setattr(traversal, "_pack_exact", lambda *args: taken.append(1) or exact(*args))
    for beyond in (2**63, -(2**63) - 1, 2**64 + 3, -(2**100)):
        found = (*edges[:-1], beyond)
        assert pack(form, found, bits) == _packed_by_position(form, found, bits)
    assert len(taken) == 4


def test_unpack_inverts_pack_at_every_width_signedness_and_lane_count():
    """The operand read back from its words is the operand, for every traversal that
    presents each element (replays among them), lanes of 1 to 64 bits, signed or not."""
    rng = random.Random(11)
    checked = 0
    while checked < 400:
        form = _random_form(rng)
        if set(offsets(form).tolist()) != set(range(prod(form.shape))):
            continue  # an element it never presents: no operand to read back
        signed = rng.random() < 0.5
        bits = rng.choice((1, 2, 3, 4, 7, 8, 9, 16, 31, 32, 33, 62, 63, *((64,) if signed else ())))
        low, high = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
        operand = np.array([rng.randint(low, high) for _ in range(prod(form.shape))], np.int64)
        words = pack(form, operand, bits)
        found = unpack(form, words, bits, signed=signed)
        assert found.shape == form.shape and np.array_equal(found.ravel(), operand), (form, bits)
        assert pack(form, found.ravel(), bits) == words
        checked += 1


def test_unpack_refuses_words_that_are_no_operand():
    vector = vector_major((4,), 2)
    assert unpack(vector, (0x1F, 0x23), 4, signed=True).tolist() == [-1, 1, 3, 2]
    assert unpack(vector, (0x1F, 0x23), 4, signed=False).tolist() == [15, 1, 3, 2]
    with pytest.raises(ValueError, match="2 beats, not 3 words"):
        unpack(vector, (0, 0, 0), 4, signed=False)
    with pytest.raises(ValueError, match="wider than its 8 bits"):
        unpack(vector, (0x100, 0), 4, signed=False)
    with pytest.raises(ValueError, match="exceed an int64"):
        unpack(vector, (0, 0), 64, signed=False)
    # A replayed element read twice with two values.
    with pytest.raises(
        ValueError, match=r"element \(1,\) is presented twice .* beat 0 lane 1, .* beat 2 lane 1"
    ):
        unpack(vector.repeated(2), (0x10, 0x00, 0x20, 0x00), 4, signed=False)
    assert unpack(vector.repeated(2), (0x10, 0x32, 0x10, 0x32), 4, signed=False).tolist() == [
        0,
        1,
        2,
        3,
    ]
    with pytest.raises(ValueError, match=r"presents no value of element \(2,\)"):
        unpack(Traversal((4,), (Loop(2, 0),), (Loop(2, 1),)), (0, 0), 4, signed=False)


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
    source = design_space(
        MemStreamKernel(
            platform=FULL_DSP48E2, dtype=DataType["INT3"], form=chunked, contents=values
        )
    )
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
    """The logical part is the channel's plan (``finn.dataflow.plan``): its steps name the
    adapter, and what no chain repairs is refused as the plan says."""
    wider = compatibility(
        contract(vector_major((4,), 2), Endpoint.INITIATOR),
        contract(vector_major((4,), 4), Endpoint.TARGET),
        source_is_top=False,
        sink_is_top=False,
    )
    assert mismatch_codes(wider) == {"channel-form"}
    assert next(iter(wider)).message == "needs a width_conversion adapter"
    found = compatibility(
        contract(tile(4, 4, 1, 4), Endpoint.INITIATOR),
        contract(tile(4, 4, 4, 1), Endpoint.TARGET),
        source_is_top=False,
        sink_is_top=False,
    )
    # Equal lanes and widths, different positions per beat: a lane regroup, which no
    # chain of adapters realizes.
    assert mismatch_codes(found) == {"channel-form"}
    message = next(iter(found)).message
    assert message.startswith("no chain of adapters repairs it") and "lane axis" in message


def test_repetition_direction_and_marker_rules_are_checked():
    form = vector_major((4,), 2)
    ok = contract(form, Endpoint.INITIATOR)
    cyclic_sink = contract(form, Endpoint.TARGET, repetition=Repetition.CYCLIC)
    assert "channel-repetition" in mismatch_codes(
        compatibility(ok, cyclic_sink, source_is_top=False, sink_is_top=False)
    )
    backwards = contract(form, Endpoint.INITIATOR)
    assert "channel-direction" in mismatch_codes(
        compatibility(ok, backwards, source_is_top=False, sink_is_top=False)
    )
    last = (StreamMarker("s_m", MarkerKind.LAST),)
    produced = StreamContract(
        native("s", 6, Endpoint.INITIATOR, markers=last),
        INT3,
        form,
        markers=(("s_m", LevelEnd(1)),),
    )
    required = StreamContract(
        native("s", 6, Endpoint.TARGET, markers=last), INT3, form, markers=(("s_m", LevelEnd(2)),)
    )
    assert "channel-marker" in mismatch_codes(
        compatibility(produced, required, source_is_top=False, sink_is_top=False)
    )
    # A marker closing every beat is a constant: met by any producer, from the producer's
    # own marker where it offers one, otherwise tied high.
    every = StreamContract(
        native("s", 6, Endpoint.TARGET, markers=last), INT3, form, markers=(("s_m", LevelEnd(1)),)
    )
    unmarked = contract(form, Endpoint.INITIATOR)
    assert compatibility(unmarked, every, source_is_top=False, sink_is_top=False) == ()
    assert marker_pairs(unmarked, every) == ((None, "s_m"),)
    assert marker_pairs(produced, every) == (("s_m", "s_m"),)


def test_a_marker_after_an_adapter_is_one_the_adapter_must_make():
    """A sequence step invalidates the markers before it (``finn.dataflow.plan``): a
    consumer's marker past a width conversion is refused as one none survives."""
    last = (StreamMarker("s_m", MarkerKind.LAST),)
    produced = StreamContract(
        native("s", 6, Endpoint.INITIATOR, markers=last),
        INT3,
        vector_major((8,), 2),
        markers=(("s_m", LevelEnd(4)),),
    )
    required = StreamContract(
        native("s", 12, Endpoint.TARGET, markers=last),
        INT3,
        vector_major((8,), 4),
        markers=(("s_m", LevelEnd(2)),),
    )
    found = compatibility(produced, required, source_is_top=False, sink_is_top=False)
    assert [(each.code, each.message) for each in found] == [
        ("channel-form", "needs a width_conversion adapter"),
        (
            "channel-marker",
            "s_m requires a marker every 2 beats; none survives the width_conversion adapter",
        ),
    ]


def test_contracts_reject_lanes_wider_than_the_word_and_unknown_marker_rules():
    with pytest.raises(ValueError, match="exceed"):
        contract(vector_major((4,), 4), Endpoint.TARGET, width=8)
    with pytest.raises(ValueError, match="marker"):
        contract(vector_major((4,), 2), Endpoint.TARGET, markers=(("missing", LevelEnd(2)),))
    # A marker closes a loop level of its form: three beats close none of two.
    last = (StreamMarker("s_m", MarkerKind.LAST),)
    with pytest.raises(ValueError, match="closes no loop level"):
        StreamContract(
            native("s", 6, Endpoint.TARGET, markers=last),
            INT3,
            vector_major((4,), 2),
            markers=(("s_m", LevelEnd(3)),),
        )


# -- the delivery kernel -----------------------------------------------------------------


def delivery(form=None, values=(1, -2, 7, -8), **choices):
    form = vector_major((4,), 2) if form is None else form
    base = design_space(
        MemStreamKernel(dtype=DataType["INT4"], form=form, contents=values, platform=FULL_DSP48E2)
    )
    return base.with_choices(**choices) if choices else base


def test_delivery_publishes_a_cyclic_contract_and_waits_only_for_its_own_choice():
    base = delivery()
    output = base.output.contract
    assert output.repetition is Repetition.CYCLIC and output.form == vector_major((4,), 2)
    assert output.payload_bits == output.transport.data_width == 8
    assert base.image == (0xE1, 0x87)
    assert isinstance(base.query(MemStreamKernel.module), Unresolved)
    requirements = delivery(ram_style="block", pumped_memory=False).module
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
        MemStreamKernel(
            dtype=DataType[dtype],
            form=vector_major((4,), 2),
            contents=values,
            platform=FULL_DSP48E2,
        )
    )
    answer = point.with_choices(ram_style="auto", pumped_memory=False).query(MemStreamKernel.module)
    assert isinstance(answer, Rejected)
    if message:
        assert any(message in finding.message for finding in answer.findings)


# -- a second consumer: cyclic channel parameters into eltwise ---------------------------

CHANNELS, PE, PIXELS = 4, 2, 3
PARAMETERS = (1, -2, 7, -8)


def eltwise_with_constant(form=None):
    """Eltwise ADD whose rhs is a memory's cyclic channel vector, joined directly."""
    form = vector_major((CHANNELS,), PE) if form is None else form
    int4, int5 = DataType["INT4"], DataType["INT5"]

    class Constant(Root):
        x = Channel(tensor=Tensor((PIXELS, CHANNELS), INT4), port="in0_V", platform=FULL_DSP48E2)
        c = Channel(tensor=Tensor((CHANNELS,), INT4), adaptable=False, platform=FULL_DSP48E2)
        y = Channel(
            tensor=Tensor((PIXELS, CHANNELS), ScalarEncoding(int5)),
            port="out0_V",
            platform=FULL_DSP48E2,
        )
        rhs = MemStreamKernel(
            platform=FULL_DSP48E2, dtype=int4, form=form, contents=PARAMETERS, output_channel=c
        )
        add = EltwiseKernel(
            operation="ADD",
            pe=PE,
            lhs_dtype=int4,
            rhs_dtype=int4,
            b_scale=1.0,
            platform=FULL_DSP58,
            lhs_channel=x,
            rhs_channel=c,
            result_channel=y,
        )

    return commit(
        with_direct_transports(design_space(Constant())),
        {"rhs.ram_style": "distributed", "rhs.pumped_memory": False},
    )


def test_the_delivery_kernel_serves_a_second_consumer_through_the_same_contract():
    module = eltwise_with_constant().module
    (into,) = [link for link in module.fragment.links if link.sink.data == "bdat"]
    # Lane for lane: the memory's channels are the ones eltwise reads.
    assert (into.source.instance, into.sink.instance) == ("rhs", "add")
    assert (into.lanes, into.lane_bits) == ((0, 1), 4)
    # A delivery whose lanes carry other positions (0,2),(1,3) needs a lane regroup.
    strided = Traversal.over((CHANNELS,), ((0, 2, 1),), ((0, 2, 2),))
    refused = eltwise_with_constant(strided).c.query(Channel.netlist)
    assert isinstance(refused, Rejected)
    (plan,) = [finding for finding in refused.findings if finding.code == "channel-plan"]
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
    source = design_space(
        MemStreamKernel(
            platform=FULL_DSP48E2, dtype=DataType["INT4"], form=produced, contents=values
        )
    )
    fifo = design_space(FifoKernel(word_bits=16, depth=2, platform=FULL_DSP48E2)).with_choices(
        ram_style="auto"
    )
    # The hop connects directly, and its link crosses the lanes: no adapter.
    sink = StreamContract(fifo.input.transport, INT4, wanted)
    assert compatibility(source.output.contract, sink, source_is_top=False, sink_is_top=False) == ()
    link = wired("source", source.output.contract, "fifo", sink)
    assert (link.source.data, link.sink.data) == ("m_axis_0_tdata", "idat")
    assert (link.lanes, link.lane_bits) == ((0, 2, 1, 3), 4)


@requires_xsim
def test_eltwise_with_cyclic_constant_computes_the_broadcast_sum(tmp_path):
    inputs = [(-8 + 3 * index) % 16 - 8 for index in range(CHANNELS * PIXELS)]
    expected = [value + PARAMETERS[index % CHANNELS] for index, value in enumerate(inputs)]
    words_in = [pack_lanes(inputs[i : i + PE], 4) for i in range(0, len(inputs), PE)]
    words_out = [pack_lanes(expected[i : i + PE], 5) for i in range(0, len(expected), PE)]
    design = eltwise_with_constant()
    stream_through(
        design.module,
        tmp_path,
        inputs={"in0_V": (words_in, 4 * PE)},
        outputs={"out0_V": (words_out, 5 * PE)},
        cycles=design.cycles,
    )
