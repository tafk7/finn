# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Stream contracts, checked composition, and a reusable memory.

The traversals' own tests (pack, unpack, classification, the tiled MVU's and Shuffle's
stress cases) are in ``tests/dataflow/test_traversal.py``; the tiled MVU's weight chunks
stay here, where a memory delivers them.
"""

from itertools import product

import pytest
from qonnx.core.datatype import DataType
from support.helpers import (
    FULL_DSP48E2,
    FULL_DSP58,
    built,
    by_hand,
    cycles_of,
    fifo_node,
    keyed,
    port_transport,
    with_adapter_memories,
    with_direct_transports,
)
from support.tiled_mvu import MH, MW, NF, PE_T, SF, SIMD_T, R, T
from support.xsim import requires_xsim

from finn.core.rtlsim_bench import pack_lanes, stream_through
from finn.core.space import Inapplicable, Rejected, Unresolved, design_space
from finn.dataflow.steps import Hop
from finn.dataflow.steps import steps as between
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import (
    Adaptation,
    BeatSequence,
    LevelEnd,
    Reorder,
    Repetition,
    Traversal,
    classify,
    offsets,
    tile,
    vector_major,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.channels import Channel, wired
from finn.kernels.composer import Facts, Port, channel
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.input_generator import InputGeneratorKernel
from finn.kernels.laws import Advance, Standing
from finn.kernels.library import library_module
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.port_model import StreamPort
from finn.kernels.sides import Side
from finn.kernels.transport import (
    MarkerKind,
    ReadyValidTransport,
    StreamMarker,
    StreamPins,
    compatibility,
    marker_pairs,
)
from finn.kernels.values.semantics import IntegerTensorValue
from finn.kernels.vpc import VpcKernel
from finn.toolchain.launcher import host_toolchain

INT3 = ScalarEncoding(DataType["INT3"])
INT4 = ScalarEncoding(DataType["INT4"])


def native(name, bits, endpoint, clock="clk", reset="rst", markers=()):
    return ReadyValidTransport(
        name, bits, endpoint, f"{name}_d", f"{name}_v", f"{name}_r", clock, reset, markers
    )


def pins(transport, element, form, repetition=Repetition.ONCE, markers=()):
    """``transport``'s pins carrying ``element`` in ``form``, each marker on the pin keyed."""
    rules = tuple(rule for _, rule in sorted(markers))
    sequence = BeatSequence(form, repetition, rules)
    return StreamPins(transport, Stream(element, sequence), markers)


def contract(form, endpoint, *, element=INT3, repetition=Repetition.ONCE, width=None, **kw):
    bits = width or form.lanes * element.bits
    return pins(native("s", bits, endpoint), element, form, repetition, **kw)


def streamed(given: StreamPins) -> Stream:
    """The stream the pins carry: its element and its beat sequence."""
    return given.stream


def mismatch_codes(mismatches):
    return {item.code for item in mismatches}


def delivered(form: Traversal, hops: tuple[Hop, ...]) -> list[int]:
    """The flat offsets ``hops`` deliver from ``form``, by each step's own definition: a
    width step keeps the element order, regrouping only its lanes; a reorder emits, of each
    frame of ``frame_beats`` beats, the beat at ``sum(index * coefs)`` over ``dims``, outer
    first (``traversal.Reorder``). A coordinate oracle (``offsets``), independent of the
    forms the steps derive between."""
    at = [int(offset) for offset in offsets(form)]
    for hop in hops:
        if hop.reorder is not None:
            lanes, frame = hop.source.form.lanes, hop.reorder.frame_beats
            beats = [at[first : first + lanes] for first in range(0, len(at), lanes)]
            at = [
                offset
                for start in range(0, len(beats), frame)
                for index in product(*(range(extent) for extent in hop.reorder.dims))
                for offset in beats[start + sum(i * c for i, c in zip(index, hop.reorder.coefs))]
            ]
    return at


# -- the tiled MVU's weight chunks ---------------------------------------------------------


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
            platform=FULL_DSP48E2,
            dtype=DataType["INT3"],
            form=chunked,
            contents=IntegerTensorValue.of(values),
        )
    )
    sink = contract(chunked.repeated(R // T), Endpoint.TARGET)
    produced = source.output.stream_pins.pins
    assert compatibility(streamed(produced), streamed(sink)) == ()


# -- compatibility -----------------------------------------------------------------------


def test_equal_contracts_connect_and_cyclic_sources_repeat_into_consumer_passes():
    weights = tile(4, 4, 2, 2)
    source = contract(weights, Endpoint.INITIATOR, repetition=Repetition.CYCLIC)
    for sink_form in (weights, weights.repeated(5)):
        sink = contract(sink_form, Endpoint.TARGET)
        assert compatibility(streamed(source), streamed(sink)) == ()


def test_mismatches_name_the_adapter_that_would_repair_them():
    """The logical part is the steps between the channel's two streams
    (``finn.dataflow.steps``): they name the adapter, and what no chain repairs is
    refused as they say."""
    wider = compatibility(
        streamed(contract(vector_major((4,), 2), Endpoint.INITIATOR)),
        streamed(contract(vector_major((4,), 4), Endpoint.TARGET)),
    )
    assert mismatch_codes(wider) == {"channel-form"}
    assert next(iter(wider)).message == "needs a width_conversion adapter"
    rows, columns = tile(4, 4, 1, 4), tile(4, 4, 4, 1)
    found = compatibility(
        streamed(contract(rows, Endpoint.INITIATOR)),
        streamed(contract(columns, Endpoint.TARGET)),
    )
    # Equal lanes and widths, different positions per beat: a lane regroup of the same
    # positions, through one lane a beat (the only count dividing both at which the two
    # orders differ by a reorder), the adapter this direct connection lacks.
    assert [(item.code, item.message) for item in found] == [
        (
            "channel-form",
            "needs a width_conversion -> reorder -> width_conversion adapter: "
            "Reorder(frame_beats=16, dims=(4, 4), coefs=(1, 4))",
        )
    ]
    regroup = between(
        contract(rows, Endpoint.INITIATOR).sequence, contract(columns, Endpoint.TARGET).sequence
    )
    assert delivered(rows, regroup.hops) == [int(offset) for offset in offsets(columns)]
    # Rows 0 and 2 alone, the columns read: positions the source never presents, which no
    # chain of adapters repairs.
    halves = Traversal.over((4, 4), ((0, 2, 2),), ((1, 4, 1),))
    assert sorted(int(offset) for offset in offsets(halves)) == [*range(4), *range(8, 12)]
    missing = compatibility(
        streamed(contract(halves, Endpoint.INITIATOR)),
        streamed(contract(columns, Endpoint.TARGET)),
    )
    assert [(item.code, item.message) for item in missing] == [
        (
            "channel-form",
            "no chain of adapters repairs it: the sequences present different positions",
        )
    ]


def test_repetition_and_marker_rules_are_checked():
    form = vector_major((4,), 2)
    ok = contract(form, Endpoint.INITIATOR)
    cyclic_sink = contract(form, Endpoint.TARGET, repetition=Repetition.CYCLIC)
    assert "channel-repetition" in mismatch_codes(
        compatibility(streamed(ok), streamed(cyclic_sink))
    )
    last = (StreamMarker("s_m", MarkerKind.LAST),)
    produced = pins(
        native("s", 6, Endpoint.INITIATOR, markers=last),
        INT3,
        form,
        markers=(("s_m", LevelEnd(1)),),
    )
    required = pins(
        native("s", 6, Endpoint.TARGET, markers=last), INT3, form, markers=(("s_m", LevelEnd(2)),)
    )
    assert "channel-marker" in mismatch_codes(compatibility(streamed(produced), streamed(required)))
    # A marker closing every beat is a constant: met by any producer, from the producer's
    # own marker where it offers one, otherwise tied high.
    every = pins(
        native("s", 6, Endpoint.TARGET, markers=last), INT3, form, markers=(("s_m", LevelEnd(1)),)
    )
    unmarked = contract(form, Endpoint.INITIATOR)
    assert compatibility(streamed(unmarked), streamed(every)) == ()
    assert marker_pairs(unmarked, every) == ((None, "s_m"),)
    assert marker_pairs(produced, every) == (("s_m", "s_m"),)


def test_a_marker_after_an_adapter_is_one_the_adapter_must_make():
    """A sequence step invalidates the markers before it (``finn.dataflow.steps``): a
    consumer's marker past a width conversion is refused as one none survives."""
    last = (StreamMarker("s_m", MarkerKind.LAST),)
    produced = pins(
        native("s", 6, Endpoint.INITIATOR, markers=last),
        INT3,
        vector_major((8,), 2),
        markers=(("s_m", LevelEnd(4)),),
    )
    required = pins(
        native("s", 12, Endpoint.TARGET, markers=last),
        INT3,
        vector_major((8,), 4),
        markers=(("s_m", LevelEnd(2)),),
    )
    found = compatibility(streamed(produced), streamed(required))
    assert [(each.code, each.message) for each in found] == [
        ("channel-form", "needs a width_conversion adapter"),
        (
            "channel-marker",
            "the sink requires a marker every 2 beats; none survives the width_conversion adapter",
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
        pins(
            native("s", 6, Endpoint.TARGET, markers=last),
            INT3,
            vector_major((4,), 2),
            markers=(("s_m", LevelEnd(3)),),
        )


# -- the delivery kernel -----------------------------------------------------------------


def delivery(form=None, values=(1, -2, 7, -8), **choices):
    form = vector_major((4,), 2) if form is None else form
    base = design_space(
        MemStreamKernel(
            dtype=DataType["INT4"],
            form=form,
            contents=IntegerTensorValue.of(values),
            platform=FULL_DSP48E2,
        )
    )
    return base.with_choices(**choices) if choices else base


def test_delivery_publishes_a_cyclic_contract_and_waits_only_for_its_own_choice():
    base = delivery()
    output = base.output.stream_pins.pins
    assert output.sequence.repetition is Repetition.CYCLIC and output.form == vector_major((4,), 2)
    assert output.payload_bits == output.transport.data_width == 8
    assert base.image == (0xE1, 0x87)
    assert isinstance(base.query(MemStreamKernel.netlist), Unresolved)
    requirements = library_module(delivery(ram_style="block", pumped_memory=False))
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
            contents=IntegerTensorValue.of(values),
            platform=FULL_DSP48E2,
        )
    )
    answer = point.with_choices(ram_style="auto", pumped_memory=False).query(
        MemStreamKernel.netlist
    )
    assert isinstance(answer, Rejected)
    if message:
        assert any(message in finding.message for finding in answer.findings)


# -- a second consumer: cyclic channel parameters into eltwise ---------------------------

CHANNELS, PE, PIXELS = 4, 2, 3
PARAMETERS = (1, -2, 7, -8)
STRIDED = Traversal.over((CHANNELS,), ((0, 2, 1),), ((0, 2, 2),))
"""Two lanes a beat carrying positions (0, 2), then (1, 3): eltwise's under another lane axis."""
DOUBLED = Traversal.over((CHANNELS,), ((0, 2, 1),), ((0, 2, 1),))
"""Two lanes a beat carrying positions (0, 1), then (1, 2): position 3 is never presented."""


def eltwise_with_constant(form=None):
    """Eltwise ADD whose rhs is a memory's cyclic channel vector, joined directly."""
    return with_direct_transports(constant_network(form))


def constant_network(form=None, *, adaptable=False):
    """``eltwise_with_constant``'s partition, its transports open, its constant's channel
    admitting an adapter where ``adaptable``; its kernels' choices stored as it is bound (a
    stored set that breaks a channel's rule binds, and the channel reads refused)."""
    form = vector_major((CHANNELS,), PE) if form is None else form
    int4, int5 = DataType["INT4"], DataType["INT5"]

    x_tensor = Tensor((PIXELS, CHANNELS), INT4)
    c_tensor = Tensor((CHANNELS,), INT4)
    y_tensor = Tensor((PIXELS, CHANNELS), ScalarEncoding(int5))

    return by_hand(
        {
            "x": channel(
                Side.boundary("in0_V"), Port("add", "lhs"), tensor=x_tensor, platform=FULL_DSP48E2
            ),
            "c": channel(
                Port("rhs", "output"),
                Port("add", "rhs"),
                tensor=c_tensor,
                adaptable=adaptable,
                platform=FULL_DSP48E2,
            ),
            "y": channel(
                Port("add", "result"),
                Side.boundary("out0_V"),
                tensor=y_tensor,
                platform=FULL_DSP48E2,
            ),
            "rhs": Facts(
                MemStreamKernel,
                {
                    "platform": FULL_DSP48E2,
                    "dtype": int4,
                    "form": form,
                    "contents": IntegerTensorValue.of(PARAMETERS),
                },
            ),
            "add": Facts(
                EltwiseKernel,
                {
                    "operation": "ADD",
                    "b_scale": 1.0,
                    "platform": FULL_DSP58,
                    "lhs_tensor": x_tensor,
                    "rhs_tensor": c_tensor,
                },
            ),
        },
        keyed({"add.pe": PE, "rhs.ram_style": "distributed", "rhs.pumped_memory": False}),
    )


def test_the_delivery_kernel_serves_a_second_consumer_through_the_same_contract():
    module = built(eltwise_with_constant(), "constant")
    (into,) = [link for link in module.fragment.links if link.sink.data == "bdat"]
    # Lane for lane: the memory's channels are the ones eltwise reads.
    assert (into.source.instance, into.sink.instance) == ("rhs", "add")
    assert (into.lanes, into.lane_bits) == ((0, 1), 4)
    # A delivery whose lanes carry other positions (0,2),(1,3) needs a lane regroup, a
    # width step, a reorder and a width step, on a channel that admits no adapter: the
    # channel is refused with its transport open, and joining it directly is refused by
    # the rule it breaks, naming the same chain.
    strided = constant_network(STRIDED)
    refused = strided.member("c").query(Channel.netlist)
    assert isinstance(refused, Rejected)
    (refusal,) = [finding for finding in refused.findings if finding.code == "channel-steps"]
    assert refusal.message == (
        "the streams need width_conversion -> reorder -> width_conversion, "
        "and this channel admits no adapter"
    )
    trial = strided.trial(keyed({"c.transport": "direct"}))
    assert not trial.accepted
    ((rule, (form,)),) = trial.refused.items()
    assert (rule, form.code) == (("c", "compatible"), "channel-form")
    assert form.message == (
        "channel-form: needs a width_conversion -> reorder -> width_conversion adapter: "
        "Reorder(frame_beats=4, dims=(2, 2), coefs=(1, 2))"
    )


def test_a_lane_regroup_on_an_adaptable_channel_is_built_from_existing_stages():
    # The same lane regroup on a channel that admits adapters: its steps are the width step
    # before the transport (a vpc) and the reorder and width step after it (an input_gen,
    # then a vpc), delivering what eltwise reads, position for position, each pixel's pass.
    # Only physical choices are open (its transport, its adapter's memory); the first of
    # each builds them, with each stage's parameters the steps'.
    strided = constant_network(STRIDED, adaptable=True)
    validity = strided.validity("c")
    assert validity.status == "open"
    assert {(f.code, f.owner) for piece in validity.pieces for f in piece.result.findings} == {
        ("decision-unassigned", "transport"),
        ("decision-unassigned", "adapter_ram_style"),
    }
    found = strided.member("c")
    hops = found.steps.hops
    assert found.steps.describe() == "width_conversion -> reorder -> width_conversion"
    assert hops[0].source.form == STRIDED.repeated(PIXELS)
    assert hops[-1].sink.form == vector_major((CHANNELS,), PE).repeated(PIXELS)
    assert [hop.reorder for hop in hops] == [None, Reorder(4, (2, 2), (1, 2)), None]
    assert delivered(hops[0].source.form, hops) == [
        int(offset) for offset in offsets(vector_major((CHANNELS,), PE).repeated(PIXELS))
    ]
    assert found.output_adapting and found.input_chain == "input_gen_vpc"
    assert [stage.kind for stage in found.output_adapter.realization] == ["vpc"]
    assert [stage.kind for stage in found.adapter.realization] == ["input_gen", "vpc"]
    defaults = with_adapter_memories(with_direct_transports(strided))
    assert defaults.validity("c").status == "valid"
    module = built(defaults, "constant")
    leaves = {
        name: (leaf.implementation_id, dict(leaf.parameters))
        for name, leaf in module.fragment.instances
        if name.startswith("c.")
    }
    assert leaves == {
        "c.output_adapter.vpc.vpc": (
            "finnlib.vpc",
            {"N": 2, "PAD_ZEROS": 1, "PI": 2, "PO": 1, "RELAX_THROUGHPUT": 0, "W": 4},
        ),
        "c.adapter.input_gen_vpc.input_gen": (
            "finnlib.input_generator",
            {
                "COEFS": "'{1, 2}",
                "D": 2,
                "DATA_WIDTH": 4,
                "DIMS": "'{2, 2}",
                "FM_SIZE": 4,
                "RAM_STYLE": '"distributed"',
            },
        ),
        "c.adapter.input_gen_vpc.vpc": (
            "finnlib.vpc",
            {"N": 2, "PAD_ZEROS": 1, "PI": 1, "PO": 2, "RELAX_THROUGHPUT": 0, "W": 4},
        ),
    }
    path = [
        (link.source.instance, link.sink.instance, link.sink.data)
        for link in module.fragment.links
        if "c." in f"{link.source.instance}{link.sink.instance}"
    ]
    assert path == [
        ("rhs", "c.output_adapter.vpc.vpc", "idat"),
        ("c.output_adapter.vpc.vpc", "c.adapter.input_gen_vpc.input_gen", "idat"),
        ("c.adapter.input_gen_vpc.input_gen", "c.adapter.input_gen_vpc.vpc", "idat"),
        ("c.adapter.input_gen_vpc.vpc", "add", "bdat"),
    ]


def test_a_lane_regroup_s_stages_are_library_kernels_that_keep_their_own_contracts():
    # The regroup adds no kernel and no rule: its three stages are the library's vpc and
    # input generator, each port buffered (CONTRACT.md § 3, L3), none with a pace, and the
    # channel lists each stage's intake below the stage, structural from its RTL (§ 5).
    point = with_adapter_memories(with_direct_transports(constant_network(STRIDED, adaptable=True)))
    found = point.member("c")
    stages = (found.output_adapter.vpc, found.adapter.input_gen, found.adapter.vpc)
    assert [type(stage) for stage in stages] == [VpcKernel, InputGeneratorKernel, VpcKernel]
    for stage in stages:
        for port in (stage.input, stage.output):
            assert port.advance is Advance.BUFFERED
            assert isinstance(port.query(StreamPort.pace), Inapplicable)
    intake = found.intake
    assert intake.names == (
        "output_adapter.vpc.input",
        "adapter.input_gen_vpc.input_gen.input",
        "adapter.input_gen_vpc.vpc.input",
    )
    assert [(entry.before_stop, entry.most) for entry in intake.ports] == [(1, 2), (7, 8), (3, 3)]
    assert {entry.standing for entry in intake.ports} == {Standing.STRUCTURAL}
    assert [entry.source.split(":")[0] for entry in intake.ports] == [
        "rtl/shape/vpc.sv",
        "rtl/shape/input_gen.sv",
        "rtl/shape/vpc.sv",
    ]


def test_steps_that_refuse_refuse_the_input_adapter_once():
    # A delivery presenting position 1 twice and position 3 never, on a channel that admits
    # adapters: no chain presents a position it is not given, so its steps refuse
    # (channel-steps), and the picker of its input adapter refuses with them, one piece
    # keyed by the selector, as the guard of the chosen selector it replaces did; its
    # adapter's memory, which the picker guards, one guard piece; and that memory's
    # UltraRAM rule, reading the memory first, refused through that read with no guard
    # piece of its own.
    assert sorted(int(offset) for offset in offsets(DOUBLED)) == [0, 1, 1, 2]
    doubled = constant_network(DOUBLED, adaptable=True)
    (refusal,) = doubled.member("c").query(Channel.steps).findings
    assert (refusal.code, refusal.message) == (
        "channel-steps",
        "no adapter can join the streams: the sequences present different positions",
    )
    validity = doubled.validity("c")
    assert validity.status == "refused"
    pieces = {
        piece.owner: {(f.code, f.owner) for f in piece.result.findings} for piece in validity.pieces
    }
    steps = ("channel-steps", "steps")
    assert pieces == {
        "realizable": {steps},
        "compatible": {steps},
        **{
            guarded: {steps, ("guard-refused", guarded)}
            for guarded in ("adapter", "output_adapter", "adapter_ram_style")
        },
        "adapter_uram_present": {steps},
    }


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
            platform=FULL_DSP48E2,
            dtype=DataType["INT4"],
            form=produced,
            contents=IntegerTensorValue.of(values),
        )
    )
    fifo = design_space(fifo_node(word_bits=16, depth=2)).with_choices(ram_style="auto")
    # The hop connects directly, and its link crosses the lanes: no adapter.
    sink = pins(port_transport(fifo, "input"), INT4, wanted)
    assert compatibility(streamed(source.output.stream_pins.pins), streamed(sink)) == ()
    link = wired("source", source.output.stream_pins.pins, "fifo", sink)
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
        built(design, "constant"),
        tmp_path,
        inputs={"in0_V": (words_in, 4 * PE)},
        outputs={"out0_V": (words_out, 5 * PE)},
        cycles=cycles_of(design).value,
        toolchain=host_toolchain(),
    )
