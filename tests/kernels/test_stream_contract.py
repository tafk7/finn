# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Stream contracts, checked composition, and a reusable memory.

The traversals' own tests (pack, unpack, classification, the tiled MVU's and Shuffle's
stress cases) are in ``tests/dataflow/test_traversal.py``; the tiled MVU's weight chunks
stay here, where a memory delivers them.
"""

import pytest
from qonnx.core.datatype import DataType
from support.helpers import (
    FULL_DSP48E2,
    FULL_DSP58,
    built,
    choose,
    cycles_of,
    fifo_node,
    keyed,
    port_transport,
    with_direct_transports,
)
from support.tiled_mvu import MH, MW, NF, PE_T, SF, SIMD_T, R, T
from support.xsim import requires_xsim

from finn.core.rtlsim_bench import pack_lanes, stream_through
from finn.core.space import Rejected, Unresolved, design_space, inspection
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import (
    Adaptation,
    LevelEnd,
    Repetition,
    Traversal,
    classify,
    tile,
    vector_major,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.channels import Channel, wired
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.library import library_module
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.network import Interface, Network, wire
from finn.kernels.transport import (
    MarkerKind,
    ReadyValidTransport,
    StreamContract,
    StreamMarker,
    compatibility,
    marker_pairs,
)
from finn.toolchain.launcher import host_toolchain

INT3 = ScalarEncoding(DataType["INT3"])
INT4 = ScalarEncoding(DataType["INT4"])


def native(name, bits, endpoint, clock="clk", reset="rst", markers=()):
    return ReadyValidTransport(
        name, bits, endpoint, f"{name}_d", f"{name}_v", f"{name}_r", clock, reset, markers
    )


def contract(form, endpoint, *, element=INT3, repetition=Repetition.ONCE, width=None, **kw):
    bits = width or form.lanes * element.bits
    return StreamContract(native("s", bits, endpoint), element, form, repetition, **kw)


def streamed(given: StreamContract) -> Stream:
    """The stream a contract states: its element and its beat sequence."""
    return Stream(given.element, given.sequence)


def mismatch_codes(mismatches):
    return {item.code for item in mismatches}


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
            platform=FULL_DSP48E2, dtype=DataType["INT3"], form=chunked, contents=values
        )
    )
    sink = contract(chunked.repeated(R // T), Endpoint.TARGET)
    produced = source.output_contract.contract
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
    found = compatibility(
        streamed(contract(tile(4, 4, 1, 4), Endpoint.INITIATOR)),
        streamed(contract(tile(4, 4, 4, 1), Endpoint.TARGET)),
    )
    # Equal lanes and widths, different positions per beat: a lane regroup, which no
    # chain of adapters realizes.
    assert mismatch_codes(found) == {"channel-form"}
    message = next(iter(found)).message
    assert message.startswith("no chain of adapters repairs it") and "lane axis" in message


def test_repetition_and_marker_rules_are_checked():
    form = vector_major((4,), 2)
    ok = contract(form, Endpoint.INITIATOR)
    cyclic_sink = contract(form, Endpoint.TARGET, repetition=Repetition.CYCLIC)
    assert "channel-repetition" in mismatch_codes(
        compatibility(streamed(ok), streamed(cyclic_sink))
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
    assert "channel-marker" in mismatch_codes(compatibility(streamed(produced), streamed(required)))
    # A marker closing every beat is a constant: met by any producer, from the producer's
    # own marker where it offers one, otherwise tied high.
    every = StreamContract(
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
    output = base.output_contract.contract
    assert output.repetition is Repetition.CYCLIC and output.form == vector_major((4,), 2)
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
            contents=values,
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


def eltwise_with_constant(form=None):
    """Eltwise ADD whose rhs is a memory's cyclic channel vector, joined directly."""
    return with_direct_transports(constant_network(form))


def constant_network(form=None, *, adaptable=False):
    """``eltwise_with_constant``'s network, its transports open, its constant's channel
    admitting an adapter where ``adaptable``."""
    form = vector_major((CHANNELS,), PE) if form is None else form
    int4, int5 = DataType["INT4"], DataType["INT5"]

    x_tensor = Tensor((PIXELS, CHANNELS), INT4)
    c_tensor = Tensor((CHANNELS,), INT4)
    y_tensor = Tensor((PIXELS, CHANNELS), ScalarEncoding(int5))

    network = Network(
        {
            "x": wire(
                tensor=x_tensor,
                port="in0_V",
                consumer=Interface("add", "lhs_contract"),
                platform=FULL_DSP48E2,
            ),
            "c": wire(
                tensor=c_tensor,
                adaptable=adaptable,
                producer=Interface("rhs", "output_contract"),
                consumer=Interface("add", "rhs_contract"),
                platform=FULL_DSP48E2,
            ),
            "y": wire(
                tensor=y_tensor,
                port="out0_V",
                producer=Interface("add", "result_contract"),
                platform=FULL_DSP48E2,
            ),
            "rhs": MemStreamKernel(
                platform=FULL_DSP48E2,
                dtype=int4,
                form=form,
                contents=PARAMETERS,
            ),
            "add": EltwiseKernel(
                operation="ADD",
                b_scale=1.0,
                platform=FULL_DSP58,
                lhs_tensor=x_tensor,
                rhs_tensor=c_tensor,
            ),
        }
    )
    return choose(
        network, {"add.pe": PE, "rhs.ram_style": "distributed", "rhs.pumped_memory": False}
    )


def test_the_delivery_kernel_serves_a_second_consumer_through_the_same_contract():
    module = built(eltwise_with_constant(), "constant")
    (into,) = [link for link in module.fragment.links if link.sink.data == "bdat"]
    # Lane for lane: the memory's channels are the ones eltwise reads.
    assert (into.source.instance, into.sink.instance) == ("rhs", "add")
    assert (into.lanes, into.lane_bits) == ((0, 1), 4)
    # A delivery whose lanes carry other positions (0,2),(1,3) needs a lane regroup: the
    # channel is refused with its transport open, and joining it directly is refused by
    # the rule it breaks, for the same reason.
    strided = constant_network(Traversal.over((CHANNELS,), ((0, 2, 1),), ((0, 2, 2),)))
    refused = strided.member("c").query(Channel.netlist)
    assert isinstance(refused, Rejected)
    (refusal,) = [finding for finding in refused.findings if finding.code == "channel-plan"]
    assert "another lane axis" in refusal.message
    trial = strided.trial(keyed({"c.transport": "direct"}))
    assert not trial.accepted
    ((rule, (form,)),) = trial.refused.items()
    assert (rule, form.code) == (("c", "compatible"), "channel-form")
    assert "another lane axis" in form.message


def test_steps_that_refuse_refuse_the_input_adapter_once():
    # The same lane regroup on a channel that admits adapters: its steps refuse
    # (channel-plan), so the picker of its input adapter refuses with them, one piece keyed
    # by the selector, as the guard of the chosen selector it replaces did; and its
    # adapter's memory and that memory's UltraRAM rule, which the picker guards, one
    # guard piece each.
    strided = constant_network(
        Traversal.over((CHANNELS,), ((0, 2, 1),), ((0, 2, 2),)), adaptable=True
    )
    validity = inspection.validity(strided.member("c"))
    assert validity.status == "refused"
    pieces = {
        piece.owner: {(f.code, f.owner) for f in piece.result.findings} for piece in validity.pieces
    }
    steps = ("channel-plan", "steps")
    assert pieces == {
        "realizable": {steps},
        "compatible": {steps},
        **{
            guarded: {steps, ("guard-refused", guarded)}
            for guarded in (
                "adapter",
                "output_adapter",
                "adapter_ram_style",
                "adapter_uram_present",
            )
        },
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
            platform=FULL_DSP48E2, dtype=DataType["INT4"], form=produced, contents=values
        )
    )
    fifo = design_space(fifo_node(word_bits=16, depth=2)).with_choices(ram_style="auto")
    # The hop connects directly, and its link crosses the lanes: no adapter.
    sink = StreamContract(port_transport(fifo, "input"), INT4, wanted)
    assert compatibility(streamed(source.output_contract.contract), streamed(sink)) == ()
    link = wired("source", source.output_contract.contract, "fifo", sink)
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
