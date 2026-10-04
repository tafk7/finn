# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Non-stream interfaces: exported control buses, tie-offs and child padding.

dotp feeds thresholding in one root; the activation stream's adapter replays
each row for dotp. dotp's padded AXIS result feeds a child: the padding bits
stay unconnected and the consumer's padding is zero. Thresholding's AXI-Lite
bus is presented through a ``ControlBus`` when its thresholds are
runtime-writable and otherwise held idle by the module, as is the set
selector of a single threshold set.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Available, Rejected, Space, design_space
from finn.kernels.artifacts.abi import Bus, Endpoint, StandardProtocol
from finn.kernels.artifacts.build import netlist
from finn.kernels.control import ControlBus
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.streams import Stream
from finn.kernels.target import DspBlock
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernels.helpers import Root, placed, with_adapter_memories
from kernels.xsim import requires_xsim, stream_through

REPETITIONS, WIDTH, HEIGHT, SIMD = 2, 4, 2, 2
FOLDS = WIDTH // SIMD
A, W, R = DataType["INT3"], DataType["INT3"], DataType["INT9"]
THRESHOLDS = (((-5, 0, 7), (-2, 3, 10)),)
X = Tensor((REPETITIONS, WIDTH), ScalarEncoding(A))
WEIGHT_TENSOR = Tensor((WIDTH, HEIGHT), ScalarEncoding(W))
RESULT_TENSOR = Tensor((REPETITIONS, HEIGHT), ScalarEncoding(R))
LEVEL_TENSOR = Tensor((REPETITIONS, HEIGHT), ScalarEncoding(DataType["UINT2"]))


class Activated(Root):
    """dotp, then thresholding: a padded child result feeding a child."""

    activations = Stream(tensor=X, port="in0_V")
    weights = Stream(tensor=WEIGHT_TENSOR, port="in1_V")
    results = Stream(tensor=RESULT_TENSOR)
    levels = Stream(tensor=LEVEL_TENSOR, port="out0_V")
    config = ControlBus(port="s_axilite")
    compute = PackedDotpKernel(
        target_dsp=DspBlock.DSP48E2,
        target_period_ns=5.0,
        result_dtype=R,
        x_stream=activations,
        w_stream=weights,
        y_stream=results,
    )
    activate = ThresholdingAxiKernel(
        input_dtype=R,
        threshold_dtype=R,
        thresholds=THRESHOLDS,
        bias=0,
        pe=1,
        ram_style="auto",
        ultra_stages=0,
        input_stream=results,
        output_stream=levels,
        control=config,
    )


def activated(*, writable: bool):
    return with_adapter_memories(
        design_space(Activated()).with_choices(
            {
                Activated.compute.pe: 1,
                Activated.compute.simd: SIMD,
                Activated.compute.compute_pumping: False,
                Activated.activate.use_axilite: writable,
                Activated.activate.deep_pipeline: False,
            }
        )
    )


def test_a_padded_child_result_feeds_a_child_and_its_padding_stays_unconnected():
    # Probe P3: one INT9 lane rides a 16-bit AXIS word into thresholding.
    module = activated(writable=False).module
    (fed,) = [link for link in module.fragment.links if link.sink.instance == "activate"]
    assert (fed.source.instance, fed.source.data_bits, fed.payload_bits) == ("compute", 16, 9)
    text = netlist(module, "top")
    assert (
        "assign n__u_activate__s_axis_tdata[8:0] = n__u_compute__m_axis_output_tdata[8:0];" in text
    )
    assert "assign n__u_activate__s_axis_tdata[15:9] = 7'h0;" in text
    assert "n__u_compute__m_axis_output_tdata[15:9]" not in text


def test_read_only_thresholds_tie_their_control_and_set_interfaces():
    # Probe P4: every thresholding input is driven, and nothing is presented.
    module = activated(writable=False).module
    assert [port.name for port in module.pins.ports] == [
        "ap_clk",
        "ap_rst_n",
        "in0_V",
        "in1_V",
        "out0_V",
    ]
    held = placed(module, "activate").held
    tied = dict(held.inputs)
    assert tied["s_axilite_AWVALID"] == tied["s_axis_set_tvalid"] == 0
    assert {"s_axilite_AWREADY", "s_axilite_RDATA", "s_axis_set_tready"} <= set(held.unused)


def test_writable_thresholds_export_their_bus_through_the_control_node():
    module = activated(writable=True).module
    (bus,) = [
        port for port in module.pins.ports if isinstance(port, Bus) and port.name == "s_axilite"
    ]
    assert bus.protocol is StandardProtocol.AXILITE and bus.endpoint is Endpoint.TARGET
    assert (bus.associated_clock, bus.associated_reset) == ("ap_clk", "ap_rst_n")
    assert "assign n__u_activate__s_axilite_AWVALID = s_axilite_AWVALID;" in netlist(module, "top")


def test_writable_thresholds_without_a_control_bus_are_refused():
    class Unexported(Activated):
        activate = ThresholdingAxiKernel(
            input_dtype=R,
            threshold_dtype=R,
            thresholds=THRESHOLDS,
            bias=0,
            pe=1,
            ram_style="auto",
            ultra_stages=0,
            input_stream=Activated.results,
            output_stream=Activated.levels,
        )

    point = design_space(Unexported()).with_choices(
        {
            Unexported.compute.compute_pumping: False,
            Unexported.activate.use_axilite: True,
            Unexported.activate.deep_pipeline: False,
        }
    )
    refused = point.activate.query(ThresholdingAxiKernel.module)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"threshold-control"}


@requires_xsim
@pytest.mark.parametrize("writable", (False, True))
def test_the_composed_module_computes_thresholded_dot_products(tmp_path, writable):
    # Writable, the exported AXI-Lite bus is held idle (every other top input is
    # held at zero): the thresholds are still the initial table.
    module = activated(writable=writable).module
    x = [[(3 * r + 5 * k) % 8 - 4 for k in range(WIDTH)] for r in range(REPETITIONS)]
    w = [[(7 * h + 3 * k) % 8 - 4 for k in range(WIDTH)] for h in range(HEIGHT)]
    levels = [
        sum(t <= sum(x[r][k] * w[h][k] for k in range(WIDTH)) for t in THRESHOLDS[0][h])
        for r in range(REPETITIONS)
        for h in range(HEIGHT)
    ]

    def word(values):
        return sum((v & 7) << (3 * i) for i, v in enumerate(values))

    # Each row once: the stream's input_gen presents it HEIGHT times, framed.
    activation_words = [
        word(x[r][f * SIMD : (f + 1) * SIMD]) for r in range(REPETITIONS) for f in range(FOLDS)
    ]
    weight_words = [
        word(w[h][f * SIMD : (f + 1) * SIMD])
        for _ in range(REPETITIONS)
        for h in range(HEIGHT)
        for f in range(FOLDS)
    ]
    stream_through(
        module,
        tmp_path,
        inputs={"in0_V": (activation_words, 3 * SIMD), "in1_V": (weight_words, 3 * SIMD)},
        outputs={"out0_V": (levels, 2)},
    )


def test_several_threshold_sets_take_a_set_selector_stream():
    # A sideband is an ordinary stream: one set index per input beat.
    selectors = Tensor((REPETITIONS * HEIGHT,), ScalarEncoding(DataType["UINT1"]))
    two_sets = (THRESHOLDS[0], ((-4, 1, 8), (-3, 2, 9)))

    class Selected(Space):
        values = Stream(tensor=RESULT_TENSOR, port="in0_V")
        sets = Stream(tensor=selectors, port="in1_V")
        levels = Stream(tensor=LEVEL_TENSOR, port="out0_V")
        activate = ThresholdingAxiKernel(
            input_dtype=R,
            threshold_dtype=R,
            thresholds=two_sets,
            bias=0,
            pe=1,
            ram_style="auto",
            ultra_stages=0,
            input_stream=values,
            output_stream=levels,
            set_stream=sets,
        )

    point = design_space(Selected()).with_choices(
        {Selected.activate.use_axilite: False, Selected.activate.deep_pipeline: False}
    )
    assert isinstance(point.sets.query(Stream.netlist), Available)
    assert point.sets.endpoints.sink.transport.name == "s_axis_set"
    # One set index for every input beat: a shorter selector stream is refused.
    short = Tensor((HEIGHT,), ScalarEncoding(DataType["UINT1"]))

    class Short(Selected):
        sets = Stream(tensor=short, port="in1_V")

    refused = design_space(Short()).sets.query(Stream.netlist)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"threshold-set-stream"}
