# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One MatMulKernel space for dense and depthwise forms.

The form is a fact of the operation; what follows from it is derived: whether
activations are broadcast, whether rows are replayed, the activation traversal.
A depthwise operation is realized natively (only the INT8 DSP58 core reads one
channel per lane) or densely, with block-diagonal weights on any core; the
dense realization needs the weights, so a weight memory. Weights are
stored (k, n): window by channel.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Available, Rejected, Unresolved
from finn.dataflow.plan import Step
from finn.kernels.base import Kernel
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel
from kernels.helpers import labels, matmul_point, matmul_root, placed, with_adapter_memories
from finn.dataflow.gemm import Form
from kernels.helpers import WeightDelivery, matmul_assembly
from finn.dataflow.traversal import Traversal
from finn.kernels.target import DspBlock

FACTS = dict(
    m=2,
    k=9,  # the window
    n=4,  # the channels
    form=Form.DEPTHWISE,
    activation_dtype=DataType["INT4"],
    weights_dtype=DataType["INT4"],
    target_dsp=DspBlock.DSP58,
)


def point(core="int8_dsp58", **facts):
    facts = {**FACTS, "target_period_ns": 5.0, **facts}
    choices = {
        "matmul.memory": "none",
        "w.transport": "direct",
        "matmul.compute": core,
        f"matmul.compute.{core}.pe": 2,
        f"matmul.compute.{core}.simd": 3,
        f"matmul.compute.{core}.compute_pumping": False,
    }
    native = "native" if facts["form"] is Form.DEPTHWISE else None
    return with_adapter_memories(commit(matmul_point(realization=native, **facts), choices))


def parameters(module, label):
    return dict(placed(module, label).parameters)


def test_depthwise_rows_pass_once_with_a_frame_per_window():
    configured = point()
    # Nothing is replayed: the stream only closes each window's frame.
    assert configured.x.plan.steps == (Step.MARKERS,)
    boundary = configured.x.endpoints.source
    # Rows, then channel folds, then window folds; lane s * PE + p is window
    # position s of channel p.
    channel_tile = Traversal.over(
        (2, 9, 4), ((0, 2, 1), (2, 2, 2), (1, 3, 3)), ((1, 3, 1), (2, 2, 1))
    )
    assert boundary.transport.name == "in0_V" and boundary.form == channel_tile
    framed = configured.x.endpoints.sink
    assert framed.form == boundary.form
    assert [level.beats for level in framed.rules.values()] == [3]
    module = configured.module
    compute = parameters(module, "matmul.compute.int8_dsp58")
    # An input_gen passing three-beat frames in order, closing each.
    markers = parameters(module, "x.adapter.input_gen.input_gen")
    assert (markers["FM_SIZE"], markers["DIMS"], markers["COEFS"]) == (3, "'{3}", "'{1}")
    assert compute["ACTIVATION_BROADCASTING"] == 0
    assert compute["CORE"] == '"dotp_8sx9_dsp58"'
    widths = {
        port.name: next(item.width for item in port.signals if item.logical == "tdata")
        for port in module.pins.ports
        if port.name.endswith("_V")
    }
    # PE channels of SIMD window positions per activation beat; the result is exact.
    assert widths == {"in0_V": 24, "in1_V": 24, "out0_V": 24}
    (framing,) = [
        link.markers
        for link in module.fragment.links
        if link.sink.instance == "matmul.compute.int8_dsp58" and link.markers
    ]
    assert framing == (("olst", 0, "s_axis_input_tlast", None),)


def test_a_dense_form_replays_each_row_per_output_fold():
    dense = point(form=Form.DENSE)
    assert dense.x.plan.steps == (Step.REORDER, Step.MARKERS)
    compute = parameters(dense.module, "matmul.compute.int8_dsp58")
    assert compute["ACTIVATION_BROADCASTING"] == 1


def test_only_the_int8_dsp58_core_reads_a_depthwise_form():
    packed = point("packed").query(Kernel.module)
    assert isinstance(packed, Rejected)
    assert "dotp-form" in {finding.code for finding in packed.findings}
    facts = {**FACTS, "pe": 2, "simd": 3}
    # External weights exclude the dense realization; natively, one core fits.
    assert matmul_assembly(**facts).module is not None
    with pytest.raises(ValueError, match="realizations compatible with this configuration: none"):
        matmul_assembly(**{**facts, "target_dsp": DspBlock.DSP48E2})


def test_depthwise_cyclic_weights_are_the_channel_tile():
    weights = [[(c + k) % 16 - 8 for c in range(4)] for k in range(9)]
    built = matmul_assembly(
        **FACTS,
        pe=2,
        simd=3,
        realization="native",
        weight_delivery=WeightDelivery.MEMSTREAM,
        weights=weights,
    )
    assert "in1_V" not in {port.name for port in built.module.pins.ports}
    # Channel folds, then window folds; PE channels of SIMD taps a beat, SIMD fastest.
    assert len(built.initializer) == 2 * 3
    assert (built.activation_beats, built.weight_beats, built.result_beats) == (12, 12, 4)


def test_one_root_carries_the_weights_of_whichever_realization_is_committed():
    # The root binds its weight stream's tensor to MatMul's weight_tensor, which
    # follows the realization: open, the tensor waits on it.
    weights = tuple(tuple((c + k) % 7 - 3 for c in range(4)) for k in range(9))
    point = commit(
        matmul_point(**FACTS, target_period_ns=5.0, weights=weights), {"matmul.memory": "memstream"}
    )
    pending = point.query(matmul_root(MatMulKernel).w.tensor)
    assert isinstance(pending, Unresolved)
    assert {finding.owner for finding in pending.findings} == {"matmul.realization"}
    for realization, shape, core in (
        ("native", (9, 4), "int8_dsp58"),
        ("dense", (36, 4), "packed"),
    ):
        committed = commit(point, {"matmul.realization": realization})
        assert committed.w.tensor.shape == shape
        built = with_adapter_memories(
            commit(
                committed,
                {
                    "w.transport": "direct",
                    "matmul.compute": core,
                    f"matmul.compute.{core}.pe": 2,
                    f"matmul.compute.{core}.simd": 3,
                    f"matmul.compute.{core}.compute_pumping": False,
                    "matmul.memory.memstream.ram_style": "auto",
                    "matmul.memory.memstream.pumped_memory": False,
                },
            )
        ).query(Kernel.module)
        assert isinstance(built, Available), built
        assert f"matmul.compute.{core}" in labels(built.value)


# By channel, then stored (k, n): window by channel.
BY_CHANNEL = ((1, -2, 3, -4), (2, 3, -1, 0), (-3, 1, 2, 1))
WEIGHTS = tuple(zip(*BY_CHANNEL))
DENSE = dict(
    m=2,
    k=4,
    n=3,
    form=Form.DEPTHWISE,
    activation_dtype=DataType["INT4"],
    weights_dtype=DataType["INT4"],
    pe=3,
    simd=4,
    weight_delivery=WeightDelivery.MEMSTREAM,
    weights=WEIGHTS,
)


def test_a_dense_realization_reads_window_by_channel_rows_against_block_diagonal_weights():
    # On DSP48E2 only the dense realization computes it: the packed core.
    built = matmul_assembly(target_dsp=DspBlock.DSP48E2, **DENSE)
    assert labels(built.module) == [
        "x.adapter.input_gen.input_gen",
        "matmul.compute.packed",
        "matmul.memory.memstream",
    ]
    compute = parameters(built.module, "matmul.compute.packed")
    assert compute["ACTIVATION_BROADCASTING"] == 1 and compute["SIMD"] == 4
    # Rows of 4 x 3 = 12 activations in three SIMD beats, replayed once (PE = C).
    assert (built.activation_beats, built.weight_beats, built.result_beats) == (6, 6, 2)
    # W'[k * 3 + c', c] = W[k, c] where c' = c: each weight beat is one tile of W'.
    blocks = [
        [BY_CHANNEL[c][k] if other == c else 0 for k in range(4) for other in range(3)]
        for c in range(3)
    ]
    mask = 0xF
    expected = tuple(
        sum((blocks[p][start + s] & mask) << (4 * (p * 4 + s)) for p in range(3) for s in range(4))
        for start in range(0, 12, 4)
    )
    assert built.initializer == expected
    # The result precision is the operation's: a window of 4, not 12.
    assert built.result_dtype == DataType["INT10"]


def test_the_dense_realization_needs_known_weights_and_either_may_be_chosen_on_dsp58():
    external = {**DENSE, "weight_delivery": WeightDelivery.EXTERNAL, "weights": None}
    with pytest.raises(ValueError, match="matmul-realization"):
        matmul_assembly(target_dsp=DspBlock.DSP48E2, realization="dense", **external)
    with pytest.raises(ValueError, match="native, dense"):
        matmul_assembly(target_dsp=DspBlock.DSP58, **DENSE)
    for realization, core in (("native", "int8_dsp58"), ("dense", "packed")):
        built = matmul_assembly(
            target_dsp=DspBlock.DSP58, realization=realization, core=core, **DENSE
        )
        assert f"matmul.compute.{core}" in labels(built.module)


@pytest.mark.parametrize(
    "weights,delivery,narrow",
    [
        (WEIGHTS, WeightDelivery.MEMSTREAM, 1),  # no weight is INT4's -8
        (((-8, 0, 0), (0,) * 3, (0,) * 3, (0,) * 3), WeightDelivery.MEMSTREAM, 0),
        (None, WeightDelivery.EXTERNAL, 0),  # weights at run time promise nothing
    ],
)
def test_narrow_weights_follow_known_weights(weights, delivery, narrow):
    facts = {**DENSE, "form": Form.DENSE, "n": 3, "pe": 3}
    built = matmul_assembly(
        target_dsp=DspBlock.DSP48E2, **{**facts, "weights": weights, "weight_delivery": delivery}
    )
    assert parameters(built.module, "matmul.compute.packed")["NARROW_WEIGHTS"] == narrow
