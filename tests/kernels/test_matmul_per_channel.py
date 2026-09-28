# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One MatMulKernel space for dense and per-channel (depthwise) contractions.

The contraction is a fact of the operation; what follows from it is derived:
whether activations are broadcast, whether rows are replayed, the activation
traversal. A per-channel operation is realized natively (only the INT8 DSP58
core reads one channel per lane) or densely, with block-diagonal weights on
any core; the dense realization needs the weights, so not external delivery.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Rejected, design_space
from finn.kernels.configure import commit
from finn.kernels.matmul import Contraction, MatMulKernel, WeightDelivery, matmul_assembly
from finn.kernels.physical.forms import channel_tile
from finn.kernels.physical.structure import PhysicalPin, PinSlice
from finn.kernels.target import DspBlock

FACTS = dict(
    rows=2,
    reduction=9,  # the window
    outputs=4,  # the channels
    contraction=Contraction.PER_CHANNEL,
    activation_dtype=DataType["INT4"],
    weights_dtype=DataType["INT4"],
    target_dsp=DspBlock.DSP58,
)
CHOICES = {
    "pe": 2,
    "simd": 3,
    "delivery": "external",
    "weight_stream.transport": "direct",
    "compute": "int8_dsp58",
    "compute_pumping": False,
}


def point(**facts):
    facts = {**FACTS, "target_period_ns": 5.0, **facts}
    choices = dict(CHOICES)
    if facts["contraction"] is Contraction.PER_CHANNEL:
        choices["realization"] = "native"
    return commit(design_space(MatMulKernel(**facts)), choices)


def test_per_channel_rows_pass_once_with_a_frame_per_window():
    configured = point()
    assert configured.reuse == 1
    assert configured.activations.spec.form == channel_tile(2, 9, 4, 2, 3)
    assert configured.replayed.spec.form == configured.activations.spec.form
    assert [every.period for every in configured.replayed.spec.markers] == [3]
    structure = configured.structure.structure
    replay, compute = (dict(item.requirements.parameters) for item in structure.instances)
    assert replay == {"LEN": 3, "REP": 1, "W": 24}
    assert compute["ACTIVATION_BROADCASTING"] == 0
    assert compute["CORE"] == '"dotp_8sx9_dsp58"'
    widths = {
        port.name: next(item.width for item in port.signals if item.logical == "tdata")
        for port in structure.top_abi.ports
        if port.name.endswith("_V")
    }
    # PE channels of SIMD window positions per activation beat; the result is exact.
    assert widths == {"in0_V": 24, "in1_V": 24, "out0_V": 24}
    # One repetition: the replay's sequence and final-repetition markers coincide.
    (last,) = [
        wire.source
        for wire in structure.wires
        if wire.destination.pin == PhysicalPin("u_compute_int8_dsp58", "s_axis_input_tlast")
    ]
    assert isinstance(last, PinSlice) and last.pin.instance_id == "u_replay"
    assert last.pin.signal_id in {"olast", "ofin"}


def test_a_dense_contraction_replays_each_row_per_output_fold():
    dense = point(contraction=Contraction.DENSE)
    assert dense.reuse == 2
    compute = dict(dense.structure.structure.instances[1].requirements.parameters)
    assert compute["ACTIVATION_BROADCASTING"] == 1


def test_only_the_int8_dsp58_core_reads_a_per_channel_contraction():
    packed = point().with_choices(compute="packed").query(MatMulKernel.structure)
    assert isinstance(packed, Rejected)
    assert "dotp-contraction" in {finding.code for finding in packed.findings}
    facts = {**FACTS, "pe": 2, "simd": 3}
    # External weights exclude the dense realization; natively, one core fits.
    assert matmul_assembly(**facts).requirements is not None
    with pytest.raises(ValueError, match="realizations compatible with this configuration: none"):
        matmul_assembly(**{**facts, "target_dsp": DspBlock.DSP48E2})


def test_per_channel_cyclic_weights_are_the_channel_tile():
    weights = [[(c + k) % 16 - 8 for k in range(9)] for c in range(4)]
    built = matmul_assembly(
        **FACTS,
        pe=2,
        simd=3,
        realization="native",
        weight_delivery=WeightDelivery.CYCLIC,
        weights=weights,
    )
    assert "in1_V" not in {port.name for port in built.structure.top_abi.ports}
    # Channel folds, then window folds; PE channels of SIMD taps a beat, SIMD fastest.
    assert len(built.initializer) == 2 * 3
    assert (built.activation_beats, built.weight_beats, built.result_beats) == (12, 12, 4)


WEIGHTS = ((1, -2, 3, -4), (2, 3, -1, 0), (-3, 1, 2, 1))
DENSE = dict(
    rows=2,
    reduction=4,
    outputs=3,
    contraction=Contraction.PER_CHANNEL,
    activation_dtype=DataType["INT4"],
    weights_dtype=DataType["INT4"],
    pe=3,
    simd=4,
    weight_delivery=WeightDelivery.CYCLIC,
    weights=WEIGHTS,
)


def test_a_dense_realization_reads_window_by_channel_rows_against_block_diagonal_weights():
    # On DSP48E2 only the dense realization computes it: the packed core.
    built = matmul_assembly(target_dsp=DspBlock.DSP48E2, **DENSE)
    assert [item.instance_id for item in built.structure.instances] == [
        "u_replay",
        "u_compute_packed",
        "u_delivery_cyclic",
    ]
    compute = dict(built.structure.instances[1].requirements.parameters)
    assert compute["ACTIVATION_BROADCASTING"] == 1 and compute["SIMD"] == 4
    # Rows of 4 x 3 = 12 activations in three SIMD beats, replayed once (PE = C).
    assert (built.activation_beats, built.weight_beats, built.result_beats) == (6, 6, 2)
    # W'[c, k * 3 + c'] = W[c, k] where c' = c: each weight beat is one tile of W'.
    blocks = [
        [WEIGHTS[c][k] if other == c else 0 for k in range(4) for other in range(3)]
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
        assert built.structure.instances[1].instance_id == f"u_compute_{core}"


@pytest.mark.parametrize(
    "weights,delivery,narrow",
    [
        (WEIGHTS, WeightDelivery.CYCLIC, 1),  # no weight is INT4's -8
        (((-8, 0, 0, 0), (0,) * 4, (0,) * 4), WeightDelivery.CYCLIC, 0),
        (None, WeightDelivery.EXTERNAL, 0),  # weights at run time promise nothing
    ],
)
def test_narrow_weights_follow_known_weights(weights, delivery, narrow):
    facts = {**DENSE, "contraction": Contraction.DENSE, "outputs": 3, "pe": 3}
    built = matmul_assembly(
        target_dsp=DspBlock.DSP48E2, **{**facts, "weights": weights, "weight_delivery": delivery}
    )
    assert dict(built.structure.instances[1].requirements.parameters)["NARROW_WEIGHTS"] == narrow
