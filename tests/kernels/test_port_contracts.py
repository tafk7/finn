# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""dotp's ports derive from its schedule, and each refusal stays on its own channel.

dotp takes its extents from the streams it sits on, and its folding factors are its own
Decisions; every port presents what its schedule derives, so a wrong lane
count or a transposed tile can no longer be written into dotp. A producer
presenting another order is the channel's to judge: it names the steps between
its two streams and its adapter carries them out, a lane order is wires, and a
channel admitting no adapter refuses the steps. A wrong lane count and a transposed
tile become steps; a folding factor that does not divide its extent is refused
where it is committed; a frame crossing rows, results that swap frames and
depthwise operands under a dense core cannot be written, since dotp derives
its own schedule and the form is its own.
"""

from __future__ import annotations

from typing import Any

import pytest
from qonnx.core.datatype import DataType
from support.helpers import (
    FULL_DSP48E2,
    FULL_DSP58,
    choose,
    dotp_extents,
    keyed,
    matmul_point,
    with_adapter_memories,
    with_channel_defaults,
    with_direct_transports,
)

from finn.core.space import Available, Rejected, inspection
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.gemm import Form
from finn.dataflow.steps import Step
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import (
    Adaptation,
    Traversal,
    classify,
    regrouped,
    tile,
    vector_major,
)
from finn.kernels.channels import Channel
from finn.kernels.dotp import Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.library import library_module
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.network import Interface, Network, wire
from finn.kernels.values.semantics import IntegerTensorValue

A, W, R = DataType["INT3"], DataType["INT3"], DataType["INT8"]
ROWS, REDUCTION, OUTPUTS, PE, SIMD = 2, 4, 4, 2, 2


def shapes(form: Form) -> tuple[tuple[int, ...], ...]:
    x = (ROWS, REDUCTION, OUTPUTS) if form is Form.DEPTHWISE else (ROWS, REDUCTION)
    return x, (REDUCTION, OUTPUTS), (ROWS, OUTPUTS)


def weights_tile() -> Traversal:
    """dotp's weights, stored (k, n): PE x SIMD tiles, n folds outer, SIMD fastest in a beat."""
    return Traversal.over(
        (REDUCTION, OUTPUTS),
        ((1, OUTPUTS // PE, PE), (0, REDUCTION // SIMD, SIMD)),
        ((1, PE, 1), (0, SIMD, 1)),
    )


def weight_values(shape: tuple[int, ...]) -> Any:
    return tuple(tuple((3 * i + j) % 7 - 3 for j in range(shape[1])) for i in range(shape[0]))


def placed(
    form: Form = Form.DENSE,
    *,
    weights_form: Traversal | None = None,
    adaptable: bool = True,
    pe: int = PE,
):
    """dotp between boundary activations and results, its weights from a cyclic ROM."""
    x, w, y = shapes(form)
    core = Int8Dsp58DotpKernel if form is Form.DEPTHWISE else PackedDotpKernel
    tiled = weights_tile() if weights_form is None else weights_form

    a_tensor = Tensor(x, ScalarEncoding(A))
    w_tensor = Tensor(w, ScalarEncoding(W))
    r_tensor = Tensor(y, ScalarEncoding(R))

    network = Network(
        {
            "a": wire(
                tensor=a_tensor,
                port="in0_V",
                consumer=Interface("compute", "x_contract"),
                platform=FULL_DSP48E2,
            ),
            "w_s": wire(
                platform=FULL_DSP48E2,
                tensor=w_tensor,
                adaptable=adaptable,
                producer=Interface("weights", "output_contract"),
                consumer=Interface("compute", "w_contract"),
            ),
            "r": wire(
                tensor=r_tensor,
                port="out0_V",
                producer=Interface("compute", "y_contract"),
                platform=FULL_DSP48E2,
            ),
            "weights": MemStreamKernel(
                platform=FULL_DSP48E2,
                dtype=W,
                form=tiled,
                contents=IntegerTensorValue.of(weight_values(w)),
            ),
            "compute": core(
                form=form,
                result_range=ordinary_integer_bounds(R),
                x_tensor=a_tensor,
                w_tensor=w_tensor,
                **dotp_extents(w_tensor),
                pe=pe,
                simd=SIMD,
                platform=FULL_DSP58,
            ),
        }
    )
    chosen = choose(
        network,
        {
            "compute.compute_pumping": False,
            **({"compute.reducer": "tree"} if core is PackedDotpKernel else {}),
            "weights.ram_style": "auto",
            "weights.pumped_memory": False,
        },
    )
    # The core is given its folding, so no change brings its channels new facts: their
    # adapters are set as the cut sets them.
    return with_channel_defaults(with_direct_transports(chosen))


def test_every_port_presents_what_the_schedule_derives():
    point = placed()
    compute = point.member("compute")
    x, w, y = (compute.x_stream.sequence, compute.w_stream.sequence, compute.y_stream.sequence)
    assert x.form == vector_major((ROWS, REDUCTION), SIMD).replayed(
        OUTPUTS // PE, inner_beats=REDUCTION // SIMD
    )
    assert w.form == weights_tile().repeated(ROWS)
    # The same positions, beat for beat, as MVAU's (n, k) tile transposed.
    assert [tuple(p[::-1] for p in beat) for beat in weights_tile().positions()] == list(
        tile(OUTPUTS, REDUCTION, PE, SIMD).positions()
    )
    assert y.form == vector_major((ROWS, OUTPUTS), PE)
    assert [rule.beats for rule in x.markers] == [REDUCTION // SIMD]
    # Each port's contract is labelled after the port, from the root.
    assert point.member("r").producer_label == "compute.y"
    # The cyclic weights and the results connect as derived.
    assert isinstance(point.member("w_s").query(Channel.netlist), Available)
    assert isinstance(point.member("r").query(Channel.netlist), Available)


def test_depthwise_activations_carry_pe_channels_of_simd_window_positions():
    point = placed(Form.DEPTHWISE)
    form = point.member("compute").x_stream.sequence.form
    assert form.lanes == PE * SIMD and form.shape == (ROWS, REDUCTION, OUTPUTS)
    # Lane s * PE + p is window position s of channel p (FinnLib's order).
    assert next(form.positions()) == ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1))
    assert isinstance(point.member("w_s").query(Channel.netlist), Available)


def test_a_folding_factor_that_does_not_divide_its_extent_refuses_the_core_given_it():
    # A core is given its folding (its role's choice, refused there by its domain): one
    # that does not divide its extent refuses the core's schedule.
    found = inspection.validity(placed(pe=3).member("compute"))
    assert found.status == "refused"
    assert {(f.code, f.owner) for f in found.result.findings} == {("kernel-schedule", "schedule")}


def test_a_producer_presenting_another_order_is_steps_its_stream_adapts():
    # Probe: the weight tile walked column fold first; the same PE x SIMD lanes.
    columns_first = Traversal.over(
        (REDUCTION, OUTPUTS),
        ((0, REDUCTION // SIMD, SIMD), (1, OUTPUTS // PE, PE)),
        ((1, PE, 1), (0, SIMD, 1)),
    )
    point = placed(weights_form=columns_first)
    assert point.member("w_s").steps.kinds == (Step.REORDER,)
    assert isinstance(with_adapter_memories(point).member("w_s").query(Channel.netlist), Available)
    # Probe: the tile's own sequence, one weight a beat where dotp reads PE x SIMD.
    narrow = placed(weights_form=regrouped(weights_tile(), 1))
    assert narrow.member("w_s").steps.kinds == (Step.WIDTH,)
    # Another order at other lanes, the stored weights row-major, one a beat.
    rows = placed(weights_form=vector_major((REDUCTION, OUTPUTS), 1))
    assert rows.member("w_s").steps.kinds == (Step.REORDER, Step.WIDTH)
    # dotp's own port is untouched by the producer's order.
    compute = narrow.member("compute")
    assert isinstance(compute.query(type(compute).w_contract), Available)
    # A channel that admits no adapter refuses the folding that needs one, naming the
    # adapter it would need (a change that breaks a rule is refused).
    with pytest.raises(ValueError, match="channel-form: needs a reorder adapter"):
        placed(weights_form=columns_first, adaptable=False)


def test_a_producer_s_lane_order_is_wires():
    # Probe: SIMD lanes outer, PE lanes inner; the same positions in each beat.
    transposed = Traversal.over(
        (REDUCTION, OUTPUTS),
        ((1, OUTPUTS // PE, PE), (0, REDUCTION // SIMD, SIMD)),
        ((0, SIMD, 1), (1, PE, 1)),
    )
    assert isinstance(
        placed(weights_form=transposed).member("w_s").query(Channel.netlist), Available
    )
    # hlslib's per-channel order (window positions fastest) is likewise a
    # lane permutation of FinnLib's.
    finnlib = placed(Form.DEPTHWISE).member("compute").x_stream.sequence.form
    window_fastest = Traversal(
        finnlib.shape, finnlib.beat_loops, tuple(reversed(finnlib.lane_loops))
    )
    assert classify(window_fastest, finnlib).adaptation is Adaptation.LANE_PERMUTATION


def test_one_kernel_refusal_reaches_the_change_that_selects_it():
    # Unsigned weights are refused by dotp's weight port: no core is possible, each
    # refused by its port's rule, and the change selecting the packed core is refused,
    # naming it (where it used to leave MatMul, and every port contract and channel on it,
    # refused).
    point = matmul_point(
        m=3,
        n=4,
        k=4,
        activation_dtype=DataType["INT3"],
        weights_dtype=DataType["UINT3"],
        platform=FULL_DSP48E2,
    )
    ((_, compute),) = point.viable([("matmul", "compute")])
    assert compute.cases == ()
    trial = point.trial(
        keyed(
            {
                "matmul.compute": "packed",
                "matmul.pe": 2,
                "matmul.simd": 2,
                "matmul.compute.packed.compute_pumping": False,
                "matmul.compute.packed.reducer": "tree",
            }
        )
    )
    assert not trial.accepted
    assert {(f.code, f.owner) for why in trial.refused.values() for f in why} == {
        ("dtype-family", "compute.packed.w_element_taken")
    }


def eltwise_between(rhs_shape: tuple[int, ...], rhs_dtype: str = "INT4", pe: int | None = 2) -> Any:
    """An ADD between boundary channels: lhs (3, 4), rhs of ``rhs_shape``, its PE ``pe``
    (committed where given)."""
    int4 = DataType["INT4"]

    lhs_tensor = Tensor((3, 4), ScalarEncoding(int4))
    rhs_tensor = Tensor(rhs_shape, ScalarEncoding(DataType[rhs_dtype]))
    out_tensor = Tensor((3, 4), ScalarEncoding(DataType["INT5"]))

    network = Network(
        {
            "lhs": wire(
                platform=FULL_DSP48E2,
                tensor=lhs_tensor,
                port="in0_V",
                consumer=Interface("add", "lhs_contract"),
            ),
            "rhs": wire(
                tensor=rhs_tensor,
                port="in1_V",
                consumer=Interface("add", "rhs_contract"),
                platform=FULL_DSP48E2,
            ),
            "out": wire(
                tensor=out_tensor,
                port="out0_V",
                producer=Interface("add", "result_contract"),
                platform=FULL_DSP48E2,
            ),
            "add": EltwiseKernel(
                operation="ADD",
                b_scale=1.0,
                platform=FULL_DSP58,
                lhs_tensor=lhs_tensor,
                rhs_tensor=rhs_tensor,
            ),
        }
    )
    return with_direct_transports(network if pe is None else choose(network, {"add.pe": pe}))


def test_eltwise_broadcasts_a_channel_vector_once_per_pixel():
    point = eltwise_between((4,))
    # The rhs port presents the channel vector once per pixel it meets: a whole
    # pass repeated, which a boundary presents as it is.
    repeated = vector_major((4,), 2).repeated(3)
    assert point.member("add").rhs_stream.sequence.form == repeated
    assert point.member("rhs").sides.source.sequence.form == repeated
    assert all(
        stream.steps.kinds == ()
        for stream in (point.member("lhs"), point.member("rhs"), point.member("out"))
    )
    assert dict(library_module(point.member("add")).parameters)["PE"] == 2


def test_eltwise_refuses_an_operand_it_cannot_broadcast_or_does_not_carry():
    # A channel vector of another length disagrees with lhs on the channels.
    misshaped = eltwise_between((3,), pe=None)  # no PE divides what the extents refuse
    refused = misshaped.member("add").query(EltwiseKernel.extents)
    assert isinstance(refused, Rejected)
    assert {(finding.code, finding.message) for finding in refused.findings} == {
        ("kernel-extents", "c is 4 (lhs axis 1) and 3 (rhs axis 0)")
    }
    # An operand of another element: its port carries its tensor's, and the kernel
    # refuses the pair on its facts, its PE still open; setting PE breaks no rule that
    # was not broken before, so it is accepted, and the kernel stays refused.
    other = eltwise_between((4,), rhs_dtype="INT3", pe=None)
    assert other.member("add").rhs_element == ScalarEncoding(DataType["INT3"])
    pair = {("eltwise-integer-pair", "implementation_supported")}
    assert {(f.code, f.owner) for f in other.checked(["add"])[("add", "")]} == pair
    trial = other.trial(keyed({"add.pe": 2}))
    assert trial.accepted
    assert {(f.code, f.owner) for f in trial.network.checked(["add"])[("add", "")]} == pair
