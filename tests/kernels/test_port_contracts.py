# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""dotp's ports derive from its schedule, and each refusal stays on its own stream.

dotp takes its extents from the streams it sits on, and its folding factors are its own
Decisions; every port presents what its schedule derives, so a wrong lane
count or a transposed tile can no longer be written into dotp. A producer
presenting another order is the stream's to judge: its plan names the steps
and its adapter carries them out, a lane order is wires, and a stream
admitting no adapter refuses the plan. The B1 probes map as follows: a wrong
lane count and a transposed tile become plans; a folding factor that does not
divide its extent is refused where it is committed; a frame crossing rows,
results that swap frames and depthwise operands under a dense core have no
analogue, since dotp derives its own schedule and the form is its own.
"""

from __future__ import annotations

from typing import Any

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Available, Rejected, Space, design_space
from finn.dataflow.gemm import Form
from finn.dataflow.plan import Step
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import (
    Adaptation,
    Traversal,
    classify,
    regrouped,
    tile,
    vector_major,
)
from finn.kernels.configure import commit
from finn.kernels.dotp import Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.port import AxiStreamPort
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.streams import Stream
from finn.kernels.target import DspBlock
from kernels.helpers import matmul_point, with_adapter_memories

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

    class Placed(Space):
        a = Stream(tensor=Tensor(x, ScalarEncoding(A)), port="in0_V")
        w_s = Stream(tensor=Tensor(w, ScalarEncoding(W)), adaptable=adaptable)
        r = Stream(tensor=Tensor(y, ScalarEncoding(R)), port="out0_V")
        weights = MemStreamKernel(dtype=W, form=tiled, contents=weight_values(w), output_stream=w_s)
        compute = core(
            target_dsp=DspBlock.DSP58,
            target_period_ns=5.0,
            form=form,
            result_dtype=R,
            x_stream=a,
            w_stream=w_s,
            y_stream=r,
        )

    return commit(
        design_space(Placed()),
        {
            "compute.pe": pe,
            "compute.simd": SIMD,
            "compute.compute_pumping": False,
            "weights.ram_style": "auto",
            "weights.pumped_memory": False,
        },
    )


def codes(answer: object) -> set[str]:
    assert isinstance(answer, (Available, Rejected)), answer
    return {finding.code for finding in answer.findings} if isinstance(answer, Rejected) else set()


def test_every_port_presents_what_the_schedule_derives():
    point = placed()
    x, w, y = (port.presented for port in (point.compute.x, point.compute.w, point.compute.y))
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
    # Each end is presented by a port node, named after it on the stream.
    (end,) = point.r.users
    assert (end.node, end.member) == ("compute.y", "stream")
    # The cyclic weights and the results connect as derived.
    assert codes(point.w_s.query(Stream.netlist)) == set()
    assert codes(point.r.query(Stream.netlist)) == set()


def test_depthwise_activations_carry_pe_channels_of_simd_window_positions():
    point = placed(Form.DEPTHWISE)
    form = point.compute.x.presented.form
    assert form.lanes == PE * SIMD and form.shape == (ROWS, REDUCTION, OUTPUTS)
    # Lane s * PE + p is window position s of channel p (FinnLib's order).
    assert next(form.positions()) == ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1))
    assert codes(point.w_s.query(Stream.netlist)) == set()


def test_a_folding_factor_that_does_not_divide_its_extent_is_refused_where_it_is_committed():
    with pytest.raises(ValueError, match="compute.pe"):
        placed(pe=3)


def test_a_producer_presenting_another_order_is_a_plan_its_stream_adapts():
    # Probe: the weight tile walked column fold first; the same PE x SIMD lanes.
    columns_first = Traversal.over(
        (REDUCTION, OUTPUTS),
        ((0, REDUCTION // SIMD, SIMD), (1, OUTPUTS // PE, PE)),
        ((1, PE, 1), (0, SIMD, 1)),
    )
    point = placed(weights_form=columns_first)
    assert point.w_s.plan.steps == (Step.REORDER,)
    assert codes(with_adapter_memories(point).w_s.query(Stream.netlist)) == set()
    # Probe: the tile's own sequence, one weight a beat where dotp reads PE x SIMD.
    narrow = placed(weights_form=regrouped(weights_tile(), 1))
    assert narrow.w_s.plan.steps == (Step.WIDTH,)
    # Another order at other lanes, the stored weights row-major, one a beat.
    rows = placed(weights_form=vector_major((REDUCTION, OUTPUTS), 1))
    assert rows.w_s.plan.steps == (Step.REORDER, Step.WIDTH)
    # dotp's own port is untouched by the producer's order.
    assert isinstance(narrow.compute.w.query(AxiStreamPort.contract), Available)
    # A stream that admits no adapter refuses the plan, naming it.
    fixed = placed(weights_form=columns_first, adaptable=False)
    refused = fixed.w_s.query(Stream.netlist)
    assert "stream-plan" in codes(refused) and "reorder" in str(refused)


def test_a_producer_s_lane_order_is_wires():
    # Probe: SIMD lanes outer, PE lanes inner; the same positions in each beat.
    transposed = Traversal.over(
        (REDUCTION, OUTPUTS),
        ((1, OUTPUTS // PE, PE), (0, REDUCTION // SIMD, SIMD)),
        ((0, SIMD, 1), (1, PE, 1)),
    )
    assert codes(placed(weights_form=transposed).w_s.query(Stream.netlist)) == set()
    # E-048: hlslib's per-channel order (window positions fastest) is likewise a
    # lane permutation of FinnLib's.
    finnlib = placed(Form.DEPTHWISE).compute.x.presented.form
    window_fastest = Traversal(
        finnlib.shape, finnlib.beat_loops, tuple(reversed(finnlib.lane_loops))
    )
    assert classify(window_fastest, finnlib).adaptation is Adaptation.LANE_PERMUTATION


def test_one_kernel_refusal_reaches_only_its_own_stream():
    # Probe P2: unsigned weights are refused by dotp's weight port. The
    # activations and the results are untouched.
    point = commit(
        matmul_point(
            m=3,
            n=4,
            k=4,
            activation_dtype=DataType["INT3"],
            weights_dtype=DataType["UINT3"],
            target_dsp=DspBlock.DSP48E2,
            target_period_ns=5.0,
        ),
        {
            "w.transport": "direct",
            "matmul.compute": "packed",
            "matmul.compute.packed.pe": 2,
            "matmul.compute.packed.simd": 2,
            "matmul.compute.packed.compute_pumping": False,
        },
    )
    point = with_adapter_memories(point)
    assert isinstance(point.x.query(Stream.netlist), Available)
    assert isinstance(point.y.query(Stream.netlist), Available)
    assert codes(point.w.query(Stream.netlist)) == {"dtype-family"}


def eltwise_between(rhs_shape: tuple[int, ...], rhs_dtype: str = "INT4") -> Any:
    """An ADD between boundary streams: lhs (3, 4), rhs of ``rhs_shape``, PE 2."""
    int4 = DataType["INT4"]

    class Added(Space):
        lhs = Stream(tensor=Tensor((3, 4), ScalarEncoding(int4)), port="in0_V")
        rhs = Stream(tensor=Tensor(rhs_shape, ScalarEncoding(DataType[rhs_dtype])), port="in1_V")
        out = Stream(tensor=Tensor((3, 4), ScalarEncoding(DataType["INT5"])), port="out0_V")
        add = EltwiseKernel(
            operation="ADD",
            pe=2,
            lhs_dtype=int4,
            rhs_dtype=int4,
            b_scale=1.0,
            target_dsp=DspBlock.DSP58,
            lhs_stream=lhs,
            rhs_stream=rhs,
            result_stream=out,
        )

    return design_space(Added())


def test_eltwise_broadcasts_a_channel_vector_once_per_pixel():
    point = eltwise_between((4,))
    # The rhs port presents the channel vector once per pixel it meets: a whole
    # pass repeated, which a boundary presents as it is.
    repeated = vector_major((4,), 2).repeated(3)
    assert point.add.rhs.presented.form == repeated
    assert point.rhs.endpoints.source.form == repeated
    assert all(stream.plan.steps == () for stream in (point.lhs, point.rhs, point.out))
    assert dict(point.add.module.parameters)["PE"] == 2


def test_eltwise_refuses_an_operand_it_cannot_broadcast_or_does_not_carry():
    # A channel vector of another length disagrees with lhs on the channels.
    misshaped = eltwise_between((3,))
    refused = misshaped.add.query(EltwiseKernel.extents)
    assert isinstance(refused, Rejected)
    assert {(finding.code, finding.message) for finding in refused.findings} == {
        ("kernel-extents", "c is 4 (lhs axis 1) and 3 (rhs axis 0)")
    }
    # An operand stream of another element: the stream refuses the port's end.
    other = eltwise_between((4,), rhs_dtype="INT3").rhs.query(Stream.netlist)
    assert isinstance(other, Rejected)
    assert "stream-tensor" in {finding.code for finding in other.findings}
