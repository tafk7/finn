# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""dotp's ports derive from its schedule, and each refusal stays on its own stream.

A parent hands dotp the schedule it computes (the extents of ``m``, ``n`` and
``k``, their folds and the beat order) and its form; every port presents what
they derive, so a wrong lane count or a transposed tile can no longer be
written into dotp. What dotp cannot compute is refused by one admission rule
(``dotp-schedule``). A producer presenting another order is the stream's to
judge: its plan names the steps and its adapter carries them out, a field
order is wires, and a stream admitting no adapter refuses the plan. The B1
probes map as follows: a wrong lane count and a transposed tile become plans,
a frame crossing rows becomes an admission refusal, and results that swap
frames have no analogue, since results are derived. Depthwise operands read
by a dense dotp have none either: the form is dotp's, not the schedule's.
"""

from __future__ import annotations

from typing import Any

from qonnx.core.datatype import DataType

from finn.core.space import Available, Rejected, Space, design_space
from finn.dataflow.gemm import Form, k, m, n
from finn.dataflow.plan import Step
from finn.dataflow.schedule import Schedule
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import (
    Adaptation,
    Traversal,
    classify,
    regrouped,
    tile,
    vector_major,
)
from finn.kernels.delivery import CyclicDelivery
from finn.kernels.dotp import DotpAxiKernel, Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.matmul import MatMulKernel
from finn.kernels.streams import Stream, commit_adapters
from finn.kernels.target import DspBlock

A, W, R = DataType["INT3"], DataType["INT3"], DataType["INT8"]
ROWS, REDUCTION, OUTPUTS, PE, SIMD = 2, 4, 4, 2, 2


def schedule(beats: tuple = (m, n, k)) -> Schedule:
    """``n`` folded by PE and ``k`` by SIMD, in the beat order ``beats``."""
    return Schedule({m: ROWS, n: OUTPUTS, k: REDUCTION}, folds={n: PE, k: SIMD}, beats=beats)


def shapes(form: Form) -> tuple[tuple[int, ...], ...]:
    extents = {m: ROWS, n: OUTPUTS, k: REDUCTION}
    return tuple(tuple(extents[i] for i in operand) for operand in (form.x, form.w, form.y))


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
    scheduled: Schedule | None = None,
    weights_form: Traversal | None = None,
    pe: int = PE,
    adaptable: bool = True,
):
    """dotp between boundary activations and results, its weights from a cyclic ROM."""
    scheduled = schedule() if scheduled is None else scheduled
    x, w, y = shapes(form)
    core = Int8Dsp58DotpKernel if form is Form.DEPTHWISE else PackedDotpKernel
    tiled = weights_tile() if weights_form is None else weights_form

    class Placed(Space):
        a = Stream(tensor=Tensor(x, ScalarEncoding(A)), port="in0_V")
        w_s = Stream(tensor=Tensor(w, ScalarEncoding(W)), adaptable=adaptable)
        r = Stream(tensor=Tensor(y, ScalarEncoding(R)), port="out0_V")
        weights = CyclicDelivery(dtype=W, form=tiled, values=weight_values(w), output_stream=w_s)
        compute = core(
            activation_dtype=A,
            weights_dtype=W,
            result_dtype=R,
            pe=pe,
            simd=SIMD,
            target_dsp=DspBlock.DSP58,
            target_period_ns=5.0,
            form=form,
            activation_stream=a,
            weights_stream=w_s,
            result_stream=r,
            schedule=scheduled,
        )

    return design_space(Placed()).with_choices(
        {Placed.compute.compute_pumping: False, Placed.weights.rom_style: "auto"}
    )


def codes(answer: object) -> set[str]:
    assert isinstance(answer, (Available, Rejected)), answer
    return {finding.code for finding in answer.findings} if isinstance(answer, Rejected) else set()


def test_every_port_presents_what_the_schedule_derives():
    point = placed()
    presented = point.compute.sequences
    assert presented.activation.form == vector_major((ROWS, REDUCTION), SIMD).replayed(
        OUTPUTS // PE, inner_beats=REDUCTION // SIMD
    )
    assert presented.weights.form == weights_tile().repeated(ROWS)
    # The same positions, beat for beat, as MVAU's (n, k) tile transposed.
    assert [tuple(p[::-1] for p in beat) for beat in weights_tile().positions()] == list(
        tile(OUTPUTS, REDUCTION, PE, SIMD).positions()
    )
    assert presented.result.form == vector_major((ROWS, OUTPUTS), PE)
    assert [rule.beats for rule in presented.activation.markers] == [REDUCTION // SIMD]
    # The cyclic weights and the results connect as derived.
    assert codes(point.w_s.query(Stream.connection)) == set()
    assert codes(point.r.query(Stream.connection)) == set()


def test_depthwise_activations_carry_pe_channels_of_simd_window_positions():
    point = placed(Form.DEPTHWISE)
    form = point.compute.sequences.activation.form
    assert form.lanes == PE * SIMD and form.shape == (ROWS, REDUCTION, OUTPUTS)
    # Field s * PE + p is window position s of channel p (FinnLib's order).
    assert next(form.positions()) == ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1))
    assert codes(point.w_s.query(Stream.connection)) == set()


def test_what_dotp_cannot_compute_is_one_admission_refusal():
    # Probe: a frame crossing activation rows, the schedule walking m inside k.
    crossing = placed(scheduled=schedule(beats=(n, k, m)))
    refused = crossing.compute.query(DotpAxiKernel.sequences)
    assert codes(refused) == {"dotp-schedule"} and "innermost" in str(refused)
    # The schedule's folds must be dotp's PE x SIMD.
    assert codes(placed(pe=1).compute.query(DotpAxiKernel.sequences)) == {"dotp-schedule"}
    # dotp folds nothing but n and k.
    folded_rows = Schedule(
        {m: ROWS, n: OUTPUTS, k: REDUCTION}, folds={m: 2, n: PE, k: SIMD}, beats=(m, n, k)
    )
    refused = placed(scheduled=folded_rows).compute.query(DotpAxiKernel.sequences)
    assert codes(refused) == {"dotp-schedule"}


def test_a_producer_presenting_another_order_is_a_plan_its_stream_adapts():
    # Probe: the weight tile walked column fold first; the same PE x SIMD lanes.
    columns_first = Traversal.over(
        (REDUCTION, OUTPUTS),
        ((0, REDUCTION // SIMD, SIMD), (1, OUTPUTS // PE, PE)),
        ((1, PE, 1), (0, SIMD, 1)),
    )
    point = placed(weights_form=columns_first)
    assert point.w_s.plan.steps == (Step.REORDER,)
    assert codes(commit_adapters(point).w_s.query(Stream.connection)) == set()
    # Probe: the tile's own sequence, one weight a beat where dotp reads PE x SIMD.
    narrow = placed(weights_form=regrouped(weights_tile(), 1))
    assert narrow.w_s.plan.steps == (Step.WIDTH,)
    # Another order at other lanes, the stored weights row-major, one a beat.
    rows = placed(weights_form=vector_major((REDUCTION, OUTPUTS), 1))
    assert rows.w_s.plan.steps == (Step.REORDER, Step.WIDTH)
    # dotp's own port is untouched by the producer's order.
    assert isinstance(narrow.compute.query(DotpAxiKernel.weights_port), Available)
    # A stream that admits no adapter refuses the plan, naming it.
    fixed = placed(weights_form=columns_first, adaptable=False)
    refused = fixed.w_s.query(Stream.connection)
    assert "stream-plan" in codes(refused) and "reorder" in str(refused)


def test_a_producer_s_field_order_is_wires():
    # Probe: SIMD fields outer, PE fields inner; the same positions in each beat.
    transposed = Traversal.over(
        (REDUCTION, OUTPUTS),
        ((1, OUTPUTS // PE, PE), (0, REDUCTION // SIMD, SIMD)),
        ((0, SIMD, 1), (1, PE, 1)),
    )
    assert codes(placed(weights_form=transposed).w_s.query(Stream.connection)) == set()
    # E-048: hlslib's per-channel order (window positions fastest) is likewise a
    # field permutation of FinnLib's.
    finnlib = placed(Form.DEPTHWISE).compute.sequences.activation.form
    window_fastest = Traversal(
        finnlib.shape, finnlib.beat_loops, tuple(reversed(finnlib.lane_loops))
    )
    assert classify(window_fastest, finnlib).adaptation is Adaptation.LANE_PERMUTATION


def test_one_kernel_refusal_reaches_only_its_own_stream():
    # Probe P2: unsigned weights are refused by dotp's weight port. The
    # activations and the results are untouched.
    point = design_space(
        MatMulKernel(
            rows=3,
            reduction=4,
            outputs=4,
            activation_dtype=DataType["INT3"],
            weights_dtype=DataType["UINT3"],
            target_dsp=DspBlock.DSP48E2,
            target_period_ns=5.0,
        )
    ).with_choices(
        {
            MatMulKernel.pe: 2,
            MatMulKernel.simd: 2,
            MatMulKernel.delivery: "external",
            MatMulKernel.weight_stream.transport: "direct",
            MatMulKernel.compute: "packed",
            MatMulKernel.compute_pumping: False,
        }
    )
    point = commit_adapters(point)
    assert isinstance(point.activations.query(Stream.connection), Available)
    assert isinstance(point.results.query(Stream.connection), Available)
    assert codes(point.weight_stream.query(Stream.connection)) == {"dtype-family"}
