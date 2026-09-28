# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""dotp's ports derive from its iteration, and each refusal stays on its own stream.

A parent hands dotp the nest of the contraction it computes and the accesses of
its activations, weights and results; every port presents what that nest
derives, so a wrong lane count or a transposed tile can no longer be written
into dotp. What dotp cannot compute is refused by one admission rule
(``dotp-iteration``). A producer presenting another order is the stream's to
judge: its plan names the steps and its adapter carries them out, a field
order is wires, and a stream admitting no adapter refuses the plan. The B1
probes map as follows: a wrong lane count and a transposed tile become plans,
a frame crossing rows becomes an admission refusal, and results that swap
frames have no analogue, since results are derived.
"""

from __future__ import annotations

from typing import Any

from qonnx.core.datatype import DataType

from finn.core.space import Available, Rejected, Space, design_space
from finn.dataflow.nest import Einsum, Iteration, Nest, accesses, fold
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
from finn.kernels.delivery import CyclicDelivery
from finn.kernels.dotp import Contraction, DotpAxiKernel, Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.matmul import MatMulKernel
from finn.kernels.streams import Stream, commit_adapters
from finn.kernels.target import DspBlock

A, W, R = DataType["INT3"], DataType["INT3"], DataType["INT8"]
ROWS, REDUCTION, OUTPUTS, PE, SIMD = 2, 4, 4, 2, 2


def iteration(contraction: Contraction, *, order: tuple[int, ...] | None = None) -> Iteration:
    """The contraction folded by PE and SIMD; ``order`` permutes the beat levels."""
    einsum = Einsum(contraction.einsum)
    output = einsum.output[-1]
    extents = {"r": ROWS, "k": REDUCTION, output: OUTPUTS}
    nest = fold(einsum, extents, {output: PE, "k": SIMD})
    x, w, y = accesses(einsum, nest, extents)
    if order is not None:
        nest = Nest(tuple(nest.beats[i] for i in order), nest.lanes)
    return Iteration(nest, (x, w, y))


def weight_values(shape: tuple[int, ...]) -> Any:
    return tuple(tuple((3 * i + j) % 7 - 3 for j in range(shape[1])) for i in range(shape[0]))


def placed(
    contraction: Contraction = Contraction.DENSE,
    *,
    nested: Iteration | None = None,
    weights_form: Traversal | None = None,
    pe: int = PE,
    adaptable: bool = True,
):
    """dotp between boundary activations and results, its weights from a cyclic ROM."""
    nested = iteration(contraction) if nested is None else nested
    x, w, y = nested.operands
    core = Int8Dsp58DotpKernel if contraction is Contraction.PER_CHANNEL else PackedDotpKernel
    form = tile(w.tensor[0], w.tensor[1], PE, SIMD) if weights_form is None else weights_form

    class Placed(Space):
        a = Stream(tensor=Tensor(x.tensor, ScalarEncoding(A)), port="in0_V")
        w_s = Stream(tensor=Tensor(w.tensor, ScalarEncoding(W)), adaptable=adaptable)
        r = Stream(tensor=Tensor(y.tensor, ScalarEncoding(R)), port="out0_V")
        weights = CyclicDelivery(
            dtype=W, form=form, values=weight_values(w.tensor), output_stream=w_s
        )
        compute = core(
            activation_dtype=A,
            weights_dtype=W,
            result_dtype=R,
            pe=pe,
            simd=SIMD,
            target_dsp=DspBlock.DSP58,
            target_period_ns=5.0,
            contraction=contraction,
            activation_stream=a,
            weights_stream=w_s,
            result_stream=r,
            iteration=nested,
        )

    return design_space(Placed()).with_choices(
        {Placed.compute.compute_pumping: False, Placed.weights.rom_style: "auto"}
    )


def codes(answer: object) -> set[str]:
    assert isinstance(answer, (Available, Rejected)), answer
    return {finding.code for finding in answer.findings} if isinstance(answer, Rejected) else set()


def test_every_port_presents_what_the_nest_derives():
    point = placed()
    presented = point.compute.presentations
    assert presented.activation.form == vector_major((ROWS, REDUCTION), SIMD).replayed(
        OUTPUTS // PE, inner_beats=REDUCTION // SIMD
    )
    assert presented.weights.form == tile(OUTPUTS, REDUCTION, PE, SIMD).repeated(ROWS)
    assert presented.result.form == vector_major((ROWS, OUTPUTS), PE)
    assert [rule.beats for rule in presented.activation.markers] == [REDUCTION // SIMD]
    # The cyclic weights and the results connect as derived.
    assert codes(point.w_s.query(Stream.connection)) == set()
    assert codes(point.r.query(Stream.connection)) == set()


def test_per_channel_activations_carry_pe_channels_of_simd_window_positions():
    point = placed(Contraction.PER_CHANNEL)
    form = point.compute.presentations.activation.form
    assert form.lanes == PE * SIMD and form.shape == (ROWS, REDUCTION, OUTPUTS)
    # Field s * PE + p is window position s of channel p (FinnLib's order).
    assert next(form.positions()) == ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1))
    assert codes(point.w_s.query(Stream.connection)) == set()


def test_what_dotp_cannot_compute_is_one_admission_refusal():
    # Probe: a frame crossing activation rows, the nest walking r inside k.
    crossing = placed(nested=iteration(Contraction.DENSE, order=(1, 2, 0)))
    refused = crossing.compute.query(DotpAxiKernel.presentations)
    assert codes(refused) == {"dotp-iteration"} and "innermost" in str(refused)
    # The nest's lanes must be dotp's PE x SIMD.
    assert codes(placed(pe=1).compute.query(DotpAxiKernel.presentations)) == {"dotp-iteration"}
    # Per-channel accesses are not a broadcast: the dense mode refuses them.
    point = placed(Contraction.DENSE, nested=iteration(Contraction.PER_CHANNEL))
    assert codes(point.compute.query(DotpAxiKernel.presentations)) == {"dotp-iteration"}


def test_a_producer_presenting_another_order_is_a_plan_its_stream_adapts():
    # Probe: the weight tile walked column fold first; the same PE x SIMD lanes.
    columns_first = Traversal.over(
        (OUTPUTS, REDUCTION),
        ((1, REDUCTION // SIMD, SIMD), (0, OUTPUTS // PE, PE)),
        ((0, PE, 1), (1, SIMD, 1)),
    )
    point = placed(weights_form=columns_first)
    assert point.w_s.plan.steps == (Step.REORDER,)
    assert codes(commit_adapters(point).w_s.query(Stream.connection)) == set()
    # Probe: the tile's own sequence, one weight a beat where dotp reads PE x SIMD.
    narrow = placed(weights_form=regrouped(tile(OUTPUTS, REDUCTION, PE, SIMD), 1))
    assert narrow.w_s.plan.steps == (Step.WIDTH,)
    # Another order at other lanes, the row-major weights one a beat.
    rows = placed(weights_form=vector_major((OUTPUTS, REDUCTION), 1))
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
        (OUTPUTS, REDUCTION),
        ((0, OUTPUTS // PE, PE), (1, REDUCTION // SIMD, SIMD)),
        ((1, SIMD, 1), (0, PE, 1)),
    )
    assert codes(placed(weights_form=transposed).w_s.query(Stream.connection)) == set()
    # E-048: hlslib's per-channel order (window positions fastest) is likewise a
    # field permutation of FinnLib's.
    finnlib = placed(Contraction.PER_CHANNEL).compute.presentations.activation.form
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
