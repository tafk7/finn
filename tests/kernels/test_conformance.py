# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every kernel on channels, checked against its RTL by the conformance harness.

Each case places one kernel between boundary channels over sampled folding factors
(``kernels.conformance``): dotp on both cores (packed; INT8, dense and
depthwise), thresholding, eltwise with a broadcast operand, transpose,
memstream with an identity reference, the channel stages (fifo, vpc and
input_gen) and MatMul, a kernel with children. Every folding factor is a
Decision (dotp's and MatMul's PE and SIMD sampled; thresholding's and eltwise's
PE and transpose's SIMD given as configurations); memstream's ``form``, a fact
of its consumer, is given.

A channel stage's ports are opaque words, and the channel's adapter states what
they carry (``finn.kernels.adapters.realize``). Its case places the stage on
two channels (``StagedFifo``, ``StagedVpc``, ``StagedInputGenerator``: its
module, its native pins, ports presenting what its realization gives them), so
XSim checks that the RTL walks the realization: a FIFO in each storage its RTL
implements, width conversions, and input_gen's reorders (a transpose, and a
replay). MatMul's activations are replayed and framed by its channel's adapter.

The harness's reason to exist: a thresholding kernel that declares an order
its RTL does not walk passes every Python check and fails in XSim, for the
loop order (channel folds outer, rows inner, where the RTL walks rows outer)
and for the lane order (the two levels of a split channel index swapped). The
same kernels declaring the RTL's orders pass both.

The checks refuse: a module whose pins its sources contradict, and a kernel
whose ``parameters()`` omit a module parameter, including for modules with a
parameter whose value the RTL checker does not establish (thresholding's
array, eltwise's real, which is itself the omission checked).

transpose's stalled samples and its adapter sample (behind a ``vpc``) exercise
``inner_shuffle``'s page guard (``finn.kernels.transpose``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Param, Rejected, derived, design_space
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.gemm import Form
from finn.dataflow.plan import plan
from finn.dataflow.schedule import Index, Schedule
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, Traversal, tile, vector_major
from finn.kernels.adapters import Convert, Generate, RealizedStage, realize
from finn.kernels.artifacts.abi import Direction, Endpoint, Signal
from finn.kernels.artifacts.module import Held
from finn.kernels.base import NATIVE_CLOCKING
from finn.kernels.channels import Channel
from finn.kernels.dotp import Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.fifo import FifoKernel
from finn.kernels.input_generator import InputGeneratorKernel
from finn.kernels.matmul import MatMulKernel, exact_result_dtype
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.port import AxiStreamPort
from finn.kernels.target import DspBlock
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.transpose import TransposeKernel
from finn.kernels.values.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.vpc import VpcKernel
from kernels.adapted import columns_first
from kernels.conformance import (
    MODES,
    NonConformance,
    Sample,
    _settle_known,
    _values,
    conformance,
    place,
    samples,
)
from kernels.helpers import FULL_DSP48E2, FULL_DSP58, full_platform
from kernels.xsim import requires_xsim


def tensor(shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor(shape, ScalarEncoding(DataType[dtype]))


# -- dotp ----------------------------------------------------------------------------------

ROWS, REDUCTION, OUTPUTS = 2, 6, 4


def dotp(
    space_type: type[Any],
    dsp: DspBlock,
    bits: int,
    form: Form = Form.DENSE,
    reducer: str | None = None,
) -> dict[str, Any]:
    """Y = X @ W, or per channel (depthwise); the accumulator type is the parent's fact.
    ``reducer`` is the packed core's, which only it declares."""
    a = w = DataType[f"INT{bits}"]
    depthwise = form is Form.DEPTHWISE
    reduction = 3 if depthwise else REDUCTION
    x_shape = (ROWS, reduction, OUTPUTS) if depthwise else (ROWS, reduction)

    def reference(x_channel: np.ndarray, w_channel: np.ndarray) -> dict[str, np.ndarray]:
        y = (x_channel * w_channel).sum(axis=1) if depthwise else x_channel @ w_channel
        return {"y_channel": y}

    return dict(
        space_type=space_type,
        inputs={
            "x_channel": Tensor(x_shape, ScalarEncoding(a)),
            "w_channel": Tensor((reduction, OUTPUTS), ScalarEncoding(w)),
        },
        outputs={"y_channel": (ROWS, OUTPUTS)},
        reference=reference,
        choices={"compute_pumping": False, **({"reducer": reducer} if reducer else {})},
        facts={
            "platform": full_platform(dsp),
            "form": form,
            "result_dtype": exact_result_dtype(reduction, a, w),
        },
    )


# -- thresholding --------------------------------------------------------------------------

PIXELS, CHANNELS = 3, 6
# Three thresholds a channel, two apart from the next channel's, so that a value
# thresholded against another channel's row often lands on another level: the
# planted-error tests below need their stimulus to tell channels apart.
THRESHOLDS = (tuple((-8 + 2 * c, -7 + 2 * c, -6 + 2 * c) for c in range(CHANNELS)),)


def levels(values: np.ndarray) -> np.ndarray:
    return (values[..., None] >= np.array(THRESHOLDS[0])).sum(axis=-1)


THRESHOLDING_FACTS = dict(
    input_dtype=DataType["INT4"],
    threshold_dtype=DataType["INT4"],
    thresholds=THRESHOLDS,
    bias=0,
    platform=FULL_DSP48E2,
)
# Its memories: of the two stages (N = 3), the deeper in block RAM, the other distributed.
THRESHOLDING_CHOICES = {
    "use_axilite": False,
    "deep_pipeline": False,
    "ram_style": "distributed",
    "block_stages": 1,
    "ultra_stages": 0,
}
PE_FACTORS = ({"pe": 1}, {"pe": 3}, {"pe": CHANNELS})


def thresholding() -> dict[str, Any]:
    return dict(
        space_type=ThresholdingAxiKernel,
        inputs={"input_channel": tensor((PIXELS, CHANNELS), "INT4")},
        outputs={"output_channel": (PIXELS, CHANNELS)},
        reference=lambda input_channel: {"output_channel": levels(input_channel)},
        factors=PE_FACTORS,
        choices=THRESHOLDING_CHOICES,
        facts=THRESHOLDING_FACTS,
    )


# -- the loop-order proof: thresholding on a schedule ---------------------------------------

r, c = Index("r"), Index("c")


class RowsFirst(ThresholdingAxiKernel):
    """thresholding_axi declaring its rows outer, channels split by PE inner: the RTL's order."""

    id = "test.thresholding_axi.rows_first"
    channels_outer: ClassVar[bool] = False

    @derived
    def schedule(self) -> Schedule | Rejected:
        *outer, last = self.indices
        order = (last, *outer) if type(self).channels_outer else (*outer, last)
        return self.bound_schedule(tuple(order), self.factors, extents={last: self.channels})


class ChannelsFirst(RowsFirst):
    """The same module declared with the wrong loop order: channel folds outer, rows inner."""

    id = "test.thresholding_axi.channels_first"
    channels_outer: ClassVar[bool] = True


def scheduled(space_type: type[RowsFirst]) -> dict[str, Any]:
    """Two channel folds or more, so the two orders differ in every sample."""
    return dict(
        thresholding(),
        space_type=space_type,
        outputs={"output_channel": tensor((PIXELS, CHANNELS), "UINT2")},
        factors=({"pe": 1}, {"pe": 2}, {"pe": 3}),
    )


# -- the lane-order proof: every channel in one beat, the channel index split ---------------

co, ci = Index("co"), Index("ci")


def split_ports(schedule: Any, lanes: tuple[Index, ...]) -> tuple[AxiStreamPort, AxiStreamPort]:
    """Input and output reading channel ``c = 3 co + ci``, lanes in ``lanes`` order."""
    input = AxiStreamPort(
        name="s_axis",
        endpoint=Endpoint.TARGET,
        channel=ThresholdingAxiKernel.input_channel,
        schedule=schedule,
        index=(r, co * 3 + ci),
        lanes=lanes,
        dtype=ThresholdingAxiKernel.input_dtype,
    )
    output = AxiStreamPort(
        name="m_axis",
        endpoint=Endpoint.INITIATOR,
        channel=ThresholdingAxiKernel.output_channel,
        schedule=schedule,
        index=(r, co * 3 + ci),
        lanes=lanes,
        dtype=ThresholdingAxiKernel.result_dtype,
    )
    return input, output


class LanesInOrder(ThresholdingAxiKernel):
    """PE = C: all channels a beat, lane ``3 co + ci`` holding channel ``3 co + ci``."""

    id = "test.thresholding_axi.lanes_in_order"

    @derived
    def schedule(self) -> Schedule | Rejected:
        # The window axis binds nothing: the split's extents are the author's.
        split = {co: self.channels // 3, ci: 3}
        return self.bound_schedule((r, co, ci), factors=split, extents=split)

    input, output = split_ports(schedule, (co, ci))


class LanesReversed(LanesInOrder):
    """The same module declared with its lane levels swapped: lane ``2 ci + co``."""

    id = "test.thresholding_axi.lanes_reversed"
    input, output = split_ports(LanesInOrder.schedule, (ci, co))


def split(space_type: type[LanesInOrder]) -> dict[str, Any]:
    return dict(
        thresholding(),
        space_type=space_type,
        outputs={"output_channel": tensor((PIXELS, CHANNELS), "UINT2")},
        factors=({"pe": CHANNELS},),
    )


# -- eltwise, transpose, memstream ----------------------------------------------------------


def eltwise() -> dict[str, Any]:
    """lhs + rhs, rhs a channel vector broadcast over the rows."""
    return dict(
        space_type=EltwiseKernel,
        inputs={
            "lhs_channel": tensor((PIXELS, CHANNELS), "INT4"),
            "rhs_channel": tensor((CHANNELS,), "INT4"),
        },
        outputs={"result_channel": (PIXELS, CHANNELS)},
        reference=lambda lhs_channel, rhs_channel: {"result_channel": lhs_channel + rhs_channel},
        factors=PE_FACTORS,
        facts=dict(
            operation="ADD",
            lhs_dtype=DataType["INT4"],
            rhs_dtype=DataType["INT4"],
            b_scale=1.0,
            platform=FULL_DSP58,
        ),
    )


MATRICES = (2, 6, 6)


def transpose() -> dict[str, Any]:
    """Rows in, columns out: the same tensor in another order, so the reference is identity.

    The stalled input (and a ``vpc``
    feeding the adapter sample) lets the output catch up with the input, which
    FinnLib's ``inner_shuffle`` survives only with its page-guard fix.
    """
    return dict(
        space_type=TransposeKernel,
        inputs={"input_channel": tensor(MATRICES, "INT4")},
        outputs={"output_channel": MATRICES},
        reference=lambda input_channel: {"output_channel": input_channel},
        factors=tuple({"simd": simd} for simd in (1, 3, 6)),
        choices={"ram_style": "auto"},
        facts={"platform": FULL_DSP48E2},
    )


STORED = (4, 6)
CONTENTS = tuple(tuple((7 * row + 5 * col) % 16 - 8 for col in range(6)) for row in range(4))


def memstream() -> dict[str, Any]:
    """The stored operand streamed in each consumer form: identity against its contents."""
    return dict(
        space_type=MemStreamKernel,
        inputs={},
        outputs={"output_channel": STORED},
        reference=lambda: {"output_channel": np.array(CONTENTS)},
        factors=(
            *({"form": vector_major(STORED, lanes)} for lanes in (1, 3, 6)),
            {"form": tile(*STORED, 2, 3)},
        ),
        choices={"ram_style": "auto", "pumped_memory": False},
        facts={"dtype": DataType["INT4"], "contents": CONTENTS, "platform": FULL_DSP48E2},
    )


# -- the channel stages: fifo, vpc, input_gen ------------------------------------------------


def stage_ports(
    input_channel: Any, output_channel: Any, arriving: Any, leaving: Any, moved: Any
) -> tuple[AxiStreamPort, AxiStreamPort]:
    """A channel stage's native pins on the two channels the harness places, presenting the
    beat sequences its realization gives them (``arriving``, ``leaving``).

    On a channel a stage's ports are opaque words (``WordPort``) and its adapter states
    what they carry (``finn.kernels.adapters.realize``); here the ports sit on channels
    and present exactly that, so XSim checks that the RTL walks it.
    """
    input = AxiStreamPort(
        name="input",
        endpoint=Endpoint.TARGET,
        channel=input_channel,
        sequence=arriving,
        signals=("idat", "ivld", "irdy"),
        clock=NATIVE_CLOCKING.clock,
        reset=NATIVE_CLOCKING.reset,
    )
    output = AxiStreamPort(
        name="output",
        endpoint=Endpoint.INITIATOR,
        channel=output_channel,
        sequence=leaving,
        dtype=moved,
        signals=("odat", "ovld", "ordy"),
        clock=NATIVE_CLOCKING.clock,
        reset=NATIVE_CLOCKING.reset,
    )
    return input, output


class StagedFifo(FifoKernel):
    """FifoKernel between two channels: what arrives leaves unchanged."""

    id = "test.staged.fifo"
    input_channel: Channel = Param(required=False)
    output_channel: Channel = Param(required=False)
    arriving: BeatSequence = Param()
    leaving: BeatSequence = Param()

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def moved(self) -> QONNXDataType:
        return self.input_channel.tensor.element.dtype

    input, output = stage_ports(input_channel, output_channel, arriving, leaving, moved)


class StagedVpc(VpcKernel):
    """VpcKernel between two channels: a width conversion."""

    id = "test.staged.vpc"
    input_channel: Channel = Param(required=False)
    output_channel: Channel = Param(required=False)
    arriving: BeatSequence = Param()
    leaving: BeatSequence = Param()

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def moved(self) -> QONNXDataType:
        return self.input_channel.tensor.element.dtype

    input, output = stage_ports(input_channel, output_channel, arriving, leaving, moved)


class StagedInputGenerator(InputGeneratorKernel):
    """InputGeneratorKernel between two channels: a reorder. ``olst``, its loop-completion
    marker, is left unused: the harness's boundaries carry no marker. The dotp and MatMul
    cases check it, as the TLAST that frames their activations."""

    id = "test.staged.input_generator"
    input_channel: Channel = Param(required=False)
    output_channel: Channel = Param(required=False)
    arriving: BeatSequence = Param()
    leaving: BeatSequence = Param()

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def moved(self) -> QONNXDataType:
        return self.input_channel.tensor.element.dtype

    input, output = stage_ports(input_channel, output_channel, arriving, leaving, moved)

    def other_pins(self) -> tuple[Signal, ...]:
        return (Signal("olst", Direction.OUT, len(self.dims)),)

    def held(self) -> Held:
        return Held((), ("olst",))


STAGED = (4, 6)


def realized(source: Traversal, sink: Traversal) -> RealizedStage:
    """The one stage a channel's adapter realizes between ``source`` and ``sink``."""
    (stage,) = realize(plan(BeatSequence(source), BeatSequence(sink)))
    return stage


def staged(stage: RealizedStage) -> dict[str, object]:
    """Its two sequences; the markers its output guarantees are left unused here."""
    return {"arriving": stage.source, "leaving": BeatSequence(stage.sink.form)}


def fifo() -> dict[str, Any]:
    """The identity in each storage the RTL implements: shift, LUTRAM, block and UltraRAM."""

    def storing(lanes: int, depth: int, ram_style: str) -> dict[str, object]:
        same = BeatSequence(vector_major(STAGED, lanes))
        return {
            "word_bits": 4 * lanes,
            "depth": depth,
            "ram_style": ram_style,
            "arriving": same,
            "leaving": same,
        }

    return dict(
        space_type=StagedFifo,
        inputs={"input_channel": tensor(STAGED, "INT4")},
        outputs={"output_channel": STAGED},
        reference=lambda input_channel: {"output_channel": input_channel},
        factors=(
            storing(1, 2, "auto"),
            storing(3, 100, "auto"),
            storing(6, 600, "auto"),
            storing(2, 64, "ultra"),
        ),
        facts={"platform": FULL_DSP48E2},
    )


def vpc() -> dict[str, Any]:
    """The same elements in the same order, at another lane count."""

    def converting(lanes_in: int, lanes_out: int) -> dict[str, object]:
        stage = realized(vector_major(STAGED, lanes_in), vector_major(STAGED, lanes_out))
        assert stage.module == Convert(lanes_in, lanes_out)
        return {"lanes_in": lanes_in, "lanes_out": lanes_out, **staged(stage)}

    return dict(
        space_type=StagedVpc,
        inputs={"input_channel": tensor(STAGED, "INT4")},
        outputs={"output_channel": STAGED},
        reference=lambda input_channel: {"output_channel": input_channel},
        factors=(converting(1, 3), converting(2, 3), converting(3, 2), converting(6, 1)),
        facts={"element_bits": 4},
    )


def input_gen() -> dict[str, Any]:
    """A reorder as the channel's adapter realizes it: a transpose, word by word and two
    lanes a word, and each row replayed."""

    def generating(source: Traversal, sink: Traversal) -> dict[str, object]:
        stage = realized(source, sink)
        module = stage.module
        assert isinstance(module, Generate)
        return {
            "word_bits": 4 * source.lanes,
            "frame_words": module.frame,
            "dims": module.dims,
            "strides": module.coefs,
            **staged(stage),
        }

    rows, cols = STAGED
    return dict(
        space_type=StagedInputGenerator,
        inputs={"input_channel": tensor(STAGED, "INT4")},
        outputs={"output_channel": STAGED},
        reference=lambda input_channel: {"output_channel": input_channel},
        factors=(
            generating(vector_major(STAGED, 1), columns_first(rows, cols, 1)),
            generating(vector_major(STAGED, 2), columns_first(rows, cols, 2)),
            generating(vector_major(STAGED, 3), vector_major(STAGED, 3).replayed(2, inner_beats=2)),
        ),
        choices={"ram_style": "auto"},
        facts={"platform": FULL_DSP48E2},
    )


# -- MatMul: a kernel with children --------------------------------------------------------

MATMUL_FACTS = dict(
    m=ROWS,
    n=OUTPUTS,
    k=REDUCTION,
    activation_dtype=DataType["INT4"],
    weights_dtype=DataType["INT4"],
    platform=FULL_DSP48E2,
)


def matmul() -> dict[str, Any]:
    """Y = X @ W on the packed core, its weights streamed: its activations replayed and framed
    by the channel's adapter, as its core's port states; its result element its own view."""
    result = design_space(MatMulKernel(**MATMUL_FACTS)).result_tensor
    return dict(
        space_type=MatMulKernel,
        inputs={
            "x_channel": tensor((ROWS, REDUCTION), "INT4"),
            "w_channel": tensor((REDUCTION, OUTPUTS), "INT4"),
        },
        outputs={"y_channel": result},
        reference=lambda x_channel, w_channel: {"y_channel": x_channel @ w_channel},
        choices={
            "compute": "packed",
            "compute.packed.compute_pumping": False,
            "compute.packed.reducer": "tree",
        },
        facts=MATMUL_FACTS,
    )


CASES = {
    "dotp-packed": lambda: dotp(PackedDotpKernel, DspBlock.DSP48E2, 4, reducer="tree"),
    "dotp-packed-compressor": lambda: dotp(
        PackedDotpKernel, DspBlock.DSP48E2, 4, reducer="compressor"
    ),
    "dotp-int8": lambda: dotp(Int8Dsp58DotpKernel, DspBlock.DSP58, 8),
    "dotp-int8-depthwise": lambda: dotp(Int8Dsp58DotpKernel, DspBlock.DSP58, 8, Form.DEPTHWISE),
    "thresholding": thresholding,
    "thresholding-rows-first": lambda: scheduled(RowsFirst),
    "thresholding-lanes-in-order": lambda: split(LanesInOrder),
    "eltwise": eltwise,
    "transpose": transpose,
    "memstream": memstream,
    "fifo": fifo,
    "vpc": vpc,
    "input_gen": input_gen,
    "matmul": matmul,
}


def test_sampled_folding_factors_are_smallest_interior_largest_then_an_adapter() -> None:
    case = dotp(PackedDotpKernel, DspBlock.DSP48E2, 4, reducer="tree")
    del case["reference"]
    chosen = samples(**case)
    assert [(sample.label, dict(sample.factors), sample.adapter) for sample in chosen] == [
        ("smallest", {"pe": 1, "simd": 1}, False),
        ("interior", {"pe": 2, "simd": 3}, False),
        ("largest", {"pe": 4, "simd": 6}, False),
        ("adapter, interior", {"pe": 2, "simd": 3}, True),
    ]


def test_sampling_deduplicates_and_a_kernel_without_inputs_has_no_adapter_sample() -> None:
    case = memstream()
    del case["reference"]
    case["factors"] = (*case["factors"], case["factors"][0])
    assert [sample.adapter for sample in samples(**case)] == [False] * 4


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_kernel_conforms(case: str) -> None:
    conformance(**CASES[case]())


@requires_xsim
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_kernel_conforms_in_xsim(case: str, tmp_path: Path) -> None:
    conformance(**CASES[case](), xsim=tmp_path)


# A declared order the RTL does not walk: channel folds outer, or the lane levels swapped.
WRONG = {"loop-order": lambda: scheduled(ChannelsFirst), "lane-order": lambda: split(LanesReversed)}


@pytest.mark.parametrize("wrong", sorted(WRONG))
def test_a_wrong_order_passes_every_python_check(wrong: str) -> None:
    conformance(**WRONG[wrong]())


@pytest.mark.parametrize("wrong", sorted(WRONG))
def test_the_stimulus_tells_the_wrong_order_apart(wrong: str) -> None:
    """In every sample, the RTL walking its own order computes some other level.

    thresholding_axi applies the thresholds of the channel at each position of
    its own order (row-major, PE channels a beat) to whatever arrives there. A
    sample whose random values happened to give the same levels either way
    would pass XSim and prove nothing.
    """
    case = WRONG[wrong]()
    space_type, inputs, table = case["space_type"], case["inputs"], np.array(THRESHOLDS[0])
    for sample in samples(**{key: value for key, value in case.items() if key != "reference"}):
        values = _values(space_type, sample, inputs)
        placed = place(
            space_type,
            sample,
            inputs,
            case["outputs"],
            choices=case["choices"],
            facts=case["facts"],
            values=values,
        )
        x = values["input_channel"]
        declared = placed.input_channel.endpoints.sink.form
        walked = vector_major(declared.shape, sample.factors["pe"])
        arrive = [p for beat in declared.positions() for p in beat]
        applied = [p[1] for beat in walked.positions() for p in beat]
        differ = sum(
            int((x[p] >= table[c]).sum() != (x[p] >= table[p[1]]).sum())
            for p, c in zip(arrive, applied)
        )
        assert differ, f"{sample.label}: the values give every level either way"


@requires_xsim
@pytest.mark.parametrize("wrong", sorted(WRONG))
def test_a_wrong_order_fails_in_xsim(wrong: str, tmp_path: Path) -> None:
    case = WRONG[wrong]()
    with pytest.raises(NonConformance) as caught:
        conformance(**case, xsim=tmp_path)
    failed = {(sample.label, mode) for sample, mode, _ in caught.value.failures}
    # Every failure is an output word the RTL computed differently, not a build error.
    assert all("output_channel word" in message for _, _, message in caught.value.failures)
    chosen = samples(**{key: value for key, value in case.items() if key != "reference"})
    assert failed == {(sample.label, mode) for sample in chosen for mode in MODES}


def test_known_failures_are_strict() -> None:
    a, b = Sample("a", {}), Sample("b", {})
    _settle_known((a, b), [(a, "free", "word 1")], {("a", "free"): "a defect"})
    with pytest.raises(NonConformance, match="b \\(free\\)"):
        _settle_known((a, b), [(b, "free", "word 1")], {})
    with pytest.raises(AssertionError, match="now pass: a \\(free\\): a defect"):
        _settle_known((a, b), [], {("a", "free"): "a defect"})
    with pytest.raises(ValueError, match="name no simulation"):
        _settle_known((a,), [], {("c", "free"): "a defect"})


# -- the checks refuse ----------------------------------------------------------------------


class Misnamed(MemStreamKernel):
    """memstream_axi with its output bus misnamed: the source has no such pins."""

    id = "test.memstream_axi.misnamed"
    output = AxiStreamPort(
        name="m_axis_1",
        endpoint=Endpoint.INITIATOR,
        channel=MemStreamKernel.output_channel,
        sequence=MemStreamKernel.output_sequence,
        dtype=MemStreamKernel.dtype,
        clock="clk",
        reset="rst",
    )


class Unbound(MemStreamKernel):
    """memstream_axi without RAM_STYLE, which the module declares with a default."""

    id = "test.memstream_axi.unbound"

    def parameters(self) -> Mapping[str, int | str]:
        return {key: value for key, value in super().parameters().items() if key != "RAM_STYLE"}


class Unpipelined(ThresholdingAxiKernel):
    """thresholding_axi without DEEP_PIPELINE, which the module declares with a default.

    Checked at all because THRESHOLDS, an array, does not decline the module: the
    checker names it without a value. Omitting THRESHOLDS itself cannot show it: slang refuses the
    module's default for it (``'{default: ...}`` "invalid target type"), so
    that binding declines.
    """

    id = "test.thresholding_axi.unpipelined"

    def parameters(self) -> Mapping[str, int | str]:
        return {key: value for key, value in super().parameters().items() if key != "DEEP_PIPELINE"}


class Unscaled(EltwiseKernel):
    """eltwise without B_SCALE: a real, named by the checker but not valued."""

    id = "test.eltwise.unscaled"

    def parameters(self) -> Mapping[str, int | str]:
        return {key: value for key, value in super().parameters().items() if key != "B_SCALE"}


def test_a_module_whose_sources_contradict_its_pins_is_refused() -> None:
    with pytest.raises(AssertionError, match="memstream_axi refuses its ABI: .*m_axis_1_tdata"):
        conformance(**dict(memstream(), space_type=Misnamed))


@pytest.mark.parametrize(
    ("case", "space_type", "omitted"),
    [
        (memstream, Unbound, "RAM_STYLE"),
        # Modules with a parameter whose value the checker does not establish.
        (thresholding, Unpipelined, "DEEP_PIPELINE"),
        (eltwise, Unscaled, "B_SCALE"),
    ],
)
def test_parameters_must_name_every_module_parameter(
    case: Any, space_type: type[Any], omitted: str
) -> None:
    with pytest.raises(AssertionError, match=rf"omits \['{omitted}'\]"):
        conformance(**dict(case(), space_type=space_type))
