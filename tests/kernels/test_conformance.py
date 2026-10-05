# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every kernel on streams, checked against its RTL by the conformance harness.

Each case places one kernel between boundary streams over sampled folding factors
(``kernels.conformance``): dotp on both cores (packed; INT8, dense and
depthwise), thresholding, eltwise with a broadcast operand, transpose, and
memstream with an identity reference. Every folding factor is a Decision (dotp's PE and
SIMD sampled; thresholding's and eltwise's PE and transpose's SIMD given as
configurations); memstream's ``form``, a fact of its consumer, is given.

The harness's reason to exist: a thresholding kernel that declares an order
its RTL does not walk passes every Python check and fails in XSim, for the
loop order (channel folds outer, rows inner, where the RTL walks rows outer)
and for the lane order (the two levels of a split channel index swapped). The
same kernels declaring the RTL's orders pass both.

The checks refuse: a module whose pins its sources contradict, and a kernel
whose ``parameters()`` omit a module parameter, including for modules with a
parameter whose value the RTL checker does not establish (thresholding's
array, eltwise's real, which is itself the omission checked).

transpose needs FinnLib with the ``inner_shuffle`` page-guard fix (pinned at
``99d75e8``; ``finn.kernels.transpose``): at ``d03f2fc`` its samples at SIMD 3
and 6 failed stalled, and its adapter sample failed in both modes.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Rejected, derived
from finn.dataflow.gemm import Form
from finn.dataflow.schedule import Index, Schedule
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import tile, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.dotp import Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.matmul import exact_result_dtype
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.port import AxiStreamPort
from finn.kernels.target import DspBlock
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.transpose import TransposeKernel
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

    def reference(x_stream: np.ndarray, w_stream: np.ndarray) -> dict[str, np.ndarray]:
        y = (x_stream * w_stream).sum(axis=1) if depthwise else x_stream @ w_stream
        return {"y_stream": y}

    return dict(
        space_type=space_type,
        inputs={
            "x_stream": Tensor(x_shape, ScalarEncoding(a)),
            "w_stream": Tensor((reduction, OUTPUTS), ScalarEncoding(w)),
        },
        outputs={"y_stream": (ROWS, OUTPUTS)},
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
        inputs={"input_stream": tensor((PIXELS, CHANNELS), "INT4")},
        outputs={"output_stream": (PIXELS, CHANNELS)},
        reference=lambda input_stream: {"output_stream": levels(input_stream)},
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
        outputs={"output_stream": tensor((PIXELS, CHANNELS), "UINT2")},
        factors=({"pe": 1}, {"pe": 2}, {"pe": 3}),
    )


# -- the lane-order proof: every channel in one beat, the channel index split ---------------

co, ci = Index("co"), Index("ci")


def split_ports(schedule: Any, lanes: tuple[Index, ...]) -> tuple[AxiStreamPort, AxiStreamPort]:
    """Input and output reading channel ``c = 3 co + ci``, lanes in ``lanes`` order."""
    input = AxiStreamPort(
        name="s_axis",
        endpoint=Endpoint.TARGET,
        stream=ThresholdingAxiKernel.input_stream,
        schedule=schedule,
        index=(r, co * 3 + ci),
        lanes=lanes,
        dtype=ThresholdingAxiKernel.input_dtype,
    )
    output = AxiStreamPort(
        name="m_axis",
        endpoint=Endpoint.INITIATOR,
        stream=ThresholdingAxiKernel.output_stream,
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
        outputs={"output_stream": tensor((PIXELS, CHANNELS), "UINT2")},
        factors=({"pe": CHANNELS},),
    )


# -- eltwise, transpose, memstream ----------------------------------------------------------


def eltwise() -> dict[str, Any]:
    """lhs + rhs, rhs a channel vector broadcast over the rows."""
    return dict(
        space_type=EltwiseKernel,
        inputs={
            "lhs_stream": tensor((PIXELS, CHANNELS), "INT4"),
            "rhs_stream": tensor((CHANNELS,), "INT4"),
        },
        outputs={"result_stream": (PIXELS, CHANNELS)},
        reference=lambda lhs_stream, rhs_stream: {"result_stream": lhs_stream + rhs_stream},
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
        inputs={"input_stream": tensor(MATRICES, "INT4")},
        outputs={"output_stream": MATRICES},
        reference=lambda input_stream: {"output_stream": input_stream},
        factors=tuple({"simd": simd} for simd in (1, 3, 6)),
        choices={"ram_style": "auto"},
    )


STORED = (4, 6)
CONTENTS = tuple(tuple((7 * row + 5 * col) % 16 - 8 for col in range(6)) for row in range(4))


def memstream() -> dict[str, Any]:
    """The stored operand streamed in each consumer form: identity against its contents."""
    return dict(
        space_type=MemStreamKernel,
        inputs={},
        outputs={"output_stream": STORED},
        reference=lambda: {"output_stream": np.array(CONTENTS)},
        factors=(
            *({"form": vector_major(STORED, lanes)} for lanes in (1, 3, 6)),
            {"form": tile(*STORED, 2, 3)},
        ),
        choices={"ram_style": "auto", "pumped_memory": False},
        facts={"dtype": DataType["INT4"], "contents": CONTENTS, "platform": FULL_DSP48E2},
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
        x = values["input_stream"]
        declared = placed.input_stream.endpoints.sink.form
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
    assert all("output_stream word" in message for _, _, message in caught.value.failures)
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
        stream=MemStreamKernel.output_stream,
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

    Checked at all only because THRESHOLDS, an array, no longer declines the
    module. Omitting THRESHOLDS itself cannot show it: slang refuses the
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
