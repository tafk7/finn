# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every kernel on streams, checked against its RTL by the conformance harness.

Each case places one kernel between boundary streams over sampled folds
(``kernels.conformance``): dotp on both cores (packed; INT8, dense and
depthwise), thresholding, eltwise with a broadcast operand, transpose, and
memstream with an identity reference. Folds that are still Params
(thresholding's and eltwise's ``pe``, transpose's SIMD inside ``input_form``,
memstream's ``form``) are given explicitly.

The harness's reason to exist: a thresholding kernel that declares its
channels outer and its rows inner, where the RTL walks rows outer, passes
every Python check and fails in XSim. The same kernel declaring the RTL's
order passes both.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.core.space import derived
from finn.dataflow.gemm import Form
from finn.dataflow.schedule import SCHEDULE, Index, Schedule
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import tile, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.dotp import Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.matmul import exact_result_dtype
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.port import GivenPort, ScheduledPort
from finn.kernels.target import DspBlock
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.transpose import TransposeKernel
from kernels.conformance import NonConformance, conformance, samples
from kernels.xsim import requires_xsim


def tensor(shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor(shape, ScalarEncoding(DataType[dtype]))


# -- dotp ----------------------------------------------------------------------------------

ROWS, REDUCTION, OUTPUTS = 2, 6, 4


def dotp(family: type[Any], dsp: DspBlock, bits: int, form: Form = Form.DENSE) -> dict[str, Any]:
    """Y = X @ W, or per channel (depthwise); the result element is the stream's until A6."""
    a = w = DataType[f"INT{bits}"]
    depthwise = form is Form.DEPTHWISE
    reduction = 3 if depthwise else REDUCTION
    x_shape = (ROWS, reduction, OUTPUTS) if depthwise else (ROWS, reduction)

    def reference(x_stream: np.ndarray, w_stream: np.ndarray) -> dict[str, np.ndarray]:
        y = (x_stream * w_stream).sum(axis=1) if depthwise else x_stream @ w_stream
        return {"y_stream": y}

    return dict(
        family=family,
        inputs={
            "x_stream": Tensor(x_shape, ScalarEncoding(a)),
            "w_stream": Tensor((reduction, OUTPUTS), ScalarEncoding(w)),
        },
        outputs={
            "y_stream": Tensor((ROWS, OUTPUTS), ScalarEncoding(exact_result_dtype(reduction, a, w)))
        },
        reference=reference,
        choices={"compute_pumping": False},
        facts={"target_dsp": dsp, "target_period_ns": 5.0, "form": form},
    )


# -- thresholding --------------------------------------------------------------------------

PIXELS, CHANNELS = 3, 6
# Three thresholds a channel, each channel's its own: a level names its channel's thresholds.
THRESHOLDS = (tuple((-6 + c, -1 + c, 1 + c) for c in range(CHANNELS)),)


def levels(values: np.ndarray) -> np.ndarray:
    return (values[..., None] >= np.array(THRESHOLDS[0])).sum(axis=-1)


THRESHOLDING_FACTS = dict(
    input_dtype=DataType["INT4"],
    threshold_dtype=DataType["INT4"],
    thresholds=THRESHOLDS,
    bias=0,
    depth_trigger_bram=0,
    depth_trigger_uram=0,
)
THRESHOLDING_CHOICES = {"use_axilite": False, "deep_pipeline": False}
PE_FOLDS = ({"pe": 1}, {"pe": 3}, {"pe": CHANNELS})


def thresholding() -> dict[str, Any]:
    return dict(
        family=ThresholdingAxiKernel,
        inputs={"input_stream": tensor((PIXELS, CHANNELS), "INT4")},
        outputs={"output_stream": (PIXELS, CHANNELS)},
        reference=lambda input_stream: {"output_stream": levels(input_stream)},
        folds=PE_FOLDS,
        choices=THRESHOLDING_CHOICES,
        facts=THRESHOLDING_FACTS,
    )


# -- the loop-order proof: thresholding on a schedule ---------------------------------------

r, c = Index("r"), Index("c")


class RowsFirst(ThresholdingAxiKernel):
    """thresholding_axi over a schedule of rows, then channels folded by PE: the RTL's order."""

    id = "test.thresholding_axi.rows_first"
    beats: ClassVar[tuple[Index, ...]] = (r, c)

    @derived(semantics=SCHEDULE)
    def schedule(self) -> Schedule:
        rows, channels = self.input_stream.tensor.shape
        return Schedule({r: rows, c: channels}, folds={c: self.pe}, beats=type(self).beats)

    input = ScheduledPort(
        name="s_axis",
        endpoint=Endpoint.TARGET,
        stream=ThresholdingAxiKernel.input_stream,
        schedule=schedule,
        index=(r, c),
        lanes=(c,),
    )
    output = ScheduledPort(
        name="m_axis",
        endpoint=Endpoint.INITIATOR,
        stream=ThresholdingAxiKernel.output_stream,
        schedule=schedule,
        index=(r, c),
        lanes=(c,),
    )


class ChannelsFirst(RowsFirst):
    """The same module declared with the wrong loop order: channel folds outer, rows inner."""

    id = "test.thresholding_axi.channels_first"
    beats: ClassVar[tuple[Index, ...]] = (c, r)


def scheduled(family: type[RowsFirst]) -> dict[str, Any]:
    """Two channel folds or more, so the two orders differ in every sample."""
    return dict(
        thresholding(),
        family=family,
        outputs={"output_stream": tensor((PIXELS, CHANNELS), "UINT2")},
        folds=({"pe": 1}, {"pe": 2}, {"pe": 3}),
    )


# -- eltwise, transpose, memstream ----------------------------------------------------------


def eltwise() -> dict[str, Any]:
    """lhs + rhs, rhs a channel vector broadcast over the rows."""
    return dict(
        family=EltwiseKernel,
        inputs={
            "lhs_stream": tensor((PIXELS, CHANNELS), "INT4"),
            "rhs_stream": tensor((CHANNELS,), "INT4"),
        },
        outputs={"result_stream": (PIXELS, CHANNELS)},
        reference=lambda lhs_stream, rhs_stream: {"result_stream": lhs_stream + rhs_stream},
        folds=PE_FOLDS,
        facts=dict(
            operation="ADD",
            lhs_dtype=DataType["INT4"],
            rhs_dtype=DataType["INT4"],
            b_scale=1.0,
            target_dsp=DspBlock.DSP58,
        ),
    )


MATRICES = (2, 6, 6)


def transpose() -> dict[str, Any]:
    """Rows in, columns out: the same tensor in another order, so the reference is identity.

    Its output stream is required, so the harness cannot read the element with it
    unplaced: the output is given a Tensor.
    """
    return dict(
        family=TransposeKernel,
        inputs={"input_stream": tensor(MATRICES, "INT4")},
        outputs={"output_stream": tensor(MATRICES, "INT4")},
        reference=lambda input_stream: {"output_stream": input_stream},
        folds=tuple({"input_form": vector_major(MATRICES, simd)} for simd in (1, 3, 6)),
        choices={"ram_style": "auto"},
    )


STORED = (4, 6)
CONTENTS = tuple(tuple((7 * row + 5 * col) % 16 - 8 for col in range(6)) for row in range(4))


def memstream() -> dict[str, Any]:
    """The stored operand streamed in each consumer form: identity against its contents."""
    return dict(
        family=MemStreamKernel,
        inputs={},
        outputs={"output_stream": STORED},
        reference=lambda: {"output_stream": np.array(CONTENTS)},
        folds=(
            *({"form": vector_major(STORED, lanes)} for lanes in (1, 3, 6)),
            {"form": tile(*STORED, 2, 3)},
        ),
        choices={"ram_style": "auto", "pumped_memory": False},
        facts={"dtype": DataType["INT4"], "contents": CONTENTS},
    )


CASES = {
    "dotp-packed": lambda: dotp(PackedDotpKernel, DspBlock.DSP48E2, 4),
    "dotp-int8": lambda: dotp(Int8Dsp58DotpKernel, DspBlock.DSP58, 8),
    "dotp-int8-depthwise": lambda: dotp(Int8Dsp58DotpKernel, DspBlock.DSP58, 8, Form.DEPTHWISE),
    "thresholding": thresholding,
    "thresholding-rows-first": lambda: scheduled(RowsFirst),
    "eltwise": eltwise,
    "transpose": transpose,
    "memstream": memstream,
}


def test_sampled_folds_are_smallest_interior_largest_then_an_adapter() -> None:
    case = dotp(PackedDotpKernel, DspBlock.DSP48E2, 4)
    del case["reference"]
    chosen = samples(**case)
    assert [(sample.label, dict(sample.folds), sample.adapter) for sample in chosen] == [
        ("smallest", {"pe": 1, "simd": 1}, False),
        ("interior", {"pe": 2, "simd": 3}, False),
        ("largest", {"pe": 4, "simd": 6}, False),
        ("adapter, interior", {"pe": 2, "simd": 3}, True),
    ]


def test_sampling_deduplicates_and_a_kernel_without_inputs_has_no_adapter_sample() -> None:
    case = memstream()
    del case["reference"]
    case["folds"] = (*case["folds"], case["folds"][0])
    assert [sample.adapter for sample in samples(**case)] == [False] * 4


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_kernel_conforms(case: str) -> None:
    conformance(**CASES[case]())


# FinnLib d03f2fc: inner_shuffle.sv:294 reads read_addr before line 309 declares it;
# xvlog refuses the file (VRFC 10-3380), as the RTL checker's slang does.
UNCOMPILED = pytest.mark.xfail(
    raises=NonConformance, strict=True, reason="inner_shuffle.sv does not compile in xvlog"
)


@requires_xsim
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, marks=UNCOMPILED) if case == "transpose" else case
        for case in sorted(CASES)
    ],
)
def test_the_kernel_conforms_in_xsim(case: str, tmp_path: Path) -> None:
    conformance(**CASES[case](), xsim=tmp_path)


def test_a_wrong_loop_order_passes_every_python_check() -> None:
    conformance(**scheduled(ChannelsFirst))


@requires_xsim
def test_a_wrong_loop_order_fails_in_xsim(tmp_path: Path) -> None:
    case = scheduled(ChannelsFirst)
    with pytest.raises(NonConformance) as caught:
        conformance(**case, xsim=tmp_path)
    failed = {(sample.label, mode) for sample, mode, _ in caught.value.failures}
    # Every failure is an output word the RTL computed differently, not a build error.
    assert all("output_stream word" in message for _, _, message in caught.value.failures)
    chosen = samples(**{key: value for key, value in case.items() if key != "reference"})
    assert failed == {(sample.label, mode) for sample in chosen for mode in ("free", "stalled")}


# -- the checks refuse ----------------------------------------------------------------------


class Misnamed(MemStreamKernel):
    """memstream_axi with its output bus misnamed: the source has no such pins."""

    id = "test.memstream_axi.misnamed"
    output = GivenPort(
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


def test_a_module_whose_sources_contradict_its_pins_is_refused() -> None:
    with pytest.raises(AssertionError, match="memstream_axi refuses its ABI: .*m_axis_1_tdata"):
        conformance(**dict(memstream(), family=Misnamed))


def test_parameters_must_name_every_module_parameter() -> None:
    with pytest.raises(AssertionError, match=r"omits \['RAM_STYLE'\]"):
        conformance(**dict(memstream(), family=Unbound))
