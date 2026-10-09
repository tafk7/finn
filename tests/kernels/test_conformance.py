# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every kernel on channels, checked against its RTL by the conformance harness.

Each case, stated by its kernel's spec (``kernels.specs``), places one kernel
between boundary channels over sampled folding factors (``kernels.conformance``):
dotp on both cores (packed; INT8, dense and depthwise), thresholding, eltwise
with a broadcast operand, transpose, memstream with an identity reference, the
channel stages (fifo, vpc and input_gen) and MatMul, a kernel with children.
Every folding factor is a Decision (dotp's and MatMul's PE and SIMD sampled;
thresholding's and eltwise's PE and transpose's SIMD given as configurations);
memstream's ``form``, a fact of its consumer, is given.

A channel stage's ports are opaque words, and the channel's adapter states what
they carry (``finn.kernels.adapters.realize``). Its case places the stage on
two channels (``kernels.specs.stages``'s ``StagedFifo``, ``StagedVpc``,
``StagedInputGenerator``: its module, its native pins, ports presenting what its
realization gives them), so XSim checks that the RTL walks the realization: a
FIFO in each storage its RTL implements, width conversions, and input_gen's
reorders (a transpose, and a replay). MatMul's activations are replayed and
framed by its channel's adapter.

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

The platform's capabilities, a few cases each (KT10): pumped compute (INT8 dotp
on DSP58, one interior configuration: pumping needs SIMD >= 2) and a pumped
memory (memstream), each simulated with ``ap_clk2x`` driven aligned with
``ap_clk``; and AXI-Lite, thresholding's runtime-writable rows, one shared row
and a row a channel over two channel folds of three lanes, each table written
through the bus as its kernel declares it before the streams start.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from finn.core.space import Rejected, derived
from finn.dataflow.schedule import Index, Schedule
from finn.dataflow.traversal import offsets, pack, vector_major
from finn.harness.orders import decode
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.port import AxiStreamPort
from finn.kernels.target import DspBlock
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernels.conformance import (
    MODES,
    NonConformance,
    Sample,
    _ends,
    _settle_known,
    _stream,
    _values,
    conformance,
    place,
    samples,
)
from kernels.specs import conformance_cases
from kernels.specs.dotp import dotp
from kernels.specs.eltwise import eltwise
from kernels.specs.memstream import memstream
from kernels.specs.thresholding import CHANNELS, PIXELS, THRESHOLDS, tensor, thresholding
from kernels.xsim import requires_xsim

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
        return self.bound_schedule(tuple(order), self.factors)


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
    def extents(self) -> dict[Index, int] | Rejected:
        # The window axis binds nothing: the split's extents are the author's, a row a channel.
        return self._bound({co: self.rows // 3, ci: 3})

    @derived
    def schedule(self) -> Schedule | Rejected:
        return self.bound_schedule((r, co, ci), factors={co: self.rows // 3, ci: 3})

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


# The kernels' cases (``kernels.specs``), and the harness's own: thresholding
# declaring the RTL's orders, beside the wrong orders below.
CASES = {
    **conformance_cases(),
    "thresholding-rows-first": lambda: scheduled(RowsFirst),
    "thresholding-lanes-in-order": lambda: split(LanesInOrder),
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


def rtl_decoded(case: dict[str, Any], sample: Sample) -> str:
    """The wrong order's words as the RTL computes them, walking its own order on both
    ports (row-major, PE channels a beat; ``test_the_stimulus_tells_the_wrong_order_apart``),
    decoded: the decoding names that order on both ports, and its message is returned."""
    space_type, inputs = case["space_type"], case["inputs"]
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
    names = ("input_channel", "output_channel")
    streams = {
        name: _stream(placed, name, ends.form, ends)
        for name in names
        for ends in [_ends(placed, space_type, sample, [name])[name]]
    }
    declared = streams["input_channel"].form
    walked = vector_major(declared.shape, sample.factors["pe"])
    assert walked != declared
    seen = np.zeros(declared.shape, dtype=np.int64).ravel()
    seen[offsets(walked)] = values["input_channel"].ravel()[offsets(declared)]
    levels = case["reference"](input_channel=seen.reshape(declared.shape))["output_channel"]
    words = pack(walked, levels.ravel().tolist(), streams["output_channel"].bits)
    decoded = decode(
        {"output_channel": streams["output_channel"]},
        {"output_channel": words},
        inputs={"input_channel": streams["input_channel"]},
        values=values,
        reference=lambda read: case["reference"](**read),
    )
    assert decoded is not None, sample.label
    assert dict(decoded.walked) == {name: walked for name in names}, sample.label
    return decoded.message


@pytest.mark.parametrize("wrong", sorted(WRONG))
def test_the_wrong_order_decodes_as_the_order_the_rtl_walks(wrong: str) -> None:
    """Offline: the words the RTL computes walking its own order decode as that order,
    named in the port's indices, in every sample."""
    case = WRONG[wrong]()
    for sample in samples(**{key: value for key, value in case.items() if key != "reference"}):
        message = rtl_decoded(case, sample)
        assert message.startswith("the words are another order's: input_channel: beat "), message
        assert "; output_channel: beat " in message


@requires_xsim
@pytest.mark.parametrize("wrong", sorted(WRONG))
def test_a_wrong_order_fails_in_xsim(wrong: str, tmp_path: Path) -> None:
    case = WRONG[wrong]()
    with pytest.raises(NonConformance) as caught:
        conformance(**case, xsim=tmp_path)
    failed = {(sample.label, mode) for sample, mode, _ in caught.value.failures}
    # Every failure is decoded as the order the RTL walks, not left a word that differs.
    for sample, mode, message in caught.value.failures:
        assert message.startswith(rtl_decoded(case, sample) + " | "), (sample.label, mode)
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
