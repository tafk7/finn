# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Softmax kernel offline: its rows folded by SIMD, its FLOAT32 output, its cycles and
the HLS request it writes from its point; its reference; the conformance harness's
float extension. Its top's C simulation (tier T1½) is marked ``vitis``: a check to run,
which no gate decides on; its cycles in XSim, measured against its statement.

The conformance cases (``kernels.specs.softmax``) are simulated in XSim by
``test_conformance.py``, and its numeric sweep (``kernels.sweeps.softmax_numeric``) by
the XSim sweep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from qonnx.core.datatype import DataType

from finn.core.rtlsim_bench import Receive, measure, stream_out
from finn.core.space import Available
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import pack, unreplayed, vector_major
from finn.harness.hls import c_simulate
from finn.kernels.artifacts.contributions import HlsSource
from finn.kernels.artifacts.module import Leaf
from finn.kernels.softmax import FLOAT32, SoftmaxKernel, lane_type
from finn.toolchain import hls
from finn.toolchain.launcher import ToolUnavailable, host_toolchain
from kernels.conformance import (
    FLOAT_BOUND,
    KERNEL,
    Sample,
    _beyond,
    _candidates,
    _ends,
    _values,
    _words,
    design_cycles,
    design_member,
    design_module,
    place,
    samples,
    unchosen,
)
from kernels.specs.softmax import TOLERANCE, case, softmax, softmax_float, softmax_int8
from kernels.sweeps.softmax_numeric import received
from kernels.xsim import requires_hls

CHANNELS = ["input_channel", "output_channel"]


def placed(given: dict[str, Any], simd: int) -> Any:
    sample = Sample(f"simd={simd}", {"simd": simd})
    values = _values(SoftmaxKernel, sample, given["inputs"])
    point = place(
        SoftmaxKernel,
        sample,
        given["inputs"],
        given["outputs"],
        facts=given["facts"],
        values=values,
    )
    return point, sample, values


def kernel_leaf(point: Any) -> Leaf:
    module = design_module(point, SoftmaxKernel)
    assert isinstance(module, Available)
    (leaf,) = (leaf for label, leaf in module.value.fragment.instances if label == KERNEL)
    assert isinstance(leaf, Leaf)
    return leaf


def request(point: Any) -> HlsSource:
    (found,) = kernel_leaf(point).sources
    assert isinstance(found, HlsSource)
    return found


def test_each_row_is_presented_simd_elements_a_beat_in_float32() -> None:
    point, sample, _ = placed(softmax_int8(), 4)
    ends = _ends(point, SoftmaxKernel, sample, CHANNELS)
    source, result = ends["input_channel"], ends["output_channel"]
    # Six rows of eight, two beats a row of four lanes.
    assert (source.form.beats, source.form.lanes) == (12, 4)
    assert (result.form.beats, result.form.lanes) == (12, 4)
    first = [position for beat in list(source.form.positions())[:2] for position in beat]
    assert first == [(0, 0, column) for column in range(8)]
    assert source.element == ScalarEncoding(DataType["INT8"])
    assert result.element == ScalarEncoding(DataType["FLOAT32"])
    assert not source.markers and not result.markers
    kernel = design_member(point, KERNEL)
    assert kernel.output_tensor == Tensor((2, 3, 8), ScalarEncoding(DataType["FLOAT32"]))


def test_simd_divides_the_row() -> None:
    assert _candidates(unchosen(**softmax_float()), f"{KERNEL}.simd") == (1, 2, 3, 6)


def test_its_cycles_are_a_beat_a_cycle_and_a_cycle_a_row() -> None:
    """The exponentiation stage takes each row's maximum in a cycle of its own."""
    for simd, beats in ((1, 48), (4, 12), (8, 6)):
        point, _, _ = placed(softmax_int8(), simd)
        assert design_member(point, KERNEL).schedule.beat_count == beats
        assert design_cycles(point) == Available(beats + 6), simd


def test_its_request_is_written_from_the_point() -> None:
    given = softmax_int8()
    first, fourth = (request(placed(given, simd)[0]) for simd in (1, 4))
    assert first.function.startswith("finn_softmax_")
    assert first.function != fourth.function
    assert request(placed(given, 4)[0]) == fourth  # deterministic
    assert "using Element = ap_int<8>;" in fourth.top
    assert "static SoftMax<Element, float, 8, 4>  core;" in fourth.top
    assert "using InBeat = ap_uint<32>;" in fourth.top
    assert "using OutBeat = ap_uint<128>;" in fourth.top
    assert [header.path for header in fourth.headers] == [
        "hls/nonlin/softmax.hpp",
        "hls/util/util.hpp",
    ]
    assert fourth.period_ns == given["facts"]["platform"].period_ns
    leaf = kernel_leaf(placed(given, 4)[0])
    assert not leaf.parameters
    assert [pin.name for pin in leaf.abi.pins] == ["ap_clk", "ap_rst_n", "s_axis", "m_axis"]


def test_a_float_lane_is_read_as_its_bits() -> None:
    top = request(placed(softmax_float(), 3)[0]).top
    assert "using Element = float;" in top
    assert "lane.i = ap_uint<32>(beat.range(32*i + 31, 32*i)).to_uint();" in top
    assert "using InBeat = ap_uint<96>;" in top


def test_lanes_that_fill_no_whole_byte_are_read_from_a_padded_beat() -> None:
    point, _, _ = placed(case((2, 6), "INT4"), 3)
    top = request(point).top
    assert "using InBeat = ap_uint<16>;" in top
    assert "lanes[i] = Element(ap_uint<4>(beat.range(4*i + 3, 4*i)));" in top


def test_lane_types() -> None:
    assert lane_type(FLOAT32) == "float"
    assert lane_type(DataType["INT4"]) == "ap_int<4>"
    assert lane_type(DataType["UINT8"]) == "ap_uint<8>"
    assert lane_type(DataType["BINARY"]) == "ap_uint<1>"


# -- the reference --------------------------------------------------------------------------


def test_the_reference_normalizes_each_row() -> None:
    x = np.array([[0.0, 1.0, 2.0], [5.0, 5.0, 5.0]], dtype=np.float32)
    found = softmax(x)
    assert np.allclose(found.sum(axis=-1), 1.0)
    assert np.allclose(found[1], 1 / 3)
    powers = np.exp([0.0, 1.0, 2.0])
    assert np.allclose(found[0], powers / powers.sum())


def test_infinities_share_one_and_negative_infinities_take_none() -> None:
    inf = np.inf
    x = np.array([[inf, 1, inf, -inf], [-inf, 0, 1, -inf], [inf, inf, inf, inf]], np.float32)
    found = softmax(x)
    assert found[0].tolist() == [0.5, 0.0, 0.5, 0.0]
    assert found[1, 0] == found[1, 3] == 0.0 and np.isclose(found[1, 1:3].sum(), 1.0)
    assert found[2].tolist() == [0.25] * 4
    assert np.isnan(softmax(np.full((1, 3), -inf))).all()


def test_an_integer_is_read_as_binary32_first() -> None:
    """2**31 - 1 and 2**31 - 2 are one binary32: they share as equals."""
    found = softmax(np.array([[2**31 - 1, 2**31 - 2]], dtype=np.int64))
    assert found.tolist() == [[0.5, 0.5]]


# -- the harness's float extension ------------------------------------------------------------


def test_a_float_input_is_drawn_as_binary32_in_its_bound() -> None:
    inputs = softmax_float()["inputs"]
    values = _values(SoftmaxKernel, Sample("any", {}), inputs)["input_channel"]
    assert values.dtype == np.float32 and values.shape == (4, 6)
    assert np.abs(values).max() <= FLOAT_BOUND and len(np.unique(values)) == values.size


def test_a_float_is_streamed_as_its_bits() -> None:
    form = vector_major((2, 2), 2)
    element = ScalarEncoding(DataType["FLOAT32"])
    words, bits = _words(form, np.array([[1.0, -2.0], [0.5, np.inf]]), element)
    assert bits == 64
    assert words == [0xC0000000_3F800000, 0x7F800000_3F000000]


def test_a_float_first_input_has_no_adapter_sample() -> None:
    given = softmax_float()
    chosen = samples(
        SoftmaxKernel, inputs=given["inputs"], outputs=given["outputs"], facts=given["facts"]
    )
    assert [sample.label for sample in chosen] == ["smallest", "interior", "largest"]
    given = softmax_int8()
    chosen = samples(
        SoftmaxKernel, inputs=given["inputs"], outputs=given["outputs"], facts=given["facts"]
    )
    assert chosen[-1].adapter and chosen[-1].factors == {"simd": 4}


def test_floats_that_differ_within_the_tolerance_pass_and_others_are_named() -> None:
    form = vector_major((2, 2), 2)
    expected = np.array([[0.25, 0.75], [0.5, 0.5]])
    element = ScalarEncoding(DataType["FLOAT32"])

    def words(found: list[list[float]]) -> list[int]:
        return _words(form, np.array(found), element)[0]

    assert _beyond(form, words([[0.25, 0.75], [0.5, 0.5]]), expected, TOLERANCE) is None
    near = words([[0.25 + TOLERANCE / 2, 0.75], [0.5, 0.5 - TOLERANCE / 2]])
    assert _beyond(form, near, expected, TOLERANCE) is None
    far = _beyond(form, words([[0.25, 0.75], [0.5, 0.501]]), expected, TOLERANCE)
    assert far is not None and far.startswith("1 lanes beyond 1e-05")
    assert "beat 1 lane 1" in far
    nan = _beyond(form, words([[np.nan, 0.75], [0.5, 0.5]]), expected, TOLERANCE)
    assert nan is not None and "beat 0 lane 0" in nan
    assert _beyond(form, words([[0.25, 0.75], [0.5, 0.5]])[:1], expected, TOLERANCE) == (
        "1 words of 2"
    )


# -- with the toolchain -------------------------------------------------------------------------


def _c_simulator() -> bool:
    toolchain = host_toolchain()
    try:
        hls.installation(toolchain)
        toolchain.command("g++")
    except ToolUnavailable:
        return False
    return True


def _found(form: Any, words: Any) -> npt.NDArray[np.float64]:
    return received(form, tuple(words))


@pytest.mark.vitis
@pytest.mark.skipif(not _c_simulator(), reason="no HLS installation's headers or g++")
@pytest.mark.parametrize("name", ["softmax_int8", "softmax_float"])
def test_the_top_computes_the_case_in_c_simulation(name: str, tmp_path: Path) -> None:
    given = {"softmax_int8": softmax_int8, "softmax_float": softmax_float}[name]()
    point, sample, values = placed(given, 2)
    ends = _ends(point, SoftmaxKernel, sample, CHANNELS)
    source, result = ends["input_channel"], ends["output_channel"]
    x = values["input_channel"]
    found = c_simulate(
        kernel_leaf(point),
        tmp_path,
        inputs={"s_axis": _words(unreplayed(source.form), x, source.element)},
        lasts={},
        outputs={"m_axis": result.form.beats},
        toolchain=host_toolchain(),
    )
    assert len(found["m_axis"]) == result.form.beats
    assert np.allclose(_found(result.form, found["m_axis"]), softmax(x), rtol=0, atol=TOLERANCE)


@requires_hls
def test_it_takes_a_beat_a_cycle_and_a_cycle_a_row_as_it_states(tmp_path: Path) -> None:
    """Frames back to back, never stalled: the input takes its beats at the rate the
    kernel states, a cycle a row more than its beats (``cycles``). The output's words are
    taken first as they arrive (and checked within the tolerance), then measured."""
    point, sample, values = placed(softmax_int8(), 4)
    ends = _ends(point, SoftmaxKernel, sample, CHANNELS)
    source, result = ends["input_channel"], ends["output_channel"]
    x = values["input_channel"]
    module = design_module(point, SoftmaxKernel)
    cycles = design_cycles(point)
    assert isinstance(module, Available) and isinstance(cycles, Available)
    driven = {"input_channel": _words(unreplayed(source.form), x, source.element)}
    bits = result.form.lanes * 32
    words = stream_out(
        module.value,
        tmp_path / "once",
        inputs=driven,
        outputs={"output_channel": Receive(result.form.beats, bits)},
        cycles=cycles.value,
        toolchain=host_toolchain(),
    )["output_channel"]
    found = _found(result.form, words)
    assert np.allclose(found, softmax(x), rtol=0, atol=TOLERANCE)
    measured = measure(
        module.value,
        tmp_path / "measured",
        inputs=driven,
        outputs={"output_channel": (list(words), bits)},
        frames=4,
        cycles=cycles.value,
        toolchain=host_toolchain(),
    )
    core = f"{KERNEL}.s_axis_TDATA"
    assert measured.per_frame(core) == 12
    assert measured.interval() == cycles.value == 18


def test_the_words_a_sweep_reads_back_are_its_tensor() -> None:
    form = vector_major((2, 2), 2)
    element = ScalarEncoding(DataType["FLOAT32"])
    x = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
    words = list(pack(form, x.view(np.uint32).astype(np.int64).ravel().tolist(), 32))
    assert words == _words(form, x, element)[0]
    assert _found(form, words).tolist() == x.tolist()
