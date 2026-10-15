# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The LayerNorm kernel offline: its rows folded by SIMD, its FLOAT32 output, its
epsilon, its refusal of integers, and the HLS request it writes from its point; its
reference, and the tolerance against a model of the core's binary32 arithmetic. Its
top's C simulation (tier T1½) is marked ``vitis``: a check to run, which no gate decides
on; its cycles in XSim, measured against its statement.

The conformance cases (``kernels.specs.layernorm``) are simulated in XSim by
``test_conformance.py``, and its numeric sweep (``kernels.sweeps.layernorm_numeric``) by
the XSim sweep.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.core.rtlsim_bench import Receive, measure, stream_out
from finn.core.space import Available, Rejected, inspection
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import unreplayed
from finn.harness.hls import c_simulate
from finn.kernels.artifacts.contributions import HlsSource
from finn.kernels.artifacts.module import Leaf
from finn.kernels.layernorm import LayerNormKernel, binary32, float_literal
from finn.toolchain import hls
from finn.toolchain.launcher import ToolUnavailable, host_toolchain
from kernels.conformance import (
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
from kernels.specs.base import placed as kernel_placed
from kernels.specs.layernorm import (
    EPSILON,
    TOLERANCE,
    _mean,
    binary32_core,
    bounds,
    case,
    condition,
    layernorm,
    layernorm_epsilon,
    layernorm_rows,
)
from kernels.sweeps.layernorm_numeric import CASES, received, rows
from kernels.xsim import requires_hls

CHANNELS = ["input_channel", "output_channel"]
CONFORMANCE = {"layernorm_rows": layernorm_rows, "layernorm_epsilon": layernorm_epsilon}


def placed(given: dict[str, Any], simd: int) -> Any:
    sample = Sample(f"simd={simd}", {"simd": simd})
    values = _values(LayerNormKernel, sample, given["inputs"])
    point = place(
        LayerNormKernel,
        sample,
        given["inputs"],
        given["outputs"],
        facts=given["facts"],
        values=values,
    )
    return point, sample, values


def kernel_leaf(point: Any) -> Leaf:
    module = design_module(point, LayerNormKernel)
    assert isinstance(module, Available)
    (leaf,) = (leaf for label, leaf in module.value.fragment.instances if label == KERNEL)
    assert isinstance(leaf, Leaf)
    return leaf


def request(point: Any) -> HlsSource:
    (found,) = kernel_leaf(point).sources
    assert isinstance(found, HlsSource)
    return found


def test_each_row_is_presented_simd_elements_a_beat_in_float32() -> None:
    point, sample, _ = placed(layernorm_rows(), 2)
    ends = _ends(point, LayerNormKernel, sample, CHANNELS)
    source, result = ends["input_channel"], ends["output_channel"]
    # Six rows of eight, four beats a row of two lanes.
    assert (source.form.beats, source.form.lanes) == (24, 2)
    assert (result.form.beats, result.form.lanes) == (24, 2)
    first = [position for beat in list(source.form.positions())[:4] for position in beat]
    assert first == [(0, 0, column) for column in range(8)]
    assert source.element == ScalarEncoding(DataType["FLOAT32"])
    assert result.element == ScalarEncoding(DataType["FLOAT32"])
    assert not source.markers and not result.markers
    kernel = design_member(point, KERNEL)
    assert kernel.output_tensor == Tensor((2, 3, 8), ScalarEncoding(DataType["FLOAT32"]))


def test_simd_divides_the_row() -> None:
    assert _candidates(unchosen(**layernorm_epsilon()), f"{KERNEL}.simd") == (1, 2, 3, 4, 6, 12)


def test_its_cycles_are_stated_as_a_beat_a_cycle() -> None:
    """The least a frame takes; the core's own rate is measured in XSim (below)."""
    for simd, beats in ((1, 48), (2, 24), (8, 6)):
        point, _, _ = placed(layernorm_rows(), simd)
        assert design_member(point, KERNEL).schedule.beat_count == beats
        assert design_cycles(point) == Available(beats), simd


def test_its_request_is_written_from_the_point() -> None:
    given = layernorm_rows()
    first, second = (request(placed(given, simd)[0]) for simd in (1, 2))
    assert first.function.startswith("finn_layernorm_")
    assert first.function != second.function
    assert request(placed(given, 2)[0]) == second  # deterministic
    assert "using Lanes = hls::vector<float, 2>;" in second.top
    assert f"layernorm<8>(src0, dst0, {float_literal(float(np.float32(EPSILON)))});" in second.top
    assert "using InBeat = ap_uint<64>;" in second.top
    assert "using OutBeat = ap_uint<64>;" in second.top
    assert [header.path for header in second.headers] == [
        "hls/nonlin/layernorm.hpp",
        "hls/util/util.hpp",
    ]
    assert second.period_ns == given["facts"]["platform"].period_ns
    leaf = kernel_leaf(placed(given, 2)[0])
    assert not leaf.parameters
    assert [pin.name for pin in leaf.abi.pins] == ["ap_clk", "ap_rst_n", "s_axis", "m_axis"]


def test_its_epsilon_is_the_tops_binary32_constant() -> None:
    """Two epsilons, two tops; the literal is the binary32 value, not the one given."""
    tops = {
        epsilon: request(placed(case((2, 8), epsilon=epsilon), 8)[0]).top
        for epsilon in (1e-5, 1e-3)
    }
    assert tops[1e-5] != tops[1e-3]
    assert "layernorm<8>(src0, dst0, 9.999999747378752e-06f);" in tops[1e-5]
    assert "layernorm<8>(src0, dst0, 0.0010000000474974513f);" in tops[1e-3]


def test_a_float_literal_rounds_back_to_its_binary32_value() -> None:
    for value in (1e-5, 1e-3, 0.0, 0.25, 2.0, 1e20, -1e-12, float(np.finfo(np.float32).max)):
        rounded = binary32(value)
        assert isinstance(rounded, float)
        literal = float_literal(rounded)
        assert literal.endswith("f") and ("." in literal or "e" in literal)
        assert (
            np.float32(literal[:-1]) == np.float32(rounded)
            and float(np.float32(rounded)) == rounded
        )


def test_an_epsilon_is_rounded_to_binary32_or_refused() -> None:
    assert binary32(1e-5) == float(np.float32(1e-5)) != 1e-5
    assert binary32(1e-50) == 0.0
    for value in (float("inf"), float("-inf"), float("nan"), 1e300):
        found = binary32(value)
        assert isinstance(found, Rejected)
        assert [finding.code for finding in found.findings] == ["layernorm-epsilon"]


def test_an_integer_input_is_refused_for_its_mean_stage() -> None:
    found = inspection.admission(kernel_placed(case((2, 8), "INT8")))
    assert isinstance(found, Rejected)
    (finding,) = found.findings
    assert finding.code == "dtype-family"
    assert "FLOAT32 only" in finding.message and "overflows" in finding.message
    assert not isinstance(inspection.admission(kernel_placed(layernorm_rows())), Rejected)


def test_a_float_first_input_has_no_adapter_sample() -> None:
    given = layernorm_rows()
    chosen = samples(
        LayerNormKernel, inputs=given["inputs"], outputs=given["outputs"], facts=given["facts"]
    )
    assert [sample.label for sample in chosen] == ["smallest", "interior", "largest"]


# -- the reference and the tolerance ------------------------------------------------------------


def test_the_reference_normalizes_each_row() -> None:
    rng = np.random.default_rng(0)
    x = (3 + rng.standard_normal((4, 64)) * 2).astype(np.float32)
    found = layernorm(x, EPSILON)
    assert np.allclose(found.mean(axis=-1), 0.0, atol=1e-12)
    assert np.allclose(found.var(axis=-1), 1.0, atol=1e-5)
    assert (layernorm(np.full((1, 8), 0.1, np.float32), EPSILON) == 0.0).all()


def test_epsilon_is_read_as_binary32() -> None:
    x = np.array([[0.0, 2.0**-10]], dtype=np.float32)  # centred +-2**-11, variance 2**-22
    found = layernorm(x, 1e-5)[0, 1]
    assert np.isclose(found, 2.0**-11 / np.sqrt(2.0**-22 + float(np.float32(1e-5))), 0, 1e-16)
    assert not np.isclose(found, 2.0**-11 / np.sqrt(2.0**-22 + 1e-5), 0, 1e-12)


def test_epsilon_decides_the_near_constant_rows() -> None:
    """Each near-constant row's results with its epsilon and without are further apart
    than a hundred times its bound: the sweep tells an ignored epsilon apart."""
    (near,) = (each for each in CASES if each.label == "near_constant")
    x = rows(near, np.random.default_rng(zlib.crc32(near.label.encode())))
    apart = np.abs(layernorm(x, near.epsilon) - np.nan_to_num(layernorm(x, 0.0)))
    for row in range(3, 7):
        assert apart[row].max() > 100 * bounds(x, near.epsilon)[row].max(), row


def test_the_bound_grows_with_a_rows_offset_against_its_spread() -> None:
    x = np.array([[1.0, -1.0] * 4, [101.0, 99.0] * 4, [0.0] * 8], dtype=np.float32)
    found = condition(x, 0.0)[:, 0]
    assert found[0] == 1.0 and np.isclose(found[1], np.sqrt(10001.0)) and found[2] == 1.0
    assert bounds(x, 0.0).shape == x.shape


def _model_errors(x: Any, simd: int, epsilon: float) -> Any:
    """The model's error at each result, over its bound."""
    error = np.abs(binary32_core(x, simd, epsilon).astype(np.float64) - layernorm(x, epsilon))
    return error / bounds(x, epsilon)


def test_the_tolerance_is_ten_times_what_the_cores_arithmetic_reaches_on_the_sweep() -> None:
    """The model of the core's binary32 arithmetic stays within a tenth of each bound on
    every sweep row (``kernels.specs.layernorm``: the tolerance's grounds)."""
    for each in CASES:
        x = rows(each, np.random.default_rng(zlib.crc32(each.label.encode())))
        assert _model_errors(x, each.simd, each.epsilon).max() < 0.1, each.label


@pytest.mark.parametrize("name", sorted(CONFORMANCE))
def test_the_tolerance_is_ten_times_what_the_cores_arithmetic_reaches_on_a_case(
    name: str,
) -> None:
    """The conformance harness compares within ``TOLERANCE`` alone: its rows' error, as
    the model reaches it at every SIMD the case samples, stays within a tenth of it."""
    given = CONFORMANCE[name]()
    epsilon = given["facts"]["epsilon"]
    chosen = samples(
        LayerNormKernel, inputs=given["inputs"], outputs=given["outputs"], facts=given["facts"]
    )
    for sample in chosen:
        x = _values(LayerNormKernel, sample, given["inputs"])["input_channel"]
        simd = sample.factors["simd"]
        assert isinstance(simd, int)
        error = np.abs(binary32_core(x, simd, epsilon) - layernorm(x, epsilon))
        assert error.max() < TOLERANCE / 10, sample.label


def test_the_model_adds_as_the_core_does() -> None:
    """A beat's lanes as a tree, the beats in order: of 1, 0, 2**-24 and 2**-24, one lane a
    beat loses each 2**-24 to 1 (a tie, to even); four lanes a beat add them as a pair
    first, which 1 keeps."""
    half = 2.0**-24
    x = np.array([[1.0, 0.0, half, half]], dtype=np.float32)
    assert _mean(x, 1)[0, 0] == np.float32(0.25)
    assert _mean(x, 4)[0, 0] == np.float32((1 + 2 * half) / 4)
    assert _mean(x, 2)[0, 0] == np.float32((1 + 2 * half) / 4)  # (1 + 0) + (half + half)


def test_floats_beyond_the_tolerance_are_named() -> None:
    point, sample, values = placed(layernorm_rows(), 8)
    form = _ends(point, LayerNormKernel, sample, CHANNELS)["output_channel"].form
    expected = layernorm(values["input_channel"], EPSILON)
    element = ScalarEncoding(DataType["FLOAT32"])
    exact = _words(form, expected, element)[0]
    assert _beyond(form, exact, expected, TOLERANCE) is None
    far = _words(form, expected + 1e-3, element)[0]
    found = _beyond(form, far, expected, TOLERANCE)
    assert found is not None and found.startswith("48 lanes beyond 1e-05")


# -- with the toolchain -------------------------------------------------------------------------


def _c_simulator() -> bool:
    toolchain = host_toolchain()
    try:
        hls.installation(toolchain)
        toolchain.command("g++")
    except ToolUnavailable:
        return False
    return True


@pytest.mark.vitis
@pytest.mark.skipif(not _c_simulator(), reason="no HLS installation's headers or g++")
@pytest.mark.parametrize("name", sorted(CONFORMANCE))
def test_the_top_computes_the_case_in_c_simulation(name: str, tmp_path: Path) -> None:
    given = CONFORMANCE[name]()
    point, sample, values = placed(given, 2)
    ends = _ends(point, LayerNormKernel, sample, CHANNELS)
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
    expected = layernorm(x, given["facts"]["epsilon"])
    assert np.allclose(received(result.form, tuple(found["m_axis"])), expected, 0, TOLERANCE)


@requires_hls
def test_it_takes_at_least_a_beat_a_cycle_as_it_states(tmp_path: Path) -> None:
    """Frames back to back, never stalled: the input takes a frame's beats in no fewer
    cycles than the kernel states (a beat a cycle). The core's own interval is printed:
    its binary32 accumulations and its stages' wait for each row's statistic may make it
    longer, which the statement does not yet reach. The output's words are taken first as
    they arrive (and checked within the tolerance), then measured."""
    point, sample, values = placed(layernorm_rows(), 2)
    ends = _ends(point, LayerNormKernel, sample, CHANNELS)
    source, result = ends["input_channel"], ends["output_channel"]
    x = values["input_channel"]
    module = design_module(point, LayerNormKernel)
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
    found = received(result.form, tuple(words))
    assert np.allclose(found, layernorm(x, EPSILON), rtol=0, atol=TOLERANCE)
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
    assert measured.per_frame(core) == 24
    interval = measured.interval()
    print(f"layernorm: {interval} cycles a frame of 24 beats in 6 rows; stated {cycles.value}")
    assert cycles.value == 24 and interval >= cycles.value
