# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The piecewise-polynomial activation kernel offline: its lanes folded by PE, its FLOAT32
ports, its cycles, the module it binds and its refusals; its reference, its tolerances and
what its numeric sweep covers. Its cycles in XSim, measured against its statement.

The conformance case (``support.kernel_specs.pwpolyf``) is simulated in XSim by
``test_conformance.py``, and its numeric sweep (``kernels.sweeps.pwpolyf_numeric``) by
the XSim sweep.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qonnx.core.datatype import DataType
from support.conformance import (
    KERNEL,
    Sample,
    _contracts,
    _possible,
    _values,
    _words,
    design_cycles,
    design_member,
    design_module,
    place,
    samples,
    unchosen,
)
from support.kernel_specs.base import placed as placed_case
from support.kernel_specs.pwpolyf import CLAMP, TOLERANCE, activation, case, clamped, pwpolyf_gelu
from support.xsim import requires_xsim

from finn.core.rtlsim_bench import Receive, measure, stream_out
from finn.core.space import Available
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import unreplayed
from finn.harness.points import rejected
from finn.kernels.artifacts.abi import Signal
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.module import Leaf
from finn.kernels.pwpolyf import FUNCTIONS, PwPolyKernel
from finn.toolchain.launcher import host_toolchain
from kernels.sweeps import pwpolyf_numeric

CHANNELS = ["input_channel", "output_channel"]


def placed(given: dict[str, Any], pe: int) -> Any:
    sample = Sample(f"pe={pe}", {"pe": pe})
    values = _values(PwPolyKernel, sample, given["inputs"])
    point = place(
        PwPolyKernel,
        sample,
        given["inputs"],
        given["outputs"],
        facts=given["facts"],
        values=values,
    )
    return point, sample, values


def kernel_leaf(point: Any) -> Leaf:
    module = design_module(point, PwPolyKernel)
    assert isinstance(module, Available)
    (leaf,) = (leaf for label, leaf in module.value.fragment.instances if label == KERNEL)
    assert isinstance(leaf, Leaf)
    return leaf


def test_each_row_is_presented_pe_elements_a_beat_in_float32() -> None:
    point, sample, _ = placed(pwpolyf_gelu(), 4)
    contracts = _contracts(point, PwPolyKernel, sample, CHANNELS)
    source, result = contracts["input_channel"], contracts["output_channel"]
    # Six rows of eight, two beats a row of four lanes.
    assert (source.form.beats, source.form.lanes) == (12, 4)
    assert (result.form.beats, result.form.lanes) == (12, 4)
    first = [position for beat in list(source.form.positions())[:2] for position in beat]
    assert first == [(0, 0, column) for column in range(8)]
    float32 = ScalarEncoding(DataType["FLOAT32"])
    assert source.element == float32 and result.element == float32
    assert not source.markers and not result.markers
    kernel = design_member(point, KERNEL)
    assert kernel.output_tensor == Tensor((2, 3, 8), float32)


def test_pe_divides_the_innermost_axis() -> None:
    assert _possible(unchosen(**pwpolyf_gelu()), (KERNEL, "pe")) == (1, 2, 4, 8)


def test_a_float_input_has_no_adapter_sample() -> None:
    given = pwpolyf_gelu()
    chosen = samples(
        PwPolyKernel,
        inputs=given["inputs"],
        outputs=given["outputs"],
        facts=given["facts"],
    )
    assert [sample.label for sample in chosen] == ["smallest", "interior", "largest"]


def test_its_cycles_are_its_beats() -> None:
    """Elementwise, a beat a cycle: its pace is exact."""
    for pe, beats in ((1, 48), (4, 12), (8, 6)):
        point, _, _ = placed(pwpolyf_gelu(), pe)
        assert design_member(point, KERNEL).schedule.beat_count == beats
        assert design_cycles(point) == Available(beats), pe


def test_its_module_takes_pe_and_the_function_on_native_pins() -> None:
    leaf = kernel_leaf(placed(case("tanh", (2, 3, 8)), 4)[0])
    assert dict(leaf.parameters) == {"PE": 4, "FUNC": '"tanh"'}
    widths = {pin.name: pin.width for pin in leaf.abi.pins if isinstance(pin, Signal)}
    assert widths["xdat"] == widths["ydat"] == 4 * 32
    assert {"clk", "rst", "xvld", "xrdy", "yvld", "yrdy"} <= set(widths)
    sources = [source for source in leaf.sources if isinstance(source, CopiedSource)]
    assert [(source.root, source.path) for source in sources] == [
        ("finnlib", "rtl/infra/fifo.sv"),
        ("finnlib", "rtl/nonlin/pwpolyf_pkg.sv"),
        ("finnlib", "rtl/nonlin/pwpolyf.sv"),
    ]


@pytest.mark.parametrize("function", FUNCTIONS)
def test_each_function_is_valid_on_dsp58(function: str) -> None:
    assert not rejected(placed_case(case(function, (2, 8))))


# -- the reference ----------------------------------------------------------------------------


def test_the_reference_is_each_function() -> None:
    x = np.array([-2.0, 0.0, 1.0, 3.0], dtype=np.float32)
    phi = np.array([0.5 * (1 + math.erf(v / math.sqrt(2))) for v in x.tolist()])
    sigmoid = 1 / (1 + np.exp(-x.astype(np.float64)))
    assert np.allclose(activation("gelu", x), x * phi)
    assert np.allclose(activation("silu", x), x * sigmoid)
    assert np.allclose(activation("sigmoid", x), sigmoid)
    assert np.allclose(activation("tanh", x), np.tanh(x.astype(np.float64)))
    assert activation("gelu", np.array([1.0]))[0] == pytest.approx(0.8413447460685429)
    with pytest.raises(ValueError, match="relu"):
        activation("relu", x)


def test_at_an_infinity_the_reference_is_the_limit() -> None:
    x = np.array([-np.inf, np.inf], dtype=np.float32)
    assert activation("gelu", x).tolist() == [0.0, np.inf]
    assert activation("silu", x).tolist() == [0.0, np.inf]
    assert activation("sigmoid", x).tolist() == [0.0, 1.0]
    assert activation("tanh", x).tolist() == [-1.0, 1.0]


def test_the_reference_reads_its_input_as_binary32() -> None:
    x = np.array([0.1], dtype=np.float64)
    assert activation("tanh", x)[0] == math.tanh(float(np.float32(0.1)))


def test_the_clamps_are_the_values_the_rtl_states() -> None:
    x = np.array([-np.inf, -CLAMP, -7.5, 0.0, 7.5, CLAMP, 1000.0, np.inf], dtype=np.float32)
    nan = np.nan
    assert np.array_equal(
        clamped("gelu", x), [0.0, 0.0, nan, nan, nan, 8.0, 1000.0, np.inf], equal_nan=True
    )
    assert np.array_equal(
        clamped("silu", x), [0.0, 0.0, nan, nan, nan, 8.0, 1000.0, np.inf], equal_nan=True
    )
    assert np.array_equal(
        clamped("sigmoid", x), [0.0, 0.0, nan, nan, nan, 1.0, 1.0, 1.0], equal_nan=True
    )
    assert np.array_equal(
        clamped("tanh", x), [-1.0, -1.0, nan, nan, nan, 1.0, 1.0, 1.0], equal_nan=True
    )


@pytest.mark.parametrize("function", FUNCTIONS)
def test_each_tolerance_covers_its_clamp(function: str) -> None:
    """At |x| = 8 the hardware gives the function's limit: the tolerance holds the gap."""
    x = np.array([-CLAMP, CLAMP], dtype=np.float32)
    gap = np.abs(activation(function, x) - clamped(function, x)).max()
    assert gap <= TOLERANCE[function], (function, gap)


def test_the_tolerances_tell_the_functions_apart() -> None:
    """A module computing another function fails: on the conformance case's random
    inputs, each pair differs by far more than both tolerances."""
    x = _values(PwPolyKernel, Sample("any", {}), pwpolyf_gelu()["inputs"])
    values = x["input_channel"]
    for first in FUNCTIONS:
        for second in FUNCTIONS:
            if first != second:
                apart = np.abs(activation(first, values) - activation(second, values)).max()
                assert apart > 10 * (TOLERANCE[first] + TOLERANCE[second]), (first, second)


# -- the numeric sweep -------------------------------------------------------------------------


def test_the_sweep_covers_each_function_and_pe_one_a_divisor_and_the_whole_axis() -> None:
    cases = pwpolyf_numeric.CASES
    assert sorted(case.function for case in cases) == sorted(FUNCTIONS)
    width = pwpolyf_numeric.SHAPE[-1]
    pes = {case.pe for case in cases}
    assert {1, width} <= pes and pes - {1, width}
    assert all(width % pe == 0 for pe in pes)


def test_the_sweep_streams_each_edge_its_neighbour_and_the_clamps() -> None:
    x = pwpolyf_numeric.values("gelu")
    assert x.dtype == np.float32 and x.shape == pwpolyf_numeric.SHAPE
    assert not np.isnan(x).any()
    found = set(x.ravel().tolist())
    edges = pwpolyf_numeric.edges()
    assert edges.size == 41 and edges[0] == 0.25 and edges[-1] == CLAMP
    for edge in edges.tolist():
        below = float(np.nextafter(np.float32(edge), np.float32(0)))
        assert {edge, -edge, below, -below} <= found, edge
    assert {np.inf, -np.inf, 1000.0, -1000.0} <= found
    assert np.signbit(x).any() and (x == 0).sum() == 2


def test_the_sweep_names_a_result_beyond_the_tolerance_and_a_wrong_clamp() -> None:
    sweep = pwpolyf_numeric.Case("sigmoid", pe=1)
    x = np.array([[0.0, 1.0, CLAMP, -CLAMP]], dtype=np.float32)
    exact = activation("sigmoid", x)
    stated = clamped("sigmoid", x)
    found = np.where(np.isnan(stated), exact, stated)
    assert pwpolyf_numeric.checked(sweep, x, found) <= TOLERANCE["sigmoid"]
    far = found.copy()
    far[0, 1] += 1e-2
    with pytest.raises(AssertionError, match="1 results beyond"):
        pwpolyf_numeric.checked(sweep, x, far)
    wrong = found.copy()
    wrong[0, 2] = exact[0, 2]  # within the tolerance, but not the RTL's clamp
    with pytest.raises(AssertionError, match="the clamp gives"):
        pwpolyf_numeric.checked(sweep, x, wrong)


# -- with the toolchain -------------------------------------------------------------------------


@requires_xsim
def test_it_takes_a_beat_a_cycle_as_it_states(tmp_path: Path) -> None:
    """Frames back to back, never stalled: the input takes its beats at the rate the
    kernel states (``cycles``). The output's words are taken first as they arrive (and
    checked within the tolerance), then measured."""
    point, sample, values = placed(pwpolyf_gelu(), 4)
    contracts = _contracts(point, PwPolyKernel, sample, CHANNELS)
    source, result = contracts["input_channel"], contracts["output_channel"]
    x = values["input_channel"]
    module = design_module(point, PwPolyKernel)
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
    found = pwpolyf_numeric.received(result.form, tuple(words))
    assert np.allclose(found, activation("gelu", x), rtol=0, atol=TOLERANCE["gelu"])
    measured = measure(
        module.value,
        tmp_path / "measured",
        inputs=driven,
        outputs={"output_channel": (list(words), bits)},
        frames=4,
        cycles=cycles.value,
        toolchain=host_toolchain(),
    )
    assert measured.per_frame(f"{KERNEL}.xdat") == 12
    assert measured.interval() == cycles.value == 12
