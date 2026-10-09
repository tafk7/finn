# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Pool kernel offline: its windows, their markers and its output's element, and the
HLS request it writes from its point; its top's C simulation (tier T1½), marked
``vitis``: a check to run, which no gate decides on; and its cycles in XSim, measured
against its schedule's.

The conformance case (``kernels.specs.pool``) is simulated in XSim by
``test_conformance.py``; the C simulation here runs the same case's interior sample,
fed the words and TLASTs the kernel's input port declares (what its channel's adapter
presents), and compares the output with the same reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.core.executors.xsim.rtl import measure
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import pack, unreplayed
from finn.harness.hls import c_simulate
from finn.kernels.artifacts.contributions import HlsSource
from finn.kernels.artifacts.module import Leaf
from finn.kernels.pool import PoolKernel, element_type
from finn.util.toolchain import machine_toolchain
from kernels.conformance import KERNEL, Sample, _ends, _values, _words, place
from kernels.specs.pool import max_pool, pool
from kernels.xsim import requires_hls


def placed(case: dict[str, Any], pe: int) -> Any:
    sample = Sample(f"pe={pe}", {"pe": pe})
    values = _values(PoolKernel, sample, case["inputs"])
    point = place(
        PoolKernel, sample, case["inputs"], case["outputs"], facts=case["facts"], values=values
    )
    return point, sample, values


def kernel_leaf(point: Any) -> Leaf:
    (leaf,) = (leaf for label, leaf in point.module.fragment.instances if label == KERNEL)
    assert isinstance(leaf, Leaf)
    return leaf


def test_the_input_presents_each_window_whole_and_closes_it() -> None:
    case = pool()
    point, sample, _ = placed(case, 3)
    ends = _ends(point, PoolKernel, sample, ["input_channel", "output_channel"])
    window = ends["input_channel"]
    # Two by three windows, two channel folds each, four positions a window, three lanes.
    assert (window.form.beats, window.form.lanes) == (2 * 3 * 2 * 4, 3)
    assert [marker.beats for _, marker in window.markers] == [4]
    first = [position for beat in list(window.form.positions())[:4] for position in beat]
    assert first == [
        (row, column, channel)
        for row, column in ((0, 0), (0, 1), (1, 0), (1, 1))
        for channel in range(3)
    ]
    output = ends["output_channel"]
    assert (output.form.beats, output.form.lanes) == (2 * 3 * 2, 3)
    assert output.element == ScalarEncoding(DataType["INT4"])
    assert point.query(type(point).cycles).value == 2 * 3 * 2 * 4


def test_a_strided_window_counts_the_windows_that_fit_and_drops_the_rest() -> None:
    case = pool(shape=(5, 7, 2), window=(2, 3), stride=(2, 2))
    assert case["outputs"]["output_channel"] == (2, 3, 2)
    point, sample, values = placed(case, 2)
    ends = _ends(point, PoolKernel, sample, ["input_channel"])
    assert ends["input_channel"].form.beats == 2 * 3 * 6
    reference = max_pool(values["input_channel"], (2, 3), (2, 2))
    assert reference.shape == (2, 3, 2)


def test_an_output_shape_other_than_the_windows_is_refused() -> None:
    case = dict(pool(), outputs={"output_channel": (3, 3, 6)})
    with pytest.raises(ValueError, match="oh is 2 \\(given\\) and 3 \\(output axis 0\\)"):
        placed(case, 1)


def test_its_request_is_written_from_the_point() -> None:
    case = pool()
    leaves = {pe: kernel_leaf(placed(case, pe)[0]) for pe in (1, 3)}
    requests = {pe: leaf.sources[0] for pe, leaf in leaves.items()}
    assert all(isinstance(request, HlsSource) for request in requests.values())
    first, third = requests[1], requests[3]
    assert isinstance(first, HlsSource) and isinstance(third, HlsSource)
    assert first.function.startswith("finn_pool_") and leaves[1].name == first.function
    assert first.function != third.function
    assert kernel_leaf(placed(case, 3)[0]).sources[0] == third  # deterministic
    assert "using Element = ap_int<4>;" in third.top
    assert "hls::vector<Element, 3>" in third.top
    assert "hls::axis_data<ap_uint<16>, AXIS_ENABLE_LAST>" in third.top
    assert third.period_ns == case["facts"]["platform"].period_ns
    assert not leaves[3].parameters
    pins = [pin.name for pin in leaves[3].abi.pins]
    assert pins == ["ap_clk", "ap_rst_n", "s_axis", "m_axis"]


def test_element_types_follow_signedness() -> None:
    assert element_type(DataType["INT4"]) == "ap_int<4>"
    assert element_type(DataType["UINT8"]) == "ap_uint<8>"
    assert element_type(DataType["BINARY"]) == "ap_uint<1>"


def _c_simulator() -> bool:
    toolchain = machine_toolchain()
    try:
        toolchain.hls_installation()
        toolchain.command("g++")
    except (LookupError, FileNotFoundError):
        return False
    return True


@pytest.mark.vitis
@pytest.mark.skipif(not _c_simulator(), reason="no HLS installation's headers or g++")
def test_the_top_computes_the_case_in_c_simulation(tmp_path: Path) -> None:
    case = pool()
    point, sample, values = placed(case, 3)
    ends = _ends(point, PoolKernel, sample, ["input_channel", "output_channel"])
    window, output = ends["input_channel"], ends["output_channel"]
    x = values["input_channel"]
    words = list(pack(window.form, x.ravel().tolist(), window.element.bits))
    ((_, marker),) = window.markers
    lasts = [marker.asserted(beat) for beat in range(len(words))]
    expected = case["reference"](input_channel=x)["output_channel"]
    expected_words = list(
        pack(output.form, np.asarray(expected).ravel().tolist(), output.element.bits)
    )
    found = c_simulate(
        kernel_leaf(point),
        tmp_path,
        inputs={"s_axis": (words, window.form.lanes * window.element.bits)},
        lasts={"s_axis": lasts},
        outputs={"m_axis": len(expected_words)},
    )
    assert found == {"m_axis": expected_words}


@requires_hls
def test_it_takes_a_beat_a_cycle_as_its_schedule_states(tmp_path: Path) -> None:
    """Frames back to back, never stalled: the pool's input takes its schedule's beats in
    as many cycles, a frame's windows without a bubble, and the design leaves a frame per
    its slowest member's cycles (``cycles``)."""
    case = pool()
    point, sample, values = placed(case, 3)
    ends = _ends(point, PoolKernel, sample, ["input_channel", "output_channel"])
    window = ends["input_channel"]
    x = values["input_channel"]
    expected = case["reference"](input_channel=x)["output_channel"]
    beats = getattr(point, KERNEL).schedule.beat_count
    measured = measure(
        point.module,
        tmp_path,
        inputs={"input_channel": _words(unreplayed(window.form), x, window.element)},
        outputs={
            "output_channel": _words(
                ends["output_channel"].form, np.asarray(expected), ends["output_channel"].element
            )
        },
        frames=4,
        cycles=point.cycles,
    )
    core = f"{KERNEL}.s_axis_TDATA"
    assert measured.per_frame(core) == beats == 48
    assert measured.busy(core)[-1] == beats
    assert measured.interval() == point.cycles
