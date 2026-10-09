# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The HLS source kind end to end: FinnLib's own pooling top through FINN's HLS runner
and the stream testbench.

``finnlib_pooling_top`` is FinnLib's ``hls/nonlin/pooling_top.cpp`` at its default
values (``pooling_top.hpp``: a 24x36 map of 9 unsigned channels, a 2x3 window, three
channels a beat), its top renamed as a request names it; ``pooling_variant`` is the
same file, its pool's map and window substituted. Both read FinnLib's text as the test
runs: none of it is written here. Each is a leaf whose ABI is declared, as a
kernel declares one: HLS's AXIS pins (``src_TDATA``, upper case), ``ap_clk`` and
``ap_rst_n``. Synthesized by ``built_hls`` (cached under ``$FINN_HOME/hls``, its
exported top checked against those pins), simulated free and stalled against a max
pool (``kernels.specs.pool.max_pool``): alone, and the two configurations chained in
one netlist, whose submodules HLS names apart by their tops (``config_rtl
-module_auto_prefix``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from qonnx.core.datatype import DataType

from finn.core.executors.xsim.pacing import FREE, STALLED, Pacing
from finn.core.executors.xsim.rtl import pack_lanes, stream_through
from finn.harness.toolchain import finnlib_root
from finn.kernels.artifacts.abi import Bus, Clock, Direction, Endpoint, Free, Reset, Signal
from finn.kernels.artifacts.contributions import CopiedSource, HlsSource
from finn.kernels.artifacts.module import Abi, Composed, Fragment, Leaf, Link, LinkEnd
from finn.kernels.transport import AxisBeat, AxisSpelling
from kernels.specs.pool import max_pool
from kernels.xsim import requires_hls

Array = npt.NDArray[np.int64]

#: The pooling top's headers, by their FinnLib paths.
HEADERS = tuple(
    CopiedSource("finnlib", path)
    for path in (
        "hls/nonlin/pooling.hpp",
        "hls/util/util.hpp",
        "hls/shape/input_gen.hpp",
    )
)
#: FinnLib's own header of its pooling top: its values, and its prototype.
TOP_HEADER = CopiedSource("finnlib", "hls/nonlin/pooling_top.hpp")
DIRECTIVES = ("config_rtl -module_auto_prefix", "config_rtl -deadlock_detection none")
# pooling_top.hpp's values.
H, W, C, KH, KW, SIMD = 24, 36, 9, 2, 3, 3
WINDOW = (KH, KW)
BITS = 32  # T = unsigned
PACINGS = {"free": FREE, "stalled": STALLED}


#: The pooling top's call of the pool, at ``pooling_top.hpp``'s values.
POOL_CALL = "max_pool<H, W, KH, KW, C/SIMD>("


def pooling_top_text() -> str:
    """FinnLib's ``hls/nonlin/pooling_top.cpp``, read as the test runs."""
    text = (finnlib_root() / "hls/nonlin/pooling_top.cpp").read_text()
    assert text.count("void pooling_top(") == 1
    assert text.count(POOL_CALL) == 1
    return text


def finnlib_pooling_top() -> HlsSource:
    """FinnLib's pooling top at its default values, its function renamed."""
    text = pooling_top_text()
    return HlsSource.request(
        "finnlib_pooling_top",
        lambda name: text.replace("void pooling_top(", f"void {name}("),
        (TOP_HEADER, *HEADERS),
        5.0,
        DIRECTIVES,
    )


def pooling_variant(h: int, w: int, kh: int, kw: int) -> HlsSource:
    """FinnLib's pooling top, its function renamed, pooling at other values: ``T``,
    ``SIMD`` and ``C`` its header's, the map and window given."""
    text = pooling_top_text().replace(POOL_CALL, f"max_pool<{h}, {w}, {kh}, {kw}, {C // SIMD}>(")
    return HlsSource.request(
        "pooling_variant",
        lambda name: text.replace("void pooling_top(", f"void {name}("),
        (TOP_HEADER, *HEADERS),
        5.0,
        DIRECTIVES,
    )


def _stream(name: str, endpoint: Endpoint, spelling: AxisSpelling) -> Bus:
    beat = AxisBeat(name, DataType["UINT32"], SIMD, endpoint=endpoint, spelling=spelling)
    return beat.bus(clock="ap_clk", reset="ap_rst_n")


CLOCKING = (
    Signal("ap_clk", Direction.IN, 1, Clock(Free())),
    Signal("ap_rst_n", Direction.IN, 1, Reset(True, True, ("ap_clk",))),
)


def leaf(request: HlsSource) -> Leaf:
    pins = (
        *CLOCKING,
        _stream("src", Endpoint.TARGET, AxisSpelling.UPPER),
        _stream("dst", Endpoint.INITIATOR, AxisSpelling.UPPER),
    )
    return Leaf("test.hls.pooling", "1", request.function, (), Abi(pins), (request,))


def _end(instance: str | None, name: str) -> LinkEnd:
    suffix = ("_TDATA", "_TVALID", "_TREADY") if instance else ("_tdata", "_tvalid", "_tready")
    data, valid, ready = (f"{name}{part}" for part in suffix)
    return LinkEnd(instance, data, BITS * SIMD, valid, ready)


def chained(first: HlsSource, second: HlsSource) -> Composed:
    """``first`` feeding ``second``, between the root's ``s_axis`` and ``m_axis``."""
    lanes = tuple(range(SIMD))
    pins = (
        *CLOCKING,
        _stream("s_axis", Endpoint.TARGET, AxisSpelling.LOWER),
        _stream("m_axis", Endpoint.INITIATOR, AxisSpelling.LOWER),
    )
    links = (
        Link(_end(None, "s_axis"), _end("first", "src"), BITS, lanes),
        Link(_end("first", "dst"), _end("second", "src"), BITS, lanes),
        Link(_end("second", "dst"), _end(None, "m_axis"), BITS, lanes),
    )
    return Composed(
        "test.hls.chain",
        "1",
        "hls_chain",
        Abi(pins),
        Fragment((("first", leaf(first)), ("second", leaf(second))), links),
    )


def words(image: Array) -> tuple[list[int], int]:
    """An HWC image's beats, SIMD channels a beat, lane zero the lowest channel."""
    flat = image.reshape(-1, SIMD).tolist()
    return [pack_lanes(beat, BITS) for beat in flat], BITS * SIMD


def image() -> Array:
    return np.random.default_rng(7).integers(0, 1 << 4, size=(H, W, C), dtype=np.int64)


@requires_hls
@pytest.mark.parametrize("mode", sorted(PACINGS))
def test_finnlib_pooling_top_synthesizes_and_streams(mode: str, tmp_path: Path) -> None:
    request = finnlib_pooling_top()
    x = image()
    stream_through(
        leaf(request),
        tmp_path,
        inputs={"src": words(x)},
        outputs={"dst": words(max_pool(x, WINDOW, WINDOW))},
        pacing=PACINGS[mode],
    )


@requires_hls
@pytest.mark.parametrize("mode", sorted(PACINGS))
def test_two_configurations_share_one_netlist(mode: str, tmp_path: Path) -> None:
    first = finnlib_pooling_top()
    second = pooling_variant(H // KH, W // KW, 2, 2)
    x = image()
    pacing: Pacing = PACINGS[mode]
    stream_through(
        chained(first, second),
        tmp_path,
        inputs={"s_axis": words(x)},
        outputs={"m_axis": words(max_pool(max_pool(x, WINDOW, WINDOW), (2, 2), (2, 2)))},
        pacing=pacing,
    )
