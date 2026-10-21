# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``built_hls`` runs as many HLS syntheses at once as the toolchain's selection says Vivado
launches runs (``Selection.vivado_jobs``), never one per request. The synthesis is
stubbed with one that counts the calls in flight, so no tool runs."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

import finn.transformation.kernels.hls as hls
from finn.kernels.artifacts.abi import Clock, Direction, Free, Reset, Signal
from finn.kernels.artifacts.contributions import CopiedSource, HlsSource
from finn.kernels.artifacts.module import Abi, Composed, Fragment, Leaf
from finn.toolchain.hls import HlsProduct
from finn.toolchain.launcher import Selection, Toolchain

HEADERS = (CopiedSource("fixture", "core.hpp"),)
PINS = (
    Signal("ap_clk", Direction.IN, 1, Clock(Free())),
    Signal("ap_rst_n", Direction.IN, 1, Reset(True, True, ("ap_clk",))),
)
#: How long a stubbed synthesis holds its place: long enough that runs a pool lets
#: together overlap.
HELD = 0.2


def _request(value: int) -> HlsSource:
    return HlsSource.request(
        "core",
        lambda name: f'#include "core.hpp"\nvoid {name}() {{ core<{value}>(); }}\n',
        HEADERS,
        5.0,
    )


def _module(requests: list[HlsSource]) -> Composed:
    leaves = tuple(
        (f"u{index}", Leaf("test.hls", request.function, (), Abi(PINS), (request,)))
        for index, request in enumerate(requests)
    )
    return Composed("test.hls.many", "1", "many", Abi(PINS), Fragment(leaves))


class Flight:
    """The stubbed synthesis: counts the calls in flight and the most there were at once."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0
        self.names: list[str] = []

    def synthesize(
        self, directory: Path, part: str, *, name: str, toolchain: Toolchain, cache: Path | None
    ) -> HlsProduct:
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
            self.names.append(name)
        time.sleep(HELD)
        with self.lock:
            self.now -= 1
        product = HlsProduct(directory)
        product.verilog.mkdir(parents=True, exist_ok=True)
        return product


@pytest.mark.parametrize("jobs", [1, 2])
def test_hls_runs_go_at_most_as_many_at_once_as_the_selection_states(
    jobs: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "finnlib"
    root.mkdir()
    (root / "core.hpp").write_text("template<int N> void core() {}\n")
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "build"))
    flight = Flight()
    monkeypatch.setattr(hls, "synthesize", flight.synthesize)
    monkeypatch.setattr(hls, "check_abi", lambda pins, files, top: ())
    requests = [_request(value) for value in range(4)]
    toolchain = Toolchain(Selection(vivado_jobs=jobs), {"PATH": os.defpath})

    built = hls.built_hls(
        _module(requests), "xc7z020clg400-1", toolchain=toolchain, roots={"fixture": root}
    )

    assert set(built) == {request.function for request in requests}
    assert sorted(flight.names) == sorted(built)
    assert flight.peak == jobs
