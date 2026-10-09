# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The HLS source kind without a tool: a request named by its digest, staged
relocatably, a leaf built from it, and its product emitted beside copied sources."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from finn.kernels.artifacts.abi import Clock, Direction, Free, Reset, Signal
from finn.kernels.artifacts.build import emit_module
from finn.kernels.artifacts.contributions import (
    ContributionError,
    CopiedSource,
    GeneratedData,
    HlsSource,
)
from finn.kernels.artifacts.hls import SCRIPT, TOP, quoted_includes, stage_hls
from finn.kernels.artifacts.module import Abi, BuildError, Composed, Fragment, Leaf

HEADERS = (
    CopiedSource("fixture", "hls/nonlin/core.hpp"),
    CopiedSource("fixture", "hls/util/util.hpp"),
)
PINS = (
    Signal("ap_clk", Direction.IN, 1, Clock(Free())),
    Signal("ap_rst_n", Direction.IN, 1, Reset(True, True, ("ap_clk",))),
)


def _top(value: int = 3) -> HlsSource:
    return HlsSource.request(
        "core",
        lambda name: f'#include "core.hpp"\nvoid {name}() {{ core<{value}>(); }}\n',
        HEADERS,
        5.0,
        ("config_rtl -module_auto_prefix",),
    )


def _fixture(root: Path) -> Path:
    (root / "hls/nonlin").mkdir(parents=True, exist_ok=True)
    (root / "hls/util").mkdir(parents=True, exist_ok=True)
    (root / "hls/nonlin/core.hpp").write_text(
        '#include "util.hpp"\n#include <ap_int.h>\ntemplate<int N> void core() {}\n'
    )
    (root / "hls/util/util.hpp").write_text("// nothing it includes\n")
    return root


def test_a_request_is_named_by_its_digest() -> None:
    request = _top()
    stem, key = request.function[:-17], request.function[-16:]
    assert request.function == f"{stem}_{key}" and stem == "core"
    assert _top() == request
    assert _top(4).function != request.function
    assert replace(request, period_ns=5).function == request.function  # an int period is a float
    for other in (
        HlsSource.request("core", lambda name: f"void {name}() {{}}\n", HEADERS, 5.0),
        HlsSource.request("core", lambda name: _top().top.replace(_top().function, name), (), 5.0),
        HlsSource.request(
            "core", lambda name: _top().top.replace(_top().function, name), HEADERS, 4.0
        ),
    ):
        assert other.function != request.function


def test_a_request_whose_name_is_not_its_digest_is_refused() -> None:
    request = _top()
    with pytest.raises(ContributionError, match="not the request's own digest"):
        replace(request, period_ns=4.0)
    renamed = "core_0000000000000000"
    with pytest.raises(ContributionError, match="not the request's own digest"):
        replace(request, function=renamed, top=request.top.replace(request.function, renamed))
    with pytest.raises(ContributionError, match="does not define"):
        replace(request, top="void other() {}\n")
    with pytest.raises(ContributionError, match="not a C or C\\+\\+ header"):
        HlsSource.request("core", lambda name: name, (CopiedSource("fixture", "a.cpp"),), 5.0)
    with pytest.raises(ContributionError, match="no '__'"):
        HlsSource.request("a__b", lambda name: name, (), 5.0)
    with pytest.raises(ContributionError, match="one line of Tcl"):
        HlsSource.request("core", lambda name: name, (), 5.0, ("a\nb",))


def test_staging_writes_a_relocatable_request(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "finnlib")
    request = _top()
    one = stage_hls(request, tmp_path / "one", roots={"fixture": root})
    two = stage_hls(request, tmp_path / "two", roots={"fixture": root})
    assert one.files == (TOP, "hls/nonlin/core.hpp", "hls/util/util.hpp", SCRIPT)
    for name in one.files:
        assert (one.directory / name).read_bytes() == (two.directory / name).read_bytes()
    script = (one.directory / SCRIPT).read_text()
    assert str(tmp_path) not in script
    assert f"set_top {request.function}" in script
    assert "set_part $part" in script
    assert "-Ihls/nonlin -Ihls/util" in script
    assert "config_rtl -module_auto_prefix" in script
    assert script.index("config_rtl") < script.index("csynth_design")


def test_staging_refuses_an_include_no_declared_header_satisfies(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "finnlib")
    undeclared = HlsSource.request(
        "core", lambda name: f'#include "core.hpp"\nvoid {name}() {{}}\n', HEADERS[:1], 5.0
    )
    with pytest.raises(BuildError, match="hls/nonlin/core.hpp includes 'util.hpp'"):
        stage_hls(undeclared, tmp_path / "out", roots={"fixture": root})
    with pytest.raises(BuildError, match="no source root resolves 'fixture'"):
        stage_hls(_top(), tmp_path / "out", roots={})


def test_quoted_includes_are_the_requests_and_angle_brackets_the_tools() -> None:
    text = '#include "a.hpp"\n  #  include "b.h"\n#include <ap_int.h>\n// #include "c.hpp"\n'
    assert quoted_includes(text) == ("a.hpp", "b.h")


def _leaf(request: HlsSource) -> Leaf:
    return Leaf("test.hls", "1", request.function, (), Abi(PINS), (request,))


def test_an_hls_leaf_is_named_by_its_top_and_takes_no_parameters() -> None:
    request = _top()
    assert _leaf(request).name == request.function
    with pytest.raises(BuildError, match="is not its HLS request's top"):
        Leaf("test.hls", "1", "core", (), Abi(PINS), (request,))
    with pytest.raises(BuildError, match="compiled into its top"):
        Leaf(
            "test.hls",
            "1",
            request.function,
            (("N", 3),),
            Abi(PINS, (("N", "3"),)),
            (request,),
        )
    with pytest.raises(BuildError, match="its module's only source"):
        Leaf(
            "test.hls",
            "1",
            request.function,
            (),
            Abi(PINS),
            (request, CopiedSource("fixture", "core.sv")),
        )


def _product(directory: Path, request: HlsSource, *extra: str) -> Path:
    directory.mkdir(parents=True)
    (directory / f"{request.function}.v").write_text(f"module {request.function}; endmodule\n")
    (directory / f"{request.function}_rom.v").write_text("module rom; endmodule\n")
    (directory / f"{request.function}_rom.dat").write_text("00\n")
    for name in extra:
        (directory / name).write_text("\n")
    return directory


def test_emission_stages_the_product_below_its_top_and_its_images_as_data(
    tmp_path: Path,
) -> None:
    request = _top()
    product = _product(tmp_path / "product", request)
    emitted = emit_module(
        _leaf(request), tmp_path / "out", roots={}, built={request.function: product}
    )
    function = request.function
    assert emitted.entry_point == function
    assert set(emitted.sources) == {f"{function}/{function}.v", f"{function}/{function}_rom.v"}
    assert emitted.data == (f"{function}_rom.dat",)
    assert (emitted.directory / f"{function}_rom.dat").read_text() == "00\n"


def test_emission_refuses_a_missing_or_foreign_product(tmp_path: Path) -> None:
    request = _top()
    with pytest.raises(BuildError, match="no HLS product is given"):
        emit_module(_leaf(request), tmp_path / "out", roots={})
    product = _product(tmp_path / "product", request, "ip.tcl")
    with pytest.raises(BuildError, match="does not stage: \\['ip.tcl'\\]"):
        emit_module(_leaf(request), tmp_path / "out", roots={}, built={request.function: product})


def test_two_configurations_compose_without_sharing_a_file(tmp_path: Path) -> None:
    first, second = _top(3), _top(4)
    products = {
        request.function: _product(tmp_path / request.function, request)
        for request in (first, second)
    }
    module = Composed(
        "test.hls.pair",
        "1",
        "pair",
        Abi(PINS),
        Fragment((("a", _leaf(first)), ("b", _leaf(second)))),
    )
    emitted = emit_module(module, tmp_path / "out", roots={}, built=products)
    staged = [path for path in emitted.sources if path.endswith(".v")]
    assert len(staged) == 4 and len(set(staged)) == 4
    assert len(emitted.data) == 2


def test_a_leaf_with_data_and_an_hls_product_keeps_both(tmp_path: Path) -> None:
    request = _top()
    product = _product(tmp_path / "product", request)
    leaf = replace(_leaf(request), data=(GeneratedData("table.dat", b"01\n"),))
    emitted = emit_module(leaf, tmp_path / "out", roots={}, built={request.function: product})
    assert set(emitted.data) == {"table.dat", f"{request.function}_rom.dat"}
