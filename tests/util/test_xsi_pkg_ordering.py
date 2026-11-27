# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import pytest

import shlex
import sys

from finn import resources

# finn_xsi is not an installed package: finn.xsi adds the finn-xsi resource to
# sys.path at runtime. srcutil has no xsi C-extension import, so we can add the
# dir and import it even where finn_xsi is not built yet (finn_xsi.adapter could
# not, since it imports sim_engine).
_XSI_ROOT = resources.path("finn-xsi")
if _XSI_ROOT not in sys.path:
    sys.path.insert(0, _XSI_ROOT)

from finn_xsi.srcutil import is_pkg_src, order_pkg_first, prj_text  # noqa: E402


@pytest.mark.util
@pytest.mark.parametrize(
    "path",
    [
        # a generic *_pkg suffix should catch any new finn-rtllib
        # package without manually editing a whitelist
        "/path/to/swg_pkg.sv",
        "/path/to/mvu_pkg.v",
        "/path/to/some_other_pkg.sv",
        "foo_pkg.sv",
    ],
)
def test_is_pkg_src_matches_pkg_basenames(path):
    assert is_pkg_src(path)


@pytest.mark.util
def test_is_pkg_src_does_not_match_non_pkg_or_header_files():
    # compile_sim_obj skips header files (.svh / .vh) when writing the .prj, so
    # the predicate must not pull them into the package group either
    assert not is_pkg_src("/path/to/foo.sv")
    assert not is_pkg_src("/path/to/foo.v")
    assert not is_pkg_src("/path/to/foo.vhd")
    assert not is_pkg_src("/path/to/swg_pkg.svh")
    assert not is_pkg_src("/path/to/swg_pkg_top.sv")
    assert not is_pkg_src("/path/to/_pkg.svh")


@pytest.mark.util
def test_order_pkg_first_floats_packages_and_is_stable():
    # packages must come before modules, and order within each group (including
    # the interleaved headers) must be preserved so dependency order survives
    source_list = [
        "/path/to/top.sv",
        "/path/to/swg_pkg.sv",
        "/path/to/foo.vh",
        "/path/to/mvu_pkg.v",
        "/path/to/bar.v",
    ]
    assert order_pkg_first(source_list) == [
        "/path/to/swg_pkg.sv",
        "/path/to/mvu_pkg.v",
        "/path/to/top.sv",
        "/path/to/foo.vh",
        "/path/to/bar.v",
    ]


@pytest.mark.util
def test_order_pkg_first_preserves_list_with_no_packages():
    source_list = ["/path/to/top.sv", "/path/to/bar.v"]
    assert order_pkg_first(source_list) == source_list


@pytest.mark.util
def test_prj_names_sources_with_spaces_whole():
    """xelab reads a .prj line as words; a double-quoted path is one word."""
    lib = "/home/x/rtl lib"
    prj = prj_text(
        [f"{lib}/mvu/mvu_pkg.sv", f"{lib}/fifo/hdl/fifo.sv", f"{lib}/inc/defs.vh", "/b/top.v"],
        glbl="/opt/Xilinx 2025/glbl.v",
    )
    assert [shlex.split(line) for line in prj.splitlines()] == [
        ["verilog", "work", "/opt/Xilinx 2025/glbl.v"],
        ["sv", "work", "--include", f"{lib}/inc", f"{lib}/mvu/mvu_pkg.sv"],
        ["sv", "work", "--include", f"{lib}/inc", f"{lib}/fifo/hdl/fifo.sv"],
        ["verilog", "work", "--include", f"{lib}/inc", "/b/top.v"],
    ]
    # Paths without whitespace are written as before.
    assert prj_text(["/a/x.sv", "/a/y.vhd"]) == "sv work  /a/x.sv\nvhdl2008 work /a/y.vhd\n"
