# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os.path

# these helpers live in a separate module from adapter.py so they can be
# imported without the xsi C-extension (pulled in via sim_engine), keeping them
# unit-testable in a checkout where finn_xsi has not been built yet


def is_pkg_src(path: str) -> bool:
    """Return True for ``*_pkg.{sv,v}`` files (SystemVerilog/Verilog packages).

    ``compile_sim_obj`` uses this to move packages ahead of the modules that
    import them, which xelab requires. Matching the ``_pkg`` basename suffix is
    just a useful heuristic that may catch new finn-rtllib packages without
    manually editing a whitelist.
    """
    base = os.path.basename(path)
    return base.endswith("_pkg.sv") or base.endswith("_pkg.v")


def order_pkg_first(source_list):
    """Return ``source_list`` reordered so ``*_pkg.{sv,v}`` files come first.

    The partition is stable, so files keep their original relative order within
    each group. This only guarantees packages before modules. A package that
    imports another package still depends on ``source_list`` already listing
    them in dependency order.
    """
    pkg_srcs = []
    other_srcs = []
    for src in source_list:
        (pkg_srcs if is_pkg_src(src) else other_srcs).append(src)
    return pkg_srcs + other_srcs


def prj_word(path: str) -> str:
    """A path as one word of an xelab .prj line: like Vivado's own .prj files, a
    path with whitespace is double-quoted."""
    return '"%s"' % path if any(c.isspace() for c in path) else path


def prj_text(source_list, glbl=None) -> str:
    """The rtlsim.prj that ``compile_sim_obj`` gives xelab for ``source_list``."""
    lines = []
    if glbl is not None:
        lines.append(f"verilog work {prj_word(glbl)}")
    # extract (unique, by using a set) verilog headers for inclusion
    verilog_headers = {
        os.path.dirname(x) for x in source_list if x.endswith(".vh") or x.endswith(".svh")
    }
    verilog_header_incl_str = " ".join(["--include " + prj_word(x) for x in verilog_headers])
    # packages must come before the modules that import them
    for src_line in order_pkg_first(source_list):
        src = prj_word(src_line)
        if src_line.endswith(".v"):
            lines.append(f"verilog work {verilog_header_incl_str} {src}")
        elif src_line.endswith(".vhd"):
            # note that Verilog header incls are not added for VHDL
            lines.append(f"vhdl2008 work {src}")
        elif src_line.endswith(".sv"):
            lines.append(f"sv work {verilog_header_incl_str} {src}")
        elif src_line.endswith(".vh") or src_line.endswith(".svh"):
            # skip adding Verilog headers directly (see verilog_header_incl_str)
            continue
        else:
            raise Exception(f"Unknown extension for .prj file sources: {src_line}")
    return "".join(line + "\n" for line in lines)
