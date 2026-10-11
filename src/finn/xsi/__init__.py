############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################
"""FINN XSI (Xilinx Simulation Interface) support module

This module provides utilities for RTL simulation support via finn_xsi. The
finn_xsi C++ extension is built against the Vivado of the toolchain it is given on
first use; ``python -m finn.xsi.setup`` builds it ahead of time, against the
machine's. The kernel path simulates through ``finn.core.rtlsim_bench`` and does not
use this package.

Usage:
    from finn import xsi
    from finn.toolchain.launcher import host_toolchain
    toolchain = host_toolchain()
    if xsi.is_available(toolchain):
        sim = xsi.load_sim_obj(..., toolchain=toolchain)
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from finn.toolchain.launcher import Toolchain


def is_available(toolchain: Toolchain) -> bool:
    """Whether RTL simulation can be used here: the finn_xsi extension can be built
    against ``toolchain``'s Vivado.

    Cheap: checks prerequisites (a selected Vivado with XSim headers, a C++
    compiler, pybind11) without building, importing or running anything.
    """
    from finn.xsi.setup import check_prerequisites  # noqa: PLC0415

    return not check_prerequisites(toolchain)


# Cache for the loaded adapter
_adapter_module: Optional[Any] = None


def _load_modules(toolchain: Toolchain) -> None:
    """Build finn_xsi for ``toolchain`` if needed and import it; raise with the reason
    if impossible."""
    global _adapter_module

    if _adapter_module is not None:
        return

    from finn.xsi.setup import ensure_built  # noqa: PLC0415

    xsi_so = ensure_built(toolchain)
    # The Python adapter is installed normally; only the native artifact is external.
    added = str(xsi_so.parent) not in sys.path
    if added:
        sys.path.insert(0, str(xsi_so.parent))
    try:
        # Imports must be inside function: the native module is only importable
        # from the artifact directory. xsi is imported for its effect: it loads the
        # native module the adapter needs.
        import xsi  # noqa: F401, PLC0415

        import finn_xsi.adapter  # noqa: PLC0415
        import finn_xsi.sim_engine  # noqa: F401, PLC0415

        _adapter_module = finn_xsi.adapter
    finally:
        if added:
            sys.path.remove(str(xsi_so.parent))


def load_sim_obj(
    sim_out_dir: Any, out_so_relative_path: Any, tracefile: Any = None, *, toolchain: Toolchain
) -> Any:
    """``finn_xsi.adapter.load_sim_obj``, the extension built for ``toolchain`` first
    (``_load_modules``)."""
    _load_modules(toolchain)
    assert _adapter_module is not None
    return _adapter_module.load_sim_obj(
        sim_out_dir, out_so_relative_path, tracefile, toolchain=toolchain
    )


# The functions of finn_xsi.adapter that act on a loaded simulation: they import
# nothing native, so they need no extension built.
_ADAPTER_FUNCTIONS = [
    "reset_rtlsim",
    "close_rtlsim",
]


def __getattr__(name: str) -> Any:
    """Dynamically wrap finn_xsi.adapter functions."""
    if name in {"compile_sim_obj", "get_simkernel_so"}:
        from finn.xsi import compile as compilation  # noqa: PLC0415

        return getattr(compilation, name)
    if name in _ADAPTER_FUNCTIONS:
        import finn_xsi.adapter  # noqa: PLC0415

        return getattr(finn_xsi.adapter, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
