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
finn_xsi C++ extension is built against the selected Vivado on first use;
``python -m finn.xsi.setup`` builds it ahead of time. The kernel path simulates
through ``finn.core.executors.xsim`` and does not use this package.

Usage:
    from finn import xsi
    if xsi.is_available():
        sim = xsi.SimEngine(...)
"""

import sys
from typing import Any, Optional


def is_available() -> bool:
    """Whether RTL simulation can be used here: the finn_xsi extension can be built.

    Cheap: checks prerequisites (a selected Vivado with XSim headers, a C++
    compiler, pybind11) without building, importing or running anything.
    """
    from finn.xsi.setup import check_prerequisites  # noqa: PLC0415

    return not check_prerequisites()


# Cache for loaded modules
_adapter_module: Optional[Any] = None
_sim_engine_module: Optional[Any] = None


def _load_modules() -> None:
    """Build finn_xsi if needed and import it; raise with the reason if impossible."""
    global _adapter_module, _sim_engine_module

    if _adapter_module is not None:
        return

    from finn.xsi.setup import ensure_built  # noqa: PLC0415

    xsi_so = ensure_built()
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
        import finn_xsi.sim_engine  # noqa: PLC0415

        _adapter_module = finn_xsi.adapter
        _sim_engine_module = finn_xsi.sim_engine
    finally:
        if added:
            sys.path.remove(str(xsi_so.parent))


# List of functions to wrap from finn_xsi.adapter
_ADAPTER_FUNCTIONS = [
    "load_sim_obj",
    "reset_rtlsim",
    "close_rtlsim",
    "rtlsim_multi_io",
]


def __getattr__(name: str) -> Any:
    """Dynamically wrap finn_xsi.adapter functions."""
    if name in {"locate_glbl", "compile_sim_obj", "get_simkernel_so"}:
        from finn.xsi import compile as compilation  # noqa: PLC0415

        return getattr(compilation, name)
    if name in _ADAPTER_FUNCTIONS:

        def wrapper(*args, **kwargs):
            _load_modules()
            return getattr(_adapter_module, name)(*args, **kwargs)

        wrapper.__name__ = name
        wrapper.__doc__ = f"Wrapper for finn_xsi.adapter.{name}"
        return wrapper
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# SimEngine class wrapper
class SimEngine:
    """Wrapper for finn_xsi.sim_engine.SimEngine."""

    def __init__(self, *args, **kwargs):
        _load_modules()
        self._engine = _sim_engine_module.SimEngine(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._engine, name)
