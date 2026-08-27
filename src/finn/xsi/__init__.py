############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################
"""FINN XSI (Xilinx Simulation Interface) support module

This module provides utilities for RTL simulation support via finn_xsi.
The finn_xsi extension must be built separately using the setup command.

Usage:
    # Check if XSI support is available
    from finn import xsi
    if xsi.is_available():
        import finn_xsi.adapter
"""

import logging
import sys
from typing import Any, Optional

from finn.xsi.paths import find_xsi_so, xsi_source_dir


def is_available() -> bool:
    """Check if XSI (RTL simulation) support is available.

    Returns:
        bool: True if finn_xsi can be imported, False otherwise
    """
    # Check if xsi.so exists
    if find_xsi_so() is None:
        return False

    # Try loading the modules (this will cache them if successful)
    return _load_modules()


# Cache for loaded modules
_adapter_module: Optional[Any] = None
_sim_engine_module: Optional[Any] = None
_xsi_module: Optional[Any] = None


def _load_modules() -> bool:
    """Load finn_xsi modules if available."""
    global _adapter_module, _sim_engine_module, _xsi_module

    if _adapter_module is not None:
        return True

    xsi_so = find_xsi_so()
    if xsi_so is None:
        return False

    # The compiled `xsi` extension and the `finn_xsi` Python package now live in
    # separate directories, so both have to be importable. When the artifact was
    # found in its legacy in-tree location these two coincide, and the set
    # collapses back to a single entry.
    import_paths = [str(xsi_so.parent), str(xsi_source_dir())]
    added_paths = [p for p in dict.fromkeys(import_paths) if p not in sys.path]
    for p in reversed(added_paths):
        sys.path.insert(0, p)

    try:
        # Imports must be inside function: modules require dynamic path setup
        # and may not exist if finn_xsi extension is not built
        import finn_xsi.adapter  # noqa: PLC0415
        import finn_xsi.sim_engine  # noqa: PLC0415
        import xsi  # noqa: PLC0415

        _xsi_module = xsi
        _adapter_module = finn_xsi.adapter
        _sim_engine_module = finn_xsi.sim_engine

        return True
    except ImportError as e:
        # Log the specific import error for debugging
        logging.debug(f"Failed to import finn_xsi modules: {e}")
        return False
    except Exception as e:
        # Catch any unexpected errors during module loading
        logging.warning(f"Unexpected error loading finn_xsi: {type(e).__name__}: {e}")
        return False
    finally:
        # Remove whatever we added, leaving any pre-existing entries alone
        for p in added_paths:
            try:
                sys.path.remove(p)
            except ValueError:
                pass  # Path was already removed somehow


# List of functions to wrap from finn_xsi.adapter
_ADAPTER_FUNCTIONS = [
    "locate_glbl",
    "compile_sim_obj",
    "get_simkernel_so",
    "load_sim_obj",
    "reset_rtlsim",
    "close_rtlsim",
    "rtlsim_multi_io",
]


def __getattr__(name: str) -> Any:
    """Dynamically wrap finn_xsi.adapter functions."""
    if name in _ADAPTER_FUNCTIONS:

        def wrapper(*args, **kwargs):
            if not _load_modules():
                raise ImportError("finn_xsi not available. Run: python -m finn.xsi.setup")
            return getattr(_adapter_module, name)(*args, **kwargs)

        wrapper.__name__ = name
        wrapper.__doc__ = f"Wrapper for finn_xsi.adapter.{name}"
        return wrapper
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# SimEngine class wrapper
class SimEngine:
    """Wrapper for finn_xsi.sim_engine.SimEngine."""

    def __init__(self, *args, **kwargs):
        if not _load_modules():
            raise ImportError("finn_xsi not available. Run: python -m finn.xsi.setup")
        self._engine = _sim_engine_module.SimEngine(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._engine, name)
