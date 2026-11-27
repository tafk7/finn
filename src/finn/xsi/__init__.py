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
The finn_xsi extension is built for the Vivado installation in use the first time
it is needed (python -m finn.xsi.setup builds it ahead of time).

Usage:
    # Check if XSI support is available
    from finn import xsi
    if xsi.is_available():
        import finn_xsi.adapter
"""

import logging
import sys
from typing import Any, Optional

from finn.util.launcher import amd_path, not_set_up
from finn.xsi.paths import build_dir, source_dir


def is_available() -> bool:
    """Check if XSI (RTL simulation) support is available, building it if needed.

    Returns:
        bool: True if finn_xsi can be imported, False otherwise
    """
    # Try loading the modules (this will cache them if successful)
    return _load_modules()


# Cache for loaded modules
_adapter_module: Optional[Any] = None
_sim_engine_module: Optional[Any] = None
_xsi_module: Optional[Any] = None
# Why the built extension did not import, if it did not.
_import_error: Optional[str] = None
# Whether this process built the extension on first use, and whether that failed.
_build_attempted = False
_build_failed = False


def _load_modules() -> bool:
    """Load finn_xsi, building it for this Vivado installation on first use."""
    if _adapter_module is not None:
        return True
    xsi_so = build_dir() / "xsi.so"
    if xsi_so.exists() and _import_built():
        return True
    # Missing, or built but not loadable here (an older build): build once per process.
    return _build_on_first_use(force=xsi_so.exists()) and _import_built()


def _build_on_first_use(force: bool) -> bool:
    """Build the extension where Vivado is set up; once per process."""
    global _build_attempted, _build_failed
    if _build_attempted or not amd_path("XILINX_VIVADO"):
        return False
    _build_attempted = True
    from finn.xsi.setup import build_xsi  # noqa: PLC0415

    logging.info("Building finn_xsi for %s in %s", amd_path("XILINX_VIVADO"), build_dir())
    # finn_xsi is optional, so a failed build is non-fatal
    _build_failed = not build_xsi(force=force, verbose=False)
    return not _build_failed


def _import_built() -> bool:
    """Import the built extension and its Python package."""
    global _adapter_module, _sim_engine_module, _xsi_module, _import_error

    xsi_path = build_dir()
    if not (xsi_path / "xsi.so").exists():
        return False

    # Temporarily add to path for import
    path_added = str(xsi_path) not in sys.path
    if path_added:
        sys.path.insert(0, str(xsi_path))

    source_added = str(source_dir()) not in sys.path
    if source_added:
        sys.path.insert(1, str(source_dir()))
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
        _import_error = str(e)
        return False
    except Exception as e:
        # Catch any unexpected errors during module loading
        logging.warning(f"Unexpected error loading finn_xsi: {type(e).__name__}: {e}")
        return False
    finally:
        if source_added:
            sys.path.remove(str(source_dir()))
        # Remove from path if we added it
        if path_added and str(xsi_path) in sys.path:
            try:
                sys.path.remove(str(xsi_path))
            except ValueError:
                pass  # Path was already removed somehow


def unavailable_message() -> str:
    """Why finn_xsi cannot be used here, and what to do about it."""
    if not amd_path("XILINX_VIVADO"):
        return str(
            not_set_up(
                "vivado",
                "RTL simulation needs Vivado, but XILINX_VIVADO is not set "
                "(in a container, use `docker/run --fpga`)",
            )
        )
    if _build_failed:
        return (
            "FINN's XSim binding (finn_xsi) did not build for this Vivado installation and "
            f"Python environment; the compiler's output is in "
            f"{build_dir() / 'compile_xsi.stderr.log'}. Retry with python -m finn.xsi.setup --force"
        )
    if _import_error and (build_dir() / "xsi.so").exists():
        return (
            f"FINN's XSim binding (finn_xsi) is built ({build_dir() / 'xsi.so'}) but does not "
            f"load here: {_import_error}. Rebuild it with python -m finn.xsi.setup --force"
        )
    return (
        "FINN's XSim binding (finn_xsi) is not built for this Vivado installation and Python "
        f"environment (expected {build_dir() / 'xsi.so'}). Build it with python -m finn.xsi.setup"
    )


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

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _load_modules():
                raise ImportError(unavailable_message())
            return getattr(_adapter_module, name)(*args, **kwargs)

        wrapper.__name__ = name
        wrapper.__doc__ = f"Wrapper for finn_xsi.adapter.{name}"
        return wrapper
    if name == "SimulationError":
        if not _load_modules():
            raise ImportError(unavailable_message())
        assert _sim_engine_module is not None
        return _sim_engine_module.SimulationError
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# SimEngine class wrapper
class SimEngine:
    """Wrapper for finn_xsi.sim_engine.SimEngine."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _load_modules():
            raise ImportError(unavailable_message())
        assert _sim_engine_module is not None
        self._engine = _sim_engine_module.SimEngine(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)
