############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################
"""Installed XSI sources, external, writable native build artifacts, and the
environment a process that simulates through XSI starts in."""

import hashlib
import os
import sysconfig
from pathlib import Path
from typing import Optional

from finn import resources
from finn.toolchain.launcher import Toolchain, identity

# Stable, read-only paths to FINN's own sources: the xsi bridge's. Wheels must be
# unpacked (as pip normally installs them). Generated projects may retain these
# absolute paths: keep this installation in place for their lifetime. No temporary
# extraction or cache is created in the installed package.

# Families of FINN's own sources, by the name of the resource that supplies each.
_FAMILIES = {"xsi": "xsi"}


def resource_path(family: str, *parts: str) -> str:
    """Return an existing path in a family of FINN's own sources (xsi).

    Each family is a FINN resource (finn/resources.toml), so
    FINN_RESOURCES_<NAME> or a project declaration can replace it.
    """
    if family not in _FAMILIES:
        raise KeyError(family)
    if any(Path(part).is_absolute() or ".." in Path(part).parts for part in parts):
        raise ValueError("Resource paths must stay inside their resource family")
    resource = Path(resources.path(_FAMILIES[family], fetch=False)).joinpath(*parts)
    if not resource.exists():
        raise FileNotFoundError(resource)
    return str(resource.resolve())


def xsi_source_dir() -> Path:
    """Stable installed C++ sources, never a destination for generated files."""
    return Path(resource_path("xsi"))


def xsi_artifact_dir(toolchain: Toolchain) -> Path:
    """Directory the compiled ``xsi.so`` for ``toolchain`` lives in.

    One directory per Vivado installation, by what it is (``identity``), and Python
    ABI, under ``$FINN_HOME/xsi``, so switching between them reuses each build
    instead of rebuilding. FINN_XSI_BUILD_DIR selects an exact directory instead.
    ``ToolUnavailable`` when the toolchain selects no Vivado.
    """
    override = os.environ.get("FINN_XSI_BUILD_DIR")
    if override:
        return Path(override)
    tool = identity(toolchain, "xsim")
    abi = sysconfig.get_config_var("SOABI") or "unknown"
    key = hashlib.sha256(f"{tool}\0{abi}".encode()).hexdigest()[:16]
    return resources.home() / "xsi" / f"{abi}-{key}"


def find_xsi_so(toolchain: Toolchain) -> Optional[Path]:
    """Return the compiled extension for ``toolchain``, if it was built."""
    candidate = xsi_artifact_dir(toolchain) / "xsi.so"
    return candidate if candidate.is_file() else None


#: The simulator libraries an XSI simulation loads, by the installation that has
#: them: the Vivado simulation kernel (``finn_xsi``), and the floating-point
#: operators HLS-generated code links. docker/finn-toolchain.sh puts the same
#: directories on a native shell's loader path.
SIMULATION_LIBRARIES = (
    ("XILINX_VIVADO", "lib/lnx64.o"),
    ("XILINX_VITIS", "lnx64/tools/fpo_v7_1"),
    ("XILINX_HLS", "lnx64/tools/fpo_v7_1"),
)


def simulation_environment(toolchain: Toolchain) -> dict[str, str]:
    """``toolchain``'s environment with the loader path an XSI simulation needs: the
    simulator libraries of its installations (``SIMULATION_LIBRARIES``, those that
    exist) ahead of its ``LD_LIBRARY_PATH``. The loader reads the path when a process
    starts, so this is the environment of a new process that simulates: the build
    process of ``build_dataflow_directory``, an XSI C++ driver."""
    environment = dict(toolchain.environment)
    libraries = [
        str(Path(environment[variable]) / suffix)
        for variable, suffix in SIMULATION_LIBRARIES
        if environment.get(variable) and (Path(environment[variable]) / suffix).is_dir()
    ]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    if libraries:
        environment["LD_LIBRARY_PATH"] = ":".join(libraries)
    return environment
