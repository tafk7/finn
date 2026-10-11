############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################
"""Installed XSI sources and external, writable native build artifacts."""

import hashlib
import os
import sysconfig
from pathlib import Path
from typing import Optional

from finn import resources

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


def xsi_artifact_dir() -> Path:
    """Directory the compiled ``xsi.so`` for the selected toolchain lives in.

    One directory per Vivado installation and Python ABI, under
    ``$FINN_HOME/xsi``, so switching between them reuses each build instead of
    rebuilding. FINN_XSI_BUILD_DIR selects an exact directory instead.
    """
    override = os.environ.get("FINN_XSI_BUILD_DIR")
    if override:
        return Path(override)
    vivado = os.environ.get("XILINX_VIVADO", "")
    vivado = os.path.realpath(vivado) if vivado else "none"
    abi = sysconfig.get_config_var("SOABI") or "unknown"
    key = hashlib.sha256(f"{vivado}\0{abi}".encode()).hexdigest()[:16]
    return resources.home() / "xsi" / f"{abi}-{key}"


def find_xsi_so() -> Optional[Path]:
    """Return the compiled extension for the selected toolchain, if it was built."""
    candidate = xsi_artifact_dir() / "xsi.so"
    return candidate if candidate.is_file() else None
