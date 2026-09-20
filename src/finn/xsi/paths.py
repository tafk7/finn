############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################
"""Installed XSI sources and external, writable native build artifacts."""

import os
from pathlib import Path
from typing import Optional

from finn.util._legacy_build_env import build_directory
from finn.util.resources import resource_path


def xsi_source_dir() -> Path:
    """Stable installed C++ sources, never a destination for generated files."""
    return Path(resource_path("xsi"))


def xsi_artifact_dir() -> Path:
    """Directory the compiled ``xsi.so`` is written to and loaded from.

    Defaults under ``$FINN_BUILD_DIR`` so the artifact is scoped to the same
    build tree as everything else FINN generates, and never to the workspace.
    """
    override = os.environ.get("FINN_XSI_BUILD_DIR")
    if override:
        return Path(override)
    return Path(build_directory()) / "finn_xsi"


def find_xsi_so() -> Optional[Path]:
    """Return the selected compiled extension, or None if it has not been built."""
    candidate = xsi_artifact_dir() / "xsi.so"
    return candidate if candidate.is_file() else None
