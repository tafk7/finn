# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Installed RTL and templates for physical Kernel construction."""

from importlib.resources import files
from pathlib import Path


def resource_root() -> Path:
    """Return the installed directory used by the explicit ``kernels`` source root."""

    root = files("finn.kernels.resources")
    path = Path(str(root))
    if not path.is_dir():
        raise FileNotFoundError(f"kernel resource root is absent: {path}")
    return path


def template_root() -> Path:
    """Return the same packaged directory for assembly template resolution."""

    return resource_root()


__all__ = ["resource_root", "template_root"]
