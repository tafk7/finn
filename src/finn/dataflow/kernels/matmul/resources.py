# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Installed resources for matrix-multiplication Kernel construction."""

from importlib.resources import files
from pathlib import Path


def template_root() -> Path:
    """Return the installed filesystem root containing the family templates."""

    root = files("finn.dataflow.kernels.matmul").joinpath("templates")
    path = Path(str(root))
    if not path.is_dir():
        raise FileNotFoundError(f"matrix-multiplication template root is absent: {path}")
    return path


__all__ = ["template_root"]
