# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Templates shipped with the package: the composed module's wrapper."""

from importlib.resources import files
from pathlib import Path


def template_root() -> Path:
    """The installed directory holding the package's templates."""

    path = Path(str(files("finn.kernels.resources")))
    if not path.is_dir():
        raise FileNotFoundError(f"kernel template directory is absent: {path}")
    return path


__all__ = ["template_root"]
