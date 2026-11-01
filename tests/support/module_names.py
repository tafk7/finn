# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A module's name as a build names it now (``live_name``): the snapshot of the files it stages
read from FinnLib as FINN resolves it (``finn.transformation.kernels.package.source_roots``,
``FINN_RESOURCES_FINNLIB`` overriding it), so it moves with the FinnLib checked out."""

from __future__ import annotations

from finn.kernels.artifacts.build import capture
from finn.kernels.artifacts.module import Module, module_name
from finn.transformation.kernels.package import source_roots


def live_name(module: Module) -> str:
    """``module``'s name as a build names it from FinnLib now; no HLS leaf (nothing here
    synthesizes)."""
    return module_name(module, capture(module, roots=source_roots()))
