# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Model-free module artifact services at the historical Kernel import path.

The implementations live in :mod:`finn.dataflow.artifacts.build`.  This module
contains no Kernel adapter: callers pass immutable ``ModuleBuildRequirements``
or ``PreparedModuleBuild`` values directly.
"""

from finn.dataflow.artifacts.build import (
    KERNEL_SOURCE_SCHEMA,
    MODULE_SOURCE_SCHEMA,
    BuildError,
    PreparedModuleBuild,
    RenderedModuleSources,
    materialize_module_sources,
    module_source_derivation,
    portable_module_component,
    prepare_module_build,
    render_module_sources,
)

__all__ = [
    "KERNEL_SOURCE_SCHEMA",
    "MODULE_SOURCE_SCHEMA",
    "BuildError",
    "PreparedModuleBuild",
    "RenderedModuleSources",
    "materialize_module_sources",
    "module_source_derivation",
    "portable_module_component",
    "prepare_module_build",
    "render_module_sources",
]
