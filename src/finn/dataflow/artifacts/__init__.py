# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Artifact identity, sources, manifests, store, and packaging.

Everything here is value machinery below ``ComponentABI``: derivations,
canonical projection, blobs, objects, attempts, manifests, lifecycle states,
source closures, packagers.  None of it needs a Kernel, a Region, a design
point, or an Operation, and the boundary that says so is a **one-way import
rule**, tested rather than documented:

    ``artifacts`` imports the standard library and the approved dependencies.
    ``hardware`` may import ``artifacts``; never the reverse.

``artifacts`` is a leaf.  ``ComponentABI``, ``ModuleDefinition``,
``ModuleInstance`` and ``BuildUnit`` therefore live here rather than under
``hardware`` as the design's §19 table first placed them -- a portable
packaging boundary cannot depend on the semantic stack it is meant to be
portable across.

The package is empty on purpose while it is built.  Nothing outside it imports
it until the shadow-mode adapter (A9), so an increment that lands here cannot
move a recorded value anywhere else.
"""
