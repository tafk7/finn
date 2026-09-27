# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Artifact identity, sources, manifests, store, and packaging.

Everything here is value machinery below ``ComponentABI``: derivations,
canonical projection, blobs, objects, attempts, manifests, lifecycle states,
source closures, packagers.  None of it needs a Kernel, a Region, a design
point, or an Operation, and the boundary that says so is a **one-way import
rule**, tested rather than documented:

    ``artifacts`` imports the standard library and the approved dependencies.
    Concrete kernels may import ``artifacts``; never the reverse.

``artifacts`` is a leaf.  ``ComponentABI`` therefore lives here rather than
beside the Kernel that declares one -- a portable packaging boundary cannot
depend on the semantic stack it is meant to be portable across.

The kernel boundary and artifact isolation tests walk every module in this
package and refuse imports of concrete kernels, Space, the engine and dataflow
models.  What crosses into it is a detached ``ModuleBuildRequirements`` value,
never a Space occurrence.
"""
