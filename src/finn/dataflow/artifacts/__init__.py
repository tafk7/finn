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

``artifacts`` is a leaf.  ``ComponentABI`` therefore lives here rather than
beside the Kernel that declares one -- a portable packaging boundary cannot
depend on the semantic stack it is meant to be portable across.

The rule is checked, not merely stated: ``test_package_boundaries`` walks every
module in this package and refuses an import of ``model``, ``kernels``,
``designs``, ``ops`` or ``_engine``.  What crosses *into* it is a detached
value -- ``ModuleBuildSpec`` today, and whatever U5's composition
contract settles on for a Design -- never an occurrence.
"""
