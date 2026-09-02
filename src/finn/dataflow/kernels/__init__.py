# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public physical-Kernel authoring surface.

A ``Kernel`` covers one or more selected Region families and the edges
between them.  It owns microarchitecture, target coverage, physical-only
choices, physical parameter derivation, elaboration, and a source manifest.  It
owns no logical dataflow: the Regions it covers were selected before it, and it
imports the folding they were built from rather than choosing its own.

To contribute one, subclass ``Kernel`` and place immutable declaration objects
in its class body. A ``DataflowDesign`` owns which Kernel classes are
candidates at each placement; compilation and candidate selection are private.

``CompiledKernelDeclaration`` is deliberately absent from this surface.  It is
the *compiled* form -- a Kernel class with its scoped declarations already
built -- and a contributor never names it. Generic assembly code that
genuinely needs the type imports it from the private
``finn.dataflow.kernels._declaration`` leaf.
"""

from finn.dataflow.kernels.kernel import (
    Kernel,
    PhysicalComponent,
    scalar_parameters,
)


__all__ = [
    "Kernel",
    "PhysicalComponent",
    "scalar_parameters",
]
