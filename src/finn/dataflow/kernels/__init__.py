# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The physical hardware layer: Kernels, coverage, and bindings.

A ``Kernel`` covers one or more selected Region families and the edges
between them.  It owns microarchitecture, target coverage, physical-only
choices, physical parameter derivation, elaboration, and a source manifest.  It
owns no logical dataflow: the Regions it covers were selected before it, and it
imports the folding they were built from rather than choosing its own.

To contribute one, subclass ``Kernel``, declare its design through
``KernelScope``, and hand it to ``declare_kernel``.  Add a
``KernelCandidateSelection`` only when several Kernels cover one point and
something has to choose between them.

``CompiledKernelDeclaration`` is deliberately absent from this surface.  It is
the *compiled* form -- a Kernel class with its scoped declarations already
built -- and a contributor never names it.  Assembly code that genuinely needs
the type imports it from ``finn.dataflow.kernels.kernel``.

**This contract is not finished.**  Two of the nine obligations the migration
plan lists for a physical Kernel are stubs here: artifact requirements and
implementation evidence.  Both are shaped by what the first real consumer
needs, so they are added when the MVAU migration supplies one, not guessed at
from a synthetic case.
"""

from finn.dataflow.kernels.authoring import (
    COVERAGE,
    KernelScope,
    declare_kernel,
    kernel_namespace,
)
from finn.dataflow.kernels.kernel import (
    BINDING_PATH,
    BoundRegion,
    CoveragePattern,
    EdgeCoverage,
    Kernel,
    KernelOrigin,
    KernelParameter,
    PhysicalComponent,
    RegionCoverage,
    SourceFile,
    audit_elaboration,
    bind_kernel,
    bound_regions,
    check_declared_references,
    scalar_parameters,
)


__all__ = [
    "BINDING_PATH",
    "COVERAGE",
    "BoundRegion",
    "CoveragePattern",
    "EdgeCoverage",
    "KernelScope",
    "Kernel",
    "KernelOrigin",
    "KernelParameter",
    "PhysicalComponent",
    "RegionCoverage",
    "SourceFile",
    "audit_elaboration",
    "bind_kernel",
    "bound_regions",
    "check_declared_references",
    "declare_kernel",
    "kernel_namespace",
    "scalar_parameters",
]
