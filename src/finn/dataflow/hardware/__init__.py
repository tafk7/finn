# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The physical hardware layer: Kernels, coverage, and bindings.

A ``HardwareKernel`` covers one or more selected Region families and the edges
between them.  It owns microarchitecture, target coverage, physical-only
choices, physical parameter derivation, elaboration, and a source manifest.  It
owns no logical dataflow: the Regions it covers were selected before it, and it
imports the folding they were built from rather than choosing its own.

To contribute one, subclass ``HardwareKernel``, declare its design through
``HardwareDesign``, and hand it to ``declare_hardware_kernel``.  Add a
``HardwareKernelSelection`` only when several Kernels cover one point and
something has to choose between them.

``HardwareKernelDeclaration`` is deliberately absent from this surface.  It is
the *compiled* form -- a Kernel class with its scoped declarations already
built -- and a contributor never names it.  Assembly code that genuinely needs
the type imports it from ``finn.dataflow.hardware.kernel``.

**This contract is not finished.**  Two of the nine obligations the migration
plan lists for a physical Kernel are stubs here: artifact requirements and
implementation evidence.  Both are shaped by what the first real consumer
needs, so they are added when the MVAU migration supplies one, not guessed at
from a synthetic case.
"""

from finn.dataflow.hardware.authoring import (
    COVERAGE,
    HardwareDesign,
    declare_hardware_kernel,
    hardware_namespace,
)
from finn.dataflow.hardware.identity import (
    COMPOSED_ARTIFACT_SCHEMA_VERSION,
    DEFAULT_BUILDER,
    DEFAULT_VLNV,
    KERNEL_ARTIFACT_SCHEMA_VERSION,
    PACKAGED_ARTIFACT_SCHEMA_VERSION,
    SYNTHESIS_ARTIFACT_SCHEMA_VERSION,
    ArtifactIdentityError,
    BuilderIdentity,
    ComposedArtifactIdentity,
    KernelArtifactIdentity,
    PackagedArtifactIdentity,
    SourceIdentity,
    IpPackageArtifactIdentity,
    SynthesisArtifactIdentity,
    TargetIdentity,
    VlnvIdentity,
    composed_artifact_identity,
    kernel_artifact_identity,
)
from finn.dataflow.hardware.kernel import (
    BINDING_PATH,
    BoundRegion,
    ComputationContract,
    CoveragePattern,
    EdgeCoverage,
    HardwareKernel,
    KernelBinding,
    KernelOrigin,
    KernelParameter,
    PhysicalComponent,
    RegionCoverage,
    SourceFile,
    audit_elaboration,
    bind_hardware_kernel,
    bound_regions,
    check_declared_references,
    scalar_parameters,
)
from finn.dataflow.hardware.selection import (
    HARDWARE_KERNEL_ID_SEMANTICS,
    HardwareKernelSelection,
)
from finn.dataflow.hardware.store import (
    NO_ARTIFACT_STORE,
    ArtifactKey,
    ArtifactStore,
    ArtifactStoreError,
    EmptyArtifactStore,
    StoredArtifact,
    checked_lookup,
)

__all__ = [
    "BINDING_PATH",
    "COMPOSED_ARTIFACT_SCHEMA_VERSION",
    "COVERAGE",
    "DEFAULT_BUILDER",
    "DEFAULT_VLNV",
    "HARDWARE_KERNEL_ID_SEMANTICS",
    "KERNEL_ARTIFACT_SCHEMA_VERSION",
    "NO_ARTIFACT_STORE",
    "PACKAGED_ARTIFACT_SCHEMA_VERSION",
    "SYNTHESIS_ARTIFACT_SCHEMA_VERSION",
    "ArtifactIdentityError",
    "ArtifactKey",
    "ArtifactStore",
    "ArtifactStoreError",
    "BoundRegion",
    "BuilderIdentity",
    "ComposedArtifactIdentity",
    "ComputationContract",
    "CoveragePattern",
    "EdgeCoverage",
    "EmptyArtifactStore",
    "HardwareDesign",
    "HardwareKernel",
    "HardwareKernelSelection",
    "KernelArtifactIdentity",
    "KernelBinding",
    "KernelOrigin",
    "KernelParameter",
    "PackagedArtifactIdentity",
    "PhysicalComponent",
    "RegionCoverage",
    "SourceFile",
    "SourceIdentity",
    "StoredArtifact",
    "IpPackageArtifactIdentity",
    "SynthesisArtifactIdentity",
    "TargetIdentity",
    "VlnvIdentity",
    "audit_elaboration",
    "bind_hardware_kernel",
    "bound_regions",
    "check_declared_references",
    "checked_lookup",
    "composed_artifact_identity",
    "declare_hardware_kernel",
    "hardware_namespace",
    "kernel_artifact_identity",
    "scalar_parameters",
]
