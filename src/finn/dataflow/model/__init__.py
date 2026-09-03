# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The generic declarative frontend for dataflow design spaces.

One authoring language, one compiler, one occurrence lifecycle, lowering to one
flat ``DesignSpaceSpec``:

```text
Space          ordinary declarations, direct Subspace composition,
               and Variant structural choice
   |
SpaceModel[S]  the compiled, reusable model of one authored root
   |
occurrence     an attached instance of S over one immutable point
```

This package is **layer-neutral**.  ``Kernel`` lives in
``finn.dataflow.kernels``, ``DataflowDesign`` in ``finn.dataflow.designs``, and
operation-owned Designs with their operation -- each is an ordinary ``Space``
subclass and none of them is privileged here.  Concrete implementations do not
incubate in this namespace.

``finn.dataflow._engine`` stays the only validator, evaluator, point, answer,
readiness, and constraint runtime.  Nothing here introduces a nested Engine, a
nested DesignPoint, or a second answer lattice.

**Fixed child versus structural choice.**  `Subspace(Child, ...)` used directly
as a class member places one child Space; the same declaration used inside
`Variant({"a": Subspace(A, ...), "b": Subspace(B, ...)})` places exactly one of
several, keyed by the alternative id the mapping already gives it.  Several
alternatives add one ordinary selector `Decision` over those ids and gate every
alternative fragment through it.  A singleton adds no selector but keeps the
same selected-output paths, so adding an alternative later renames nothing that
already existed.  Both are descriptors: `pipeline.fixed` is the child
occurrence and `pipeline.implementation` is its bound `VariantView`.

**Selection policy is not here.**  A Variant declaration stores no search
callback.  The compiler publishes a `BranchCatalog` of paths and case structure;
an external algorithm reads it, trials immutable successor points, and commits
the ordinary selector.  `BranchInfo` carries no evaluator, point, cost, or
measurement service, and works the same for a plain Space branch and a layer
specialization's segment.

**Construction hooks, not layer knowledge.**  A specialization customizes
compilation through ``_finalize_compilation`` and its own declaration types.
That is how the Kernel layer enforces its Region ownership rule and the Design
layer builds its Network property, without this package naming either.

**Boundaries.**  Artifact projection, ONNX lowering, persistence, and
selection policy are all deliberately absent.
"""

from finn.dataflow.model.branching import (
    BranchCatalog,
    BranchInfo,
    BranchOutputInfo,
    CaseInfo,
)
from finn.dataflow.model.compiler import SpaceModel, compile_space, compile_space_model
from finn.dataflow.model.declarations import (
    RESERVED_LIFECYCLE_NAMES,
    RESERVED_PROTOCOL_NAMES,
    AuthoringError,
    CanonicalValueCodec,
    ConstraintGroup,
    Decision,
    Input,
    OccurrenceContext,
    Problem,
    Projection,
    Readiness,
    Space,
    Subspace,
    Variant,
    constraint,
    derived,
    divisors_of,
    domain,
    finite,
    reject,
    unresolved,
)
from finn.dataflow.model.occurrence import (
    OccurrenceDiagnostic,
    ProjectionAssessment,
    VariantView,
)

__all__ = [
    # authoring vocabulary shared by every layer
    "RESERVED_LIFECYCLE_NAMES",
    "RESERVED_PROTOCOL_NAMES",
    "AuthoringError",
    "ConstraintGroup",
    "Decision",
    "Input",
    "CanonicalValueCodec",
    "OccurrenceContext",
    "OccurrenceDiagnostic",
    "Problem",
    "Projection",
    "ProjectionAssessment",
    "Readiness",
    "Space",
    "Subspace",
    "Variant",
    "VariantView",
    "constraint",
    "derived",
    "divisors_of",
    "domain",
    "finite",
    "reject",
    "unresolved",
    # lowering, and the policy-neutral seam specialization code reads
    "BranchCatalog",
    "BranchInfo",
    "BranchOutputInfo",
    "CaseInfo",
    "SpaceModel",
    "compile_space",
    "compile_space_model",
]
