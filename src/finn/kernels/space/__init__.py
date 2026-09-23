# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The generic declarative frontend for design spaces.

One authoring language, one compiler, one occurrence lifecycle, lowering to one
flat ``DesignSpaceSpec``:

```text
Space          ordinary declarations, direct Subspace composition,
               and SubspaceChoice structural choice
   |
SpaceModel[S]  the compiled, reusable model of one authored root
   |
occurrence     an attached instance of S over one immutable point
```

This package is **layer-neutral**. ``finn.kernels.base.Kernel`` is an ordinary
``Space`` subclass. Concrete components live in ``finn.kernels.dotp`` and
``finn.kernels.mvau``; this frontend imports neither those components nor the
artifact and modeling layers that consume its declarations.

``finn.kernels._engine`` stays the only validator, evaluator, point, answer,
readiness, and constraint runtime.  Nothing here introduces a nested Engine, a
nested DesignPoint, or a second answer lattice.

**Fixed child versus structural choice.**  `Subspace(Child, ...)` used directly
as a class member places one child Space; the same declaration used inside
`SubspaceChoice({"a": Subspace(A, ...), "b": Subspace(B, ...)})` places exactly
one of several, keyed by the alternative id the mapping already gives it.  Several
alternatives add one ordinary selector `Decision` over those ids and gate every
alternative fragment through it.  A singleton adds no selector but keeps the
same selected-output paths, so adding an alternative later renames nothing that
already existed.  Both are descriptors: `pipeline.fixed` is the child
occurrence and `pipeline.implementation` is its bound `ChoiceView`.

**Selection policy is not here.**  A SubspaceChoice declaration stores no search
callback.  The compiler publishes a `BranchCatalog` of paths and case structure;
an external algorithm reads it, trials immutable successor points, and commits
the ordinary selector.  `BranchInfo` carries no evaluator, point, cost, or
measurement service, and works the same for a plain Space branch and a layer
specialization's segment.

**Construction hooks, not layer knowledge.**  A specialization customizes
compilation through ``_finalize_compilation`` and its own declaration types.
That is how the Kernel layer adds domain conveniences without this package
naming Regions, Networks, or physical artifacts.

**Boundaries.**  ``View`` assesses a declared value through generic Space
readiness and constraints. The consumer owns the value's meaning and any
artifact generation. Domain value semantics, graph lowering, persistence, and
selection policy are supplied by consumers; importing this frontend loads none
of those layers.
"""

from finn.kernels.space.branching import (
    BranchCatalog,
    BranchInfo,
    BranchOutputInfo,
    CaseInfo,
)
from finn.kernels.space.compiler import SpaceModel, compile_space, compile_space_model
from finn.kernels.space.capabilities import View
from finn.kernels.space.declarations import (
    RESERVED_LIFECYCLE_NAMES,
    RESERVED_PROTOCOL_NAMES,
    AuthoringError,
    CanonicalValueCodec,
    ConstraintGroup,
    Decision,
    Input,
    OccurrenceContext,
    PersistentCodec,
    Problem,
    Projection,
    Readiness,
    Space,
    Subspace,
    SubspaceChoice,
    allow_absent,
    allow_inapplicable,
    constraint,
    derived,
    divisors_of,
    domain,
    finite,
    reject,
    unresolved,
)
from finn.kernels.space.occurrence import (
    OccurrenceDiagnostic,
    ProjectionAssessment,
    RootFactory,
    ChoiceView,
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
    # the contributor-facing half of Decision(..., canonical=...)
    "PersistentCodec",
    "Problem",
    "Projection",
    "ProjectionAssessment",
    "Readiness",
    "Space",
    "Subspace",
    "RootFactory",
    "SubspaceChoice",
    "View",
    "ChoiceView",
    "allow_absent",
    "allow_inapplicable",
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
