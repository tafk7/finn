# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative frontend for constructing ordinary dataflow design-space specs.

Three layers share one frontend and lower to one flat ``DesignSpaceSpec``:

```text
Space                 ordinary declarations, direct child composition,
                      and OneOf(Case(...), ...) exclusive branching
   |
Kernel(Space)         semantic Inputs, physical-only Decisions,
                      one Region(family, version, construct, **deps)
   |
DataflowDesign(Space) semantic Decisions, Kernels segments,
                      explicit Connections and Boundaries,
                      one selected canonical DataflowNetwork
```

`finn.dataflow._engine` stays the only validator, evaluator, point, answer,
readiness, and constraint runtime.  Nothing here introduces a nested Engine, a
nested DesignPoint, or a second answer lattice.

**Direct child composition versus `OneOf`.**  `Use(Child, ...)` places one child
Space unconditionally; `OneOf(Case(A, ...), Case(B, ...))` places exactly one of
several.  Several cases add one ordinary selector `Decision` over stable case
ids and gate every case fragment through it.  A singleton adds no selector but
keeps the same selected-output paths, so adding an alternative later renames
nothing that already existed.

**Selection policy is not here.**  A branch declaration stores no search
callback.  The compiler publishes a `BranchCatalog` of paths and case structure;
an external algorithm reads it, trials immutable successor points, and commits
the ordinary selector.  `BranchInfo` carries no evaluator, point, cost, or
measurement service, and works the same for a plain Space branch and a Kernel
segment.

**Design-owned semantics versus Kernel-owned physics.**  A choice that changes
any selected Region belongs to the enclosing Design and reaches a Kernel as a
typed `Input`; a Kernel-local `Decision` may only change physical realization.
The Kernel compiler enforces this by walking the Region property's transitive
closure, values and applicability alike: a local Decision that merely *gates*
what the Region reads still decides whether the Region is there at all.  A
Region constructor signals an infeasible request by raising `RegionRefused`,
which becomes a rejecting absence; any other exception stays an
`EvaluationError`, because a defect must not read as an infeasible point.

**Region family and version.**  `Region(...)` is one ordinary `DerivedProperty`
that also names the compact semantic family the resolved value belongs to.  That
name, plus the configured Design's role-to-node metadata, is the seam a future
annotated-ONNX carrier would need; no ONNX object enters the engine.

**Boundaries.**  Kernel artifact projection is downstream and one-way; Design
artifact topology, `DataflowOp` integration, ONNX lowering, persistence, and
input-supply policy are all deliberately absent.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from finn.dataflow.model.branching import (
    BranchCatalog,
    BranchInfo,
    BranchOutputInfo,
    CaseInfo,
)
from finn.dataflow.model.compiler import SpaceModel, compile_space, compile_space_model
from finn.dataflow.model.declarations import (
    AuthoringError,
    Case,
    ConstraintGroup,
    Decision,
    Input,
    OneOf,
    Problem,
    Readiness,
    Space,
    Use,
    constraint,
    derived,
    divisors_of,
    domain,
    finite,
    reject,
    unresolved,
)
from finn.dataflow.model.occurrence import BranchView, OccurrenceError

if TYPE_CHECKING:
    from finn.dataflow.model.design import (
        Boundary,
        Connection,
        DataflowDesign,
        Kernels,
        Sink,
        configure_design,
    )
    from finn.dataflow.model.dot_product_design import DotProductDesign
    from finn.dataflow.model.kernel import (
        Kernel,
        Parameter,
        Region,
        RegionRefused,
        configure_kernel,
    )
    from finn.dataflow.model.replay_buffer import ReplayBufferKernel

_LAZY_EXPORTS = {
    name: ("finn.dataflow.model.kernel", name)
    for name in ("Kernel", "Parameter", "Region", "RegionRefused", "configure_kernel")
}
_LAZY_EXPORTS.update(
    {
        name: ("finn.dataflow.model.design", name)
        for name in (
            "Boundary",
            "Connection",
            "DataflowDesign",
            "Kernels",
            "Sink",
            "configure_design",
        )
    }
)
_LAZY_EXPORTS.update(
    {
        name: ("finn.dataflow.model.kernel_artifacts", name)
        for name in (
            "kernel_source_derivation",
            "portable_kernel_component",
            "resolve_kernel_contributions",
        )
    }
)
_LAZY_EXPORTS.update(
    {name: ("finn.dataflow.model.dotp_axi", name) for name in ("DspBlock", "DotpAxiKernel")}
)
_LAZY_EXPORTS["ReplayBufferKernel"] = (
    "finn.dataflow.model.replay_buffer",
    "ReplayBufferKernel",
)
_LAZY_EXPORTS["DotProductDesign"] = (
    "finn.dataflow.model.dot_product_design",
    "DotProductDesign",
)


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    # authoring vocabulary shared by every layer
    "AuthoringError",
    "BranchView",
    "Case",
    "ConstraintGroup",
    "Decision",
    "Input",
    "OneOf",
    "OccurrenceError",
    "Problem",
    "Readiness",
    "Space",
    "Use",
    "constraint",
    "derived",
    "divisors_of",
    "domain",
    "finite",
    "reject",
    "unresolved",
    # the Kernel specialization
    "Kernel",
    "Parameter",
    "Region",
    "RegionRefused",
    "configure_kernel",
    # the Design specialization
    "Boundary",
    "Connection",
    "DataflowDesign",
    "Kernels",
    "Sink",
    "configure_design",
    # lowering, and the policy-neutral seam specialization code reads
    "BranchCatalog",
    "BranchInfo",
    "BranchOutputInfo",
    "CaseInfo",
    "SpaceModel",
    "compile_space",
    "compile_space_model",
    # downstream artifact projection, one-way
    "kernel_source_derivation",
    "portable_kernel_component",
    "resolve_kernel_contributions",
    # the concrete Kernels and Design this experiment authored
    "DotProductDesign",
    "DotpAxiKernel",
    "DspBlock",
    "ReplayBufferKernel",
]
