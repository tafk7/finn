# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The policy-neutral inspection seam for compiled exclusive branches.

A ``OneOf`` declaration lowers to ordinary engine declarations, and the engine
deliberately does not model "these five decisions belong to one alternative".
That relationship is what a specialization algorithm needs, so the compiler
publishes it here as immutable paths and case structure.

These values carry no evaluator, no point, no cost, no measurement service, and
no chosen search policy.  An algorithm reads a :class:`BranchInfo`, enumerates
and commits the ordinary selector through the public ``Engine`` operations, and
evaluates the case's own constraints and readiness profiles.  That is the whole
seam; nothing in a reusable ``Space`` declaration knows which algorithm ran.

The declarations themselves live beside the rest of the authoring vocabulary in
:mod:`finn.dataflow.model.declarations`, because ``declared_members`` has to
recognize them; keeping the info values here avoids an import cycle and gives
specialization code one module to depend on.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from finn.dataflow._engine import QualifiedPath, ValueSemantics


@dataclass(frozen=True, slots=True)
class BranchOutputInfo:
    """One selected output forwarded from whichever case is live."""

    name: str
    path: QualifiedPath
    value_semantics: ValueSemantics[object]


@dataclass(frozen=True, slots=True)
class CaseInfo:
    """One alternative and the flat declarations it owns."""

    id: str
    namespace: str
    decision_paths: tuple[QualifiedPath, ...] = ()
    #: Every derived value the case owns.  Published so a cost- or
    #: measurement-guided algorithm can *find* the value it was told to score
    #: instead of rebuilding its path from a naming convention.
    property_paths: tuple[QualifiedPath, ...] = ()
    constraint_paths: tuple[QualifiedPath, ...] = ()
    readiness_profiles: tuple[str, ...] = ()
    child_branches: tuple[str, ...] = ()

    def property_named(self, name: str) -> QualifiedPath:
        """The case-owned property with this local name."""

        suffix = f".{name}"
        for path in self.property_paths:
            if path.value.endswith(suffix):
                return path
        raise KeyError(f"case {self.id!r} owns no property named {name!r}")


@dataclass(frozen=True, slots=True)
class BranchInfo:
    """One exclusive branch: its selector, its cases, and its selected outputs.

    ``selector`` is ``None`` for a singleton, which adds no decision.  The one
    case stays visible so an algorithm can inspect a branch uniformly without
    inventing a selector that does not exist.
    """

    namespace: str
    selector: QualifiedPath | None
    cases: tuple[CaseInfo, ...]
    outputs: tuple[BranchOutputInfo, ...] = ()

    def case(self, case_id: str) -> CaseInfo:
        for candidate in self.cases:
            if candidate.id == case_id:
                return candidate
        raise KeyError(f"branch {self.namespace!r} has no case {case_id!r}")


@dataclass(frozen=True, slots=True)
class BranchCatalog:
    """Every branch in one compiled Space, outermost first, in declaration order.

    Nesting is represented flatly: a nested branch appears in ``branches`` in
    its own right and is named by the owning ``CaseInfo.child_branches``.
    """

    branches: tuple[BranchInfo, ...] = ()

    def __len__(self) -> int:
        return len(self.branches)

    def __iter__(self) -> Iterator[BranchInfo]:
        return iter(self.branches)

    def branch(self, namespace: str) -> BranchInfo:
        for candidate in self.branches:
            if candidate.namespace == namespace:
                return candidate
        raise KeyError(f"no branch is compiled at {namespace!r}")

    @property
    def namespaces(self) -> tuple[str, ...]:
        return tuple(branch.namespace for branch in self.branches)


__all__ = [
    "BranchCatalog",
    "BranchInfo",
    "BranchOutputInfo",
    "CaseInfo",
]
