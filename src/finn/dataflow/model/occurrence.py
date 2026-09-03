# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One lifecycle-bearing occurrence over one private root runtime.

A compiled :class:`~finn.dataflow.model.compiler.SpaceModel` describes a family.
Starting it against one frozen problem produces an :class:`Occurrence`: an
object that can be asked more questions as its point becomes more complete,
without turning into a different public type each time it can answer one more.

```text
SpaceModel.start(problem)
    -> root Occurrence            one private point, one frozen problem
        .child(Use / OneOf)
            -> child Occurrence   the same point, a bound namespace
        .assign(declaration, v)
            -> successor          a new immutable point, the same view
```

**One point, several views.**  Exactly one private runtime owns the compiled
model, the engine, and the immutable point.
A child occurrence stores that runtime, its own compiled namespace, and nothing
else; it holds no nested Engine, no copy of the point, and no independently
mutable state.  Assigning through a child returns *the same child view over the
successor point*, which is why a caller can stay where it is instead of
re-navigating from the root after every choice.

**The capability boundary is the public method list.**  There is no `.point()`,
no `.engine()`, no path query, and no accessor for a compiled record.  A
contributor names declarations -- the very objects it wrote in a class body --
and the compiler resolves them.  ``QualifiedPath`` remains the engine's stable
identity and still appears inside every ``Finding``; what is withheld is the
ability to *construct* one and reach a declaration nobody offered.

**Declaration, not class, identifies an occurrence.**  One Kernel class may be
placed at several roles.  A view therefore accepts only declarations its own
class declares, and refuses one belonging to a descendant by naming every
namespace where that class occurs.  Inferring "you probably meant the only
one" is precisely the behaviour that breaks the day a second one appears.

**Two error kinds, one rule.**  A mistake about *declarations* -- naming
something outside the view's scope, assigning a derived property, selecting a
case that does not exist -- is an :class:`AuthoringError`, because the caller's
source is wrong.  A mistake about *state* -- a refused value, an uncommitted
selector -- is a ``RequestError`` carrying findings, because the caller's source
is fine and the point is not where they thought.

**Nothing here changes ``_engine``.**  Every query and commit goes through the
public ``Engine`` operations; this module adds resolution, scope, and a lock,
and subtracts capability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast, overload

from finn.dataflow._engine import (
    Answer,
    ConstraintAssessment,
    DesignPoint,
    Finding,
    FindingKind,
    ItemOutcome,
    QualifiedPath,
    ReadinessAssessment,
    RequestError,
)
from finn.dataflow.model.branching import BranchInfo
from finn.dataflow.model.compiler import (
    _CompiledBranch,
    _CompiledSpace,
    _members_of,
    _ModelSupport,
    _Ref,
    answer_for,
    resolve_value_source,
)
from finn.dataflow.model.declarations import (
    DECLARATION_TYPES,
    AuthoringError,
    ConstraintGroup,
    Decision,
    OneOf,
    Readiness,
    Space,
    Use,
    ValueSource,
)

if TYPE_CHECKING:
    from finn.dataflow.model.compiler import SpaceModel

T = TypeVar("T")
S = TypeVar("S", bound=Space)

#: Dispositions ``commit_assignments`` reports for a value that was accepted.
_ACCEPTED = ("committed", "unchanged")


# -- the private runtime ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Runtime:
    """One point and everything its lineage shares.

    Private by construction: no public occurrence method returns this object or
    any of its fields.  ``support`` carries the one Engine and the one lock the
    whole lineage uses; ``point`` is the only thing a successor replaces.
    """

    model: SpaceModel
    tree: _CompiledSpace[Space]
    point: DesignPoint

    def successor(self, point: DesignPoint) -> _Runtime:
        return _Runtime(self.model, self.tree, point)


def make_root_occurrence(
    model: SpaceModel,
    tree: _CompiledSpace[Space],
    point: DesignPoint,
) -> Occurrence:
    """Bind one frozen point to its root occurrence; called only by ``SpaceModel``."""

    return Occurrence(_Runtime(model, tree, point), tree, (tree.namespace,))


# -- the occurrence -----------------------------------------------------------


class Occurrence:
    """One namespace-bound view of one immutable point.

    Constructed by ``SpaceModel.start`` and by ``child``; there is no supported
    way to build one around a point of your own.
    """

    __slots__ = ("_runtime", "_compiled", "_scope")

    def __init__(
        self,
        runtime: _Runtime,
        compiled: _CompiledSpace[Space],
        scope: tuple[str, ...],
    ) -> None:
        self._runtime = runtime
        self._compiled = compiled
        self._scope = scope

    def __repr__(self) -> str:
        return f"Occurrence({self._compiled.owner.__name__} at {self._compiled.namespace})"

    # -- identity -------------------------------------------------------------

    @property
    def space_type(self) -> type[Space]:
        """The authored class this view is an occurrence of."""

        return self._compiled.owner

    @property
    def namespace(self) -> str:
        """This occurrence's compiled namespace, for diagnostics and logging."""

        return self._compiled.namespace

    @property
    def scope(self) -> tuple[str, ...]:
        """The root-to-here occurrence chain, in declaration vocabulary."""

        return self._scope

    @property
    def root(self) -> Occurrence:
        """The root occurrence over the same point."""

        tree = self._runtime.tree
        return Occurrence(self._runtime, tree, (tree.namespace,))

    # -- queries --------------------------------------------------------------

    def answer(self, declaration: ValueSource[T]) -> Answer[T]:
        """The current answer for one value this occurrence's class declares."""

        reference = self._value_ref(declaration, "answer")
        support = self._support
        with support.lock:
            return cast("Answer[T]", answer_for(support.engine, self._runtime.point, reference))

    @overload
    def assess(self, target: Readiness) -> ReadinessAssessment: ...

    @overload
    def assess(self, target: ConstraintGroup) -> ConstraintAssessment: ...

    def assess(
        self, target: Readiness | ConstraintGroup
    ) -> ReadinessAssessment | ConstraintAssessment:
        """Assess one readiness profile or one constraint group.

        Two return types rather than one, because they answer two different
        questions and a common supertype would only invite treating them as
        interchangeable.
        """

        support = self._support
        if isinstance(target, Readiness):
            profile = self._declared_name(
                target, (Readiness,), self._compiled.readiness_names, "readiness"
            )
            with support.lock:
                return support.engine.check_readiness(self._runtime.point, profile)
        if isinstance(target, ConstraintGroup):
            group = self._declared_name(
                target, (ConstraintGroup,), self._compiled.constraint_set_names, "constraint group"
            )
            with support.lock:
                return support.engine.evaluate_constraint_set(self._runtime.point, group)
        raise AuthoringError("assess takes a Readiness or a ConstraintGroup declaration")

    # -- navigation -----------------------------------------------------------

    def branch(self, declaration: OneOf) -> BranchInfo:
        """The policy-neutral inspection record for one branch this class owns."""

        return self._branch_member(declaration)[1].info

    def child(self, declaration: Use[S] | OneOf, case: str | None = None) -> Occurrence:
        """Descend into one child placement of this occurrence.

        For a branch, ``case`` names the alternative.  Omitting it means "the
        one that is live": the single case of a singleton, or the committed
        selector value.  An uncommitted multi-case branch is a state error, not
        an invitation to pick one.
        """

        if isinstance(declaration, Use):
            if case is not None:
                raise AuthoringError("a Use has no cases; drop the case argument")
            name = self._member_of(declaration, (Use,), "child")
            return Occurrence(self._runtime, self._compiled.child(name), (*self._scope, name))
        if isinstance(declaration, OneOf):
            name, record = self._branch_member(declaration)
            chosen = case if case is not None else self._live_case(name, record)
            return Occurrence(
                self._runtime,
                record.case(chosen).compiled,
                (*self._scope, name, chosen),
            )
        raise AuthoringError("child takes a Use or a branch declaration")

    # -- specialization -------------------------------------------------------

    def assign(self, declaration: Decision[T] | OneOf, value: object) -> Occurrence:
        """Commit one value and return this same view over the successor point.

        The receiver is never mutated.  Returning the *view* rather than the
        root is what lets a caller keep working where it is; ``.root`` reaches
        the successor root when that is what is wanted.
        """

        if isinstance(declaration, OneOf):
            return self._select(declaration, value)
        if not isinstance(declaration, Decision):
            raise AuthoringError(
                "only a Decision or a branch can be assigned; a derived value, an Input, "
                "or a problem field is decided elsewhere"
            )
        reference = self._value_ref(declaration, "assignment")
        return self._commit({reference.path: value})

    # -- internals ------------------------------------------------------------

    @property
    def _support(self) -> _ModelSupport:
        return self._runtime.model._support

    def _select(self, declaration: OneOf, value: object) -> Occurrence:
        name, record = self._branch_member(declaration)
        if not isinstance(value, str):
            raise AuthoringError(f"branch {name!r} is selected by a case id, which is a string")
        record.case(value)
        if record.selector is None:
            # A singleton adds no selector.  Naming its one case is legal and
            # is exactly the assignment a caller writes before a second
            # alternative exists; it commits nothing because there is nothing
            # to commit, and it must not silently become an error later.
            return self
        return self._commit({record.selector.path: value})

    def _commit(self, assignments: Mapping[QualifiedPath, object]) -> Occurrence:
        support = self._support
        with support.lock:
            result = support.engine.commit_assignments(self._runtime.point, assignments)
        refused = tuple(
            outcome for outcome in result.outcomes if outcome.disposition not in _ACCEPTED
        )
        if refused:
            raise RequestError(_refusal_findings(refused))
        return Occurrence(self._runtime.successor(result.point), self._compiled, self._scope)

    def _live_case(self, name: str, record: _CompiledBranch) -> str:
        if record.selector is None:
            return record.cases[0].case_id
        committed = self._runtime.point.assignments.get(record.selector.path)
        if committed is None:
            raise RequestError(
                (
                    Finding(
                        FindingKind.REQUEST,
                        "occurrence-branch-unselected",
                        record.selector.path,
                        f"branch {name!r} has no committed case; name the one you mean",
                    ),
                )
            )
        return cast(str, committed)

    def _branch_member(self, declaration: OneOf) -> tuple[str, _CompiledBranch]:
        name = self._member_of(declaration, (OneOf,), "branch")
        return name, self._compiled.branch(name)

    def _value_ref(self, declaration: object, what: str) -> _Ref[object]:
        try:
            return resolve_value_source(
                self._compiled, cast("ValueSource[object]", declaration), what
            )
        except AuthoringError:
            raise self._out_of_scope(declaration, what) from None

    def _member_of(self, declaration: object, kinds: tuple[type, ...], what: str) -> str:
        name = _members_of(self._compiled.owner, kinds).get(id(declaration))
        if name is None:
            raise self._out_of_scope(declaration, what)
        return name

    def _declared_name(
        self,
        declaration: object,
        kinds: tuple[type, ...],
        table: tuple[tuple[str, str], ...],
        what: str,
    ) -> str:
        name = self._member_of(declaration, kinds, what)
        try:
            return dict(table)[name]
        except KeyError:
            raise self._out_of_scope(declaration, what) from None

    def _out_of_scope(self, declaration: object, what: str) -> AuthoringError:
        """Refuse a declaration this view does not own, and say where it lives.

        The message is the whole point.  ``root.assign(SomeKernel.pumping, 2)``
        is a natural thing to write and a wrong thing to mean the moment that
        Kernel is placed twice, so the refusal names both placements and tells
        the caller to go through the view for the one it meant.
        """

        tree = self._runtime.tree
        occurrences = _occurrences_of(tree, declaration, (tree.namespace,))
        here = f"{self._compiled.owner.__name__} at {self._compiled.namespace}"
        if not occurrences:
            return AuthoringError(
                f"this {what} declaration is not declared by any Space in this model; "
                f"the occurrence asked was {here}"
            )
        where = ", ".join(f"{space} at {namespace}" for space, namespace in occurrences)
        return AuthoringError(
            f"this {what} declaration is not owned by {here}; it is declared by {where}. "
            "Reach that occurrence with child() and use its own view; an occurrence is "
            "never inferred from a Python class"
        )


def _refusal_findings(outcomes: tuple[ItemOutcome, ...]) -> tuple[Finding, ...]:
    """Explain a refused commit without inventing a reason it does not have."""

    findings: list[Finding] = []
    for outcome in outcomes:
        if outcome.findings:
            findings.extend(outcome.findings)
            continue
        findings.append(
            Finding(
                FindingKind.REQUEST,
                "occurrence-assignment-refused",
                outcome.path,
                f"this value was {outcome.disposition} by the {outcome.source} check",
            )
        )
    return tuple(findings)


def _occurrences_of(
    compiled: _CompiledSpace[Space],
    declaration: object,
    scope: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Every (class, namespace) in the tree whose class declares this object."""

    found: list[tuple[str, str]] = []
    if id(declaration) in _members_of(compiled.owner, DECLARATION_TYPES):
        found.append((compiled.owner.__name__, compiled.namespace))
    for name, child in compiled.children:
        found.extend(_occurrences_of(child, declaration, (*scope, name)))
    for name, branch in compiled.branches:
        for case in branch.cases:
            found.extend(_occurrences_of(case.compiled, declaration, (*scope, name, case.case_id)))
    return tuple(found)


__all__ = ["Occurrence", "make_root_occurrence"]
