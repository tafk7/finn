# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""How an answer is formed, and detached for a reader: what every evaluator of a design
space shares, whatever it keeps.

- ``Called``: what an authored function's call came to (its result, a view's or a
  constraint's assessment, whether it refused on its own values, what a method read only
  to know whether something exists); ``_finish`` and ``_finished`` make one from what the
  function returned;
- the non-values a node answers on its own: ``_unassigned``, ``_input_missing``,
  ``_inputs_absent``, ``_guard_piece``, ``_empty_domain``; ``_blocked`` (the highest
  non-value of required inputs), ``_guard_result`` and ``_guard_assessment`` (a node under
  its guard's answer), ``_present`` (the one present source of a formal);
- a domain's helpers: ``membership_arguments`` (the inputs its membership reads),
  ``_listed_members`` and ``_first_member`` (membership asked over listed values);
- equality and detachment by a member's value semantics: ``_same_value``, ``_clone``,
  ``public_value``; ``copy_result`` and ``copy_assessment`` detach a public answer, every
  value in it copied by its node's semantics;
- the parts of an answer about choices: ``_ordered`` (a validity's pieces in key order),
  ``_waits`` (what an unknown answer waits on, by key).

An address is read for its own fields (its key, kind, semantics, owner): its member
record, as its scope reads it. ``copy_assessment`` reads one edge, a view's output, and
follows it to its forwarding source where collapse did (``_forwarded``): a rule kept for
``semantics=``'s lifetime, so that a view's output is copied by the semantics it was copied
by before.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

from finn.core.reporting.findings import Finding, FindingKind

from ..errors import EvaluationError
from ..results import (
    Available,
    ConstraintAssessment,
    Inapplicable,
    NonValue,
    QueryResult,
    ReadinessAssessment,
    Rejected,
    Unresolved,
    ViewAssessment,
    assess_view,
    constraint_result,
    merged_findings,
    owned_result,
    reject,
)
from ..semantics import equal, recognize
from ..semantics import snapshot as snapshot_value

if TYPE_CHECKING:
    from .._scopes import Address, Argument
    from ..domains import Domain
    from ..inspection import Piece
    from ..semantics import ValueSemantics

Assessment: TypeAlias = ViewAssessment[object] | ConstraintAssessment

INAPPLICABLE = Inapplicable()


@dataclass(frozen=True, slots=True)
class Called:
    """What an authored function's call came to."""

    result: QueryResult[object]
    assessment: Assessment | None = None
    own_refusal: bool = False
    existence: frozenset[Address] = frozenset()


def _same_value(node: Address, left: object, right: object) -> bool:
    """Whether two values of ``node`` are equal by its semantics, each compared as a
    detached copy (equality may not change a held value)."""
    if left is right:
        return True
    semantics = node.semantics
    if semantics is None:
        return False
    role = "early cutoff"
    owner = node.owner
    if not (
        recognize(semantics, left, owner=owner, role=role)
        and recognize(semantics, right, owner=owner, role=role)
    ):
        return False
    left = snapshot_value(semantics, left, owner=owner, role=role)
    right = snapshot_value(semantics, right, owner=owner, role=role)
    return equal(semantics, left, right, owner=owner, role=role)


def _blocked(answers: Sequence[QueryResult[object]]) -> NonValue | None:
    """Required inputs keep every distinct finding of the highest-precedence non-value:
    refused, then unresolved, then inapplicable."""
    for variant in (Rejected, Unresolved, Inapplicable):
        matches = tuple(answer for answer in answers if isinstance(answer, variant))
        if matches:
            return variant(merged_findings(matches))
    return None


def _guard_result(node: Address, answer: QueryResult[object]) -> NonValue | None:
    """What a node answers under its guard's answer: nothing (it applies), inapplicable,
    or the guard's own non-value."""
    if not isinstance(answer, Available):
        return answer
    if type(answer.value) is not bool:
        raise EvaluationError(node.owner, "applicability", "guard must return bool")
    return None if answer.value else INAPPLICABLE


def _guard_assessment(node: Address, answer: NonValue) -> Assessment | None:
    if node.kind == "view":
        return assess_view(answer, owner=node.key, applicability=answer)
    if node.kind == "constraint":
        return ConstraintAssessment({node.key: answer}, answer)
    if node.kind == "validity":
        return ConstraintAssessment({}, answer)
    return None


def _clone(node: Address, value: object, *, owner: str, role: str) -> object:
    if node.semantics is None:
        raise EvaluationError(owner, role, f"{node.key} has no value semantics")
    return snapshot_value(node.semantics, value, owner=owner, role=role)


def public_value(node: Address, answer: Available[object]) -> object:
    """A value as a reader receives it: detached from the answer by its semantics (a
    validity's answer is an immutable bool)."""
    if node.kind == "validity" and type(answer.value) is bool:
        return answer.value
    return _clone(node, answer.value, owner=node.owner, role="public value snapshot")


def _unassigned(node: Address) -> Unresolved:
    return Unresolved(
        (
            Finding(
                FindingKind.BLOCKER,
                "decision-unassigned",
                node.owner,
                "decision requires a commitment",
            ),
        )
    )


def _input_missing(node: Address) -> Unresolved:
    return Unresolved(
        (
            Finding(
                FindingKind.LIMITATION,
                "input-missing",
                node.owner,
                "optional input was omitted at start",
            ),
        )
    )


def _inputs_absent(node: Address, absent: list[str]) -> Rejected:
    """A present choice's domain whose input does not apply: the input does not exist
    for these facts (inputs under the choice's own existence guards are present whenever
    it is, so it sits under a guard over facts that is false), and nothing the
    design space could store would make the domain: refused, naming each input."""
    return Rejected(
        tuple(
            Finding(
                FindingKind.REJECTION,
                "domain-input-absent",
                node.owner,
                "an input of the domain does not apply where the choice is present",
                details=(("input", owner),),
            )
            for owner in absent
        )
    )


def _guard_piece(owner: str, refusal: Rejected) -> Rejected:
    """A validity piece for a guard refused through a refused input: the guarded scope or
    member is not in play, and the refusal is the input's, carried beside."""
    return Rejected(
        (
            Finding(
                FindingKind.REJECTION,
                "guard-refused",
                owner,
                "its guard reads a refused input",
            ),
            *refusal.findings,
        )
    )


def _empty_domain(node: Address) -> Rejected:
    return reject("domain-empty", "the domain lists no value", owner=node.owner)


def _present(node: Address, answers: list[QueryResult[object]]) -> QueryResult[object]:
    """The single present source: two known present ones refuse at once; else a refused
    source refuses, and while one is unresolved the answer is unresolved."""
    active = [answer for answer in answers if not isinstance(answer, Inapplicable)]
    failures = [answer for answer in active if not isinstance(answer, Available)]
    known = len(active) - len(failures)
    if known > 1:
        failures.append(
            reject(
                "multiple-suppliers",
                f"{known} sources are present; at most one may supply this value",
                owner=node.owner,
            )
        )
    blocked = _blocked(failures)
    if blocked is not None:
        return blocked
    if not active:
        return Unresolved(
            (
                Finding(
                    FindingKind.LIMITATION,
                    "input-unsupplied",
                    node.owner,
                    "no present source supplies this value",
                ),
            )
        )
    return active[0]


def _finish(node: Address, returned: object) -> Called:
    """A callback's returned value as the node's answer and assessment."""
    if node.kind == "constraint":
        normalized = constraint_result(cast(bool | QueryResult[bool], returned), node.owner)
        return Called(
            cast(QueryResult[object], normalized),
            ConstraintAssessment({node.key: normalized}, normalized),
            isinstance(normalized, Rejected),
        )
    if isinstance(returned, (Inapplicable, Rejected, Unresolved)):
        return Called(owned_result(returned, node.owner), None, isinstance(returned, Rejected))
    if node.semantics is None:
        raise EvaluationError(node.owner, node.kind, "derived output has no value semantics")
    value = returned.value if isinstance(returned, Available) else returned
    return Called(
        Available(snapshot_value(node.semantics, value, owner=node.owner, role=node.kind))
    )


def _finished(node: Address, returned: object) -> Called:
    """``_finish``, where a returned value that fails as an answer is the owner's error."""
    try:
        return _finish(node, returned)
    except EvaluationError:
        raise
    except Exception as cause:
        raise EvaluationError(node.owner, node.kind, str(cause)) from cause


# -- domains ----------------------------------------------------------------------------


def membership_arguments(node: Address) -> tuple[Argument, ...]:
    """A Decision's domain inputs its membership and enumeration read."""
    domain = node.domain
    if domain is None:
        return ()
    named = set(domain.membership_inputs)
    return tuple(argument for argument in node.domain_arguments if argument.name in named)


def _listed_members(
    domain: Domain[object],
    values: tuple[object, ...],
    arguments: Mapping[str, object],
    *,
    semantics: ValueSemantics[object],
    owner: str,
) -> tuple[object, ...]:
    """The listed values a domain's membership does not refuse, in list order."""
    return tuple(
        value
        for value in values
        if not isinstance(
            domain.membership(value, arguments, semantics=semantics, owner=owner), Rejected
        )
    )


def _first_member(
    domain: Domain[object],
    values: tuple[object, ...],
    arguments: Mapping[str, object],
    *,
    semantics: ValueSemantics[object],
    owner: str,
) -> bool:
    """Whether a domain's membership holds any of ``values``, asked in list order up to
    the first it holds."""
    return any(
        not isinstance(
            domain.membership(value, arguments, semantics=semantics, owner=owner), Rejected
        )
        for value in values
    )


# -- the parts of an answer about choices -------------------------------------------------


def _ordered(pieces: list[Piece]) -> tuple[Piece, ...]:
    """A validity's pieces in key order (by owner)."""
    return tuple(sorted(pieces, key=lambda piece: piece.owner))


def _waits(result: object) -> tuple[str, ...]:
    """What an unresolved answer waits on: its findings' owners (the open choices and the
    missing facts), by key."""
    if not isinstance(result, Unresolved):
        return ()
    return tuple(sorted({finding.owner for finding in result.findings}))


# -- a public answer, detached -----------------------------------------------------------


def copy_result(node: Address, answer: QueryResult[object]) -> QueryResult[object]:
    """A public answer detached from where it was kept, by its semantics."""
    if not isinstance(answer, Available):
        return answer
    if node.kind == "validity" and type(answer.value) is bool:
        return answer  # an immutable bool, the pieces' AND: shared as it is
    return Available(public_value(node, answer))


def _copy_readiness(node: Address, assessment: ReadinessAssessment) -> ReadinessAssessment:
    scope = node.scope
    return ReadinessAssessment(
        {key: copy_result(scope.find(key), answer) for key, answer in assessment.results.items()},
        assessment.result,
    )


def _forwards(node: Address) -> bool:
    """A pure forwarding member: an alias that only passes its source's answer on."""
    return node.kind == "alias" and node.output is not None and node.domain is None


def _implies(required: Address | None, guard: Address | None) -> bool:
    if required is None:
        return True
    current = guard
    while current is not None:
        if current == required:
            return True
        current = current.guard
    return False


def _forwarded(view: Address) -> Address | None:
    """What a view's output is copied by: its output, or, where the output is a forwarding
    alias whose guard holds whenever the view's does, the end of its forwarding chain (each
    hop an alias forwarding under the guard of the one before). Collapse took this hop at
    the base; it is kept for ``semantics=``'s lifetime (an alias carries the declared
    semantics of the member it names, which may differ from its source's) and goes with it."""
    target = view.output
    if target is None or not _forwards(target) or not _implies(target.guard, view.guard):
        return target
    current = target
    while True:
        output = cast("Address", current.output)
        if not (_forwards(output) and _implies(output.guard, current.guard)):
            return output
        current = output


def copy_assessment(node: Address, assessment: Assessment) -> Assessment:
    """An assessment detached from where it was kept: every value in it copied by its
    semantics (a constraint's answers are exact bools and findings, shared as they are).
    A view's output is copied by its output's semantics, followed to its forwarding source
    where its guard is implied (``_forwarded``)."""
    if isinstance(assessment, ConstraintAssessment):
        return assessment
    output_node = _forwarded(node)
    output = copy_result(output_node if output_node is not None else node, assessment.output_result)
    accepted = (
        output
        if assessment.accepted_result is assessment.output_result
        else copy_result(node, assessment.accepted_result)
    )
    return ViewAssessment(
        output, _copy_readiness(node, assessment.readiness), assessment.validity, accepted
    )


__all__ = [
    "INAPPLICABLE",
    "Assessment",
    "Called",
    "copy_assessment",
    "copy_result",
    "membership_arguments",
    "public_value",
]
