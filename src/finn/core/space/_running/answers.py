# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""How an answer is formed, and detached for a reader: what every evaluator of a design
space shares, whatever it keeps.

- ``Called``: what an authored function's call came to (its result, a view's or a
  constraint's assessment, whether it refused on its own values, what a method read only
  to know whether something exists); ``_finish`` and ``_finished`` make one from what the
  function returned;
- the non-values a node answers on its own: ``_unassigned``, ``_inputs_absent``,
  ``_guard_piece``, ``_empty_domain``; ``_blocked`` (the highest non-value of required
  inputs), ``_guard_result`` and ``_guard_assessment`` (a node under its guard's answer);
- a domain's helpers: ``membership_arguments`` (the inputs its membership reads),
  ``_listed_members`` and ``_first_member`` (membership asked over listed values);
- sameness, and what a reader receives: ``_same_value`` (of one type and equal by ``==``),
  ``handed_out`` and ``public_value`` (a value itself, a mutable container a derived value or
  a view answers copied); ``copy_result`` and ``copy_assessment`` hand out a public answer;
- the parts of an answer about choices: ``_ordered`` (a validity's pieces in key order),
  ``_waits`` (what an unknown answer waits on, by key).

An address is read for its own fields (its key, kind, semantics, owner): its member
record, as its scope reads it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
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
    Rejected,
    Unresolved,
    ViewAssessment,
    assess_view,
    constraint_result,
    merged_findings,
    owned_result,
    reject,
)
from ..semantics import holds_mutable, recognize, same, unrecognized

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
    """Whether two values of ``node`` are the same value: of one type and equal by ``==``
    (``semantics.same``); an ``==`` that raises fails ``node``."""
    return same(left, right, owner=node.owner, role="early cutoff")


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


def _guard_assessment(
    node: Address, answer: NonValue, applicability: QueryResult[bool] | None = None
) -> Assessment | None:
    if node.kind == "view":
        return assess_view(
            None, validity=None, applicability=answer if applicability is None else applicability
        )
    if node.kind == "constraint":
        return ConstraintAssessment({node.key: answer}, answer)
    if node.kind == "validity":
        return ConstraintAssessment({}, answer)
    return None


def handed_out(value: object) -> object:
    """``value`` as a reader receives it: the value itself, which is immutable, but an answer
    of a derived value or a view holding a mutable value anywhere the canonical form walks
    (a dict, a list, a set, a bytearray, a dataclass that is not frozen; itself, or in a
    tuple, a named tuple, a frozen set or a frozen dataclass: a kernel's ``parameters``,
    ``semantics.holds_mutable``) is a copy of it, so no reader changes what the design space
    holds. A fact, a choice or a constant never holds one: the canonical form refuses it
    where it enters (``semantics.canonical``)."""
    return deepcopy(value) if holds_mutable(value) else value


def public_value(node: Address, answer: Available[object]) -> object:
    """A value as a reader receives it (``handed_out``)."""
    return handed_out(answer.value)


def _unassigned(node: Address) -> Unresolved:
    return Unresolved(
        (
            Finding(
                FindingKind.BLOCKER,
                "decision-unassigned",
                node.owner,
                "no choice has set this decision",
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


def _formal_value(node: Address, value: object) -> Rejected:
    """A formal bound to a value its own type does not hold (a ``T`` formal bound to a
    ``T | None`` fact left out): refused at the formal, naming the value's type."""
    assert node.semantics is not None
    return Rejected(
        (
            Finding(
                FindingKind.REJECTION,
                "formal-value",
                node.owner,
                f"the value bound to the formal is not its type: {unrecognized(node.semantics)}",
                details=(("value type", type(value).__name__),),
            ),
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
    if not recognize(node.semantics, value, owner=node.owner, role=node.kind):
        raise TypeError(unrecognized(node.semantics))
    return Called(Available(value))


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
    """What an unresolved answer waits on: its findings' owners (the open choices), by
    key."""
    if not isinstance(result, Unresolved):
        return ()
    return tuple(sorted({finding.owner for finding in result.findings}))


# -- a public answer, handed out ----------------------------------------------------------


def copy_result(node: Address, answer: QueryResult[object]) -> QueryResult[object]:
    """A public answer as a reader receives it: its value handed out (``handed_out``)."""
    if not isinstance(answer, Available):
        return answer
    value = handed_out(answer.value)
    return answer if value is answer.value else Available(value)


def copy_assessment(node: Address, assessment: Assessment) -> Assessment:
    """Hand out detached local view values; bool constraint results are immutable."""
    if isinstance(assessment, ConstraintAssessment):
        return assessment
    output = (
        None if assessment.output_result is None else copy_result(node, assessment.output_result)
    )
    result = (
        output
        if output is not None and assessment.result is assessment.output_result
        else copy_result(node, assessment.result)
    )
    return ViewAssessment(result, assessment.applicability, assessment.validity, output)


__all__ = [
    "INAPPLICABLE",
    "Assessment",
    "Called",
    "copy_assessment",
    "copy_result",
    "handed_out",
    "membership_arguments",
    "public_value",
]
