# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A rule's declared answer: the non-values its return annotation admits.

A derived value or a constraint answers its value type, ``Rejected`` and ``Unresolved``
(neither makes a design point valid), and ``Inapplicable`` where its annotation names it
(``T | Inapplicable``): one it does not name, answered, is the rule's ``EvaluationError``
(``answer-undeclared``), since a constraint reading it counts as true. An override admitting
``Inapplicable`` where the answer it overrides does not is refused when its class is made
(``override-widens-answer``). An answer is read through its type aliases, unions and marker
subclasses, each in the class that declares it.
"""

from __future__ import annotations

from typing import TypeAlias, TypeAliasType, TypeVar

import pytest
from support.refusals import refused

from finn.core.space import (
    Available,
    EvaluationError,
    Inapplicable,
    Param,
    QueryResult,
    Rejected,
    Space,
    Unresolved,
    constraint,
    derived,
    design_space,
    inspection,
    reject,
)
from finn.core.space._signatures import declared_non_values
from finn.core.space.collection import collect_space
from finn.core.space.errors import DefinitionError


class Base(Space):
    n: int = Param()

    @derived
    def plain(self) -> int:
        return self.n

    @derived
    def refusing(self) -> int | Rejected:
        return reject("odd", "odd") if self.n % 2 else self.n

    @derived
    def any_answer(self) -> QueryResult[int]:
        return Inapplicable() if self.n % 2 else Available(self.n)

    @constraint
    def even(self) -> bool | Rejected:
        return self.n % 2 == 0


def test_an_override_within_its_base_s_answer_is_made() -> None:
    class Narrower(Base):
        @derived
        def refusing(self) -> int:
            return self.n

        @derived
        def any_answer(self) -> int | Rejected:
            return self.n

        @constraint
        def even(self) -> bool:
            return True

    assert design_space(Narrower(n=3)).refusing == 3


def test_a_derived_override_admitting_inapplicable_its_base_does_not_is_refused() -> None:
    """Refused by its class statement, before anything collects it."""
    with refused(
        "override-widens-answer",
        error=DefinitionError,
        message=r"admits Inapplicable, which the answer it overrides does not",
    ) as caught:

        class Absent(Base):
            @derived
            def refusing(self) -> int | Inapplicable | Rejected:
                return Inapplicable()

    assert str(dict(caught.finding.details)["declaration"]).endswith(".Absent.refusing")


def test_an_override_admitting_a_refusal_or_unresolved_its_base_does_not_is_made() -> None:
    """Every answer admits both. The Kernel's ``schedule``: its base's is
    ``Schedule | Inapplicable``, each Kernel stating one declares ``Schedule | Rejected``."""

    class Refusing(Base):
        @derived
        def plain(self) -> int | Rejected:
            return reject("odd", "odd") if self.n % 2 else self.n

    class Waiting(Base):
        @derived
        def plain(self) -> int | Unresolved:
            return self.n

    assert isinstance(design_space(Refusing(n=3)).query(Refusing.plain), Rejected)
    assert design_space(Waiting(n=3)).plain == 3


def test_a_constraint_override_admitting_inapplicable_is_refused() -> None:
    with refused("override-widens-answer", error=DefinitionError):

        class Wider(Base):
            @constraint
            def even(self) -> bool | Inapplicable | Rejected:
                return Inapplicable()


class Absent(Inapplicable):
    """A marker's subclass: read as the marker."""


T = TypeVar("T")
# ``type Maybe[T] = T | Inapplicable``, which the gates' Python 3.11 syntax refuses.
Maybe = TypeAliasType("Maybe", T | Inapplicable, type_params=(T,))
Refusable = TypeAliasType("Refusable", Maybe[T | Rejected], type_params=(T,))


def test_an_answer_is_read_through_its_type_aliases_and_marker_subclasses() -> None:
    """A ``type`` alias is expanded with its arguments, a union flattened, a marker's
    subclass read as the marker; a marker inside a value is no option of the answer."""
    assert declared_non_values(Maybe[bool]) == {Inapplicable}
    assert declared_non_values(Refusable[int]) == {Inapplicable, Rejected}
    assert declared_non_values(Maybe[Maybe[int]]) == {Inapplicable}
    assert declared_non_values(bool | Absent) == {Inapplicable}
    answer = QueryResult[int]
    assert declared_non_values(answer) == {Inapplicable, Rejected, Unresolved}
    assert declared_non_values(tuple[Inapplicable, ...]) == frozenset()
    with refused("override-widens-answer", error=DefinitionError):

        class Aliased(Base):
            @constraint
            def even(self) -> Maybe[bool]:
                return Inapplicable()

    with refused("override-widens-answer", error=DefinitionError):

        class Subclassed(Base):
            @constraint
            def even(self) -> bool | Absent:
                return Absent()

    class Within(Space):
        @constraint
        def even(self) -> Maybe[bool] | Refusable[bool]:
            return True

    class Narrower(Within):
        @constraint
        def even(self) -> bool:
            return True

    with refused("override-widens-answer", error=DefinitionError):

        class Rewidened(Narrower):
            @constraint
            def even(self) -> Refusable[bool]:
                return True


class Local(Space):
    Answer: TypeAlias = int | Inapplicable  # type: ignore[misc]  # a class-local alias

    @derived
    def number(self) -> Answer:
        return Inapplicable()


class LocalPlain(Space):
    Answer: TypeAlias = int  # type: ignore[misc]  # a class-local alias

    @derived
    def number(self) -> Answer:
        return 1


def test_each_answer_is_read_in_the_class_that_declares_it() -> None:
    """A name in an answer resolves in its own declaring class: a subclass's own ``Answer``
    neither widens its base's answer nor narrows it."""

    class Narrowing(Local):
        Answer: TypeAlias = int  # type: ignore[misc]  # a class-local alias

        @derived
        def number(self) -> int | Inapplicable:
            return Inapplicable()

    assert collect_space(Narrowing) is not None
    with refused("override-widens-answer", error=DefinitionError):

        class Widening(LocalPlain):
            Answer: TypeAlias = int | Inapplicable  # type: ignore[misc]  # a class-local alias

            @derived
            def number(self) -> int | Inapplicable:
                return Inapplicable()


def test_an_answer_not_resolvable_when_its_class_is_made_is_compared_when_collected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Forward(Base):
        @derived
        def plain(self) -> Later:  # type: ignore[name-defined]  # noqa: F821
            return Inapplicable()

    monkeypatch.setitem(globals(), "Later", int | Inapplicable)
    with refused("override-widens-answer", error=DefinitionError):
        collect_space(Forward)


class Undeclared(Space):
    """Rules answering non-values their annotations do not name."""

    n: int = Param()

    @derived
    def value(self) -> int | Rejected:
        return Inapplicable() if self.n < 0 else self.n  # type: ignore[return-value]

    @derived
    def plain(self) -> int:
        return reject("odd", "odd") if self.n % 2 else self.n  # type: ignore[return-value]

    @constraint
    def positive(self) -> bool | Rejected:
        return self.value > 0


def test_a_rule_answering_an_inapplicable_its_annotation_does_not_admit_fails() -> None:
    """The ``value`` read by a constraint: without the check, the constraint reading it
    would be inapplicable, counted true, and the point valid."""
    assert inspection.validity(design_space(Undeclared(n=2))).status == "valid"
    with pytest.raises(EvaluationError, match=r"answered Inapplicable.*\(answer-undeclared\)"):
        inspection.validity(design_space(Undeclared(n=-2)))
    with pytest.raises(EvaluationError, match=r"^value \(derived\): answered"):
        design_space(Undeclared(n=-2)).query(Undeclared.value)


def test_a_rule_answering_a_refusal_or_unresolved_undeclared_is_admitted() -> None:
    """Neither makes a point valid: a refusal its annotation does not name refuses the
    point as any refusal does."""
    rejected = design_space(Undeclared(n=3)).query(Undeclared.plain)
    assert isinstance(rejected, Rejected)
    findings = rejected.findings

    class Waits(Space):
        @derived
        def later(self) -> int:
            return Unresolved(findings)  # type: ignore[return-value]

    assert isinstance(design_space(Waits()).query(Waits.later), Unresolved)


def test_a_constraint_answering_inapplicable_undeclared_fails() -> None:
    class Holds(Space):
        @constraint
        def holds(self) -> bool | Rejected:
            return Inapplicable()  # type: ignore[return-value]

    class Declared(Space):
        @constraint
        def holds(self) -> bool | Inapplicable:
            return Inapplicable()

    with pytest.raises(EvaluationError, match="answered Inapplicable"):
        inspection.validity(design_space(Holds()))
    assert inspection.validity(design_space(Declared())).status == "valid"
