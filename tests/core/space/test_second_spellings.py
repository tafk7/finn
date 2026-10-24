# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""One spelling for each shape of a Space: what an author writes, and what must not move.

The Space has a few shapes that can still be written two ways: ``View(source)`` beside
``view``, ``when=`` on a derived value, a constraint or a view beside the body's own early
return, ``getattr(self, name)`` and ``__reads__`` beside a closed set of reads, and a
declaration that stays open until the first compile beside one closed with its class. A
constant beside a derived value of nothing, and integer arithmetic or an attribute read on a
reference in a class body beside a named derived value, are refused by name: those hand
cases are the first group. This file holds the hand cases of the change that leaves one
spelling each.

Five groups:

- **Refused**: a spelling that has gone, refused by name, with the spelling that remains.
- **Expected failures**, strict: what the change makes true, each written with today's
  spelling where it must be. Each fails today for the intended reason (the refusal that is
  not yet raised, or the answer that is not yet given), and flips when the change lands:
  the mark comes off then.
- **Held**: what must not move. An early return answers as its guard did; a constraint over
  explicit inputs answers the same whether its guard is a ``when=`` or an early return; the
  root declaration stays closed by ``design_space``.
- **Today's guard spelling**: the same shapes written with ``when=`` on the computation,
  which goes with the spelling. Each states the answer the early return in the held group
  gives, so the two groups together show that the answer holds.
- **Known moves**: the answers a guard stops giving once it is an early return, each stated
  with the spelling that remains, as an expected failure beside today's answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from support.refusals import refused

from core.space._possible_support import possible
from finn.core import space
from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    Inapplicable,
    Param,
    Rejected,
    Space,
    Unresolved,
    constraint,
    declarations,
    derived,
    design_space,
    inspection,
    reject,
    view,
)
from finn.core.space._reads import summary
from finn.core.space.compiler import compile_model

NOT_RAISED = pytest.fail.Exception
"""What a ``pytest.raises`` block raises where nothing is: an expected failure of a refusal
that is not made yet."""


def holds(answer: object) -> bool:
    """Whether ``answer`` is the value true: a constraint that holds."""
    return isinstance(answer, Available) and answer.value is True


@dataclass(frozen=True)
class Rate:
    per_head: int
    heads: int

    @property
    def total(self) -> int:
        return self.per_head * self.heads


class Kitchen(Space):
    """A child with a derived value to read."""

    area: int = Param(default=3)

    @derived
    def cost(self) -> int:
        return self.area * 2


# -- refused: spellings that have gone --------------------------------------------------------


def test_const_is_gone_and_a_class_body_cannot_declare_one() -> None:
    assert not hasattr(space, "Const")
    assert not hasattr(declarations, "Const")
    with pytest.raises(AttributeError):

        class Holds(Space):
            limit = space.Const(1)  # type: ignore[attr-defined]

    class Constant(Space):
        @derived
        def limit(self) -> int:
            return 1

    assert design_space(Constant()).limit == 1


def test_arithmetic_on_a_reference_is_refused_by_name() -> None:
    with refused("reference-expression", error=DefinitionError, message="derived value") as caught:

        class Wing(Space):
            kitchen = Kitchen()
            doubled = kitchen.cost * 2

    assert "Kitchen node" in caught.finding.message and ".cost" in caught.finding.message


def test_an_attribute_read_on_a_reference_is_refused_by_name() -> None:
    class Ledger(Space):
        amount: int = Param()

    with refused("reference-projection", error=DefinitionError, message="derived value") as caught:

        class Office(Space):
            rate: Rate = Param()
            ledger = Ledger(amount=rate.total)

    assert "rate" in caught.finding.message and ".total" in caught.finding.message

    # The derived value it asks for does the same, bound to the formal.
    class Booked(Space):
        rate: Rate = Param()

        @derived(rate=rate)
        def amount(*, rate: Rate) -> int:
            return rate.total

        ledger = Ledger(amount=amount)

    assert design_space(Booked(rate=Rate(10, 3))).ledger.amount == 30


# -- expected failures: spellings the change refuses ------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=DefinitionError,
    reason="flips when view takes a reference: today it is read as a function of no annotation",
)
def test_a_view_given_a_reference_answers_as_the_view_of_it_does() -> None:
    class Alike(Space):
        width: int = Param(default=3)
        by_class = space.View(width)
        by_decorator = view(width)  # type: ignore[call-overload]

    point = design_space(Alike())
    assert point.by_class == 3
    assert point.by_decorator == 3


@pytest.mark.xfail(
    strict=True,
    raises=NOT_RAISED,
    reason="flips when when= leaves a derived value: a guard is the body's early return",
)
def test_when_on_a_derived_value_is_refused_by_name() -> None:
    with pytest.raises(DefinitionError, match="when"):

        class Gated(Space):
            on: bool = Param(default=True)

            @derived(when=on)
            def doubled(self) -> int:
                return 2

        design_space(Gated())


@pytest.mark.xfail(
    strict=True,
    raises=NOT_RAISED,
    reason="flips when when= leaves a constraint: a guard is the body's early return",
)
def test_when_on_a_constraint_is_refused_by_name() -> None:
    with pytest.raises(DefinitionError, match="when"):

        class Gated(Space):
            on: bool = Param(default=True)

            @constraint(when=on)
            def holds(self) -> bool:
                return True

        design_space(Gated())


@pytest.mark.xfail(
    strict=True,
    raises=NOT_RAISED,
    reason="flips when when= leaves a view: a guard is the body's early return",
)
def test_when_on_a_view_is_refused_by_name() -> None:
    with pytest.raises(DefinitionError, match="when"):

        class Gated(Space):
            on: bool = Param(default=True)

            @view(when=on)
            def shown(self) -> int:
                return 2

        design_space(Gated())


@pytest.mark.xfail(
    strict=True,
    raises=NOT_RAISED,
    reason="flips when a method's reads are a closed set: a computed name is a definition error",
)
def test_a_member_reading_by_a_computed_name_is_a_definition_error() -> None:
    with pytest.raises(DefinitionError, match="twice"):

        class Computed(Space):
            width: int = Param(default=3)

            @derived
            def twice(self) -> int:
                return 2 * cast(int, getattr(self, "wi" + "dth"))

        design_space(Computed())


class ReadsByComputedName(Space):
    width: int = Param(default=3)

    @view
    def shown(self) -> int:
        return cast(int, getattr(self, "wi" + "dth"))


def test_a_view_reading_by_a_computed_name_answers_and_its_explanation_lists_the_read() -> None:
    """Held: the view answers, and the evidence for it names the fact it read. Its reads are
    found as it runs."""
    point = design_space(ReadsByComputedName())
    assert point.shown == 3
    evidence = inspection.explain(point, ReadsByComputedName.shown)
    read = {
        node.declaration.key: {repr(handle) for handle in node.dependencies}
        for node in evidence.nodes
    }
    assert read["shown.$output"] == {"ValueHandle('width')"}


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="flips when a view's reads are only observed: no inference is kept for it, "
    "so none is dynamic",
)
def test_a_view_reading_by_a_computed_name_has_no_inferred_reads() -> None:
    model = compile_model(ReadsByComputedName)
    assert summary(model.scopes, model.per_class.reads)["dynamic"] == 0


@pytest.mark.xfail(
    strict=True,
    raises=NOT_RAISED,
    reason="flips when a class is closed as it is created: a later declaration is refused at once",
)
def test_a_declaration_added_to_a_class_after_it_is_made_is_refused_at_once() -> None:
    class Example(Space):
        pass

    with refused("structure-finalized", error=DefinitionError):
        setattr(Example, "late", Param())


class Reusable(Space):
    lanes: int = Param(default=1)


@pytest.mark.xfail(
    strict=True,
    raises=NOT_RAISED,
    reason="flips when a declaration a class places is closed with it: assigning its formal "
    "from outside is refused at once",
)
def test_a_formal_assigned_from_outside_after_the_class_is_made_is_refused_at_once() -> None:
    class Parent(Space):
        kernel = Reusable()

    with refused("declaration-frozen", error=DefinitionError):
        Parent.kernel.lanes = 2


# -- held: what must not move -----------------------------------------------------------------


class EarlyReturns(Space):
    """Members written with a guard read first and an early return."""

    on: bool = Param(default=True)
    size: int = Param(default=4)

    @derived
    def doubled(self) -> int | Inapplicable:
        if not self.on:
            return Inapplicable()
        return self.size * 2

    @constraint
    def small(self) -> bool | Inapplicable:
        if not self.on:
            return Inapplicable()
        return self.size < 10

    @view
    def shown(self) -> int | Inapplicable:
        if not self.on:
            return Inapplicable()
        return self.size * 3


def test_an_early_return_answers_inapplicable_while_its_guard_is_false() -> None:
    point = design_space(EarlyReturns(on=False))
    for member in (EarlyReturns.doubled, EarlyReturns.small, EarlyReturns.shown):
        assert isinstance(point.query(member), Inapplicable)


def test_an_early_return_answers_its_value_while_its_guard_is_true() -> None:
    point = design_space(EarlyReturns(on=True))
    assert point.query(EarlyReturns.doubled) == Available(8)
    assert holds(point.query(EarlyReturns.small))
    assert point.query(EarlyReturns.shown) == Available(12)


class Placed(Space):
    """What ``uram_constraint`` states of a memory placed by a guard that can be refused:
    a choice guarded by ``placed``, and a constraint over explicit inputs, the choice and a
    platform fact, that refuses the ultra style where the platform has none."""

    mode: str = Param(default="adapted")
    ultra: bool = Param(default=False)

    @derived
    def chain(self) -> str | Rejected:
        if self.mode == "broken":
            return reject("chain-broken", "no chain carries this out")
        return "none" if self.mode == "direct" else "adapted"

    @derived
    def placed(self) -> bool:
        return self.chain != "none"

    style: str = Decision(values=("lut", "ultra"), when=placed)


def _absent_ultra(*, choice: str, platform: bool) -> bool | Rejected:
    if choice == "ultra" and not platform:
        return reject("uram-absent", "the platform has no UltraRAM")
    return True


def _absent_ultra_if_placed(
    *, choice: str, platform: bool, placed: bool
) -> bool | Rejected | Inapplicable:
    if not placed:
        return Inapplicable()
    return _absent_ultra(choice=choice, platform=platform)


class PlacedEarly(Placed):
    """The constraint with its guard read first, as an input, and an early return."""

    uram_present = constraint(choice=Placed.style, platform=Placed.ultra, placed=Placed.placed)(
        _absent_ultra_if_placed
    )


def test_an_explicit_input_constraint_with_an_early_return_is_inapplicable_where_unplaced() -> None:
    point = design_space(PlacedEarly(mode="direct"))
    assert isinstance(point.query(PlacedEarly.style), Inapplicable)
    assert isinstance(point.query(PlacedEarly.uram_present), Inapplicable)
    assert inspection.validity(point).status == "valid"


def test_an_explicit_input_constraint_with_an_early_return_refuses_ultra_when_placed() -> None:
    point = design_space(PlacedEarly())
    assert isinstance(point.query(PlacedEarly.uram_present), Unresolved)
    assert possible(point, PlacedEarly.style) == ("lut",)
    with_platform = design_space(PlacedEarly(ultra=True))
    assert possible(with_platform, PlacedEarly.style) == ("lut", "ultra")
    chosen = with_platform.with_choices({PlacedEarly.style: "ultra"})
    assert holds(chosen.query(PlacedEarly.uram_present))


def test_an_explicit_input_constraint_with_an_early_return_is_refused_where_its_guard_is() -> None:
    """The guard and the choice are refused together where the chain is: the constraint
    answers the choice's refusal."""
    point = design_space(PlacedEarly(mode="broken"))
    choice = point.query(PlacedEarly.style)
    assert isinstance(choice, Rejected)
    answer = point.query(PlacedEarly.uram_present)
    assert isinstance(answer, Rejected)
    assert {finding.code for finding in answer.findings} == {"chain-broken"}


def test_a_root_declaration_stays_closed_by_design_space() -> None:
    class Room(Space):
        area: int = Param()

    node = Room(area=3)
    design_space(node)
    with refused("declaration-frozen", error=DefinitionError, message="is frozen"):
        node.area = 4


# -- today's guard spelling: goes with when= on a computation ---------------------------------


class Guarded(Space):
    """The members of ``EarlyReturns``, guarded by ``when=``."""

    on: bool = Param(default=True)
    size: int = Param(default=4)

    @derived(when=on)
    def doubled(self) -> int:
        return self.size * 2

    @constraint(when=on)
    def small(self) -> bool:
        return self.size < 10

    @view(when=on)
    def shown(self) -> int:
        return self.size * 3


def test_a_guard_answers_inapplicable_while_false_and_the_value_while_true() -> None:
    """Today's answers for a guarded member: the ones ``EarlyReturns`` gives."""
    off = design_space(Guarded(on=False))
    for member in (Guarded.doubled, Guarded.small, Guarded.shown):
        assert isinstance(off.query(member), Inapplicable)
    on = design_space(Guarded(on=True))
    assert on.query(Guarded.doubled) == Available(8)
    assert holds(on.query(Guarded.small))
    assert on.query(Guarded.shown) == Available(12)


class PlacedGuarded(Placed):
    """The constraint with the guard as its ``when=``: the real one's spelling."""

    uram_present = constraint(choice=Placed.style, platform=Placed.ultra, when=Placed.placed)(
        _absent_ultra
    )


def test_an_explicit_input_constraint_with_a_guard_is_inapplicable_where_unplaced() -> None:
    point = design_space(PlacedGuarded(mode="direct"))
    assert isinstance(point.query(PlacedGuarded.style), Inapplicable)
    assert isinstance(point.query(PlacedGuarded.uram_present), Inapplicable)
    assert inspection.validity(point).status == "valid"


def test_an_explicit_input_constraint_with_a_guard_refuses_the_ultra_style_when_placed() -> None:
    point = design_space(PlacedGuarded())
    assert isinstance(point.query(PlacedGuarded.uram_present), Unresolved)
    assert possible(point, PlacedGuarded.style) == ("lut",)
    with_platform = design_space(PlacedGuarded(ultra=True))
    assert possible(with_platform, PlacedGuarded.style) == ("lut", "ultra")
    chosen = with_platform.with_choices({PlacedGuarded.style: "ultra"})
    assert holds(chosen.query(PlacedGuarded.uram_present))


# -- known moves: what a guard that stood apart no longer does --------------------------------
#
# The early return is the body's own answer, so a guard that is refused through an input, or a
# view that does not apply in a Space that is refused, answers as any member reading a refused
# input does. Each case states today's answer for the guarded spelling, then the answer the
# early return gives (and gives today, in the spelling that stays).


class RefusedSpace(Space):
    """A Space with a constraint that always refuses, and a view guarded to not apply."""

    @constraint
    def never(self) -> bool | Rejected:
        return reject("never-holds", "refused on purpose")

    @derived
    def off(self) -> bool:
        return False

    @view(when=off)
    def guarded(self) -> int:
        return 1

    @view
    def early(self) -> int | Inapplicable:
        if not self.off:
            return Inapplicable()
        return 1

    @derived
    def derived_early(self) -> int | Inapplicable:
        if not self.off:
            return Inapplicable()
        return 1


def test_a_view_returning_early_in_a_refused_space_is_refused_through_its_space() -> None:
    """Held: the early return in a Space that is refused answers refused; a derived value
    answers inapplicable, as no Space's validity is read for it."""
    point = design_space(RefusedSpace())
    answer = point.query(RefusedSpace.early)
    assert isinstance(answer, Rejected)
    assert {finding.code for finding in answer.findings} == {"never-holds"}
    assert isinstance(point.query(RefusedSpace.derived_early), Inapplicable)


def test_a_guarded_view_is_inapplicable_in_a_refused_space_today() -> None:
    point = design_space(RefusedSpace())
    assert isinstance(point.query(RefusedSpace.guarded), Inapplicable)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="flips when a view's guard is its early return: refused through its Space "
    "where the guard answered inapplicable",
)
def test_a_guarded_view_is_refused_through_its_space_in_a_refused_space() -> None:
    point = design_space(RefusedSpace())
    answer = point.query(RefusedSpace.guarded)
    assert isinstance(answer, Rejected)
    assert {finding.code for finding in answer.findings} == {"never-holds"}


class UnderOneGuard(Space):
    """Two constraints under one guard that is refused through an input."""

    @derived
    def source(self) -> bool | Rejected:
        return reject("source-refused", "the source is refused")

    @constraint(when=source)
    def first(self) -> bool:
        return False

    @constraint(when=source)
    def second(self) -> bool:
        return False


class UnderOneRead(Space):
    """The same two constraints, each reading the source first."""

    @derived
    def source(self) -> bool | Rejected:
        return reject("source-refused", "the source is refused")

    @constraint
    def first(self) -> bool | Inapplicable:
        if not self.source:
            return Inapplicable()
        return False

    @constraint
    def second(self) -> bool | Inapplicable:
        if not self.source:
            return Inapplicable()
        return False


def _piece_codes(point: Space) -> dict[str, set[str]]:
    """Each piece of a refused Space's validity, by owner, with the codes of its findings."""
    return {
        piece.owner: {finding.code for finding in piece.result.findings}
        for piece in inspection.validity(point).pieces
    }


def test_two_guarded_constraints_under_a_refused_guard_are_inapplicable_today() -> None:
    point = design_space(UnderOneGuard())
    assert isinstance(point.query(UnderOneGuard.first), Inapplicable)
    assert isinstance(point.query(UnderOneGuard.second), Inapplicable)
    assert _piece_codes(point) == {
        "first": {"guard-refused", "source-refused"},
        "second": {"guard-refused", "source-refused"},
    }


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="flips when a guard is the body's early return: each constraint is refused through "
    "its read of the source, its piece in the source's code alone",
)
def test_two_guarded_constraints_under_a_refused_guard_are_each_refused_through_the_read() -> None:
    point = design_space(UnderOneGuard())
    for member in (UnderOneGuard.first, UnderOneGuard.second):
        answer = point.query(member)
        assert isinstance(answer, Rejected)
        assert {finding.code for finding in answer.findings} == {"source-refused"}
    assert _piece_codes(point) == {"first": {"source-refused"}, "second": {"source-refused"}}


def test_two_constraints_reading_a_refused_guard_are_each_refused_through_the_read() -> None:
    """Held: the early return's answer, which the guarded spelling's takes over."""
    point = design_space(UnderOneRead())
    for member in (UnderOneRead.first, UnderOneRead.second):
        answer = point.query(member)
        assert isinstance(answer, Rejected)
        assert {finding.code for finding in answer.findings} == {"source-refused"}
    assert _piece_codes(point) == {"first": {"source-refused"}, "second": {"source-refused"}}


def test_an_explicit_input_constraint_with_a_guard_is_inapplicable_where_it_is_refused() -> None:
    """Today's answer for a guard refused through an input: inapplicable, though the choice
    it constrains is refused."""
    point = design_space(PlacedGuarded(mode="broken"))
    assert isinstance(point.query(PlacedGuarded.style), Rejected)
    assert isinstance(point.query(PlacedGuarded.uram_present), Inapplicable)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="flips when a guard is the body's early return: the constraint is refused through "
    "its read of the guard, as the choice it constrains is",
)
def test_an_explicit_input_constraint_with_a_guard_is_refused_where_its_guard_is_refused() -> None:
    point = design_space(PlacedGuarded(mode="broken"))
    answer = point.query(PlacedGuarded.uram_present)
    assert isinstance(answer, Rejected)
    assert {finding.code for finding in answer.findings} == {"chain-broken"}
