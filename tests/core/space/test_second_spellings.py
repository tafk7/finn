# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""One spelling for each shape of a Space: what an author writes, and what must not move.

A Space's shapes have one spelling each. A derived value, a constraint or a view that does
not apply returns ``Inapplicable()`` from its body, reading its guard first; ``when=`` is for
a placement and a Decision, and is refused by name on a computation. ``getattr(self, name)``
and ``__reads__`` give way to a closed set of reads, and a declaration closes with its class.
A constant beside a derived value of nothing, and integer arithmetic or an attribute read on
a reference in a class body beside a named derived value, are refused by name. A view of a
reference and a view of a function are one ``view``. This file holds the hand cases of the
shapes that have one spelling.

Four groups:

- **Refused**: a spelling that has gone, refused by name, with the spelling that remains.
- **Expected failures**, strict: what a change not yet made will make true. Each fails for
  the intended reason (the refusal not yet raised, or the answer not yet given), and flips
  when the change lands: the mark comes off then.
- **Held**: what must not move. An early return answers as its guard did; a constraint over
  explicit inputs answers the same with its guard read as an input; the root declaration
  stays closed by ``design_space``.
- **Answers that follow from an early return**: a member that reads its guard answers as any
  member reading that input does, where the input is refused or waits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

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
    composite,
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


def test_the_declaration_classes_are_not_on_the_public_surface() -> None:
    """``view``, ``derived`` and ``constraint`` are the one spelling of each; the classes they
    build are the engine's, in the module that declares them."""
    for name in ("View", "Derived", "Constraint"):
        assert not hasattr(space, name)
        assert name not in space.__all__
        assert hasattr(declarations, name)
    assert {"view", "derived", "constraint"} <= set(space.__all__)


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


def test_when_on_a_derived_value_is_refused_by_name() -> None:
    with refused("computation-when", error=DefinitionError, message="Inapplicable"):

        class Gated(Space):
            on: bool = Param(default=True)

            @derived(when=on)
            def doubled(self) -> int:
                return 2


def test_when_on_a_constraint_is_refused_by_name() -> None:
    with refused("computation-when", error=DefinitionError, message="Inapplicable"):

        class Gated(Space):
            on: bool = Param(default=True)

            @constraint(when=on)
            def holds(self) -> bool:
                return True


def test_when_on_a_view_is_refused_by_name() -> None:
    with refused("computation-when", error=DefinitionError, message="Inapplicable"):

        class Gated(Space):
            on: bool = Param(default=True)

            @view(when=on)
            def shown(self) -> int:
                return 2


def test_when_on_a_view_of_a_reference_is_refused_by_name() -> None:
    with refused("computation-when", error=DefinitionError, message="Inapplicable"):

        class Gated(Space):
            on: bool = Param(default=True)
            width: int = Param(default=3)
            shown = view(width, when=on)  # type: ignore[call-overload]


def test_when_stays_on_a_placement_and_a_decision() -> None:
    class Child(Space):
        value: int = Param(default=1)

    class Gated(Space):
        on: bool = Param(default=True)
        lanes: int = Decision(values=(1, 2), when=on)
        child = Child(when=on)

    off = design_space(Gated(on=False))
    assert isinstance(off.query(Gated.lanes), Inapplicable)
    assert isinstance(design_space(Gated(on=True)).query(Gated.lanes), Unresolved)


# -- expected failures: spellings the change refuses ------------------------------------------


# -- one view: a reference or a function ------------------------------------------------------


def test_a_view_given_a_reference_answers_as_the_view_of_it_does() -> None:
    """``view(reference)`` builds what the engine's own source view builds: the same answer, the
    same source, the check taken from the source."""

    class Alike(Space):
        width: int = Param(default=3)
        by_class = declarations.View(width)
        by_decorator = view(width)

    point = design_space(Alike())
    assert point.by_class == 3
    assert point.by_decorator == 3
    assert isinstance(Alike.by_decorator, declarations.View)
    assert Alike.by_decorator.source is Alike.width
    assert Alike.by_decorator.function is None
    assert Alike.by_decorator.semantics is Alike.by_class.semantics


def test_a_view_of_a_reference_lowers_without_an_output_node_of_its_own() -> None:
    """A view of a reference reads its source's output: its node holds no ``.$output`` key,
    where a function view's does."""

    class Both(Space):
        width: int = Param(default=3)
        by_reference = view(width)

        @view
        def by_function(self) -> int:
            return self.width

    point = design_space(Both())
    keys = {
        member: {node.declaration.key for node in inspection.explain(point, reference).nodes}
        for member, reference in (("function", Both.by_function), ("reference", Both.by_reference))
    }
    assert "by_function.$output" in keys["function"]
    assert not any(key.endswith(".$output") for key in keys["reference"])
    assert point.by_reference == point.by_function == 3


def test_a_view_of_a_reference_reads_a_view() -> None:
    class Chained(Space):
        width: int = Param(default=3)
        shown = view(width)
        again = view(shown)

    assert design_space(Chained()).again == 3


def test_a_view_of_anything_but_a_reference_or_a_function_is_refused_by_name() -> None:
    with refused("view-source-invalid", error=DefinitionError, message=r"not 3 \(int\)"):

        class Constant(Space):
            shown = view(3)

    with refused("view-source-invalid", error=DefinitionError, message="not the class int"):

        class Class(Space):
            shown = view(int)

    class Function(Space):
        width: int = Param(default=3)

        def twice(self) -> int:
            return 2 * self.width

        shown = view(twice)

    assert design_space(Function()).shown == 6


def test_a_view_of_a_reference_with_aliases_is_refused_by_name() -> None:
    with refused("view-source-aliases", error=DefinitionError, message="takes no aliases"):

        class Aliased(Space):
            width: int = Param(default=3)
            shown = view(width, base=width)  # type: ignore[call-overload]


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


def test_a_declaration_added_to_a_class_after_it_is_made_is_refused_at_once() -> None:
    class Example(Space):
        pass

    with refused("structure-finalized", error=DefinitionError):
        setattr(Example, "late", Param())


class Reusable(Space):
    lanes: int = Param(default=1)


def test_a_formal_assigned_from_outside_after_the_class_is_made_is_refused_at_once() -> None:
    class Parent(Space):
        kernel = Reusable()

    with refused("declaration-frozen", error=DefinitionError, message="Parent was created"):
        Parent.kernel.lanes = 2


def test_a_field_of_a_declaration_is_refused_once_its_class_is_made() -> None:
    def other(self: object) -> int:
        return 2

    class Example(Space):
        @derived
        def value(self) -> int:
            return 1

    with refused("declaration-frozen", error=DefinitionError, message=r"Example\.value\.function"):
        cast(Any, Example.value).function = other
    assert design_space(Example()).value == 1


def test_a_formal_assigned_in_the_class_body_after_the_node_is_still_bound() -> None:
    """Held: the class body binds forward, by assignment after the node, before the class
    is made."""

    class Parent(Space):
        kernel = Reusable()
        kernel.lanes = 3

    assert design_space(Parent()).kernel.lanes == 3


def test_a_composite_is_closed_when_it_is_made() -> None:
    """A class built from data is made whole: its structure and the declarations it holds
    are closed when ``composite`` returns, before anything compiles it."""
    child = Reusable()
    child.lanes = 2
    built = composite("Built", {"child": child, "doubled": derived(lanes=child.lanes)(_doubled)})
    with refused("structure-finalized", error=DefinitionError):
        setattr(built, "late", Param())
    with refused("declaration-frozen", error=DefinitionError, message="Built was created"):
        child.lanes = 4
    assert design_space(built()).query(cast(int, getattr(built, "doubled"))) == Available(4)


def _doubled(*, lanes: int) -> int:
    return 2 * lanes


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


# -- answers that follow from an early return -------------------------------------------------
#
# The early return is the body's own answer, so a member that reads its guard and a guard that
# is refused through an input, or a view that does not apply in a Space that is refused,
# answers as any member reading a refused input does.


class RefusedSpace(Space):
    """A Space with a constraint that always refuses, a view that does not apply and a
    derived value that does not."""

    @constraint
    def never(self) -> bool | Rejected:
        return reject("never-holds", "refused on purpose")

    @derived
    def off(self) -> bool:
        return False

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


def test_a_view_that_does_not_apply_in_a_refused_space_is_refused_through_its_space() -> None:
    """A view answers refused whenever its Space is, as its answer is the Space's too; a
    derived value answers inapplicable, as no Space's validity is read for it."""
    point = design_space(RefusedSpace())
    answer = point.query(RefusedSpace.early)
    assert isinstance(answer, Rejected)
    assert {finding.code for finding in answer.findings} == {"never-holds"}
    assert isinstance(point.query(RefusedSpace.derived_early), Inapplicable)


class UnderOneRead(Space):
    """Two constraints, each reading a source that is refused first."""

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


def test_two_constraints_reading_a_refused_guard_are_each_refused_through_the_read() -> None:
    """Each answers refused in the source's code, and the Space's pieces are each constraint's,
    in the source's code alone: the refusal is counted where the source makes it."""
    point = design_space(UnderOneRead())
    for member in (UnderOneRead.first, UnderOneRead.second):
        answer = point.query(member)
        assert isinstance(answer, Rejected)
        assert {finding.code for finding in answer.findings} == {"source-refused"}
    assert _piece_codes(point) == {"first": {"source-refused"}, "second": {"source-refused"}}


class PlacedFirst(Placed):
    """``uram_present``'s shape: a constraint that reads the choice it judges first, placed
    under ``placed`` (so the rule's presence is the choice's), then a platform fact that is
    still open, as a method."""

    open_ultra: bool = Decision(values=(False, True))

    @constraint
    def uram_present(self) -> bool | Rejected:
        style = self.style
        return _absent_ultra(choice=style, platform=self.open_ultra)


def test_a_constraint_reading_its_choice_first_is_inapplicable_where_the_platform_waits() -> None:
    """Its choice absent, it answers inapplicable whatever else waits (a method stops at its
    first read with no value); a function over explicit inputs reads them all first and
    would answer what its platform does."""
    point = design_space(PlacedFirst(mode="direct"))
    assert isinstance(point.query(PlacedFirst.open_ultra), Unresolved)
    assert isinstance(point.query(PlacedFirst.uram_present), Inapplicable)
    assert inspection.validity(point).status == "valid"
    assert _piece_codes(point) == {}


def test_a_constraint_reading_its_choice_first_is_refused_where_its_guard_is_refused() -> None:
    """The choice is refused through the chain that places it: so is the constraint, in the
    chain's code alone, and its piece is the chain's, with no guard piece of its own."""
    point = design_space(PlacedFirst(mode="broken"))
    answer = point.query(PlacedFirst.uram_present)
    assert isinstance(answer, Rejected)
    assert {finding.code for finding in answer.findings} == {"chain-broken"}
    assert _piece_codes(point)["uram_present"] == {"chain-broken"}


def test_a_constraint_reading_its_choice_first_refuses_ultra_where_the_platform_has_none() -> None:
    point = design_space(PlacedFirst()).with_choices({PlacedFirst.style: "ultra"})
    assert isinstance(point.query(PlacedFirst.uram_present), Unresolved)
    without = point.try_with_choices({PlacedFirst.open_ultra: False})
    assert not without.accepted
    with_it = point.with_choices({PlacedFirst.open_ultra: True})
    assert holds(with_it.query(PlacedFirst.uram_present))


class Chosen(Space):
    """A choice ``a`` deciding whether a memory is placed, its style placed under that, and a
    platform fact still open."""

    a: int = Decision(values=(1, 2))
    open_ultra: bool = Decision(values=(False, True))

    @derived
    def placed(self) -> bool:
        return self.a == 2

    style: str = Decision(values=("lut", "ultra"), when=placed)


class ChoiceFirst(Chosen):
    @constraint
    def uram_present(self) -> bool | Rejected:
        style = self.style
        return _absent_ultra(choice=style, platform=self.open_ultra)


class GuardFirst(Chosen):
    @constraint
    def uram_present(self) -> bool | Rejected | Inapplicable:
        if not self.placed:
            return Inapplicable()
        return _absent_ultra(choice=self.style, platform=self.open_ultra)


def test_a_constraint_reading_its_choice_first_names_only_what_it_judges_standing() -> None:
    """Refusing ``ultra`` on a platform without UltraRAM: read first through the choice it
    judges, the rule names the platform alone standing, as ``a`` only decides whether the
    rule exists (a choice is read as a value, and what decides its existence is not
    followed). Read first through its guard, ``a`` is read as a value, and stands too."""
    for form, standing in ((ChoiceFirst, ("open_ultra",)), (GuardFirst, ("a", "open_ultra"))):
        point = design_space(form()).with_choices({form.a: 2, form.open_ultra: False})
        report = point.try_with_choices({form.style: "ultra"})
        assert [(rule.rule, rule.standing) for rule in report.broken] == [
            ("uram_present", standing)
        ]
