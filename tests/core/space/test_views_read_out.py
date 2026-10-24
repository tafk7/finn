# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Views are read out, never read in: the forms the reference model cannot express.

No derived value, constraint, ``when`` guard, domain or binding of a design space reads a
view; compiling a root Space that does is a definition error, ``view-read-in``. A view
reads facts, choices, derived values, its own Space's other views and its children's views.

The reference harness's matrix holds the forms the model expresses (a constraint, a derived
value, a guard and a domain reading a view). Here are the two it cannot: a child's formal
bound to a view (the model names facts at the root only) and a ``Members`` whose entries are
views (the model refuses every ``Members`` naming a view); and what the check leaves alone.

Two groups. The refused forms answer ``view-read-in``, and what the check does not refuse:
a view's own guard reading a view, a ``Members`` over views read by views only, a view's read
checked against its source, and a cycle among views found where it is found, among them the
one cycle through a derived value that stays ``cyclic-dependency``. (A kernel declaring two
control buses is ``control-bus-repeated``: that case is with the kernels', in
``tests/kernels/test_kernel_module.py``.)
"""

from __future__ import annotations

from types import MappingProxyType

import pytest
from support.refusals import refused

from finn.core.space import (
    Decision,
    DefinitionError,
    EvaluationError,
    Inapplicable,
    Members,
    Param,
    Space,
    View,
    constraint,
    derived,
    design_space,
    domain,
    view,
)


class Room(Space):
    """A child presenting its cost as a view."""

    area: int = Param(default=12)

    @view
    def cost(self) -> int:
        return self.area

    @view
    def big(self) -> bool:
        return self.area > 10


class Sized(Space):
    """A source of a view for a sibling to read."""

    lanes: int = Param(default=2)

    @view
    def carried(self) -> int:
        return 4 * self.lanes


class Sink(Space):
    """A child with a formal."""

    width: int = Param(default=1)


# -- refused: a view read in ------------------------------------------------------------------


def test_a_childs_formal_bound_to_a_siblings_view_is_view_read_in() -> None:
    """A child's formal bound to a sibling's view: the binding reads the view, so the root
    is refused ``view-read-in``."""

    class Root(Space):
        source = Sized()
        sink = Sink(width=source.carried)

    with refused("view-read-in", error=DefinitionError):
        design_space(Root())


def test_a_view_side_members_read_by_a_derived_value_is_view_read_in() -> None:
    """A ``Members`` whose entries are views (each child's ``cost``), read by a derived value:
    refused ``view-read-in`` at the root."""

    class Wing(Space):
        kitchen = Room(area=14)
        study = Room()
        costs = Members(Room.cost)

        @derived
        def claimed(self) -> int:
            return len(self.costs)

    with refused("view-read-in", error=DefinitionError):
        design_space(Wing())


def test_a_view_side_members_read_by_a_constraint_is_view_read_in() -> None:
    """A ``Members`` whose entries are views, read by a constraint: refused ``view-read-in``
    at the root."""

    class Wing(Space):
        kitchen = Room(area=14)
        study = Room()
        costs = Members(Room.cost)

        @constraint
        def two_rooms(self) -> bool:
            return len(self.costs) == 2

    with refused("view-read-in", error=DefinitionError):
        design_space(Wing())


def test_a_domain_reading_a_view_of_a_choice_is_view_read_in() -> None:
    """A domain reading a view that reads a choice: ``view-read-in``, not
    ``domain-reads-choice``. The check runs first, so the form is refused for the new reading;
    before it the refusal was the domain's own, for the choice the view reads."""

    class Picky(Space):
        depth: int = Decision(values=(1, 2, 4))

        @view
        def storage(self) -> int:
            return 8 * self.depth

    class Root(Space):
        stage = Picky()
        width: int = Decision(
            domain=domain(
                accepts=lambda *, candidate, storage: candidate <= storage,
                candidates=lambda *, storage: range(1, 4),
                storage=stage.storage,
            )
        )

    with refused("view-read-in", error=DefinitionError):
        design_space(Root())


# -- accepted: what a view may read, and the one cycle that is still a cycle ------------------


def test_a_views_own_guard_reading_a_childs_view_is_accepted() -> None:
    """A view's ``when=`` reading a child's view: accepted, and the view answers while the
    guard holds, inapplicable while it does not. The guard governs that view alone (a design
    space's rules are the only other readers, and the model has no view guards), so it is no
    rule reading a view in. (A view of the Space's own is no reference a guard takes.)"""

    class Gated(Space):
        area: int = Param(default=14)
        room = Room(area=area)
        size: int = Param(default=3)

        @view(when=room.big)
        def by_child(self) -> int:
            return 2 * self.size

    point = design_space(Gated())
    assert point.by_child == 6
    narrow = design_space(Gated(area=4))
    assert isinstance(narrow.query(Gated.by_child), Inapplicable)


def test_a_derived_value_reading_a_childs_view_bound_back_to_it_is_cyclic_dependency() -> None:
    """A derived value reads a child's view, whose body reads the child's formal, bound to
    that derived value after the class body: refused at compile as ``cyclic-dependency``, not
    ``view-read-in``. The cycle runs through no validity, so the order of the class meets it
    before the check for views read in runs; and the model cannot express a child's formal, so
    no oracle holds another answer. Either way the form is refused at compile."""

    class Child(Space):
        width: int = Param(default=1)

        @view(width=width)
        def shown(*, width: int) -> int:
            return width

    class Root(Space):
        child = Child()

        @derived(shown=child.shown)
        def again(*, shown: int) -> int:
            return shown

        child.width = again

    with refused("cyclic-dependency", error=DefinitionError):
        design_space(Root())


# -- held at the base and after ---------------------------------------------------------------


def test_a_view_side_members_read_only_by_views_is_accepted_and_answers() -> None:
    """A ``Members`` over views read only by a view (as ``KernelSpace`` reads its members'
    netlists and cycles): the Space is accepted and the view answers the sum of the
    members' views."""

    class Wing(Space):
        kitchen = Room(area=14)
        study = Room()
        costs = Members(Room.cost)

        @view
        def total(self) -> int:
            return sum(entry.value for entry in self.costs)

    point = design_space(Wing())
    assert point.total == 26
    assert [(entry.node, entry.value) for entry in point.costs] == [("kitchen", 14), ("study", 12)]


def test_a_view_reading_a_path_its_source_does_not_name_fails_read_undeclared() -> None:
    """A view whose body reads through a name that is no path from its receiver (a loop
    variable) reads what its source does not name: its answer is an evaluation failure,
    ``read-undeclared``. The read check holds a view's inferred reads, and views keep them."""

    class Hidden(Space):
        room = Room(area=3)

        @view
        def through_a_loop(self) -> int:
            for child in (self.room,):
                return child.area
            return 0

    point = design_space(Hidden())
    with pytest.raises(EvaluationError) as caught:
        point.through_a_loop
    assert caught.value.role == "read-undeclared"
    assert "room.area" in caught.value.detail


def double(*, x: int) -> int:
    return 2 * x


def _wire(declaration: View[int], **aliases: object) -> None:
    """Close a cycle: bind ``declaration``'s arguments to declarations made after it.

    No public spelling writes such a cycle: a class body cannot name a declaration made
    later in it, so these cases reach through ``aliases``. They pin a defence no author's
    class reaches; if the alias machinery changes, they go with that defence rather than
    being respelled."""
    declaration.aliases = MappingProxyType(aliases)


def test_a_cycle_among_explicit_argument_views_is_cyclic_dependency_at_compile() -> None:
    """Two views, each naming the other as its argument (``a(b)``, ``b(a)``): found when the
    class is compiled, ``cyclic-dependency``; it is no ``view-read-in``, which a view's
    reading a view is not."""
    first = View.from_function(double, semantics=None, aliases={"x": Param(default=1)})
    second = View.from_function(double, semantics=None, aliases={"x": first})
    _wire(first, x=second)

    class Loop(Space):
        a = first
        b = second

    with refused("cyclic-dependency", error=DefinitionError):
        design_space(Loop())


def test_a_view_reading_itself_by_argument_is_cyclic_dependency_at_compile() -> None:
    """A view naming itself as its argument: ``cyclic-dependency`` at compile."""
    only = View.from_function(double, semantics=None, aliases={"x": Param(default=1)})
    _wire(only, x=only)

    class Loop(Space):
        a = only

    with refused("cyclic-dependency", error=DefinitionError):
        design_space(Loop())


def test_a_view_of_a_view_closing_a_cycle_is_cyclic_dependency_at_compile() -> None:
    """``c = View(a)`` with ``a`` reading ``c`` as its argument: ``cyclic-dependency`` at
    compile."""
    first = View.from_function(double, semantics=None, aliases={"x": Param(default=1)})
    second = View(first)
    _wire(first, x=second)

    class Loop(Space):
        a = first
        c = second

    with refused("cyclic-dependency", error=DefinitionError):
        design_space(Loop())


def test_a_cycle_among_self_method_views_is_a_dependency_cycle_at_evaluation() -> None:
    """Two ``self``-method views reading each other: the Space compiles, and the first
    read meets the cycle at evaluation, ``dependency cycle`` (a ``self`` method's reads are
    found by running it)."""

    class Loop(Space):
        @view
        def a(self) -> int:
            return self.b

        @view
        def b(self) -> int:
            return self.a

    point = design_space(Loop())
    with pytest.raises(EvaluationError, match="dependency cycle") as caught:
        point.a
    assert caught.value.role == "dependency cycle"


def test_a_self_method_view_reading_itself_is_a_dependency_cycle_at_evaluation() -> None:
    """A ``self``-method view reading itself: ``dependency cycle`` at evaluation."""

    class Loop(Space):
        @view
        def a(self) -> int:
            return self.a

    point = design_space(Loop())
    with pytest.raises(EvaluationError, match="dependency cycle"):
        point.a
