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

Two groups. The refused forms are expected failures (strict) until the check exists. The
others hold before and after it: a ``Members`` over views read by views only, a view's read
checked against its source, and a cycle among views found where it is found.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import cast

import pytest
from support.refusals import refused

from finn.core.space import (
    DefinitionError,
    EvaluationError,
    Members,
    Param,
    Space,
    View,
    ViewKey,
    constraint,
    derived,
    design_space,
    view,
)

COST = ViewKey("cost", int)


class Room(Space):
    """A child presenting its cost as a view, exported under ``COST``."""

    area: int = Param(default=12)

    @view
    def cost(self) -> int:
        return self.area

    exports = {COST: cost}


class Sized(Space):
    """A source of a view for a sibling to read."""

    lanes: int = Param(default=2)

    @view
    def carried(self) -> int:
        return 4 * self.lanes


class Sink(Space):
    """A child with a formal."""

    width: int = Param(default=1)


# -- refused at T4b: a view read in (expected failures until the check exists) ---------------

REFUSED = pytest.mark.xfail(strict=True, reason="T4b: view-read-in")


@REFUSED
def test_a_childs_formal_bound_to_a_siblings_view_is_view_read_in() -> None:
    """A child's formal bound to a sibling's view: the binding reads the view, so the root
    is refused ``view-read-in``. (Today it builds and answers.)"""

    class Root(Space):
        source = Sized()
        sink = Sink(width=cast(int, source.carried))

    with refused("view-read-in", error=DefinitionError):
        design_space(Root())


@REFUSED
def test_a_view_side_members_read_by_a_derived_value_is_view_read_in() -> None:
    """A ``Members`` whose entries are views (each child's ``cost``, a view exported under
    one key), read by a derived value: refused ``view-read-in`` at the root. (Today it
    builds and answers.)"""

    class Wing(Space):
        kitchen = Room(area=14)
        study = Room()
        costs = Members(COST)

        @derived
        def claimed(self) -> int:
            return len(self.costs)

    with refused("view-read-in", error=DefinitionError):
        design_space(Wing())


@REFUSED
def test_a_view_side_members_read_by_a_constraint_is_view_read_in() -> None:
    """A ``Members`` whose entries are views, read by a constraint: refused ``view-read-in``
    at the root. (Today it builds and answers.)"""

    class Wing(Space):
        kitchen = Room(area=14)
        study = Room()
        costs = Members(COST)

        @constraint
        def two_rooms(self) -> bool:
            return len(self.costs) == 2

    with refused("view-read-in", error=DefinitionError):
        design_space(Wing())


# -- held at the base and at T4b's head ------------------------------------------------------


def test_a_view_side_members_read_only_by_views_is_accepted_and_answers() -> None:
    """A ``Members`` over views read only by a view (as ``KernelSpace`` reads its members'
    netlists and cycles): the Space is accepted and the view answers the sum of the
    members' views."""

    class Wing(Space):
        kitchen = Room(area=14)
        study = Room()
        costs = Members(COST)

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
