# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Node aliases: a second name in a Space class for a node one of its members places.

``x = inner.leaf`` names the node ``inner`` places, at any depth, and through a selector
the node of the selected case (``x = compute.leaf``). The node keeps its owner: keys,
cache and stored choices are the owner's, and the alias owns nothing. Every lookup reads
one frozen table of the class's aliases (``_aliases.alias_table``): class references,
reads on a design space and in a method, read inference, presence and inspection.

Each refusal and each answer here is one the third adversarial review of the alias
amendment probed (``alias_adversarial.py``): mutation, foreign and inherited retargeting,
the reserved name, a wrapper over a wrapper over a selector, a selector over aliased
composites, a case placing nothing, a case lacking the node, and the outer-path reports.
"""

from __future__ import annotations

from typing import Any

import pytest
from support.refusals import refused

from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    Inapplicable,
    Members,
    Param,
    Rejected,
    Select,
    Space,
    constraint,
    derived,
    design_space,
    divisors_of,
    inspection,
    reject,
    view,
)
from finn.core.space._aliases import alias_table

CALLS: list[int] = []


class Leaf(Space):
    value: int = Param(default=3)
    knob: int = Decision(domain=(1, 2))

    @derived
    def answer(self) -> int:
        CALLS.append(self.value)
        return self.value * self.knob


class Holder(Space):
    value: int = Param(default=3)
    leaf = Leaf(value=value)


class Root(Space):
    inner = Holder()
    x = inner.leaf
    y = inner.leaf
    leaves = Members(Leaf.answer)

    @derived
    def double(self) -> int:
        return self.x.answer + self.y.answer


class Pick(Space):
    compute = Select(a=Holder, b=Holder)
    x = compute.leaf

    @derived
    def seen(self) -> int:
        answer: int = self.x.answer
        return answer

    @derived
    def here(self) -> bool:
        return self.present(Pick.x)


class Wrap(Space):
    mm = Pick()
    x = mm.x


class Outer(Space):
    wrap = Wrap()
    x = wrap.x

    @derived
    def seen(self) -> int:
        answer: int = self.x.answer
        return answer


class Optional(Space):
    compute = Select(real=Holder, none=None)
    x = compute.leaf

    @derived
    def seen(self) -> int:
        answer: int = self.x.answer
        return answer

    @derived
    def here(self) -> bool:
        return self.present(Optional.x)


def chosen(point: Any, **by_key: object) -> Any:
    return point.with_choices(
        {d.reference: by_key.get(d.key, 1) for d in inspection.decisions(point)}
    )


# -- identity: the node keeps its owner ----------------------------------------------------------


@pytest.mark.usefixtures("purity_unchecked")
def test_an_alias_is_its_node_under_the_owners_keys() -> None:
    point = chosen(design_space(Root()))
    assert point.x._scope is point.y._scope is point.inner.leaf._scope
    assert [d.key for d in inspection.decisions(point)] == ["inner.leaf.knob"]
    CALLS.clear()
    assert point.double == 6
    assert point.double == 6
    assert len(CALLS) == 1  # one node, evaluated once, read by two names
    assert len(point.leaves) == 0  # an alias places nothing: Members reads own children

    changed = point.x.with_choices(knob=2)
    assert [key for key, _ in inspection.states(changed.root)] == ["inner.leaf.knob"]
    assert point.x.answer == 3
    assert changed.root.x.answer == 6
    assert point.with_choices({Root.x.knob: 2}).double == 12


def test_a_class_reference_through_an_alias_is_the_owners_reference() -> None:
    assert Root.x.answer == Root.inner.leaf.answer
    assert hash(Root.y.answer) == hash(Root.inner.leaf.answer)
    assert repr(Root.x.answer).startswith("<reference x.answer (inner.leaf.answer)")
    point = chosen(design_space(Root()))
    assert point.query(Root.x.answer) == Available(3)
    assert point.present(Root.x)


def test_a_report_names_the_alias_and_its_finding_stays_the_owners() -> None:
    """A reference read through an alias describes itself by both paths; the finding the
    owner's evaluation writes is the owner's, by its canonical path (cached once)."""

    class Refusing(Space):
        @derived
        def answer(self) -> int | Rejected:
            return reject("bad", "refused here")

    class Box(Space):
        leaf = Refusing()

    class Reporting(Space):
        inner = Box()
        x = inner.leaf
        y = inner.leaf

    point = design_space(Reporting())
    for reference in (Reporting.x.answer, Reporting.y.answer):
        answer = point.query(reference)
        assert isinstance(answer, Rejected)
        assert [(f.code, f.owner) for f in answer.findings] == [("bad", "inner.leaf.answer")]
    assert "x.answer (inner.leaf.answer)" in repr(Reporting.x.answer)
    assert "y.answer (inner.leaf.answer)" in repr(Reporting.y.answer)


def test_inspection_lists_the_aliases_and_a_nodes_path() -> None:
    point = chosen(design_space(Pick()), compute="a")
    (info,) = inspection.aliases(point)
    assert info.name == "x"
    assert info.optional is False
    assert inspection.path(point.x) == "compute.a.leaf"
    entry = alias_table(Pick)["x"]
    assert entry.steps == ("compute", "leaf")
    assert [end.path for end in entry.ends] == ["compute.a.leaf", "compute.b.leaf"]


def test_an_alias_keeps_the_names_it_was_written_with() -> None:
    """Its steps are the owner's; the names written, read in the class body before the
    outer class is named, are the alias's own: an alias of an alias names the first alias."""
    assert Wrap.__dict__["x"].written == ("mm", "x")
    assert alias_table(Wrap)["x"].steps == ("mm", "compute", "leaf")
    assert Outer.__dict__["x"].written == ("wrap", "x")
    assert Root.__dict__["y"].written == ("inner", "leaf")


def test_a_parent_constraint_reads_through_an_alias() -> None:
    class Limited(Space):
        inner = Holder()
        x = inner.leaf

        @constraint
        def limit(self) -> bool:
            return self.x.knob <= 1

    point = chosen(design_space(Limited()))
    trial = point.x.try_with_choices(knob=2)
    assert not trial.accepted
    assert [item.rule for item in trial.broken] == ["limit"]


# -- mutation ------------------------------------------------------------------------------------


def test_a_class_alias_is_neither_replaced_nor_deleted() -> None:
    with refused("structure-finalized", error=DefinitionError):
        Root.x = 99  # type: ignore[assignment]
    with refused("structure-finalized", error=DefinitionError):
        del Root.x
    assert Root.x.answer == Root.inner.leaf.answer
    # A selector alias's face, set on the closed class, as a plain one (RF2 L1).
    with refused("structure-finalized", error=DefinitionError):
        Pick.z = Pick.x
    assert not any(alias.name == "z" for alias in inspection.aliases(Pick))


def test_a_configurations_alias_is_immutable() -> None:
    point = chosen(design_space(Root()))
    with pytest.raises(AttributeError, match="immutable configuration field"):
        point.x = 99
    with pytest.raises(AttributeError, match="immutable configuration field"):
        del point.x
    assert point.x.value == 3


# -- what an alias may name ----------------------------------------------------------------------


def test_an_alias_of_another_class_s_path_is_refused() -> None:
    class Foreign(Space):
        inner = Holder(value=11)

    with refused("alias-foreign", error=DefinitionError, message=r"not in .*Bad"):

        class Bad(Space):
            inner = Holder(value=22)
            x = Foreign.inner.leaf


def test_an_inherited_alias_reads_the_declaration_it_names() -> None:
    class Base(Space):
        inner = Holder(value=11)
        x = inner.leaf

    class Sub(Base):
        pass

    assert chosen(design_space(Sub())).x.value == 11


def test_an_override_never_retargets_an_alias() -> None:
    """Resolved by declaration: a subclass replacing a member the alias reads through, even
    by one of the same class, refuses the alias it inherits; it declares the alias again."""

    class Base(Space):
        inner = Holder(value=11)
        x = inner.leaf

    class Same(Base):
        inner = Holder(value=22)

    with refused("alias-target-absent", error=DefinitionError, message=r"Same\.x .*overridden"):
        design_space(Same())

    class Redeclared(Base):
        inner = Holder(value=22)
        x = inner.leaf

    assert chosen(design_space(Redeclared())).x.value == 22


def test_an_override_that_drops_the_aliased_node_is_refused() -> None:
    class Base(Space):
        inner = Holder(value=11)
        x = inner.leaf

    class Replacement(Space):
        leaf: int = Param(default=22)

    class Sub(Base):
        inner = Replacement()  # type: ignore[assignment]

    with refused("alias-target-absent", error=DefinitionError, message=r"Sub\.x"):
        design_space(Sub())

    class Selecting(Base):
        inner = Select(a=Holder, b=Holder)

    with refused("alias-target-absent", error=DefinitionError, message="overridden"):
        design_space(Selecting())


def test_a_reserved_name_is_refused() -> None:
    with refused("name-reserved", error=DefinitionError):

        class Reserved(Space):
            inner = Holder()
            query = inner.leaf  # type: ignore[assignment]


def test_an_annotated_alias_is_refused() -> None:
    with refused("alias-annotated", error=DefinitionError, message=r"write it unannotated"):

        class Annotated(Space):
            inner = Holder()
            x: Leaf = inner.leaf


def test_an_alias_takes_no_configuration_keyword() -> None:
    with refused("alias-bound", error=DefinitionError):
        Root(x=Leaf())  # type: ignore[call-arg]


def test_an_alias_of_a_selector_is_refused() -> None:
    class Box(Space):
        compute = Select(a=Holder, b=Holder)

    with refused("alias-target-not-node", error=DefinitionError, message="is a selector"):

        class Bad(Space):
            box = Box()
            x = box.compute


def test_an_alias_of_a_value_is_no_node_alias() -> None:
    """``x = inner.leaf.knob`` stays the value alias it was; a value read through a
    selector is a value reference, never a node alias."""

    class Valued(Space):
        inner = Holder()
        k = inner.leaf.knob

    assert alias_table(Valued) == {}


def test_an_alias_of_a_case_handle_is_refused() -> None:
    with refused("alias-case-handle", error=DefinitionError):

        class Bad(Space):
            compute = Select(a=Holder, b=Holder)
            x = compute["a"].leaf


def test_a_direct_child_is_placed_once_still() -> None:
    with refused("node-placed-twice", error=DefinitionError):

        class Twice(Space):
            child = Leaf()
            x = child

        design_space(Twice())


# -- through selectors ---------------------------------------------------------------------------


def test_an_alias_reads_the_selected_cases_node() -> None:
    point = design_space(Pick()).with_choices({Pick.compute: "a", Pick.compute["a"].leaf.knob: 2})
    assert inspection.path(point.x) == "compute.a.leaf"
    assert point.seen == 6
    assert point.here
    assert point.query(Pick.x.answer) == Available(6)
    assert point.present(Pick.x)
    assert repr(Pick.x) == "<node x (compute.leaf) of the selected case>"


def test_a_wrapper_over_a_wrapper_over_a_selector() -> None:
    assert repr(Outer.x) == "<node x (wrap.mm.compute.leaf) of the selected case>"
    point = design_space(Outer()).with_choices(
        {Outer.wrap.mm.compute: "b", Outer.wrap.mm.compute["b"].leaf.knob: 2}
    )
    assert inspection.path(point.x) == "wrap.mm.compute.b.leaf"
    assert point.seen == 6
    assert point.query(Outer.x.answer) == Available(6)
    assert [d.key for d in inspection.decisions(point)] == [
        "wrap.mm.compute",
        "wrap.mm.compute.a.leaf.knob",
        "wrap.mm.compute.b.leaf.knob",
    ]


def test_a_selector_over_aliased_composites() -> None:
    class Aliased(Space):
        inner = Holder()
        x = inner.leaf

    class Over(Space):
        compute = Select(a=Aliased, b=Aliased)
        x = compute.x

        @derived
        def seen(self) -> int:
            answer: int = self.x.answer
            return answer

    point = chosen(design_space(Over()), compute="b", **{"compute.b.inner.leaf.knob": 2})
    assert point.seen == 6
    assert point.x.answer == 6
    assert inspection.path(point.x) == "compute.b.inner.leaf"


def test_a_binding_down_through_a_selector_alias() -> None:
    class Fifo(Space):
        s: int = Param()

    class Bound(Space):
        compute = Select(a=Holder, b=Holder)
        x = compute.leaf
        f = Fifo(s=x.answer)

    point = design_space(Bound()).with_choices(
        {Bound.compute: "b", Bound.compute["b"].leaf.knob: 2}
    )
    assert point.f.s == 6


def test_a_dormant_case_is_not_read() -> None:
    class Broken(Leaf):
        @derived
        def answer(self) -> int:
            raise RuntimeError("dormant case was evaluated")

    class BrokenHolder(Space):
        leaf = Broken()

    class Dormant(Space):
        compute = Select(good=Holder, bad=BrokenHolder)
        x = compute.leaf

        @derived
        def seen(self) -> int:
            answer: int = self.x.answer
            return answer

    point = chosen(design_space(Dormant()), compute="good", **{"compute.bad.leaf.knob": 2})
    assert point.seen == 3


def test_a_case_placing_nothing_makes_the_alias_absent() -> None:
    point = design_space(Optional()).with_choices(compute="none")
    assert point.x is None
    assert isinstance(point.query(Optional.seen), Inapplicable)
    assert point.here is False
    assert not point.present(Optional.x)
    assert isinstance(point.query(Optional.x), Inapplicable)
    assert isinstance(point.query(Optional.x.answer), Inapplicable)
    assert alias_table(Optional)["x"].optional
    real = chosen(design_space(Optional()), compute="real")
    assert real.seen == 3
    assert real.here


def test_an_alias_of_a_guarded_member_is_absent_while_its_guard_is() -> None:
    class Guarded(Space):
        flag: bool = Param(default=False)
        inner = Holder(when=flag)
        x = inner.leaf

        @derived
        def seen(self) -> int:
            answer: int = self.x.answer
            return answer

    off = design_space(Guarded(flag=False))
    assert not off.present(Guarded.x)
    assert isinstance(off.query(Guarded.seen), Inapplicable)
    assert isinstance(off.query(Guarded.x.answer), Inapplicable)
    on = chosen(design_space(Guarded(flag=True)))
    assert on.present(Guarded.x) and on.seen == 3


def test_a_binding_down_through_a_plain_alias() -> None:
    class Fifo(Space):
        s: int = Param()

    class Bound(Space):
        inner = Holder()
        x = inner.leaf
        f = Fifo(s=x.answer)

    assert chosen(design_space(Bound())).f.s == 3


def test_the_read_rules_hold_through_an_alias() -> None:
    """A view read in, and a domain reading a choice, are refused through an alias as they
    are by the owner's path."""

    class Cost(Space):
        @view
        def amount(self) -> int:
            return 3

    class Box(Space):
        leaf = Cost()

    class Viewing(Space):
        inner = Box()
        x = inner.leaf

        @constraint
        def supported(self) -> bool:
            amount: int = self.x.amount
            return amount < 4

    with refused("view-read-in", error=DefinitionError):
        design_space(Viewing())

    class Folding(Space):
        inner = Holder()
        x = inner.leaf

        @derived
        def domain_extent(self) -> int:
            knob: int = self.x.knob
            return knob

        fold: int = Decision(domain=divisors_of(domain_extent))

    with refused("domain-reads-choice", error=DefinitionError):
        design_space(Folding())


def test_a_case_lacking_the_node_is_refused() -> None:
    class Empty(Space):
        pass

    with refused("alias-case-absent", error=DefinitionError, message=r"\['empty'\]"):

        class Missing(Space):
            compute = Select(real=Holder, empty=Empty)
            x = compute.leaf
