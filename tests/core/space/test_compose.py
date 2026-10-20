# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""What crosses a slot, composed per template (``finn.core.space._compose``): cycles through
the Spaces a template places, what a placement does not place, and the domain rule.

Each case states what the line (``6d02612cc``: one table of every placement, its cycle
check over it and its domain rule, ``_reads.check_domains``) answered, and the composed
check answers the same: the same code, the same Decision, the same members (keys relative
to the template the cycle is found in). The domain rule is also compared with the line's,
run on ``expand(C)`` (``_domains_support``), on every Space class of the space tests and
their support, and on random composites (``_compose_support``). The refusals the
templates make that the line did not are named, each with its fix; so are the
selection's type where its choice never applies and a selection's owner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from support.refusals import refused

from core.space._compose_support import composite_space
from core.space._domains_support import domain_differences, line_refusals
from core.space._templates_support import defined
from finn.core.space import (
    Available,
    Decision,
    Derived,
    Inapplicable,
    Param,
    Space,
    View,
    derived,
    design_space,
    divisors_of,
    inspection,
    supplied,
    view,
)
from finn.core.space._compose import domain_refusals
from finn.core.space._structure import expand
from finn.core.space._template import link_template
from finn.core.space.compiler import compile_model
from finn.core.space.errors import DefinitionError

HERE = Path(__file__).resolve().parent


# -- the domain rule: existence guards across a slot -------------------------------------------


class Core(Space):
    """A core whose folding factor's domain reads its formal ``width``."""

    width: int = Param()
    pe: int = Decision(domain=divisors_of(width))


class Other(Space):
    lanes: int = Param()


class Placing(Space):
    """The core is a candidate: its formal's binding sits under the case's guard, which
    reads the selector. That guard decides whether the core's Decision exists, so the
    domain reads facts only."""

    channels: int = Param()
    compute: Core | Other = Decision({"core": Core(width=channels), "other": Other(lanes=channels)})


class Realized(Space):
    """MatMul's shape: the core's ``width`` bound to ``datapath``, which reads
    ``realization``, a choice that is no existence guard of the core's Decision."""

    channels: int = Param()
    depthwise: bool = Param()
    realization: str = Decision(values=("native", "dense"), when=depthwise)

    @derived
    def datapath(self) -> int:
        return 2 * self.channels if self.realization == "dense" else self.channels

    compute: Core | Other = Decision({"core": Core(width=datapath), "other": Other(lanes=channels)})


def test_a_cores_domain_reading_a_formal_bound_under_its_case_compiles() -> None:
    # The line: compiles.
    point = design_space(Placing(channels=8)).with_choices(compute="core")
    assert point.compute.field(Core.pe).candidates() == Available((1, 2, 4, 8))


def test_a_cores_domain_reading_a_formal_bound_to_what_reads_a_choice_is_refused() -> None:
    # The line: "compute.core.pe: its domain's membership reads the choice realization
    # (datapath -> realization); ...".
    with refused(
        "domain-reads-choice",
        error=DefinitionError,
        owner="compute.core.pe",
        message=r"reads the choice realization \(datapath -> realization\)",
    ):
        compile_model(Realized)


class Leaf(Space):
    f: int = Param()
    pe: int = Decision(domain=divisors_of(f))

    @derived
    def z(self) -> int:
        return 8


class Holder(Space):
    g: int = Param()
    sel: Leaf | Other = Decision({"leaf": Leaf(f=g), "other": Other(lanes=g)})


class Reenters(Space):
    """``k.sel.leaf.pe``'s domain reads ``f``, bound to ``k.g``, which the root binds to
    ``z`` of the same case: back into the scope holding the Decision, whose case guard is
    one of its existence guards there too."""

    k = Holder()
    k.g = cast(Any, k.sel)["leaf"].z


def test_a_walk_back_into_the_decisions_own_scope_keeps_its_existence_guards() -> None:
    # The line: compiles (walked without the case guard as an existence guard, the
    # selector would be read).
    compile_model(Reenters)
    assert not line_refusals(expand(Reenters))


def test_the_domain_rule_composed_equals_the_lines_on_the_corpus() -> None:
    modules = [f"core.space.{path.stem}" for path in sorted(HERE.glob("*.py"))] + [
        "support.space_cores",
        "support.space_pipeline",
        "support.space_walk",
    ]
    compared, differences = 0, []
    for space_type in defined(modules):
        try:
            link_template(space_type)
        except DefinitionError:
            continue
        compared += 1
        found = domain_differences(space_type)
        if found is not None:
            differences.append(found)
    assert compared > 150
    assert not differences, "\n".join(differences)


@pytest.mark.parametrize("seed", range(80))
def test_the_domain_rule_composed_equals_the_lines_on_random_composites(seed: int) -> None:
    """Every Decision either refuses: the composed rule's refusals (all of them, a
    diagnostic) are the line's, on ``expand``."""
    space_type = composite_space(seed)
    try:
        link_template(space_type)
    except DefinitionError:
        return  # a cycle, refused alike (the cases above)
    assert set(domain_refusals(link_template(space_type))) == set(line_refusals(expand(space_type)))


# -- cycles through a slot ---------------------------------------------------------------------


def plus(*, v: int) -> int:
    return v + 1


class Inner(Space):
    depth: int = Param()
    doubled = Derived(plus, aliases={"v": depth})
    tripled = Derived(plus, aliases={"v": doubled})


class Loop(Space):
    """A cycle through one slot: ``budget`` reads the child's ``doubled``, which reads
    its formal, bound to ``budget``."""

    kitchen = Inner()
    budget = Derived(plus, aliases={"v": kitchen.doubled})
    kitchen.depth = cast(int, budget)


class Deep(Space):
    """Through the child's ``doubled`` on the way to ``tripled``."""

    inner = Inner()
    seen = Derived(plus, aliases={"v": inner.tripled})
    inner.depth = cast(int, seen)


class Middle(Space):
    depth: int = Param()
    inner = Inner(depth=depth)


class Top(Space):
    """Two levels down: through the middle's binding of its child's formal."""

    middle = Middle()
    seen = Derived(plus, aliases={"v": middle.inner.tripled})
    middle.depth = cast(int, seen)


class PlacesLoop(Space):
    first = Loop()
    second = Loop()


def _cycles(space_type: type[Space]) -> list[tuple[str, ...]]:
    with refused("cyclic-dependency", error=DefinitionError) as caught:
        compile_model(space_type)
    return [
        cast(tuple[str, ...], dict(found.details)["members"]) for found in caught.error.findings
    ]


@pytest.mark.parametrize(
    ("space_type", "members"),
    [
        # Each as the line named it.
        (Loop, [("budget", "kitchen.depth", "kitchen.doubled")]),
        (Deep, [("inner.depth", "inner.doubled", "inner.tripled", "seen")]),
        (
            Top,
            [
                (
                    "middle.depth",
                    "middle.inner.depth",
                    "middle.inner.doubled",
                    "middle.inner.tripled",
                    "seen",
                )
            ],
        ),
    ],
    ids=["one slot", "through the child", "two levels"],
)
def test_a_cycle_through_a_slot_is_found_in_the_placing_template(
    space_type: type[Space], members: list[tuple[str, ...]]
) -> None:
    """It names the placing template's nodes, its bindings and imports, and the nodes of
    the Spaces it places on the way (an import stands for its target), keys relative to
    that template."""
    assert _cycles(space_type) == members


def test_a_cycle_is_found_once_in_its_template_whatever_places_it() -> None:
    # The line: once per placement, by their keys: ("first.budget", "first.kitchen.depth",
    # "first.kitchen.doubled") and the same under "second". The composed check finds it
    # once, in ``Loop``'s template, relative to it.
    assert _cycles(PlacesLoop) == [("budget", "kitchen.depth", "kitchen.doubled")]


# -- placements --------------------------------------------------------------------------------


class Memory(Space):
    contents: int = Param()

    @view
    def words(self) -> int:
        return 2 * self.contents


class Fetcher(Space):
    contents: int = Param()

    @view
    def words(self) -> int:
        return self.contents


class Direct(Space):
    contents: int = Param(required=False)
    valued = supplied(contents)
    source: Memory | Fetcher = Decision(
        {"memory": Memory, "fetch": Fetcher}, when=valued, contents=contents
    )
    memory_words = View(cast(Any, source)["memory"].words)


class PlacesDirect(Space):
    on = Direct(contents=2)
    off = Direct()


def test_a_read_of_a_candidate_not_placed_is_refused_where_it_is_placed() -> None:
    # The line, word for word.
    with refused(
        "reference-not-placed",
        error=DefinitionError,
        message=r"^off\.memory_words: Memory node placed at Direct\.source\.memory .* is not "
        r"placed in off$",
    ):
        compile_model(PlacesDirect)


@dataclass(frozen=True)
class Spec:
    enabled: bool
    lanes: int


class Lanes(Space):
    contents: int = Param()
    lanes: int = Param()

    @view
    def total(self) -> int:
        return self.contents + self.lanes


class SharedGuard(Space):
    """``spec.enabled`` guards a candidate of a choice that never applies where
    ``contents`` is unsupplied, and ``other``."""

    contents: int = Param(required=False)
    spec: Spec = Param()
    valued = supplied(contents)
    source: Lanes | None = Decision(
        {"a": Lanes(lanes=2, when=spec.enabled)}, optional=True, when=valued, contents=contents
    )

    @derived(when=spec.enabled)
    def other(self) -> int:
        return 1


class SharedOperand(Space):
    """``spec.lanes`` binds a candidate's formal, and is an operand of ``total``."""

    contents: int = Param(required=False)
    spec: Spec = Param()
    valued = supplied(contents)
    source: Lanes | None = Decision(
        {"a": Lanes(lanes=spec.lanes)}, optional=True, when=valued, contents=contents
    )
    total = View(spec.lanes + 1)


class PlacesShared(Space):
    spec: Spec = Param()
    guard_on = SharedGuard(contents=2, spec=spec)
    guard_off = SharedGuard(spec=spec)
    operand_on = SharedOperand(contents=2, spec=spec)
    operand_off = SharedOperand(spec=spec)


def test_an_expression_a_candidate_shares_with_another_reader_is_placed_for_each() -> None:
    """A projection or an expression read by a candidate of a choice that may never apply
    and by another reader is one node per Decision over nodes its readers exist under, so
    each reader's is placed wherever it is (one node, made for the candidate, was not
    placed where the choice never applies: ``reference-not-placed``). The line accepted
    both; so do templates, with the line's answers."""
    point = design_space(PlacesShared(spec=Spec(enabled=True, lanes=3)))
    assert point.query(PlacesShared.guard_on.other) == Available(1)
    assert point.query(PlacesShared.guard_off.other) == Available(1)
    assert point.query(PlacesShared.operand_on.total) == Available(4)
    assert point.query(PlacesShared.operand_off.total) == Available(4)
    chosen = point.with_choices({PlacesShared.operand_on.source: "a"})
    assert cast(Lanes, chosen.operand_on.source).total == 5
    # Where the choice applies, each reader has its own node of the projection (the line
    # had the candidate's alone, ``source.a.lanes.$project.0``).
    keys = set(expand(SharedOperand).keys)
    assert {"source.a.lanes.$project.0", "total.$project.1"} <= keys


class Leafy(Space):
    contents: int = Param()

    @view
    def width(self) -> int:
        return self.contents


class Untyped(Space):
    """``width`` declares no type: it is read through a choice of its own."""

    contents: int = Param()
    inner: Leafy | None = Decision({"leaf": Leafy}, optional=True, contents=contents)
    width = View(inner.width)  # type: ignore[union-attr]


class Typed(Space):
    contents: int = Param()

    @view
    def width(self) -> int:
        return self.contents


class Through(Space):
    contents: int = Param(required=False)
    valued = supplied(contents)
    source: Untyped | Typed = Decision({"u": Untyped, "t": Typed}, when=valued, contents=contents)
    width = View(source.width)


class PlacesThrough(Space):
    on = Through(contents=2)
    off = Through()


def test_a_selections_type_where_its_choice_never_applies_is_never_observed() -> None:
    """Where ``source`` never applies (``off``), ``source.width`` is
    read through a choice whose guard, ``supplied(contents)``, never holds there: the
    selection and its readers are inapplicable, so its value is never read; its type is
    the template's, the same at every placement (its readers are checked once). The line
    typed it as the first candidate class declaring it, here no type at all, and refused
    ``off.width`` (``output-semantics-missing``): a break, named in the record."""
    point = design_space(PlacesThrough())
    assert isinstance(point.off.query(Through.width), Inapplicable)
    assert isinstance(point.off.query(Through.source), Inapplicable)
    linked = inspection.model(point).linked
    on = linked.nodes[linked.keys["on.source.$member.width"]]
    off = linked.nodes[linked.keys["off.source.$member.width"]]
    assert off.alternatives == () and on.alternatives != ()
    assert off.semantics is on.semantics is not None


# -- the refusals templates make -----------------------------------------------------------


_SPLIT: int = Decision(values=(1, 4), name="split")


class Stream(Space):
    contents: int = Param()
    lanes: int = Param()


class OwnerPerPlacement(Space):
    """``split`` is used by a candidate of a choice that may never apply and by ``plain``:
    its owner would be the candidate's scope where the choice applies, the root's where it
    does not."""

    contents: int = Param(required=False)
    valued = supplied(contents)
    source: Stream | None = Decision(
        {"fetch": Stream(lanes=_SPLIT)}, optional=True, when=valued, contents=contents
    )
    plain = Stream(contents=1, lanes=_SPLIT)


def test_a_shared_decision_whose_owner_depends_on_the_placement_is_refused() -> None:
    # A break: the line accepted it (no class in FINN or its tests does this).
    with refused(
        "template-owner-per-placement",
        error=DefinitionError,
        owner="space.allocate",
        message=r"^split: .* declare the shared Decision in the scope common to all its uses "
        r"\(a Decision member of OwnerPerPlacement\)$",
    ):
        compile_model(OwnerPerPlacement)


# -- a selection's owner ------------------------------------------------------------------------


class Pump(Space):
    ok: bool = Param()

    @view
    def cop(self) -> int:
        return 3


class Boiler(Space):
    ok: bool = Param()

    @view
    def cop(self) -> int:
        return 1


class Owned(Space):
    """``unit.ok`` is read by ``first``'s guard and by ``second``: ``second`` is lowered
    first (members before guards), ``first`` is declared first."""

    unit: Pump | None = Decision({"pump": Pump(ok=True)}, optional=True)

    @derived(when=cast(Any, unit).ok)
    def first(self) -> int:
        return 1

    second = View(cast(Any, unit).ok)
    # Read through a choice its candidates share: the choice owns it.
    heat: Pump | Boiler = Decision({"pump": Pump(ok=True), "boiler": Boiler(ok=False)})
    seen = View(heat.cop)


def test_a_selections_owner_is_its_first_reader_in_declaration_order() -> None:
    linked = expand(Owned)
    assert linked.nodes[linked.keys["unit.$member.ok"]].owner == "first"
    # A selection the candidates share: the choice's key.
    assert linked.nodes[linked.keys["heat.$member.cop"]].owner == "heat"
