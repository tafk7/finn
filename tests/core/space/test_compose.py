# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""What crosses a slot, composed per template (``finn.core.space._compose``): cycles through
the Spaces a template places, and the domain rule.

Each case states what the line (``6d02612cc``: one table of every placement, its cycle
check over it and its domain rule, ``_reads.check_domains``) answered, and the composed
check answers the same: the same code, the same Decision, the same members (keys relative
to the template the cycle is found in). The domain rule is also compared with the line's,
run over every address of a model of the class (``_domains_support``), on every Space
class of the space tests and their support, and on random composites
(``_compose_support``). The refusals the templates make that the line did not are named,
each with its fix; so is a selection's owner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from support.refusals import refused

from core.space._compose_support import composite_space
from core.space._domains_support import domain_differences, line_refusals
from core.space._line_reads import line_scopes
from core.space._templates_support import defined
from finn.core.space import (
    Available,
    Decision,
    Derived,
    Param,
    Select,
    Space,
    View,
    derived,
    design_space,
    divisors_of,
    view,
)
from finn.core.space._compose import domain_refusals
from finn.core.space._template import compiled, link_template
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
    compute: Core | Other = Select(core=Core(width=channels), other=Other(lanes=channels))


class Realized(Space):
    """MatMul's shape: the core's ``width`` bound to ``datapath``, which reads
    ``realization``, a choice that is no existence guard of the core's Decision."""

    channels: int = Param()
    depthwise: bool = Param()
    realization: str = Decision(values=("native", "dense"), when=depthwise)

    @derived
    def datapath(self) -> int:
        return 2 * self.channels if self.realization == "dense" else self.channels

    compute: Core | Other = Select(core=Core(width=datapath), other=Other(lanes=channels))


def test_a_cores_domain_reading_a_formal_bound_under_its_case_compiles() -> None:
    # The line: compiles.
    point = design_space(Placing(channels=8)).with_choices(compute="core")
    assert point.compute.field(Core.pe).candidates() == Available((1, 2, 4, 8))


def test_a_cores_domain_reading_a_formal_bound_to_what_reads_a_choice_is_refused() -> None:
    # The line: "compute.core.pe: its domain's membership reads the choice realization
    # (datapath -> realization); ...". The route keeps the hop across the slot: the core's
    # formal, then the placing Space's value it is bound to.
    with refused(
        "domain-reads-choice",
        error=DefinitionError,
        owner="compute.core.pe",
        message=r"reads the choice realization \(compute\.core\.width -> datapath -> realization\)",
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
    sel: Leaf | Other = Select(leaf=Leaf(f=g), other=Other(lanes=g))


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
    assert not line_refusals(line_scopes(Reenters))


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
    diagnostic) are the line's, over every address."""
    space_type = composite_space(seed)
    try:
        link_template(space_type)
    except DefinitionError:
        return  # a cycle, refused alike (the cases above)
    line = line_refusals(line_scopes(space_type))
    assert set(domain_refusals(compiled(space_type))) == set(line)


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
    """``unit.ok``, a member one candidate has, is read by ``first``'s guard and by
    ``second`` (``second`` lowered first, ``first`` declared first); ``heat.cop`` is a member
    both candidates have."""

    unit: Pump | None = Select(none=None, pump=Pump(ok=True))

    @derived(when=cast(Any, unit).ok)
    def first(self) -> int:
        return 1

    second = View(cast(Any, unit).ok)
    # Read through a choice its candidates share.
    heat: Pump | Boiler = Select(pump=Pump(ok=True), boiler=Boiler(ok=False))
    seen = View(heat.cop)


def test_a_selections_owner_is_its_choices_key() -> None:
    """Whoever reads it, and whether one candidate has the member or several share it: the
    owner of a selection is its choice's key (the line owned one candidate's by
    its first reader in declaration order, ``first``)."""
    scopes = line_scopes(Owned)
    assert scopes.find("unit.$member.ok").owner == "unit"
    assert scopes.find("heat.$member.cop").owner == "heat"
