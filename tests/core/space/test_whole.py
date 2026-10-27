# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The questions about a whole design space, answered from compiled forms (``_whole``): the
flow order and its loops, the presence readers, ``inspection``'s decisions, choices,
members and statistics, the scopes below a scope.

Each equals the line-derived address reference (``6d02612cc``'s algorithms, with the
explicit Group owner/count projection in ``core.space._whole_support``), by key, for every
Space class in the tests and support modules, plus the classes below, which exercise
ways the corpus does not (a name the placing Space owns read by a method; a dynamic
method's subtree holding formals, a fresh Decision and a named shared Decision; a validity
piece reached by a guard); and answering it evaluates no member. The kernels, the Chain's
members and stage 7's chain are the kernel tree's (``kernel_ops/test_space_templates``).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from core.space._templates_support import defined, links
from core.space._whole_support import differences
from core.space.test_groups import Host as GroupHost
from finn.core.space import (
    Decision,
    Group,
    Members,
    Model,
    Param,
    Select,
    Space,
    constraint,
    derived,
    design_space,
    inspection,
)
from finn.core.space._scopes import Address
from finn.core.space._template import compiled
from finn.core.space.compiler import model_of

HERE = Path(__file__).resolve().parent

# The methods whose reads, inferred per class, gained a read against the inference per
# scope (``core.space._line_reads``): two helpers calling each other (``test_reads``), each
# reading what both read (the line kept what it found first while the cycle was open, in
# the order it met them).
GAINED = {
    "Mutual": {"by_second": ["width"]},
    "PlacesMutual": {"inner.by_first": ["inner.lanes"]},
}


def _corpus() -> list[type[Space]]:
    modules = [f"core.space.{path.stem}" for path in sorted(HERE.glob("*.py"))] + [
        "support.space_cores",
        "support.space_pipeline",
        "support.space_walk",
    ]
    return [space_type for space_type in defined(modules) if links(space_type)]


def test_the_space_tests_classes_answer_as_the_line() -> None:
    classes = _corpus()
    assert len(classes) > 150
    assert GroupHost in classes and InclusionRoot in classes  # inclusion is never filtered out
    failures: list[str] = []
    gained: dict[str, dict[str, list[str]]] = {}
    loops = 0
    for space_type in classes:
        compared = differences(space_type)
        failures.extend(compared.differences)
        loops += compared.loops
        if compared.gained:
            gained[space_type.__qualname__] = compared.gained
    assert not failures, "\n".join(failures)
    assert gained == GAINED
    assert loops  # a loop is among them


# -- the ways the corpus does not take ----------------------------------------------------------


class Inner(Space):
    depth: int = Param()
    lanes: int = Decision(values=(1, 2))


#: The name a dynamic method reads by: a module global, so its source does not fix it.
LANES = "lanes"


class Dynamic(Space):
    """A dynamic method: it may read every node of its scope and below: a formal its
    placing Space binds to a Decision of its own, and a placed Space's formal bound to a
    Decision of this class."""

    width: int = Param()
    depth: int = Decision(values=(3, 4))
    inner = Inner(depth=depth)

    @derived
    def wide(self) -> bool:
        return cast(int, getattr(self.inner, LANES)) > 1

    gated: int = Decision(values=(1, 2), when=wide)


class PlacesDynamic(Space):
    pick: bool = Decision(values=(False, True))
    owned: int = Decision(values=(5, 6))
    dynamic = Dynamic(width=owned)
    later: int = Decision(values=(1, 2), when=dynamic.wide)


class Room(Space):
    depth: int = Param()


class Pieces(Space):
    """A guard reading a derived value of a placed Space's formal, which a Decision of the
    class is bound into: the walk from the guard reaches that Decision."""

    depth: int = Decision(values=(1, 2))
    room = Room(depth=depth)

    @derived
    def flag(self) -> bool:
        return self.room.depth > 1

    @derived
    def shown(self) -> bool:
        return self.flag

    extra: int = Decision(values=(1, 2), when=shown)


@pytest.mark.parametrize(
    "space_type",
    (PlacesDynamic, Pieces),
    ids=lambda space_type: space_type.__name__,
)
def test_a_way_the_corpus_does_not_take_answers_as_the_line(space_type: type[Space]) -> None:
    compared = differences(space_type)
    assert not compared.differences, "\n".join(compared.differences)
    assert not compared.gained


def test_a_dynamic_method_reads_its_scopes_subtree_and_what_its_placing_space_owns() -> None:
    keys = [item.key for item in inspection.decisions(PlacesDynamic)]
    for read in ("owned", "dynamic.inner.lanes", "dynamic.depth"):
        assert keys.index(read) < keys.index("dynamic.gated")
        assert keys.index(read) < keys.index("later")


# -- answered from compiled forms: nothing evaluated ---------------------------------------------


def test_the_flow_order_the_readers_and_the_lists_evaluate_no_member() -> None:
    model = model_of(compiled(PlacesDynamic))  # a model of its own: nothing asked yet
    inspection.decisions(model)
    inspection.choices(model)
    inspection.members(model)
    inspection.statistics(model)
    scopes = model.scopes
    for source in (*scopes.parameters, *scopes.decisions):
        model.per_class.presence_readers(source)
    assert model.per_class._session is None  # no question ran: nothing evaluated
    assert scopes.counts()["scopes made"] == 3


def test_positions_are_places_in_the_flow_order() -> None:
    point = design_space(PlacesDynamic())
    scopes = inspection.model(point).scopes
    flow = inspection.model(point).per_class.flow()
    assert [scopes.position(index) for index in flow] == list(range(len(flow)))
    assert list(scopes.flow().order) == list(flow)


# -- Group semantics absent from the original line --------------------------------------


class IncludedLeaf(Space):
    value: int = Param(default=3)

    @derived
    def metric(self) -> int:
        return self.value


class DeclarationSet(Group):
    bound: int = Param(default=2)
    pick: int = Decision(values=(1, 2))

    @constraint
    def fits(self) -> bool:
        return self.pick <= self.bound

    child = IncludedLeaf(value=bound)
    selected: IncludedLeaf | None = Select(kept=IncludedLeaf(value=bound), absent=None)
    values = Members(IncludedLeaf.metric)


class NestedDeclarationSet(Group):
    declarations = DeclarationSet()


class IncludedSpace(Space):
    group = NestedDeclarationSet()


class InclusionRoot(Space):
    group = NestedDeclarationSet()
    real = IncludedSpace()
    outside = IncludedLeaf(value=5)
    values = Members(IncludedLeaf.metric)


@pytest.mark.parametrize("permuted,eager", ((False, False), (True, False), (False, True)))
def test_inclusion_address_reference_and_explicit_owner_contract(
    permuted: bool, eager: bool
) -> None:
    compared = differences(InclusionRoot, permuted=permuted, eager=eager)
    assert not compared.differences, "\n".join(compared.differences)
    assert not compared.gained
    model = model_of(compiled(InclusionRoot), permuted=permuted, eager=eager)
    decisions = {item.key: (item.scope, item.space_type) for item in inspection.decisions(model)}
    assert decisions == {
        "group.declarations.pick": ("", InclusionRoot),
        "group.declarations.selected": ("", InclusionRoot),
        "real.group.declarations.pick": ("real", IncludedSpace),
        "real.group.declarations.selected": ("real", IncludedSpace),
    }
    assert {item.key: item.scope for item in inspection.choices(model)} == {
        "group.declarations.selected": "",
        "real.group.declarations.selected": "real",
    }
    assert {item.path: item.scope for item in inspection.groups(model)} == {
        "group": "",
        "group.declarations": "",
        "real.group": "real",
        "real.group.declarations": "real",
    }
    members = {item.key: item.scope for item in inspection.members(model)}
    assert members["group.declarations.fits"] == ""
    assert members["real.group.declarations.fits"] == "real"
    assert members["group.declarations.child.metric"] == "group.declarations.child"
    assert members["real.group.declarations.child.metric"] == "real.group.declarations.child"
    stats = inspection.statistics(model)
    assert (stats.scopes, stats.namespaces, stats.owning_decisions) == (7, 4, 4)
    assert {node.key for node in model.scopes.all_addresses() if node.kind == "validity"} == {
        "$validity",
        "real.$validity",
        "outside.$validity",
        "group.declarations.child.$validity",
        "group.declarations.selected.kept.$validity",
        "real.group.declarations.child.$validity",
        "real.group.declarations.selected.kept.$validity",
    }
    assert model.per_class._session is None  # metadata/reference traversal asks no values


def test_address_reference_still_detects_group_owner_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = inspection._decision_info

    def legacy_owner(model: Model[Space], index: Address) -> inspection.DecisionInfo[object]:
        info = original(model, index)
        return replace(
            info, scope=index.scope.name, space_type=cast(type[Space], index.scope.space_type)
        )

    monkeypatch.setattr(inspection, "_decision_info", legacy_owner)
    compared = differences(GroupHost)
    assert any(": decisions\n" in difference for difference in compared.differences)


def test_root_and_lexical_members_keep_real_space_boundaries() -> None:
    point = design_space(InclusionRoot()).with_choices(
        {
            InclusionRoot.group.declarations.selected: "kept",
            InclusionRoot.real.group.declarations.selected: "absent",
        }
    )
    assert [(item.node, item.value) for item in point.values] == [
        ("group.declarations.child", 2),
        ("group.declarations.selected.kept", 2),
        ("outside", 5),
    ]
    assert [(item.node, item.value) for item in point.real.group.declarations.values] == [
        ("child", 2)
    ]
