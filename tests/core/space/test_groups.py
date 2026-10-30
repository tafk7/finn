# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Included primitives share a real Space's ownership and existing evaluation."""

from __future__ import annotations

import pytest
from support.refusals import refused
from support.space_groups import Bundle, Host

from finn.core.space import (
    Available,
    Decision,
    Group,
    Inapplicable,
    Members,
    Param,
    Rejected,
    Select,
    Space,
    Unresolved,
    constraint,
    derived,
    design_space,
    divisors_of,
    inspection,
    reject,
    view,
)
from finn.core.space._template import compiled
from finn.core.space.compiler import model_of
from finn.core.space.errors import DefinitionError, EvaluationError, RequestError


def test_included_members_are_lazy_and_choices_rules_belong_to_host() -> None:
    point = design_space(Host())
    assert isinstance(point.first, Bundle)
    with pytest.raises(RequestError, match="present.*node"):
        point.present(Host.first)
    assert not isinstance(point.first, Space)
    assert point.first.doubled == 6
    assert point.second.doubled == 12
    assert isinstance(point.query(Host.first.shown), Unresolved)
    assert inspection.validity(point).status == "open"
    rejected = point.try_with_choices({Host.first.lanes: 4})
    assert not rejected.accepted
    assert [rule.rule for rule in rejected.broken] == ["first.fits"]
    fixed = point.with_choices({Host.first.lanes: 2, Host.second.lanes: 4})
    assert fixed.first.shown == 3
    assert fixed.second.shown == 3
    assert fixed.first.query(Bundle.shown) == Available(3)
    assert fixed.first.inspect(Bundle.shown).result == Available(3)
    assert fixed.first.field(Bundle.lanes).get() == 2
    assert inspection.validity(fixed).status == "valid"
    keys = {node.key for node in inspection.members(fixed)}
    assert "first.fits" in keys and "second.fits" in keys
    model = inspection.model(fixed)
    assert all(
        node.key not in {"first.$validity", "second.$validity"}
        for node in model.scopes.all_addresses()
    )


def test_group_cannot_be_a_root() -> None:
    with pytest.raises(RequestError, match="Group cannot be a root"):
        design_space(Bundle(extent=2))  # type: ignore[type-var]
    with refused("not-a-space", error=DefinitionError):
        model_of(compiled(Bundle))


def test_namespace_access_and_discovery_compute_nothing() -> None:
    calls: list[str] = []

    class Lazy(Group):
        @derived
        def value(self) -> list[int]:
            calls.append("value")
            return [3]

        @view
        def shown(self) -> list[int]:
            return self.value

    class Owner(Space):
        first = Lazy()
        second = Lazy()

    point = design_space(Owner())
    first = point.first
    assert isinstance(first, Lazy) and calls == []
    assert inspection.groups(point) == (
        inspection.GroupInfo("first", "", Lazy, ("value", "shown")),
        inspection.GroupInfo("second", "", Lazy, ("value", "shown")),
    )
    assert calls == []
    assert dict(inspection.presence(point)) == {"": Available(True)}
    stats = inspection.statistics(point)
    assert stats.scopes == 1 and stats.namespaces == 2
    assert first.shown == [3] and calls == ["value"]
    first.shown.append(7)
    assert point.first.shown == [3]
    assert point.second.shown == [3] and calls == ["value", "value"]
    assert all(item.scope == "" for item in inspection.members(point))


def test_effective_overrides_rebase_included_and_sibling_references() -> None:
    class Triple(Bundle):
        @derived
        def doubled(self) -> int:
            return 3 * self.extent

    class Base(Space):
        extent: int = Param(default=2)
        first = Bundle(extent=extent)
        second = Bundle(extent=first.doubled)

    class Changed(Base):
        extent: int = Param(default=4)
        first = Triple(extent=Base.extent)

    point = design_space(Changed())
    assert point.first.doubled == 12 and point.second.extent == 12
    fixed = point.with_choices({Base.first.lanes: 2, Base.second.lanes: 4})
    assert fixed.first.shown == 6 and fixed.second.shown == 6
    assert {item.space_type for item in inspection.decisions(point)} == {Changed}
    assert {item.scope for item in inspection.decisions(point)} == {""}


def test_flat_and_included_primitives_agree_by_mapped_names() -> None:
    class Flat(Space):
        extent: int = Param(default=3)
        first_lanes: int = Decision(domain=(1, 2, 4))
        second_lanes: int = Decision(domain=(1, 2, 4))

        @derived
        def first_doubled(self) -> int:
            return 2 * self.extent

        @derived
        def second_doubled(self) -> int:
            return 2 * self.first_doubled

        @constraint
        def first_fits(self) -> bool:
            return self.first_lanes <= self.extent

        @constraint
        def second_fits(self) -> bool:
            return self.second_lanes <= self.first_doubled

        @view
        def first_shown(self) -> int:
            return self.first_doubled // self.first_lanes

        @view
        def second_shown(self) -> int:
            return self.second_doubled // self.second_lanes

    flat, grouped = design_space(Flat()), design_space(Host())
    pairs = ((Flat.first_doubled, Host.first.doubled), (Flat.second_doubled, Host.second.doubled))
    for direct, included in pairs:
        assert flat.query(direct) == grouped.query(included)
    assert inspection.validity(flat).status == inspection.validity(grouped).status == "open"
    for first, second in ((1, 2), (2, 4), (4, 1)):
        left = flat.try_with_choices({Flat.first_lanes: first, Flat.second_lanes: second})
        right = grouped.try_with_choices({Host.first.lanes: first, Host.second.lanes: second})
        assert left.accepted == right.accepted
        assert [rule.rule.replace("_", ".") for rule in left.broken] == [
            rule.rule for rule in right.broken
        ]
        if left.accepted:
            assert left.instance.first_shown == right.instance.first.shown
            assert left.instance.second_shown == right.instance.second.shown


def test_nested_inclusion_real_children_and_selectors_preserve_presence() -> None:
    class Child(Space):
        choice: int = Decision(domain=(1, 2))

        @view
        def shown(self) -> int:
            return self.choice

    class Inner(Group):
        enabled: bool = Param()
        child = Child(when=enabled)
        selected: Child | None = Select(yes=Child(), no=None)

    class Outer(Group):
        enabled: bool = Param()
        inner = Inner(enabled=enabled)

    class Owner(Space):
        enabled: bool = Param(default=False)
        declarations = Outer(enabled=enabled)

    point = design_space(Owner())
    assert isinstance(point.query(Owner.declarations.inner.child.shown), Inapplicable)
    reference = Owner.declarations.inner.child.choice
    point = point.with_choices({reference: 2, Owner.declarations.inner.selected: "no"})
    assert dict(inspection.states(point))["declarations.inner.child.choice"] == inspection.Dormant(
        2
    )
    assert inspection.validity(point).status == "valid"
    assert point.with_facts({Owner.enabled: True}).declarations.inner.child.shown == 2
    assert {info.path for info in inspection.groups(point)} == {
        "declarations",
        "declarations.inner",
    }
    assert all(
        not path.endswith("inner") and path != "declarations"
        for path, _ in inspection.presence(point)
    )
    assert inspection.statistics(point).scopes == 3  # root and the two real children


def test_unused_refused_input_and_optional_output_are_sparse() -> None:
    class Optional(Group):
        enabled: bool = Param()
        sequence: int = Param()

        @derived
        def element(self) -> int:
            return 8

        @view
        def stream(self) -> int | Inapplicable:
            if not self.enabled:
                return Inapplicable()
            return self.sequence

    class Owner(Space):
        @derived
        def unused(self) -> int | Rejected:
            return reject("unused", "not needed for an absent stream")

        port = Optional(enabled=False, sequence=unused)

    point = design_space(Owner())
    assert point.port.element == 8
    assert isinstance(point.query(Owner.port.stream), Inapplicable)
    trace = inspection.explain(point, Owner.port.stream)
    assert all(node.declaration.key != "unused" for node in trace.nodes)


def test_group_placement_bindings_and_namespace_values_are_frozen() -> None:
    declared = Bundle(extent=2)

    class Owner(Space):
        group = declared

    with refused("declaration-frozen", error=DefinitionError):
        declared.extent = 4
    with refused("structure-finalized", error=DefinitionError):
        Bundle.extent = Param(default=4)
    point = design_space(Owner())
    with pytest.raises(AttributeError, match="immutable"):
        point.group.extent = 4
    with refused("node-placed-twice", error=DefinitionError):

        class Reused(Space):
            duplicate = declared


def test_foreign_same_spelled_references_and_incompatible_override_are_refused() -> None:
    class Foreign(Space):
        extent: int = Param(default=2)

    class Captured(Space):
        extent: int = Param(default=2)
        group = Bundle(extent=Foreign.extent)

    with refused("reference-not-in-scope", error=DefinitionError):
        design_space(Captured())

    class Unrelated(Group):
        extent: int = Param(default=2)

    class Wrong(Host):
        first = Unrelated()  # type: ignore[assignment]

    with refused("override-changes-kind", error=DefinitionError):
        design_space(Wrong())


def test_groups_have_no_guard_selector_case_or_members_quantifier() -> None:
    with refused("group-no-guard", error=DefinitionError):
        Bundle(extent=2, when=False)  # type: ignore[call-arg]
    with refused("group-not-space", error=DefinitionError):
        Select(group=Bundle(extent=2))

    class Wrong(Space):
        group = Bundle(extent=2)
        values = Members(Bundle.doubled)

    with refused("members-member-invalid", error=DefinitionError):
        design_space(Wrong())


def test_group_model_reads_still_cannot_take_a_view() -> None:
    class Wrong(Group):
        @view
        def shown(self) -> int:
            return 3

        @derived
        def inward(self) -> int:
            return self.shown

    class Owner(Space):
        group = Wrong()

    with refused("view-read-in", error=DefinitionError):
        design_space(Owner())


def test_group_methods_keep_explicit_edges_helpers_and_runtime_enforcement() -> None:
    class Arithmetic(Group):
        value: int = Param()

        @derived(value=value)
        def doubled(*, value: int) -> int:
            return value * 2

        @property
        def extra(self) -> int:
            return self.value + 1

        @derived
        def total(self) -> int:
            return self.doubled + self.extra

        @view
        def refusal(self) -> int | Rejected:
            return reject("own", "local readout refusal")

    def utility(value: int) -> Arithmetic:
        return Arithmetic(value=value)

    class Owner(Space):
        value: int = Param(default=4)
        one = utility(value)
        two = utility(one.total)

    point = design_space(Owner())
    assert point.one.total == 13 and point.two.total == 40
    result = point.query(Owner.one.refusal)
    assert isinstance(result, Rejected)
    assert [(finding.owner, finding.code) for finding in result.findings] == [
        ("one.refusal", "own")
    ]
    assert inspection.validity(point).status == "valid"
    assert {node.declaration.key for node in inspection.explain(point, Owner.one.total).nodes} >= {
        "one.total",
        "one.doubled",
        "value",
    }


def test_cross_group_explicit_cycle_and_local_view_cycle_are_refused() -> None:
    class Alias(Group):
        value: int = Param()

        @derived
        def forward(*, value: int) -> int:
            return value

    class Loop(Space):
        one = Alias()
        two = Alias(value=one.forward)
        one.value = two.forward

    with refused("cyclic-dependency", error=DefinitionError):
        design_space(Loop())

    class ReadoutLoop(Group):
        @view
        def recursive(self) -> int:
            return self.recursive

    class Owner(Space):
        group = ReadoutLoop()

    with pytest.raises(EvaluationError, match="dependency cycle"):
        design_space(Owner()).group.recursive


def test_group_domain_still_reads_facts_only_and_group_is_not_a_fact() -> None:
    class WrongDomain(Group):
        choice: int = Decision(domain=(2, 4))
        width: int = Decision(domain=divisors_of(choice))

    class Owner(Space):
        group = WrongDomain()

    with refused("domain-reads-choice", error=DefinitionError):
        design_space(Owner())
    with refused("formal-takes-node", error=DefinitionError):

        class WrongFact(Space):
            group: Bundle = Param()

    with refused("group-space-mixed", error=DefinitionError):

        class Mixed(Group, Space):
            pass


def test_real_children_inside_groups_are_discovered_without_counting_the_group() -> None:
    class Child(Space):
        value: int = Param()

        @derived
        def metric(self) -> int:
            return self.value

    class Component(Group):
        child = Child(value=3)
        own = Members(Child.metric)

    class Nested(Group):
        included = Component()

    class Owner(Space):
        first = Nested()
        second = Component()
        last = Child(value=5)
        values = Members(Child.metric)

    point = design_space(Owner())
    assert [(item.node, item.value) for item in point.values] == [
        ("first.included.child", 3),
        ("second.child", 3),
        ("last", 5),
    ]
    assert [(item.node, item.value) for item in point.first.included.own] == [("child", 3)]
    assert inspection.statistics(point).scopes == 4
    assert inspection.statistics(point).namespaces == 3


def test_included_constraints_keep_the_expanded_declaration_order() -> None:
    class Rules(Group):
        @constraint
        def first() -> bool:
            return True

        @constraint
        def second() -> bool:
            return True

    class Owner(Space):
        @constraint
        def before() -> bool:
            return True

        rules = Rules()

        @constraint
        def after() -> bool:
            return True

    model = inspection.model(Owner)
    assert [piece.key for piece in model.root.validity.constraints] == [
        "before",
        "rules.first",
        "rules.second",
        "after",
    ]


def test_ambiguous_inheritance_requires_an_explicit_compatible_override() -> None:
    class Left(Group):
        width: int = Param(default=2)

    class Right(Group):
        width: int = Param(default=3)

    class Ambiguous(Left, Right):
        pass

    class Wrong(Space):
        group = Ambiguous()

    with refused("inherited-declarations-conflict", error=DefinitionError):
        design_space(Wrong())

    class Resolved(Left, Right):
        width: int = Param(default=4)

    class Owner(Space):
        group = Resolved()

    assert design_space(Owner()).group.width == 4

    class Mistyped(Left):
        width: str = Param(default="bad")  # type: ignore[assignment]

    class MistypedOwner(Space):
        group = Mistyped()

    with refused("override-changes-semantics", error=DefinitionError):
        design_space(MistypedOwner())
