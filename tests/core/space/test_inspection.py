# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""External search and model-bound handles use the supported Space surface."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import assert_type

import pytest

from finn.core.space import (
    Available,
    BoundDecision,
    Const,
    Decision,
    Inapplicable,
    Param,
    Space,
    Unresolved,
    View,
    ViewKey,
    composite,
    derived,
    design_space,
    domain,
    inspection,
    view,
)
from finn.core.space.errors import RequestError
from finn.core.space.references import DecisionHandle, ValueHandle

PHYSICAL = ViewKey("physical", int)


def test_inspection_does_not_run_domains_or_evaluators() -> None:
    calls: list[str] = []

    def membership(*, candidate: int) -> bool:
        calls.append("membership")
        return candidate in (1, 2, 4)

    def candidates() -> tuple[int, ...]:
        calls.append("candidates")
        return (1, 2, 4)

    class Example(Space):
        lanes: int = Decision(domain=domain(accepts=membership, candidates=candidates))

        @derived
        def cost(*, lanes: int) -> int:
            calls.append("cost")
            return 12 // lanes

        physical = View(cost)

    point = design_space(Example())
    model = inspection.model(point)
    decisions = inspection.decisions(point)
    assert [item.key for item in decisions] == ["lanes"]
    assert {item.key for item in inspection.members(model)} == {"lanes", "cost", "physical"}
    assert inspection.choices(model) == ()
    assert inspection.dependencies(model, Example.physical)[0].key == "cost"
    assert calls == []

    # The policy is ordinary external code. Discovery and queries never adopt
    # a candidate on its behalf, and every trial uses validated assignment.
    handle = decisions[0].reference
    options = point.field(handle).candidates()
    assert isinstance(options, Available)
    scores: list[tuple[int, object]] = []
    for value in options.value:
        trial = point.with_choices({handle: value})
        result = trial.inspect(Example.physical).accepted_result
        assert isinstance(result, Available)
        scores.append((result.value, value))
    assert min(scores, key=lambda item: item[0]) == (3, 4)
    assert calls.count("membership") == 3
    assert calls.count("candidates") == 1
    assert isinstance(point.query(Example.lanes), Unresolved)


def test_discovery_reports_owning_decisions_and_author_names_for_selectors() -> None:
    class Child(Space):
        supplied: int = Param()
        local: int = Decision(values=(1, 2))

    class Root(Space):
        source: int = Decision(values=(2, 4))
        child = Child(supplied=source)
        implementation: Child = Decision({"a": Child(supplied=1), "b": Child(supplied=2)})

    point = design_space(Root())
    decisions = inspection.decisions(point)
    assert {item.key for item in decisions} == {
        "source",
        "child.local",
        "implementation",
        "implementation.a.local",
        "implementation.b.local",
    }
    assert {item.key for item in inspection.decisions(point.child)} == {"child.local"}
    selectors = [item for item in decisions if item.selector]
    assert len(selectors) == 1
    assert selectors[0].key == "implementation"
    assert selectors[0].cases == ("a", "b")
    choice = inspection.choices(point)[0]
    # None only when an enclosing body pinned the choice.
    assert_type(choice.selector, DecisionHandle[str] | None)
    assert choice.selector is not None and choice.selector == selectors[0].reference
    chosen = point.with_choices({choice.selector: "b"})
    selected = chosen.implementation
    assert isinstance(selected, Child) and selected.query(Child.supplied) == Available(2)
    # An unselected candidate is still inspectable; its members are inapplicable.
    other = inspection.candidate(chosen, Root.implementation, "a")
    assert isinstance(other, Child) and isinstance(other.query(Child.supplied), Inapplicable)
    assert [(case.name, case.scope, case.space_type) for case in choice.cases] == [
        ("a", "implementation.a", Child),
        ("b", "implementation.b", Child),
    ]


def test_typed_handles_preserve_types_and_match_repeated_discovery() -> None:
    class Example(Space):
        lanes: int = Decision(values=(1, 2))

        @view
        def physical(*, lanes: int) -> int:
            return lanes

    model = inspection.model(Example)
    point = design_space(Example())
    decision = inspection.decision_handle(model, Example.lanes)
    value = inspection.value_handle(model, Example.physical)
    assert_type(decision, DecisionHandle[int])
    assert_type(value, ValueHandle[int])
    assert_type(point.field(Example.lanes), BoundDecision[int])
    # A discovered handle is itself an edit key of the mapping form.
    trial = point.with_choices({decision: 2})
    assert trial.query(value) == Available(2)
    assert trial.query(decision) == Available(2)
    assert decision == inspection.decision_info(point, Example.lanes).reference
    assert hash(decision) == hash(inspection.decisions(point)[0].reference)
    with pytest.raises(FrozenInstanceError):
        setattr(decision, "_node", 9)


def test_foreign_handles_fail_before_callbacks_and_aliases_cannot_be_upgraded() -> None:
    calls: list[str] = []

    def membership(*, candidate: int) -> bool:
        calls.append("membership")
        return candidate > 0

    class Child(Space):
        supplied: int = Param()

    class Example(Space):
        choice: int = Decision(domain=domain(accepts=membership))
        child = Child(supplied=choice)

    class OtherExample(Example):
        pass

    first = design_space(Example())
    foreign = inspection.decision_handle(OtherExample, Example.choice)
    with pytest.raises(RequestError, match="different compiled model"):
        first.with_choices({foreign: 1})
    with pytest.raises(RequestError, match="different compiled model"):
        first.field(foreign)
    with pytest.raises(RequestError, match="different compiled model"):
        first.query(foreign)
    assert calls == []
    # A formal bound to another member's decision is an alias, not an owned decision.
    with pytest.raises(RequestError, match="not independently editable"):
        inspection.decision_handle(first.child, Child.supplied)
    with pytest.raises(RequestError, match="not independently editable"):
        inspection.decision_handle(first, Example.child.supplied)


def test_handles_follow_model_identity_across_starts_without_retaining_point_state() -> None:
    class Example(Space):
        source: int = Param()
        lanes: int = Decision(values=(1, 2))

    model = inspection.model(Example)
    first, second = design_space(Example(source=4)), design_space(Example(source=8))
    source = inspection.value_handle(model, Example.source)
    decision = inspection.decision_handle(first, Example.lanes)
    assert first.query(source) == Available(4)
    assert second.query(source) == Available(8)
    assert second.with_choices({decision: 2}).query(decision) == Available(2)
    assert isinstance(first.query(decision), Unresolved)


def test_singleton_choice_metadata_exposes_an_ordinary_editable_selector() -> None:
    # A Decision over nodes is an ordinary Decision, so even one candidate is an
    # owned, editable selector. A None candidate places nothing and has no scope or Space class.
    class Child(Space):
        value = Const(1)

    class Root(Space):
        implementation: Child = Decision({"only": Child()})
        optional: Child | None = Decision({"some": Child()}, optional=True)

    info = inspection.decision_info(Root, Root.implementation)
    assert (info.key, info.selector, info.cases) == ("implementation", True, ("only",))
    assert [item.key for item in inspection.decisions(Root)] == ["implementation", "optional"]
    implementation, optional = inspection.choices(Root)
    assert implementation.selector == info.reference
    assert [(case.name, case.scope, case.space_type) for case in implementation.cases] == [
        ("only", "implementation.only", Child)
    ]
    assert [(case.name, case.scope, case.space_type) for case in optional.cases] == [
        ("none", None, None),
        ("some", "optional.some", Child),
    ]
    point = design_space(Root())
    # Its one case is forced: read as that case, never committed.
    assert point.query(Root.implementation.value) == Available(1)
    assert [(item.key, item.value, item.refused) for item in inspection.forced(point)] == [
        ("implementation", "only", {})
    ]
    chosen = point.with_choices({implementation.selector: "only", optional.selector: "none"})
    assert isinstance(chosen.implementation, Child) and chosen.implementation.value == 1
    assert chosen.optional is None


def test_statistics_counts_instantiated_members_and_direct_structure() -> None:
    class Leaf(Space):
        value: int = Decision(values=(1, 2))
        physical = View(value)

    def repeated(count: int) -> inspection.ModelStatistics:
        space_type = composite("Repeated", {f"child{index}": Leaf() for index in range(count)})
        return inspection.statistics(space_type)

    small, large = repeated(2), repeated(4)
    # Each placement contributes its two effective members and the placement
    # itself, while the root scope is shared. A view adds one direct edge.
    assert small.authored_declarations == 6
    assert small.scopes == 3
    assert small.nodes == 4
    assert small.potential_edges == 2
    assert small.choices == 0
    assert small.owning_decisions == 2
    assert large.authored_declarations == 2 * small.authored_declarations
    assert large.nodes == 2 * small.nodes
    assert large.potential_edges == 2 * small.potential_edges
    assert large.owning_decisions == 2 * small.owning_decisions
    assert large.scopes == 2 * (small.scopes - 1) + 1
