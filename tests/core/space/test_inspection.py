# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""External search and model-bound handles use the supported Space surface."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest
from typing_extensions import assert_type

from finn.core.space import (
    Available,
    Const,
    Decision,
    DecisionRef,
    Param,
    Space,
    Subspace,
    SubspaceChoice,
    Unresolved,
    View,
    ViewKey,
    compile_space,
    derived,
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

    class Family(Space):
        lanes = Decision(int, domain=domain(accepts=membership, candidates=candidates))

        @derived
        def cost(*, lanes: int) -> int:
            calls.append("cost")
            return 12 // lanes

        physical = View(cost)

    model = compile_space(Family)
    point = model.bind()
    decisions = inspection.decisions(point)
    assert [item.key for item in decisions] == ["lanes"]
    assert {item.key for item in inspection.members(model)} == {"lanes", "cost", "physical"}
    assert inspection.choices(model) == ()
    assert inspection.dependencies(model, Family.physical)[0].key == "cost"
    assert calls == []

    # The policy is ordinary external code. Discovery and queries never adopt
    # a candidate on its behalf, and every trial uses validated assignment.
    handle = decisions[0].reference
    options = point.field(handle).candidates()
    assert isinstance(options, Available)
    scores: list[tuple[int, object]] = []
    for value in options.value:
        trial = point.with_choices(point.field(handle).change(value))
        result = trial.physical.inspect().accepted_result
        assert isinstance(result, Available)
        scores.append((result.value, value))
    assert min(scores, key=lambda item: item[0]) == (3, 4)
    assert calls.count("membership") == 3
    assert calls.count("candidates") == 1
    assert isinstance(point.query(Family.lanes), Unresolved)


def test_discovery_reports_owning_decisions_and_author_names_for_selectors() -> None:
    class Child(Space):
        supplied = Param(int)
        local = Decision(int, values=(1, 2))

    class Root(Space):
        source = Decision(int, values=(2, 4))
        child = Subspace(Child, supplied=source)
        implementation = SubspaceChoice(
            {"a": Subspace(Child, supplied=1), "b": Subspace(Child, supplied=2)}
        )

    point = Root()
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
    assert choice.selector is not None
    assert_type(choice.selector, DecisionHandle[str])
    chosen = point.with_choices(point.field(choice.selector).change("b"))
    assert chosen.implementation.alternative("b").query(Child.supplied) == Available(2)
    assert [(case.name, case.scope) for case in choice.cases] == [
        ("a", "implementation.a"),
        ("b", "implementation.b"),
    ]


def test_typed_handles_preserve_types_and_match_repeated_discovery() -> None:
    class Family(Space):
        lanes = Decision(int, values=(1, 2))

        @view
        def physical(*, lanes: int) -> int:
            return lanes

    model = compile_space(Family)
    point = model.bind()
    decision = inspection.decision_handle(model, Family.lanes)
    value = inspection.value_handle(model, Family.physical)
    assert_type(decision, DecisionHandle[int])
    assert_type(value, ValueHandle[int])
    as_decision: DecisionRef[int] = decision
    trial = point.with_choices(point.field(as_decision).change(2))
    assert trial.query(value) == Available(2)
    assert decision == inspection.decision_info(point, Family.lanes).reference
    assert hash(decision) == hash(inspection.decisions(point)[0].reference)
    with pytest.raises(FrozenInstanceError):
        setattr(decision, "_node", 9)


def test_foreign_handles_fail_before_callbacks_and_aliases_cannot_be_upgraded() -> None:
    calls: list[str] = []

    def membership(*, candidate: int) -> bool:
        calls.append("membership")
        return candidate > 0

    class Child(Space):
        supplied = Param(int)

    class Family(Space):
        choice = Decision(int, domain=domain(accepts=membership))
        child = Subspace(Child, supplied=choice)

    class OtherFamily(Family):
        pass

    first_model = compile_space(Family)
    second_model = compile_space(OtherFamily)
    first = first_model.bind()
    foreign = inspection.decision_handle(second_model, Family.choice)
    with pytest.raises(RequestError, match="different compiled model"):
        first.with_choices(first.field(foreign).change(1))
    with pytest.raises(RequestError, match="different compiled model"):
        first.field(foreign).change(1)
    with pytest.raises(RequestError, match="different compiled model"):
        first.query(foreign)
    assert calls == []
    with pytest.raises(RequestError, match="parameter alias"):
        inspection.decision_handle(first.child, cast(Decision[int], Child.supplied))
    with pytest.raises(RequestError, match="Param alias"):
        inspection.decision_handle(first, Family.child.decision_ref(Child.supplied))


def test_handles_follow_model_identity_across_starts_without_retaining_point_state() -> None:
    class Family(Space):
        source = Param(int)
        lanes = Decision(int, values=(1, 2))

    model = compile_space(Family)
    first, second = model.bind({Family.source: 4}), model.bind({Family.source: 8})
    source = inspection.value_handle(model, Family.source)
    decision = inspection.decision_handle(first, Family.lanes)
    assert first.query(source) == Available(4)
    assert second.query(source) == Available(8)
    assert second.with_choices(second.field(decision).change(2)).query(decision) == Available(2)
    assert isinstance(first.query(decision), Unresolved)


def test_singleton_choice_metadata_exposes_no_editable_selector() -> None:
    class Child(Space):
        value = Const(1)

    class Root(Space):
        implementation = SubspaceChoice({"only": Subspace(Child)})

    model = compile_space(Root)
    assert inspection.decisions(model) == ()
    choice = inspection.choices(model)[0]
    assert choice.selector is None
    assert choice.cases[0].name == "only"


def test_statistics_counts_instantiated_members_and_direct_structure() -> None:
    class Leaf(Space):
        value = Decision(int, values=(1, 2))
        physical = View(value)

    def repeated(count: int) -> inspection.ModelStatistics:
        family = cast(
            type[Space],
            type("Repeated", (Space,), {f"child{index}": Subspace(Leaf) for index in range(count)}),
        )
        return inspection.statistics(compile_space(family))

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
