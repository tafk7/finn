# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Demanded evidence is detached and equivalent across cached and fresh queries."""

from __future__ import annotations

import gc
from typing import cast
import weakref

from finn.kernels.space import (
    Const,
    Decided,
    Decision,
    Inapplicable,
    MissingInput,
    NotApplicable,
    Param,
    Rejected,
    Space,
    Subspace,
    SubspaceChoice,
    Unresolved,
    View,
    ViewKey,
    compile_space,
    constraint,
    derived,
    optional,
    view,
)
from finn.kernels.space import inspection

PHYSICAL = ViewKey("physical", int)


def test_static_alternatives_and_demanded_evidence_are_separate() -> None:
    calls: list[str] = []

    class Good(Space):
        size = Param(int)

        @view
        def physical(*, size: int) -> int:
            calls.append("good")
            return size

        exports = {PHYSICAL: physical}

    class Bad(Space):
        @view
        def physical() -> int:
            calls.append("bad")
            raise AssertionError("inactive evaluator ran")

        exports = {PHYSICAL: physical}

    class Root(Space):
        size = Param(int)
        implementation = SubspaceChoice(
            {"good": Subspace(Good, size=size), "bad": Subspace(Bad)},
            exports=(PHYSICAL,),
        )

    model = compile_space(Root)
    output = Root.implementation.accepted(PHYSICAL)
    static = inspection.dependencies(model, output)
    assert {item.key for item in static} >= {
        "implementation.good.physical",
        "implementation.bad.physical",
    }
    assert calls == []
    base = model.start({Root.size: 12})
    choice = inspection.choices(base)[0]
    assert choice.selector is not None
    point = base.assign(choice.selector, "good")
    first = inspection.explain(point, output)
    second = inspection.explain(point, output)
    assert first == second
    assert first.answer == Decided(12)
    assert calls == ["good"]
    assert not any(".bad." in node.declaration.key for node in first.nodes)
    assert any(node.selector and node.answer == Decided("good") for node in first.nodes)
    assert any(node.is_guard for node in first.nodes)
    assert any(
        node.declaration.key == "size" and node.input_presence == "supplied" for node in first.nodes
    )
    assert any(
        node.declaration.generated and node.declaration.owner == "implementation.good.physical"
        for node in first.nodes
    )


def test_optional_input_presence_distinguishes_omission_from_supplied_none() -> None:
    class Child(Space):
        size = Param(int, required=False)

        @derived(size=optional(size))
        def fallback(*, size: int | MissingInput | NotApplicable) -> int:
            return size if isinstance(size, int) else 7

    class Root(Space):
        size = Param(int, required=False)
        nil = Param(type(None), required=False)
        child = Subspace(Child, size=size)

    point = Root.start({Root.nil: None})
    missing = inspection.explain(point.child, Child.fallback)
    assert missing.answer == Decided(7)
    facts = [node for node in missing.nodes if node.input_presence is not None]
    assert len(facts) == 1
    assert facts[0].declaration.key == "size"
    assert facts[0].input_presence == "omitted"
    assert isinstance(facts[0].answer, Unresolved)
    nil = inspection.explain(point, Root.nil)
    assert nil.nodes[0].input_presence == "supplied"
    assert nil.nodes[0].answer == Decided(None)


def test_inactive_parameter_evidence_does_not_expose_unused_bound_value() -> None:
    class Child(Space):
        value = Param(int)

    class Root(Space):
        enabled = Const(False)
        child = Subspace(Child, value=Param(int), when=enabled)

    point = Root.start({Root.child.ref(Child.value): 12345})
    evidence = inspection.explain(point.child, Child.value)
    assert isinstance(evidence.answer, Inapplicable)
    param = next(node for node in evidence.nodes if node.declaration.kind == "param")
    assert param.input_presence is None
    assert isinstance(param.answer, Inapplicable)


def test_refusal_causes_stay_visible_while_another_obligation_is_unresolved() -> None:
    class Family(Space):
        chosen = Decision(int, values=(1, 2))
        value = Const(3)

        @constraint
        def supported() -> bool:
            return False

        physical = View(value, constraints=(supported,), requires=(chosen,))

    point = Family.start()
    evidence = inspection.explain(point, Family.physical)
    assert isinstance(evidence.answer, Unresolved)
    assert evidence.assessment is not None
    refused = next(node for node in evidence.nodes if node.declaration.key == "supported")
    assert isinstance(refused.answer, Rejected)
    assert refused.answer.findings[0].owner == "supported"
    choice = next(node for node in evidence.nodes if node.declaration.key == "chosen")
    assert isinstance(choice.decision_state, Decided)
    assert choice.decision_state.value.status == "unassigned"
    assert inspection.explain(point, Family.physical) == evidence


def test_evidence_values_are_detached_from_frozen_inputs_and_caches() -> None:
    class Family(Space):
        source = Param[list[int]](list)
        physical = View(source)

    source = [1, 2]
    point = Family.start({Family.source: source})
    source.append(99)
    first = inspection.explain(point, Family.physical)
    assert isinstance(first.answer, Decided)
    first.answer.value.append(3)
    node = next(node for node in first.nodes if node.declaration.key == "source")
    assert isinstance(node.answer, Decided)
    cast(list[int], node.answer.value).append(4)
    second = inspection.explain(point, Family.physical)
    assert second.answer == Decided([1, 2])
    assert point.source == [1, 2]


def test_evidence_retains_no_occurrence_or_snapshot_lifetime() -> None:
    class Family(Space):
        value = Const(2)

    point = Family.start()
    reference = weakref.ref(point)
    evidence = inspection.explain(point, Family.value)
    del point
    gc.collect()
    assert reference() is None
    assert evidence.answer == Decided(2)
