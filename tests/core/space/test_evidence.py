# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Demanded evidence is detached and equivalent across cached and fresh queries."""

from __future__ import annotations

import gc
import weakref
from typing import cast

from finn.core.space import (
    Available,
    Const,
    Decision,
    Inapplicable,
    Param,
    Rejected,
    Space,
    Unresolved,
    View,
    ViewKey,
    constraint,
    derived,
    design_space,
    inspection,
    view,
)

PHYSICAL = ViewKey("physical", int)


def test_static_alternatives_and_demanded_evidence_are_separate() -> None:
    calls: list[str] = []

    class Good(Space):
        size: int = Param()

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

    # The structural choice is a Decision over nodes; its output reads the
    # selected candidate's member by name.
    class Root(Space):
        size: int = Param()
        implementation: Good | Bad = Decision({"good": Good(size=size), "bad": Bad()})
        physical = View(implementation.physical)

    output = Root.physical
    # Static dependencies (transitively, through the generated selection) name
    # every alternative; nothing is evaluated to find them.
    pending, static = list(inspection.dependencies(Root, output)), set()
    while pending:
        item = pending.pop()
        if item.key not in static:
            static.add(item.key)
            pending.extend(inspection.dependencies(Root, item.reference))
    assert static >= {
        "implementation.good.physical",
        "implementation.bad.physical",
    }
    assert calls == []
    base = design_space(Root(size=12))
    choice = inspection.choices(base)[0]
    point = base.with_choices({choice.selector: "good"})
    first = inspection.explain(point, output)
    second = inspection.explain(point, output)
    assert first == second
    assert first.result == Available(12)
    assert calls == ["good"]
    assert not any(".bad." in node.declaration.key for node in first.nodes)
    assert any(node.selector and node.result == Available("good") for node in first.nodes)
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
        size: int = Param(required=False)

        @derived
        def doubled(self) -> int:
            return self.size * 2

    class Root(Space):
        size: int = Param(required=False)
        nil: None = Param(required=False)
        child = Child(size=size)

    point = design_space(Root(nil=None))
    missing = inspection.explain(point.child, Child.doubled)
    assert isinstance(missing.result, Unresolved)
    facts = [node for node in missing.nodes if node.input_presence is not None]
    assert len(facts) == 1
    assert facts[0].declaration.key == "size"
    assert facts[0].input_presence == "omitted"
    assert isinstance(facts[0].result, Unresolved)
    nil = inspection.explain(point, Root.nil)
    assert nil.nodes[0].input_presence == "supplied"
    assert nil.nodes[0].result == Available(None)


def test_inactive_parameter_evidence_does_not_expose_unused_bound_value() -> None:
    class Child(Space):
        value: int = Param()

    # The enclosing Space class declares the formal and binds it to the child by
    # name (``Forwarded``); a plain literal binding (``Literal``) is covered
    # alongside it.
    class Forwarded(Space):
        value: int = Param()
        enabled = Const(False)
        child = Child(value=value, when=enabled)

    class Literal(Space):
        enabled = Const(False)
        child = Child(value=12345, when=enabled)

    for point in (design_space(Forwarded(value=12345)), design_space(Literal())):
        evidence = inspection.explain(point.child, Child.value)
        assert isinstance(evidence.result, Inapplicable)
        assert not any(node.result == Available(12345) for node in evidence.nodes)
        # The enclosing formal is never demanded through the inactive child.
        assert all(node.declaration.kind != "param" for node in evidence.nodes)
        formal = next(node for node in evidence.nodes if node.declaration.key == "child.value")
        assert formal.input_presence is None
        assert isinstance(formal.result, Inapplicable)


def test_refusal_causes_stay_visible_while_another_obligation_is_unresolved() -> None:
    class Example(Space):
        chosen: int = Decision(values=(1, 2))
        value = Const(3)

        @constraint
        def supported() -> bool:
            return False

        @derived
        def selected(self) -> int:
            return self.value + self.chosen

        physical = View(selected, requires=(supported,))

    point = design_space(Example())
    evidence = inspection.explain(point, Example.physical)
    assert isinstance(evidence.result, Unresolved)
    assert evidence.assessment is not None
    refused = next(node for node in evidence.nodes if node.declaration.key == "supported")
    assert isinstance(refused.result, Rejected)
    assert refused.result.findings[0].owner == "supported"
    choice = next(node for node in evidence.nodes if node.declaration.key == "chosen")
    assert isinstance(choice.decision_state, Available)
    assert choice.decision_state.value.status == "unassigned"
    assert inspection.explain(point, Example.physical) == evidence


def test_evidence_values_are_detached_from_frozen_inputs_and_caches() -> None:
    class Example(Space):
        source: list[int] = Param()
        physical = View(source)

    source = [1, 2]
    point = design_space(Example(source=source))
    source.append(99)
    first = inspection.explain(point, Example.physical)
    assert isinstance(first.result, Available)
    first.result.value.append(3)
    node = next(node for node in first.nodes if node.declaration.key == "source")
    assert isinstance(node.result, Available)
    cast(list[int], node.result.value).append(4)
    second = inspection.explain(point, Example.physical)
    assert second.result == Available([1, 2])
    assert point.source == [1, 2]


def test_evidence_retains_no_occurrence_or_snapshot_lifetime() -> None:
    class Example(Space):
        value = Const(2)

    point = design_space(Example())
    reference = weakref.ref(point)
    evidence = inspection.explain(point, Example.value)
    del point
    gc.collect()
    assert reference() is None
    assert evidence.result == Available(2)
