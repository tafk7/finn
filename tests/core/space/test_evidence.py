# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Demanded evidence is detached and equivalent across cached and fresh queries."""

from __future__ import annotations

import gc
import weakref
from typing import cast

from finn.core.space import (
    Available,
    Decision,
    Inapplicable,
    Param,
    Rejected,
    Select,
    Space,
    constraint,
    derived,
    design_space,
    inspection,
    view,
)


def test_static_alternatives_and_demanded_evidence_are_separate() -> None:
    calls: list[str] = []

    class Good(Space):
        size: int = Param()

        @view
        def physical(*, size: int) -> int:
            calls.append("good")
            return size

    class Bad(Space):
        @view
        def physical() -> int:
            calls.append("bad")
            raise AssertionError("inactive evaluator ran")

    # The structural choice is a Decision over nodes; its output reads the
    # selected candidate's member by name.
    class Root(Space):
        size: int = Param()
        implementation: Good | Bad = Select(good=Good(size=size), bad=Bad())
        physical = view(implementation.physical)

    output = Root.physical
    # Static dependencies (transitively, through the generated selection) name
    # every alternative; nothing is evaluated to find them.
    model = inspection.model(Root)
    pending = list(model.resolve(model.root, output).dependencies)
    static: set[str] = set()
    while pending:
        node = pending.pop()
        if node.key not in static:
            static.add(node.key)
            pending.extend(node.dependencies)
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
    # Owning validity observes the unselected case's guard, but not its body.
    bad = {
        node.declaration.key: node.result for node in first.nodes if ".bad." in node.declaration.key
    }
    assert bad == {"implementation.bad.$selected": Available(False)}
    assert any(node.declaration.kind == "validity" for node in first.nodes)
    assert any(node.selector and node.result == Available("good") for node in first.nodes)
    assert any(node.is_guard for node in first.nodes)
    assert any(
        node.declaration.key == "size" and node.declaration.kind == "param" for node in first.nodes
    )
    assert any(
        node.declaration.generated and node.declaration.owner == "implementation.good.physical"
        for node in first.nodes
    )


def test_inactive_parameter_evidence_does_not_expose_unused_bound_value() -> None:
    class Child(Space):
        value: int = Param()

    # The enclosing Space class declares the formal and binds it to the child by
    # name (``Forwarded``); a plain literal binding (``Literal``) is covered
    # alongside it.
    class Forwarded(Space):
        value: int = Param()

        @derived
        def enabled(self) -> bool:
            return False

        child = Child(value=value, when=enabled)

    class Literal(Space):
        @derived
        def enabled(self) -> bool:
            return False

        child = Child(value=12345, when=enabled)

    for point in (design_space(Forwarded(value=12345)), design_space(Literal())):
        evidence = inspection.explain(point.child, Child.value)
        assert isinstance(evidence.result, Inapplicable)
        assert not any(node.result == Available(12345) for node in evidence.nodes)
        # The enclosing formal is never demanded through the inactive child.
        assert all(node.declaration.kind != "param" for node in evidence.nodes)
        formal = next(node for node in evidence.nodes if node.declaration.key == "child.value")
        assert isinstance(formal.result, Inapplicable)


def test_owning_refusal_stops_before_unresolved_output_input() -> None:
    class Example(Space):
        chosen: int = Decision(domain=(1, 2))

        @derived
        def value(self) -> int:
            return 3

        @constraint
        def supported() -> bool:
            return False

        @derived
        def selected(self) -> int:
            return self.value + self.chosen

        physical = view(selected)  # owning refusal precedes the unassigned output choice

    point = design_space(Example())
    evidence = inspection.explain(point, Example.physical)
    assert isinstance(evidence.result, Rejected)
    assert evidence.assessment is not None
    assert any(node.declaration.key == "supported" for node in evidence.nodes)
    assert all(node.declaration.key != "selected" for node in evidence.nodes)
    assert inspection.validity(point).status == "refused"
    assert all(node.declaration.key != "chosen" for node in evidence.nodes)
    assert inspection.explain(point, Example.physical) == evidence


def test_evidence_hands_out_a_container_a_derived_value_answers_as_a_copy() -> None:
    """A derived value answering a mutable container (a list): the evidence's values are
    copies of it, so a reader changing one changes nothing the design space holds."""

    class Example(Space):
        n: int = Param()

        @derived
        def source(self) -> list[int]:
            return [1, self.n]

        physical = view(source)

    point = design_space(Example(n=2))
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
        @derived
        def value(self) -> int:
            return 2

    point = design_space(Example())
    reference = weakref.ref(point)
    evidence = inspection.explain(point, Example.value)
    del point
    gc.collect()
    assert reference() is None
    assert evidence.result == Available(2)
