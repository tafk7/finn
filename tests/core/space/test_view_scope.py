# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Present views may attempt output unless their real owning Space is refused."""

from __future__ import annotations

import pytest

from finn.core.space import (
    Available,
    Decision,
    Group,
    Inapplicable,
    Members,
    NativeEvaluationError,
    Param,
    Rejected,
    Space,
    Unresolved,
    ValueUnavailableError,
    constraint,
    derived,
    design_space,
    inspection,
    reject,
    view,
)
from finn.core.space._running.evaluate import Session, evaluation
from finn.core.space.occurrence import held


@pytest.mark.parametrize("rule", ("valid", "open", "refused", "fault"))
@pytest.mark.parametrize("validity_first", (False, True))
def test_view_surfaces_observe_owner_before_output(rule: str, validity_first: bool) -> None:
    calls: list[str] = []

    class Example(Space):
        policy: str = Param()
        fact: int = Param(default=6)
        choice: int = Decision(values=(1, 2))

        @constraint
        def unrelated(self) -> bool:
            if self.policy == "fault":
                raise RuntimeError("broken rule")
            if self.policy == "open":
                return self.choice > 0
            return self.policy != "refused"

        @view
        def local(self) -> list[int]:
            calls.append("body")
            return [self.fact]

        alias = view(local)

    point = design_space(Example(policy=rule))
    if rule == "fault":
        for ref in (Example.local, Example.alias):
            with pytest.raises(NativeEvaluationError, match="broken rule"):
                point.query(ref)
        assert calls == []
        return
    if validity_first:
        assert inspection.validity(point).status == rule
    for ref in (Example.local, Example.alias):
        result = point.query(ref)
        assessment = point.inspect(ref)
        assert assessment.result == result
        assert assessment.applicability == Available(True)
        assert assessment.validity == inspection.validity(point).result
        trace = inspection.explain(point, ref)
        assert trace.result == result
        keys = {node.declaration.key for node in trace.nodes}
        assert {"$validity", "unrelated", "policy"} <= keys
        if rule == "refused":
            assert isinstance(result, Rejected)
            assert [f.owner for f in result.findings] == ["unrelated"]
            assert assessment.output_result is None
            assert "local.$output" not in keys
            with pytest.raises(ValueUnavailableError) as error:
                point.field(ref).get()
            assert error.value.result == result
            assert calls == []
        else:
            assert result == Available([6])
            assert assessment.output_result == Available([6])
            assessment.require_value().append(99)
            assert point.field(ref).get() == [6]
        at = held(point)
        fresh = evaluation(at.ds, Session()).get(at.ds.model.resolve(at.scope, ref))
        assert fresh.result == result and fresh.assessment == point.inspect(ref)

    # Same body inputs, changed validity: the gate is a real cache dependency.
    refused = point.with_facts({Example.policy: "refused"})
    assert isinstance(refused.query(Example.local), Rejected)
    allowed = refused.with_facts({Example.policy: "valid"})
    assert allowed.local == [6]
    opened = allowed.with_facts({Example.policy: "open"})
    assert opened.local == [6] and isinstance(opened.inspect(Example.local).validity, Unresolved)


@pytest.mark.parametrize("presence", ("absent", "open", "refused", "present"))
def test_scope_presence_still_precedes_an_immediately_inapplicable_body(presence: str) -> None:
    calls: list[str] = []

    class Child(Space):
        @view
        def local(self) -> int | Inapplicable:
            calls.append("body")
            return Inapplicable()

    class Owner(Space):
        mode: str = Param()
        choice: bool = Decision(values=(False, True))

        @derived
        def enabled(self) -> bool | Rejected:
            if self.mode == "refused":
                return reject("presence-refused", "cannot determine presence")
            if self.mode == "open":
                return self.choice
            return self.mode == "present"

        child = Child(when=enabled)
        entries = Members(Child.local)

        @view
        def total(self) -> int:
            return sum(entry.value for entry in self.entries)

    point = design_space(Owner(mode=presence))
    found_presence = dict(inspection.presence(point))
    assert found_presence[""] == Available(True)
    if presence == "present":
        assert found_presence["child"] == Available(True)
    elif presence == "absent":
        assert found_presence["child"] == Available(False)
    else:
        assert isinstance(found_presence["child"], Unresolved if presence == "open" else Rejected)
    assessment = point.inspect(Owner.child.local)
    expected = {"open": Unresolved, "refused": Rejected}.get(presence, Inapplicable)
    assert isinstance(assessment.result, expected)
    assert (assessment.output_result is None) == (presence != "present")
    assert calls == (["body"] if presence == "present" else [])
    if presence in ("absent", "present"):
        assert point.total == 0
    else:
        assert isinstance(point.query(Owner.total), expected)


def test_local_answers_and_detached_values_do_not_imply_completeness() -> None:
    class Producer(Space):
        choice: int = Decision(values=(1, 2))
        allowed: bool = Param(default=True)

        @constraint
        def supported(self) -> bool:
            return self.allowed

        @view
        def own(self) -> int | Rejected:
            return reject("own", "local refusal")

        @view
        def pending(self) -> int:
            return self.choice

        @view
        def constant(self) -> int:
            return 8

        @view
        def nothing(self) -> None:
            return None

        @view
        def false(self) -> bool:
            return False

    class Consumer(Space):
        fact: int = Param()

        @view
        def doubled(self) -> int:
            return self.fact * 2

    point = design_space(Producer())
    refused = point.query(Producer.own)
    assert isinstance(refused, Rejected)
    assert [finding.code for finding in refused.findings] == ["own"]
    assert isinstance(point.query(Producer.pending), Unresolved)
    with pytest.raises(ValueUnavailableError) as caught:
        _ = point.pending
    assert caught.value.result == point.query(Producer.pending)
    assert point.query(Producer.nothing) == Available(None)
    assert point.inspect(Producer.nothing).output_result == Available(None)
    assert point.query(Producer.false) == Available(False)
    receiver = design_space(Consumer(fact=point.constant))
    assert receiver.doubled == 16
    assert inspection.validity(receiver).status == "valid"
    changed = point.with_facts({Producer.allowed: False})
    assert inspection.validity(changed).status == "refused"
    assert changed.inspect(Producer.nothing).output_result is None
    assert isinstance(changed.query(Producer.false), Rejected)
    assert receiver.doubled == 16


def test_open_owner_waits_do_not_pollute_an_unresolved_view_input() -> None:
    class Example(Space):
        gate: int = Decision(values=(1, 2))
        output: int = Decision(values=(1, 2))

        @constraint
        def waiting(self) -> bool:
            return self.gate > 0

        @view
        def shown(self) -> int:
            return self.output

    point = design_space(Example())
    assessment = point.inspect(Example.shown)
    assert isinstance(assessment.validity, Unresolved)
    assert [f.owner for f in assessment.validity.findings] == ["gate"]
    assert isinstance(assessment.result, Unresolved)
    assert [f.owner for f in assessment.result.findings] == ["output"]
    assert assessment.output_result == assessment.result
    assert point.with_choices(output=2).shown == 2


def test_group_and_child_views_use_their_real_owners() -> None:
    calls: list[str] = []

    class Child(Space):
        @view
        def shown(self) -> int:
            calls.append("child")
            return 4

    class Included(Group):
        @view
        def shown(self) -> int:
            calls.append("group")
            return 4

    class Parent(Space):
        child = Child()
        included = Included()

        @constraint
        def composition() -> bool:
            return False

        @view
        def total(self) -> int:
            calls.append("parent")
            return self.child.shown + self.included.shown

    point = design_space(Parent())
    for ref in (Parent.included.shown, Parent.total):
        assessment = point.inspect(ref)
        assert isinstance(assessment.result, Rejected)
        assert assessment.output_result is None
    assert calls == []
    assert point.child.shown == 4 and calls == ["child"]
    evidence = inspection.explain(point, Parent.child.shown)
    assert "child.$validity" in {node.declaration.key for node in evidence.nodes}
    assert "$validity" not in {node.declaration.key for node in evidence.nodes}
    model = inspection.model(point)
    grouped = model.resolve(model.root, Parent.included.shown)
    assert grouped.constraints == (model.root.validity,)
    assert model.root.validity in grouped.dependencies
