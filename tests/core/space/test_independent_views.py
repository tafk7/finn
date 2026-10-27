# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Readouts demand local inputs and ordinary presence, never implicit validity."""

from __future__ import annotations

import pytest

from finn.core.space import (
    Available,
    Decision,
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
def test_local_readout_surfaces_and_cache_ignore_unrelated_rules(
    rule: str, validity_first: bool
) -> None:
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
            return [self.fact]

        alias = view(local)

    point = design_space(Example(policy=rule))

    def check_validity() -> None:
        if rule == "fault":
            with pytest.raises(NativeEvaluationError, match="broken rule"):
                inspection.validity(point)
        else:
            assert inspection.validity(point).status == rule

    if validity_first:
        check_validity()
    for ref in (Example.local, Example.alias):
        assert point.query(ref) == Available([6])
        assert point.field(ref).get() == [6]
        assessment = point.inspect(ref)
        assert assessment.applicability == Available(True)
        assert assessment.output_result == assessment.result == Available([6])
        assessment.require_value().append(99)
        assert point.query(ref) == Available([6])
        trace = inspection.explain(point, ref)
        keys = {node.declaration.key for node in trace.nodes}
        assert not keys & {"$validity", "unrelated", "policy", "choice"}
        at = held(point)
        index = at.ds.model.resolve(at.scope, ref)
        fresh = evaluation(at.ds, Session()).get(index)
        assert fresh.result == Available([6])
    check_validity()
    assert point.local == point.alias == [6]
    if rule == "fault":
        # Editing facts still compares model validity; the broken rule remains a fault.
        with pytest.raises(NativeEvaluationError, match="broken rule"):
            point.with_facts({Example.policy: "refused", Example.fact: 7})
    else:
        changed = point.with_facts({Example.policy: "refused", Example.fact: 7})
        assert changed.local == [7]
    assert point.local == [6]


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


def test_local_refusal_wait_and_detached_values_do_not_certify_producer() -> None:
    class Producer(Space):
        choice: int = Decision(values=(1, 2))

        @constraint
        def invalid() -> bool:
            return False

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
    assert point.query(Producer.false) == Available(False)
    receiver = design_space(Consumer(fact=point.constant))
    assert receiver.doubled == 16
    assert inspection.validity(receiver).status == "valid"
    assert inspection.validity(point).status == "refused"
