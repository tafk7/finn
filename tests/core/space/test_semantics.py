# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from finn.core.space import Decision, Model, Param, Space, design_space, divisors_of
from finn.core.space._runtime import Snapshot, decision_state, evaluate
from finn.core.space.errors import ConfigurationError, EvaluationError, RequestError
from finn.core.space.ir import Argument, LinkedModel, Node
from finn.core.space.results import (
    Available,
    Finding,
    FindingKind,
    Inapplicable,
    QueryResult,
    Rejected,
    Unresolved,
    ViewAssessment,
    assess_constraints,
    assess_view,
    constraint_result,
    ordered_findings,
    owned_result,
    reject,
)
from finn.core.space.semantics import ValueSemantics, default_semantics


def blocker(owner: str = "lanes") -> Unresolved:
    return Unresolved((Finding(FindingKind.BLOCKER, "decision-unassigned", owner, "choose lanes"),))


def test_answers_and_optional_markers_have_no_truth_value() -> None:
    values: tuple[object, ...] = (Available(False), Inapplicable(), reject("no", "no"), blocker())
    for value in values:
        with pytest.raises(TypeError, match="no truth value"):
            bool(value)


def test_findings_freeze_details_and_have_deterministic_owned_causes() -> None:
    details = {"shape": [1, 2], "mode": "fast"}
    cause = Finding(FindingKind.LIMITATION, "missing-input", "child.width", "no width")
    rejection = reject("bad", "unsupported", values=details, causes=(cause,))
    details["shape"] = [9]
    owned: QueryResult[object] = owned_result(rejection, "physical")
    assert isinstance(owned, Rejected)
    assert owned.findings[0].owner == "physical"
    assert owned.findings[0].causes == (cause,)
    assert owned.findings[0].details == (("mode", "fast"), ("shape", (1, 2)))
    assert ordered_findings((cause, owned.findings[0])) == ordered_findings(
        (owned.findings[0], cause)
    )
    with pytest.raises(TypeError, match="primitive"):
        reject("bad", "unsupported", values={"unsafe": object()})
    with pytest.raises(ValueError, match="unique"):
        Finding(FindingKind.REJECTION, "bad", "x", "bad", (("a", 1), ("a", 2)))


def test_unresolved_and_rejected_require_causes() -> None:
    with pytest.raises(ValueError, match="finding"):
        Unresolved(())
    with pytest.raises(ValueError, match="finding"):
        Rejected(())


def test_nominal_semantics_distinguish_bool_and_snapshot_mutable_values() -> None:
    integers = default_semantics(int)
    assert not integers.accepts(True)
    assert not integers.values_equal(True, 1)
    with pytest.raises(TypeError, match="int"):
        integers.freeze(True)
    lists: ValueSemantics[list[int]] = default_semantics(list)
    source = [1, 2]
    frozen = lists.freeze(source)
    source.append(3)
    assert frozen == [1, 2]


@dataclass(frozen=True)
class Pair:
    left: int
    right: tuple[int, ...]


@dataclass
class Bag:
    values: list[int]


def test_a_frozen_dataclass_is_its_own_snapshot_and_anything_else_is_copied() -> None:
    """The default snapshot's contract: a frozen value class holds immutable values, so
    its instance is shared; a mutable class, even one holding the same fields, is not."""

    pair = Pair(1, (2, 3))
    assert default_semantics(Pair).freeze(pair) is pair
    bag = Bag([1, 2])
    frozen = default_semantics(Bag).freeze(bag)
    assert frozen is not bag and frozen.values is not bag.values and frozen == bag


def test_declared_equality_is_independent_of_python_equality() -> None:
    bags = ValueSemantics(
        Bag,
        "unordered bag",
        lambda value: type(value) is Bag,
        lambda left, right: sorted(left.values) == sorted(right.values),
        lambda value: Bag(list(value.values)),
    )
    assert Bag([1, 2]) != Bag([2, 1])
    assert bags.values_equal(Bag([1, 2]), Bag([2, 1]))
    source = Bag([1])
    frozen = bags.freeze(source)
    source.values.append(2)
    assert frozen.values == [1]


def test_snapshot_must_preserve_declared_value_type() -> None:
    malformed: ValueSemantics[object] = ValueSemantics(
        int, "integer", lambda value: type(value) is int, lambda left, right: left == right, str
    )
    with pytest.raises(TypeError, match="changed its nominal"):
        malformed.freeze(1)


def test_only_constraints_convert_false_to_refusal() -> None:
    assert Available(False).value is False
    result = constraint_result(False, "supported")
    assert isinstance(result, Rejected)
    assert result.findings[0].owner == "supported"
    assert constraint_result(True, "supported") == Available(True)


def test_unresolved_constraints_keep_individual_refusal_inspectable() -> None:
    refused = reject("geometry", "too wide", owner="width")
    assessment = assess_constraints(
        {"lanes": blocker(), "width": refused, "unused": Inapplicable()}
    )
    assert assessment.verdict is None
    assert assessment.refused == ("width",)
    assert assessment.not_applicable == ("unused",)
    assert assessment.results["width"] is not None
    assert assessment.results["width"] == refused


def test_readiness_tracks_unresolved_results_without_erasing_refusal() -> None:
    refused = reject("no", "unsupported", owner="support")
    waiting: ViewAssessment[int] = assess_view(
        blocker(), owner="physical", constraints={"support": refused}
    )
    assert waiting.readiness.ready is None
    assert waiting.constraints.result == refused
    final = assess_view(Available(1), owner="physical", constraints={"support": refused})
    assert final.readiness.ready is True
    assert final.accepted_result == refused


def test_view_reducer_obeys_readiness_output_and_refusal_precedence() -> None:
    raw: QueryResult[int] = Available(4)
    refused = reject("geometry", "too wide", owner="width")
    waiting = assess_view(raw, owner="physical", constraints={"lanes": blocker(), "width": refused})
    assert waiting.output_result == raw
    assert isinstance(waiting.accepted_result, Unresolved)
    assert waiting.constraints.refused == ("width",)
    inactive: ViewAssessment[int] = assess_view(
        Inapplicable(), owner="physical", constraints={"width": refused}
    )
    assert isinstance(inactive.accepted_result, Inapplicable)
    denied = assess_view(raw, owner="physical", constraints={"width": refused})
    assert denied.accepted_result == refused
    accepted = assess_view(raw, owner="physical")
    assert accepted.readiness.ready is True
    assert accepted.accepted_result == raw


def test_view_applicability_precedes_unresolved_requirements() -> None:
    result: ViewAssessment[int] = assess_view(
        blocker(), owner="physical", applicability=Available(False)
    )
    assert isinstance(result.accepted_result, Inapplicable)


def test_errors_keep_boundary_and_programming_failures_distinct() -> None:
    class Lanes(Space):
        extent: int = Param()
        lanes: int = Decision(domain=divisors_of(extent))

    report = design_space(Lanes(extent=12)).try_with_choices({Lanes.lanes: 5})
    assert not report.accepted
    assert ConfigurationError(report).report is report
    assert isinstance(RequestError("missing input"), ValueError)
    cause = ZeroDivisionError("bad formula")
    try:
        raise EvaluationError("cycles", "derived", "bad formula") from cause
    except EvaluationError as error:
        assert error.owner == "cycles"
        assert error.role == "derived"
        assert error.__cause__ is cause


def prepared(*nodes: Node) -> Model[Space]:
    """Minimal prepared model for evaluator tests using hand-built explicit IR."""
    linked = LinkedModel(
        nodes,
        (),
        tuple(node.index for node in nodes),
        tuple(node.index for node in nodes if node.kind == "param"),
        tuple(node.index for node in nodes if node.kind == "decision"),
        {node.key: node.index for node in nodes},
    )

    return Model(Space, linked)


INT = cast(ValueSemantics[object], default_semantics(int))
BOOL = cast(ValueSemantics[object], default_semantics(bool))


def test_iterative_evaluation_demands_only_a_deep_query_closure() -> None:
    calls: list[int] = []

    def increment(*, previous: int) -> int:
        calls.append(previous)
        return previous + 1

    def unrelated() -> int:
        raise AssertionError("an unrelated callback ran")

    nodes = [Node(0, 0, "start", "const", semantics=INT, value=0)]
    for index in range(1, 6001):
        nodes.append(
            Node(
                index,
                0,
                f"step{index}",
                "derived",
                semantics=INT,
                arguments=(Argument("previous", index - 1),),
                function=increment,
            )
        )
    nodes.append(Node(6001, 0, "unrelated", "derived", semantics=INT, function=unrelated))
    snapshot = Snapshot(prepared(*nodes), {})
    result = evaluate(snapshot, 6000)
    assert result.result == Available(6000)
    assert result.dependencies == (5999,)
    assert len(calls) == 6000
    assert evaluate(snapshot, 6000) is result
    assert len(calls) == 6000
    assert 6001 not in snapshot.cache


def test_guard_suppresses_callbacks_and_both_decision_queries_agree() -> None:
    def inactive() -> int:
        raise AssertionError("inactive callback ran")

    model = prepared(
        Node(0, 0, "enabled", "const", semantics=BOOL, value=False),
        Node(1, 0, "lanes", "decision", semantics=INT, guard=0),
        Node(2, 0, "body", "derived", semantics=INT, guard=0, function=inactive),
        Node(3, 0, "active", "decision", semantics=INT),
    )
    snapshot = Snapshot(model, {})
    assert isinstance(evaluate(snapshot, 1).result, Inapplicable)
    assert isinstance(decision_state(snapshot, 1), Inapplicable)
    assert isinstance(evaluate(snapshot, 2).result, Inapplicable)
    assert evaluate(snapshot, 2).dependencies == (0,)
    state = decision_state(snapshot, 3)
    assert isinstance(state, Available)
    assert state.value.status == "unassigned"
    assert isinstance(evaluate(snapshot, 3).result, Unresolved)


def test_missing_optional_input_blocks_callbacks_and_differs_from_supplied_none() -> None:
    none_semantics = cast(ValueSemantics[object], default_semantics(type(None)))
    calls: list[object] = []

    def consume(*, source: object) -> bool:
        calls.append(source)
        return source is None

    def refused() -> QueryResult[object]:
        return reject("unsupported", "unsupported value")

    model = prepared(
        Node(0, 0, "source", "param", semantics=none_semantics, required=False),
        Node(
            1,
            0,
            "consumer",
            "derived",
            semantics=BOOL,
            arguments=(Argument("source", 0),),
            function=consume,
        ),
        Node(2, 0, "refused", "derived", semantics=INT, function=refused),
        Node(
            3,
            0,
            "still_refused",
            "derived",
            semantics=BOOL,
            arguments=(Argument("source", 2),),
            function=consume,
        ),
    )
    absent = Snapshot(model, {})
    supplied = Snapshot(model, {0: None})
    assert isinstance(evaluate(absent, 1).result, Unresolved)
    assert calls == []
    assert evaluate(supplied, 1).result == Available(True)
    assert calls == [None]
    propagated = evaluate(absent, 3).result
    assert isinstance(propagated, Rejected)
    assert propagated.findings[0].owner == "refused"
    assert calls == [None]


def test_callback_arguments_are_detached_from_snapshot_values() -> None:
    lists = cast(ValueSemantics[object], default_semantics(list))

    def mutate(*, source: list[int]) -> int:
        source.append(99)
        return len(source)

    model = prepared(
        Node(0, 0, "source", "param", semantics=lists),
        Node(
            1,
            0,
            "length",
            "derived",
            semantics=INT,
            arguments=(Argument("source", 0),),
            function=mutate,
        ),
    )
    first = Snapshot(model, {0: [1]})
    second = Snapshot(model, {0: [2, 3]})
    assert evaluate(first, 1).result == Available(2)
    assert evaluate(first, 0).result == Available([1])
    assert evaluate(second, 1).result == Available(3)
    successor = Snapshot(first.model, first.parameters)
    assert successor.parameters is first.parameters
    assert successor.cache == {}
    assert first.cache
    assert evaluate(successor, 1).result == Available(2)
    assert evaluate(first, 0).result == Available([1])
    assert successor.cache is not first.cache


def test_selection_preserves_selected_refusal_and_skips_other_callback() -> None:
    string = cast(ValueSemantics[object], default_semantics(str))

    def refused() -> QueryResult[object]:
        return reject("geometry", "selected implementation refused")

    def inactive() -> int:
        raise AssertionError("nonselected callback ran")

    model = prepared(
        Node(0, 0, "case", "const", semantics=string, value="chosen"),
        Node(1, 0, "chosen", "derived", semantics=INT, function=refused),
        Node(2, 0, "other", "derived", semantics=INT, function=inactive),
        Node(
            3,
            0,
            "selection",
            "select",
            semantics=INT,
            selector=0,
            alternatives=(("chosen", 1), ("other", 2)),
        ),
    )
    snapshot = Snapshot(model, {})
    selected = evaluate(snapshot, 3)
    assert selected.result == evaluate(snapshot, 1).result
    assert isinstance(selected.result, Rejected)
    assert selected.dependencies == (0, 1)
    assert 2 not in snapshot.cache


def test_view_callback_and_accepted_dependency_share_the_cached_assessment() -> None:
    model = prepared(
        Node(0, 0, "raw", "const", semantics=INT, value=4),
        Node(1, 0, "support", "constraint", semantics=BOOL, function=lambda: False),
        Node(2, 0, "physical", "view", semantics=INT, output=0, constraints=(1,)),
        Node(3, 0, "parent", "alias", semantics=INT, output=2),
    )
    snapshot = Snapshot(model, {})
    parent = evaluate(snapshot, 3)
    direct = evaluate(snapshot, 2)
    assert isinstance(direct.assessment, ViewAssessment)
    assert parent.result is direct.result
    assert direct.assessment.accepted_result == direct.result
    assert isinstance(direct.result, Rejected)
    assert direct.result.findings[0].owner == "support"


def test_bad_callback_output_raises_contextual_programming_error() -> None:
    model = prepared(Node(0, 0, "broken", "derived", semantics=INT, function=lambda: "wrong"))
    with pytest.raises(EvaluationError) as raised:
        evaluate(Snapshot(model, {}), 0)
    assert raised.value.owner == "broken"
    assert raised.value.role == "derived"
    assert isinstance(raised.value.__cause__, TypeError)


def test_required_alias_preserves_canonical_missing_parameter_owner() -> None:
    def consume(*, value: object) -> int:
        raise AssertionError("missing input must prevent invocation")

    model = prepared(
        Node(0, 0, "source", "param", semantics=INT, required=False),
        Node(1, 0, "child.alias", "alias", semantics=INT, output=0),
        Node(
            2,
            0,
            "consumer",
            "derived",
            semantics=INT,
            arguments=(Argument("value", 1),),
            function=consume,
        ),
    )
    snapshot = Snapshot(model, {})
    answer = evaluate(snapshot, 2).result
    assert isinstance(answer, Unresolved)
    assert all(finding.owner == "source" for finding in answer.findings)
    assert evaluate(snapshot, 2).dependencies == (1,)
    assert evaluate(snapshot, 1).dependencies == (0,)
