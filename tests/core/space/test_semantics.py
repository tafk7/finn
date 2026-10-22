# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import pytest

from finn.core.reporting.findings import Finding, FindingKind, ordered_findings
from finn.core.space import (
    Decision,
    Param,
    Select,
    Space,
    View,
    constraint,
    derived,
    design_space,
    divisors_of,
    inspection,
    view,
)
from finn.core.space.errors import ConfigurationError, EvaluationError, RequestError
from finn.core.space.inspection import QueryEvidence
from finn.core.space.results import (
    Available,
    Inapplicable,
    QueryResult,
    Rejected,
    Unresolved,
    ViewAssessment,
    assess_constraints,
    assess_view,
    constraint_result,
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


def test_a_refused_constraint_beside_an_unresolved_one_refuses_the_group() -> None:
    refused = reject("geometry", "too wide", owner="width")
    assessment = assess_constraints(
        {"lanes": blocker(), "width": refused, "unused": Inapplicable()}
    )
    assert assessment.verdict is False
    assert isinstance(assessment.result, Rejected)
    assert isinstance(assessment.results["lanes"], Unresolved)
    assert assessment.refused == ("width",)
    assert assessment.not_applicable == ("unused",)
    assert assessment.results["width"] is not None
    assert assessment.results["width"] == refused


def test_readiness_tracks_unresolved_results_without_erasing_refusal() -> None:
    refused = reject("no", "unsupported", owner="support")
    waiting: ViewAssessment[int] = assess_view(blocker(), owner="physical", validity=refused)
    assert waiting.readiness.ready is None
    assert waiting.validity == refused
    assert waiting.accepted_result == refused
    final = assess_view(Available(1), owner="physical", validity=refused)
    assert final.readiness.ready is True
    assert final.accepted_result == refused


def test_view_reducer_obeys_readiness_output_and_refusal_precedence() -> None:
    raw: QueryResult[int] = Available(4)
    refused = reject("geometry", "too wide", owner="width")
    # A refused Space refuses the view whatever waits; readiness still names what waits.
    waiting = assess_view(raw, owner="physical", validity=refused, waiting={"lanes": blocker()})
    assert waiting.output_result == raw
    assert waiting.accepted_result == refused
    assert waiting.readiness.ready is None
    assert waiting.readiness.results["lanes"] == blocker()
    inactive: ViewAssessment[int] = assess_view(Inapplicable(), owner="physical", validity=refused)
    assert inactive.accepted_result == refused
    # An open Space leaves the view answering its output.
    absent: ViewAssessment[int] = assess_view(Inapplicable(), owner="physical", validity=blocker())
    assert isinstance(absent.accepted_result, Inapplicable)
    undecided = assess_view(raw, owner="physical", validity=blocker())
    assert undecided.accepted_result == raw and undecided.readiness.ready is None
    denied = assess_view(raw, owner="physical", validity=refused)
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


# -- evaluation, through the public API on Space classes ---------------------------------------
#
# Each claim below was once made on a table of nodes built by hand, evaluated directly; a
# hand-built table is no design space value, so each is made here on a Space class.


def _line(depth: int, calls: list[str]) -> type[Space]:
    """``step<n>`` reads ``step<n-1>`` through its receiver, ``depth`` deep, each call that
    completes its read logged (a call the depth guard abandons in its read runs again);
    ``unrelated`` reads nothing and must never run."""
    members: dict[str, object] = {"__annotations__": {"step0": int}, "step0": Param()}

    def step(previous: str) -> Callable[[Space], int]:
        def compute(self: Space) -> int:
            value = cast(int, getattr(self, previous))
            calls.append(previous)
            return value + 1

        return compute

    def unrelated(self: Space) -> int:
        raise AssertionError("an unrelated function ran")

    for number in range(1, depth + 1):
        members[f"step{number}"] = derived(step(f"step{number - 1}"))
    members["unrelated"] = derived(unrelated)
    return cast(type[Space], type("Line", (Space,), members))


def _reads(evidence: QueryEvidence[Any], key: str) -> list[str]:
    """What the node ``key`` read in ``evidence``, by key, in order."""
    keys = {node.declaration.reference: node.declaration.key for node in evidence.nodes}
    (node,) = [node for node in evidence.nodes if node.declaration.key == key]
    return [keys[dependency] for dependency in node.dependencies]


@pytest.mark.usefixtures("purity_unchecked")
def test_a_deep_read_evaluates_only_what_it_reads_each_once() -> None:
    calls: list[str] = []
    line = _line(300, calls)
    point = design_space(line(step0=0))  # type: ignore[call-arg]
    assert point.query(line.step300) == Available(300)  # type: ignore[attr-defined]
    assert len(calls) == 300
    assert point.query(line.step300) == Available(300)  # type: ignore[attr-defined]
    assert len(calls) == 300
    evidence = inspection.explain(point, line.step300)  # type: ignore[attr-defined]
    assert _reads(evidence, "step300") == ["step299"]
    assert "unrelated" not in {node.declaration.key for node in evidence.nodes}


class Guarded(Space):
    enabled: bool = Param()
    lanes: int = Decision(values=(1, 2), when=enabled)

    @derived(when=enabled)
    def body(self) -> int:
        raise AssertionError("inactive function ran")

    active: int = Decision(values=(1, 2))


def test_a_guard_suppresses_functions_and_both_decision_questions_agree() -> None:
    point = design_space(Guarded(enabled=False))
    assert isinstance(point.query(Guarded.lanes), Inapplicable)
    assert isinstance(point.field(Guarded.lanes).state, Inapplicable)
    assert isinstance(point.query(Guarded.body), Inapplicable)
    assert _reads(inspection.explain(point, Guarded.body), "body") == ["body.$guard"]
    state = point.field(Guarded.active).state
    assert isinstance(state, Available)
    assert state.value.status == "unassigned"
    assert isinstance(point.query(Guarded.active), Unresolved)


CONSUMED: list[object] = []


class Consumer(Space):
    source: None = Param(required=False)

    @derived
    def consumer(self) -> bool:
        source = self.source
        CONSUMED.append(source)
        return source is None

    @derived
    def refused(self) -> int | Rejected:
        return reject("unsupported", "unsupported value")

    @derived
    def still_refused(self) -> bool:
        CONSUMED.append(self.refused)
        return True


def test_a_missing_optional_input_blocks_functions_and_differs_from_a_supplied_none() -> None:
    CONSUMED.clear()
    absent = design_space(Consumer())
    supplied = design_space(Consumer(source=None))
    assert isinstance(absent.query(Consumer.consumer), Unresolved)
    assert CONSUMED == []
    assert supplied.query(Consumer.consumer) == Available(True)
    assert CONSUMED == [None]
    propagated = absent.query(Consumer.still_refused)
    assert isinstance(propagated, Rejected)
    assert propagated.findings[0].owner == "refused"
    assert CONSUMED == [None]


class Lengths(Space):
    source: list[int] = Param()

    @derived
    def length(self) -> int:
        source = self.source
        source.append(99)
        return len(source)


def test_a_functions_arguments_are_detached_from_the_stored_values() -> None:
    first = design_space(Lengths(source=[1]))
    assert first.length == 2
    assert first.source == [1]
    assert design_space(Lengths(source=[2, 3])).length == 3
    again = design_space(Lengths(source=[1]))
    assert again.length == 2
    assert first.source == [1] and again.source == [1]


class Chosen(Space):
    @derived
    def value(self) -> int | Rejected:
        return reject("geometry", "selected implementation refused")


class Unchosen(Space):
    @derived
    def value(self) -> int:
        raise AssertionError("the candidate not selected ran")


class Holder(Space):
    implementation: Chosen | Unchosen = Select(chosen=Chosen, other=Unchosen)
    selection = View(implementation.value)


def test_a_selection_keeps_the_selected_refusal_and_skips_the_other_candidate() -> None:
    point = design_space(Holder()).with_choices({Holder.implementation: "chosen"})
    selected = point.query(Holder.selection)
    assert isinstance(selected, Rejected)
    assert selected == point.query(Holder.implementation["chosen"].value)  # type: ignore[index]
    assert selected.findings[0].owner == "implementation.chosen.value"
    evidence = inspection.explain(point, Holder.selection)
    assert "implementation.other.value" not in {node.declaration.key for node in evidence.nodes}


class Physical(Space):
    raw: int = Param()

    @constraint
    def support(self) -> bool:
        return False

    @view
    def physical(self) -> int:
        return self.raw


class Placed(Space):
    inner = Physical(raw=4)
    parent = View(inner.physical)


def test_a_view_and_a_view_reading_it_answer_its_accepted_result() -> None:
    point = design_space(Placed())
    direct = point.inspect(Placed.inner.physical)
    assert isinstance(direct, ViewAssessment)
    assert point.query(Placed.parent) == direct.accepted_result
    assert point.query(Placed.inner.physical) == direct.accepted_result
    assert isinstance(direct.accepted_result, Rejected)
    assert direct.accepted_result.findings[0].owner == "inner.support"


class Broken(Space):
    @derived
    def broken(self) -> int:
        return "wrong"  # type: ignore[return-value]


def test_a_functions_output_of_the_wrong_type_is_its_owners_error() -> None:
    with pytest.raises(EvaluationError) as raised:
        design_space(Broken()).query(Broken.broken)
    assert raised.value.owner == "broken"
    assert raised.value.role == "derived"
    assert isinstance(raised.value.__cause__, TypeError)


class Reader(Space):
    value: int = Param()

    @derived
    def consumer(self) -> int:
        self.value  # noqa: B018 - a missing input halts the read
        raise AssertionError("a missing input must prevent the call")


class Supplier(Space):
    source: int = Param(required=False)
    child = Reader(value=source)


def test_a_formal_bound_to_an_omitted_input_names_the_input_as_missing() -> None:
    point = design_space(Supplier())
    answer = point.query(Supplier.child.consumer)
    assert isinstance(answer, Unresolved)
    assert all(finding.owner == "source" for finding in answer.findings)
    evidence = inspection.explain(point, Supplier.child.consumer)
    assert _reads(evidence, "child.consumer") == ["source"]
