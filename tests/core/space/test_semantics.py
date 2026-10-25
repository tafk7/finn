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
from finn.core.space.semantics import NotAValue, canonical, default_semantics, same


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


def test_an_int_check_takes_no_bool_and_a_list_is_no_value() -> None:
    integers = default_semantics(int)
    assert not integers.accepts(True)
    assert not same(True, 1, owner="test", role="test")
    with pytest.raises(NotAValue, match="list is mutable"):
        canonical([1, 2])


@dataclass(frozen=True)
class Pair:
    left: int
    right: tuple[int, ...]


@dataclass
class Bag:
    values: list[int]


def test_a_frozen_dataclass_is_its_own_canonical_form_and_a_mutable_one_is_no_value() -> None:
    """A frozen value class holding immutable values is held as it is; a mutable class,
    even one holding the same fields, is refused."""

    pair = Pair(1, (2, 3))
    assert canonical(pair) is pair
    with pytest.raises(NotAValue, match="not frozen"):
        canonical(Bag([1, 2]))


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

    @derived
    def body(self) -> int | Inapplicable:
        if not self.enabled:
            return Inapplicable()
        raise AssertionError("an enabled body ran")

    active: int = Decision(values=(1, 2))


def test_a_guard_suppresses_functions_and_both_decision_questions_agree() -> None:
    point = design_space(Guarded(enabled=False))
    assert isinstance(point.query(Guarded.lanes), Inapplicable)
    assert isinstance(point.field(Guarded.lanes).state, Inapplicable)
    assert isinstance(point.query(Guarded.body), Inapplicable)
    assert _reads(inspection.explain(point, Guarded.body), "body") == ["enabled"]
    state = point.field(Guarded.active).state
    assert isinstance(state, Available)
    assert state.value.status == "unassigned"
    assert isinstance(point.query(Guarded.active), Unresolved)


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
    selection = view(implementation.value)


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
    parent = view(inner.physical)


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


def test_a_view_of_a_refused_space_refuses_and_keeps_what_waits_visible() -> None:
    class Grouped(Space):
        @derived
        def output(self) -> int:
            return 4

        lanes: int = Decision(values=(1, 2))

        @constraint
        def refused() -> bool:
            return False

        @constraint
        def pending(*, lanes: int) -> bool:
            return lanes > 0

        physical = view(output)

    point = design_space(Grouped())
    for _ in range(2):
        assessment = point.inspect(Grouped.physical)
        assert assessment.output_result == Available(4)
        assert isinstance(assessment.accepted_result, Rejected)
        assert isinstance(assessment.validity, Rejected)
        assert [finding.owner for finding in assessment.validity.findings] == ["refused"]
        assert assessment.accepted_result == assessment.validity
        # Readiness, a diagnostic, names what the refused Space still waits on.
        readiness = assessment.readiness
        assert readiness.ready is None
        assert isinstance(readiness.results["$validity"], Rejected)
        assert isinstance(readiness.results["pending"], Unresolved)
    validity = inspection.validity(point)
    assert validity.status == "refused"
    assert [piece.owner for piece in validity.pieces] == ["refused"]
    committed = point.with_choices(lanes=1)
    assert isinstance(committed.inspect(Grouped.physical).accepted_result, Rejected)
    assert committed.inspect(Grouped.physical).readiness.ready is True


def test_a_view_of_a_space_with_no_pieces_reads_its_output() -> None:
    class Empty(Space):
        @derived
        def output(self) -> int:
            return 4

        physical = view(output)

    point = design_space(Empty())
    assert point.physical == 4
    assert point.inspect(Empty.physical).readiness.ready is True
    assert inspection.validity(point) == inspection.Validity("valid", (), Available(True))
