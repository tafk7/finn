# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Forced Decisions: an open Decision with one viable case reads as that case.

Two cores share a ``width``: ``packed`` refuses more than 64 lanes, ``stub`` an
odd width, each through its ``admission``. A value Decision's cases state
requirements over a capability record. Forced values are derived at read time
and never stored; a successor reuses the verdicts its change does not reach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from finn.core.space import (
    Available,
    ConstraintGroup,
    Decision,
    DefinitionError,
    Param,
    Rejected,
    Space,
    Unresolved,
    constraint,
    derived,
    design_space,
    divisors_of,
    forcing,
    inspection,
    reject,
    requires,
    requiring,
    selections,
)


class Packed(Space):
    width: int = Param()
    pe: int = Decision(domain=divisors_of(width))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.width > 64:
            return reject("packed-width", "the packed core takes at most 64 lanes")
        return True

    admission = ConstraintGroup(fits)


class Stub(Space):
    width: int = Param()
    rows: int = Decision(values=(1, 2, 4))

    @constraint
    def even(self) -> bool | Rejected:
        if self.width % 2:
            return reject("stub-width", "the stub core takes an even width")
        return True

    admission = ConstraintGroup(even)


CORES: dict[str, type[Space] | Space] = {"packed": Packed, "stub": Stub}


class Unit(Space):
    width: int = Param()
    compute: Packed | Stub = Decision(CORES, width=width)


def commit(point: Any, choices: dict[str, object]) -> Any:
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    return point.with_choices({owned[key]: value for key, value in choices.items()})


def forced(point: Any) -> dict[str, object]:
    return {item.key: item.value for item in inspection.forced(point)}


def test_one_viable_case_reads_as_that_case_and_is_never_committed() -> None:
    point = design_space(Unit(width=128))
    assert isinstance(point.compute, Stub)
    state = point.field(Unit.compute).state
    assert isinstance(state, Available)
    assert (state.value.status, state.value.value) == ("unassigned", None)
    assert selections.capture(point).keys == ()
    (item,) = inspection.forced(point)
    assert (item.key, item.value) == ("compute", "stub")
    assert set(item.refused) == {"packed"} and "packed-width" in item.refused["packed"]


def test_several_viable_cases_stay_open_and_none_is_a_refusal_naming_each_case() -> None:
    both = design_space(Unit(width=8))
    answer = both.query(Unit.compute)
    assert isinstance(answer, Unresolved)
    assert {finding.code for finding in answer.findings} == {"decision-unassigned"}
    assert forced(both) == {}
    neither = design_space(Unit(width=129)).query(Unit.compute)
    assert isinstance(neither, Rejected)
    (finding,) = neither.findings
    assert finding.code == "decision-no-viable-case"
    assert "packed-width" in finding.message and "stub-width" in finding.message


def test_a_choice_nested_under_a_forced_selector_commits_without_it() -> None:
    # The trial reads its base's forced selector, so the nested choice applies.
    point = commit(design_space(Unit(width=128)), {"compute.stub.rows": 2})
    assert selections.capture(point).keys == ("compute.stub.rows",)
    assert point.compute.rows == 2 and forced(point) == {"compute": "stub"}


def test_a_batch_commits_a_choice_under_a_selector_another_choice_of_it_forces() -> None:
    class Batch(Space):
        lanes: int = Decision(values=(2, 3))
        compute: Packed | Stub = Decision(CORES, width=lanes)

    base = design_space(Batch())
    assert forced(base) == {}  # either core may take an open width
    # Three lanes refuse the stub core, so packed is forced, and its PE applies.
    point = commit(base, {"lanes": 3, "compute.packed.pe": 1})
    assert forced(point) == {"compute": "packed"} and point.compute.pe == 1
    refused = base.try_with_choices(commit_handles(base, {"lanes": 2, "compute.packed.pe": 1}))
    assert not refused.accepted  # two lanes leave both cores open: pe is not applicable


def commit_handles(point: Any, choices: dict[str, object]) -> dict[Any, object]:
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    return {owned[key]: value for key, value in choices.items()}


def test_an_admission_waiting_on_an_open_choice_does_not_refuse() -> None:
    class Waiting(Space):
        width: int = Param()
        pe: int = Decision(values=(1, 2, 8))

        @constraint
        def small(self) -> bool | Rejected:
            if self.pe > 4:
                return reject("waiting-pe", "at most 4 lanes a cycle")
            return True

        admission = ConstraintGroup(small)

    class Root(Space):
        width: int = Param()
        entries: dict[str, type[Space] | Space] = {"waiting": Waiting, "stub": Stub}
        compute: Waiting | Stub = Decision(entries, width=width)

    point = design_space(Root(width=7))  # odd: stub refuses; waiting waits on its pe
    assert forced(point) == {"compute": "waiting"}
    candidate = inspection.candidate(point, Root.compute, "waiting")
    assert candidate is not None and isinstance(forcing.admission(candidate), Unresolved)


def test_forcing_cascades_through_a_choice_the_forced_case_opens() -> None:
    class Chain(Space):
        width: int = Param()
        compute: Packed | Stub = Decision(CORES, width=width)

        @derived
        def after_stub(self) -> bool:
            return isinstance(self.compute, Stub)

        post: Packed | Stub = Decision(CORES, width=width, when=after_stub)

    assert forced(design_space(Chain(width=128))) == {"compute": "stub", "post": "stub"}


def test_a_value_decision_with_one_case_is_forced() -> None:
    class Lanes(Space):
        lanes: int = Decision(values=(4,))
        many: int = Decision(values=(1, 2))

    point = design_space(Lanes())
    assert point.lanes == 4 and isinstance(point.query(Lanes.many), Unresolved)
    assert [(item.key, item.value, item.refused) for item in inspection.forced(point)] == [
        ("lanes", 4, {})
    ]


@dataclass(frozen=True)
class Capabilities:
    fast: bool = True
    ports: int = 1


class Memory(Space):
    capabilities: Capabilities = Param(default=Capabilities())
    # A value formal projects its value's attributes in the class body.
    style: str = Decision(
        values=("auto", "fast"),
        requires=(
            requires(capabilities.fast, "fast-absent: no fast memory here", cases=("fast",)),
        ),
    )
    writable: bool = Decision(
        domain=requiring(
            (False, True),
            requires(capabilities.ports, "port-absent: no port to write through", cases=(True,)),
        )
    )


def test_a_value_case_whose_requirement_fails_is_refused_at_commit_and_not_viable() -> None:
    open_ = design_space(Memory())
    assert forced(open_) == {}
    assert commit(open_, {"style": "fast"}).style == "fast"
    point = design_space(Memory(capabilities=Capabilities(fast=False, ports=0)))
    assert forced(point) == {"style": "auto", "writable": False}
    reasons = {item.key: dict(item.refused) for item in inspection.forced(point)}
    assert set(reasons["style"]) == {"'fast'"} and "fast-absent" in reasons["style"]["'fast'"]
    assert "port-absent" in reasons["writable"]["True"]
    report = point.try_with_choices({Memory.style: "fast"})
    assert not report.accepted
    (outcome,) = report.outcomes
    assert isinstance(outcome.result, Rejected)
    assert [finding.code for finding in outcome.result.findings] == ["fast-absent"]
    # Enumeration stays the declared cases: the domain does not move with the facts.
    assert point.field(Memory.style).candidates() == Available(("auto", "fast"))


def test_a_requirement_is_written_code_colon_message() -> None:
    with pytest.raises(DefinitionError, match="code: message"):
        requires(Memory.capabilities, "no code here")


def test_adding_a_choice_never_switches_a_forced_case() -> None:
    """Monotonicity: a forced case stays forced, or the configuration is refused."""

    class Limited(Space):
        width: int = Param()
        limit: int = Decision(values=(1, 8))
        pe: int = Decision(domain=divisors_of(width))

        @constraint
        def within(self) -> bool | Rejected:
            if self.limit < 2:
                return reject("limited", "the limit leaves no lane")
            return True

        admission = ConstraintGroup(within)

    class Root(Space):
        width: int = Param()
        entries: dict[str, type[Space] | Space] = {"limited": Limited, "stub": Stub}
        compute: Limited | Stub = Decision(entries, width=width)

    base = design_space(Root(width=7))
    assert forced(base) == {"compute": "limited"}
    for limit in (1, 8):
        point = commit(base, {"compute.limited.limit": limit})
        answer = point.query(Root.compute)
        if isinstance(answer, Available):
            assert forced(point) == {"compute": "limited"}
        else:
            assert isinstance(answer, Rejected) and limit == 1
            assert [finding.code for finding in answer.findings] == ["decision-no-viable-case"]


def test_a_successor_reuses_the_verdicts_its_change_does_not_reach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Two(Space):
        width: int = Param()
        left: Packed | Stub = Decision(CORES, width=width)
        right: Packed | Stub = Decision(CORES, width=width)

    base = design_space(Two(width=128))
    assert forced(base) == {"left": "stub", "right": "stub"}
    found: list[str] = []
    verdict = forcing._verdict

    def counted(current: Any, index: int) -> Any:
        found.append(current.linked.nodes[index].key)
        return verdict(current, index)

    monkeypatch.setattr(forcing, "_verdict", counted)
    point = commit(base, {"left.stub.rows": 4})
    assert forced(point) == {"left": "stub", "right": "stub"}
    # Neither selector read ``left.stub.rows``; only the rest of left's own open choices is new.
    assert not any(key.endswith(("left", "right")) for key in found)
