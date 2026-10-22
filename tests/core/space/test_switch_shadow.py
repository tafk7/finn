# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The switch's shadow run (``core.space._switch_shadow``): the public API answering, and
today's engine asked every question again, held to its own contract.

- **A planted disagreement is caught and named**: today's answer made wrong (or raising)
  is a disagreement naming the question, the member asked, the design space and both
  answers, while the caller still gets the public API's answer; run as a plugin, the test
  it was met in fails and the session's count names it.
- **Transparent where both agree**: every question asked with the shadow on answers as it
  does with it off, a raise included, and the Space objects returned are the public
  API's own.
- **The count is right**: each question asked is counted once by its kind; a question a
  method asks of its receiver is left to the public API alone, and so is one asked from
  another thread.
- **Today's side changes nothing the test reads**: the model's session (its counters) and
  the design space's evaluations are as the public API left them, a public question asked
  inside today's side answers in a scratch session, and a planted change is an isolation
  break; today's engine reads its own history after a change.
- **What today's engine is asked**: a change's items as the caller gave them, snapshotted
  before the public API's run may change them; a clear in a bind, refused before any
  question, is not compared; nor is a test named in ``UNCOMPARED``.
- **Engine modules under it**: three of ``tests/core/space``'s modules that ask
  ``explain`` and the bound accessors, run with the plugin, pass with 0 disagreements.
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
import textwrap
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from core.space._running_shadow import UNCOMPARED
from core.space._switch_shadow import Shadow, form
from core.space._toys_support import House
from finn.core.space import (
    Available,
    BoundDecision,
    Decision,
    Inapplicable,
    Param,
    RequestError,
    Space,
    ValueUnavailableError,
    bind,
    derived,
    design_space,
    inspection,
    occurrence,
)
from finn.core.space._state import State
from finn.core.space.domains import Domain

TESTS = Path(__file__).parent.parent.parent

COUNT = re.compile(
    r"switch shadow: (\d+) questions compared \((.*)\); disagreements: (\d+); "
    r"isolation breaks: (\d+)"
)
SUMMARY = re.compile(r"(\d+) (passed|failed|errors?|xfailed|xpassed|skipped|deselected)")


@contextmanager
def shadowed(**today: Callable[..., object]) -> Iterator[Shadow]:
    """A shadow applied for the block (today's answers replaced by kind where given), then
    restored."""
    shadow = Shadow(today=today)
    shadow.apply()
    try:
        yield shadow
    finally:
        shadow.restore()
        shadow.end()


def _opened() -> House:
    return design_space(House(budget=100))


def _outcome(ask: Callable[[], object]) -> object:
    """What asking gives, in a form two runs share: its answer, or what it raised."""
    try:
        return ("answers", form(ask()))
    except Exception as error:  # noqa: BLE001 - a raise is an answer
        return ("raises", type(error).__name__, str(error))


def _questions(point: House) -> dict[str, Callable[[], object]]:
    """One of every question compared, asked of the house."""
    finish = point.field(House.kitchen.finish)
    return {
        "value": lambda: point.budget,
        "query": lambda: point.query(House.total),
        "present": lambda: point.present(House.garage),
        "inspect": lambda: point.inspect(House.total),
        "field": lambda: isinstance(point.field(House.kitchen.finish), BoundDecision),
        "state": lambda: finish.state,
        "candidates": finish.candidates,
        "child": lambda: point.kitchen,
        "root": lambda: point.root,
        "selected": lambda: point.heating,
        "candidate": lambda: inspection.candidate(point, House.heating, "heat_pump"),
        "validity": lambda: inspection.validity(point),
        "states": lambda: inspection.states(point),
        "viable": lambda: inspection.viable(point, [House.kitchen.finish, House.heating]),
        "explain": lambda: inspection.explain(point, House.total),
        "change": lambda: point.try_with_choices({House.kitchen.finish: 2}),
        "change of facts": lambda: point.try_with_facts({House.budget: 40}),
        "bind": lambda: bind(point, {House.kitchen.finish: 3, House.dining.finish: 3}),
    }


def _answers(point: House) -> dict[str, object]:
    return {kind: _outcome(ask) for kind, ask in _questions(point).items()}


# -- a planted disagreement ------------------------------------------------------------------


def test_a_planted_disagreement_is_caught_and_named() -> None:
    """Today's answer to ``query`` made wrong: the caller gets the public API's answer, and
    the disagreement names the question, the member, the design space and both answers."""
    expected = _opened().query(House.total)
    with shadowed(query=lambda point, reference: Available(-1)) as shadow:
        point = _opened()
        assert point.query(House.total) == expected
        assert point.budget == 100
    assert shadow.compared == Counter({"design_space": 1, "query": 1, "value": 1})
    [found] = shadow.disagreements
    assert shadow.take_pending() == [found] and shadow.take_pending() == []
    assert (found.question, found.asked) == ("query", "total")
    assert found.configuration.startswith("House {")
    differences = "\n".join(found.differences)
    assert 'kind: "value" | "unresolved"' in differences, differences
    assert 'value: -1 | {"missing": true}' in differences, differences
    text = str(found)
    assert text.startswith("query total of House {}: today's engine and the running type")
    assert all(line in text for line in found.differences)
    assert "disagreements: 1; isolation breaks: 0" in shadow.count()


def test_a_raise_on_one_side_is_a_disagreement() -> None:
    """Today's engine raising where the public API answers, and answering where it raises:
    each a disagreement; both raising alike agree."""

    def raises(*_: object) -> object:
        raise ValueUnavailableError(Inapplicable(), context="planted")

    with shadowed(value=raises, selected=lambda *_: None) as shadow:
        point = _opened()
        assert point.budget == 100
        with pytest.raises(ValueUnavailableError):
            point.heating  # noqa: B018 - a read; the selector is open
    found = {item.question: item for item in shadow.disagreements}
    assert sorted(found) == ["selected", "value"]
    assert "ValueUnavailableError" in found["value"].differences[0]
    assert '"raised": "ValueUnavailableError"' in found["selected"].differences[0]
    with shadowed() as shadow:
        point = _opened()
        with pytest.raises(ValueUnavailableError):
            point.heating  # noqa: B018
    assert shadow.compared["selected"] == 1 and not shadow.disagreements


def test_a_test_with_a_disagreement_fails_under_the_plugin(tmp_path: Path) -> None:
    """Run as a plugin: the test whose question disagrees fails with the disagreement named,
    the tests beside it pass (the purity check on, both switches, unless a test turns it
    off), and the summary counts the tests' questions."""
    module = tmp_path / "test_planted.py"
    module.write_text(
        textwrap.dedent(
            """
            from support.purity import purity_unchecked

            from finn.core.space import Decision, Param, Space, _world, design_space
            from finn.core.space import occurrence
            from finn.core.space._running import purity


            class Example(Space):
                width: int = Param()
                pe: int = Decision(values=(1, 2))


            def test_agreeing():
                assert design_space(Example(width=4)).width == 4


            def test_planted(monkeypatch):
                monkeypatch.setattr(occurrence, "_world_read_value", lambda point, ref: 5)
                assert design_space(Example(width=4)).width == 4


            def test_the_purity_check_is_on():
                assert _world._PURITY and purity._PURITY


            def test_a_test_may_turn_it_off(purity_unchecked):
                assert not _world._PURITY and not purity._PURITY
            """
        )
    )
    ran = _pytest(str(module))
    tail = _tail(ran)
    assert ran.returncode == 1, tail
    assert _counts(ran) == {"failed": 1, "passed": 3}, tail
    assert "FAILED " in ran.stdout and "test_planted.py::test_planted" in ran.stdout, tail
    assert "SwitchDisagreement: value width of Example {}" in ran.stdout, tail
    assert (
        "switch shadow: 4 questions compared (design_space 2, value 2); disagreements: 1; "
        "isolation breaks: 0" in ran.stdout
    ), tail


# -- transparent where both agree ---------------------------------------------------------


def test_the_shadow_is_transparent_where_both_agree() -> None:
    """Every question answers with the shadow on as with it off (a raise included), and the
    Space objects it returns are the public API's own."""
    without = _answers(_opened())
    with shadowed() as shadow:
        point = _opened()
        within = _answers(point)
        unchanged = point.try_with_choices({House.want_garage: False})
        assert unchanged.accepted
        assert point.try_with_choices().instance is point
        returned = [point.kitchen, point.root, point.with_choices({House.kitchen.finish: 2})]
        assert all(occurrence.held(item) is not None for item in returned)
    assert within == without
    assert cast(tuple[object, ...], within["selected"])[0] == "raises"  # the selector is open
    assert not shadow.disagreements, "\n".join(map(str, shadow.disagreements))
    assert not shadow.breaks, "\n".join(map(str, shadow.breaks))


# -- the count ------------------------------------------------------------------------------


class Doubled(Space):
    width: int = Param()
    pe: int = Decision(values=(1, 2))

    @derived
    def doubled(self) -> int:
        return 2 * self.width  # a receiver's read, left to the public API alone


def test_the_count_is_right() -> None:
    """Each question asked once is counted once, by its kind; the house's questions once
    each, and a design space's opening; a receiver's reads and another thread's questions
    are left to the public API alone, by reason."""
    with shadowed() as shadow:
        point = _opened()
        questions = _questions(point)
        for ask in questions.values():
            _outcome(ask)
    expected = Counter(dict.fromkeys(questions, 1))
    # the house opened, the field the questions share, and ``viable``'s one call
    expected.update({"design_space": 1, "field": 1})
    assert shadow.compared == expected
    assert shadow.count() == (
        f"switch shadow: {sum(expected.values())} questions compared ("
        + ", ".join(f"{kind} {number}" for kind, number in sorted(expected.items()))
        + "); disagreements: 0; isolation breaks: 0"
    )
    with shadowed() as shadow:
        doubled = design_space(Doubled(width=3))
        assert doubled.doubled == 6
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(lambda: doubled.width).result() == 3
    assert shadow.compared == Counter({"design_space": 1, "value": 1})
    assert shadow.left["a receiver's or a declaration's"] >= 1
    assert shadow.left["asked from another thread"] == 1
    assert "explain's fact reads compared as a set: 0" in shadow.left_alone()


# -- today's side changes nothing the test reads -------------------------------------------------


def _twin() -> type[Space]:
    """A Space class of its own (its own compiled model, so its own session)."""

    class Twin(Space):
        width: int = Param()
        pe: int = Decision(values=(1, 2, 4))
        fast: bool = Decision(values=(False, True))

        @derived
        def cycles(self) -> int:
            return self.width // self.pe

        @derived
        def wide(self) -> bool:
            return self.fast and self.pe > 1

    return Twin


def _twin_questions(space_type: Any) -> list[object]:
    """The same questions of a twin, asked in order: their answers' forms."""
    point = design_space(space_type(width=8))
    moved = point.with_choices(pe=2)
    found = [
        _outcome(lambda: moved.cycles),
        _outcome(lambda: moved.query(space_type.wide)),
        _outcome(lambda: inspection.validity(moved)),
        _outcome(lambda: inspection.states(moved)),
        _outcome(lambda: inspection.viable(moved, [space_type.pe, space_type.fast])),
        _outcome(lambda: inspection.explain(moved, space_type.cycles)),
        _outcome(lambda: moved.try_with_facts({space_type.width: 12}).accepted),
    ]
    for ds in (occurrence.held(point).ds, occurrence.held(moved).ds):
        session = ds.model.per_class.session()
        assert occurrence._WORLD not in ds.evaluations
        keys = cast(list[tuple[object, ...]], list(ds.evaluations))
        assert all(key[0] is session for key in keys), keys
    return found


def test_todays_side_leaves_the_models_session_and_the_design_space_as_they_were() -> None:
    """Two twins asked the same questions, one under the shadow: the same answers and the
    same counts in their models' sessions (the counters the count-of-work tests read), and
    each design space's evaluations are its model session's alone, nothing of today's
    planted there; every run of today's side found nothing changed."""
    alone, shadowed_twin = _twin(), _twin()
    without = _twin_questions(alone)
    with shadowed() as shadow:
        within = _twin_questions(shadowed_twin)
    assert within == without
    assert sum(shadow.compared.values()) >= 8 and not shadow.disagreements
    assert not shadow.breaks, "\n".join(map(str, shadow.breaks))

    def counts(space_type: Any) -> dict[str, object]:
        session = inspection.model(space_type).per_class.session()
        return dataclasses.asdict(session.counters)

    assert counts(shadowed_twin) == counts(alone)


def test_a_public_question_inside_todays_side_answers_in_a_scratch_session() -> None:
    """Inside today's side, ``occurrence.state`` is the shadow's own state (nothing planted
    on the design space), and a public question answers in a scratch session, leaving the
    model's session as it was."""
    twin: Any = _twin()
    other = design_space(twin(width=8)).with_choices(pe=2)
    model = inspection.model(other)
    assert other.cycles == 4
    before = dataclasses.asdict(model.per_class.session().counters)
    states: list[object] = []

    def asking(point: Space, reference: Any) -> object:
        states.append(occurrence.state(point))
        assert other.cycles == 4  # the public API, inside today's side
        return occurrence._world_read_value(point, reference)

    with shadowed(value=asking) as shadow:
        point = _opened()
        assert point.budget == 100
    [state] = states
    assert (
        isinstance(state, State) and occurrence._WORLD not in occurrence.held(point).ds.evaluations
    )
    assert dataclasses.asdict(model.per_class.session().counters) == before
    assert shadow.left["asked inside today's engine"] == 1
    assert not shadow.disagreements and not shadow.breaks


def test_a_planted_change_to_what_the_test_reads_is_an_isolation_break() -> None:
    """Today's side writing to the asked design space's evaluations: an isolation break,
    naming the question and what changed, failing the test as a disagreement does."""

    def planting(point: Space, reference: Any) -> object:
        occurrence.held(point).ds.evaluations["planted"] = True
        return occurrence._world_read_value(point, reference)

    with shadowed(value=planting) as shadow:
        point = _opened()
        assert point.budget == 100
    [found] = shadow.breaks
    assert found.question == "value"
    assert found.differences == ["evaluations: nothing | {'planted'}"], found.differences
    assert shadow.take_pending() == [found] and not shadow.disagreements
    assert "isolation breaks: 1" in shadow.count()


def test_today_reads_its_own_history_after_a_change() -> None:
    """A change both agree on: the design space the public API published is answered, from
    then on, by the state today's change published, not one started again."""
    with shadowed() as shadow:
        point = _opened()
        moved = point.with_choices({House.kitchen.finish: 2})
        ds = occurrence.held(moved).ds
        reference, state = shadow.worlds[id(ds)]
        assert reference() is ds and dict(state.stored) == dict(ds.stored)
        assert moved.kitchen.finish == 2
        assert shadow.worlds[id(ds)][1] is state
        assert occurrence._WORLD not in ds.evaluations
    assert not shadow.disagreements and not shadow.breaks


# -- what today's engine is asked -----------------------------------------------------------


def test_a_planted_mutation_of_the_callers_input_is_not_a_disagreement() -> None:
    """A membership callback appends to the list the caller set another choice to, while
    the public API's change runs: the public API had snapshotted it first, and today's
    engine is asked the snapshot taken before the public API's run, so the two agree."""
    payload = [1]

    def mutating(*, candidate: int) -> bool:
        payload.append(2)
        return candidate > 0

    def accepting(*, candidate: list[int]) -> bool:
        return True

    class Trial(Space):
        first: int = Decision(domain=Domain((), mutating))
        second: list[int] = Decision(domain=Domain((), accepting))

    with shadowed() as shadow:
        base = design_space(Trial())
        result = base.try_with_choices(
            base.field(Trial.first).change(1), base.field(Trial.second).change(payload)
        )
        assert result.accepted and result.instance.second == [1]
    assert payload[:2] == [1, 2]
    assert shadow.compared["change"] == 1
    assert not shadow.disagreements, "\n".join(map(str, shadow.disagreements))


def test_a_clear_inside_a_bind_is_not_compared() -> None:
    """The front end refuses a clear in a bind as it reads the bind's items: refused
    before any question, left to the public API alone, and nothing disagrees."""
    with shadowed() as shadow:
        point = design_space(Doubled(width=3)).with_choices(pe=2)
        with pytest.raises(RequestError, match="clears none"):
            bind(point, point.field(Doubled.pe).clear())
    assert shadow.compared["bind"] == 0
    assert shadow.left["refused before any question"] == 1
    assert not shadow.disagreements, "\n".join(map(str, shadow.disagreements))


def test_a_test_making_the_running_type_wrong_on_purpose_is_not_compared() -> None:
    """Each test in ``UNCOMPARED`` exists, and its questions, even a planted wrong answer,
    are left to the public API alone."""
    for test in UNCOMPARED:
        module, name = test.split("::")
        assert f"def {name}(" in (TESTS.parent / module).read_text(), test
    with shadowed(query=lambda point, reference: Available(-1)) as shadow:
        shadow.begin(min(UNCOMPARED))
        point = _opened()
        assert point.query(House.total) != Available(-1)
    assert not shadow.compared and not shadow.disagreements
    assert shadow.left["a test making the running type wrong on purpose"] == 2


# -- engine modules under the plugin ----------------------------------------------------------

ENGINE_MODULES = (
    "core/space/test_design_graph.py",
    "core/space/test_nested_parameter_bindings.py",
    "core/space/test_validity.py",
)
"""Engine modules that ask ``explain``, the bound accessors and a design space's own
questions, run under the plugin (from ``tests``)."""


def _pytest(*arguments: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:xdist",
        "-p",
        "no:cacheprovider",
        "-p",
        "core.space._switch_shadow",
        "-q",
        *arguments,
    ]
    environment = {key: value for key, value in os.environ.items() if key != "PYTEST_ADDOPTS"}
    return subprocess.run(
        command,
        cwd=str(TESTS),
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


def _tail(ran: subprocess.CompletedProcess[str]) -> str:
    return "\n".join([*ran.stdout.splitlines()[-40:], *ran.stderr.splitlines()[-20:]])


def _counts(ran: subprocess.CompletedProcess[str]) -> dict[str, int]:
    lines = ran.stdout.splitlines()
    summary = lines[-1] if lines else ""
    return {kind.rstrip("s"): int(number) for number, kind in SUMMARY.findall(summary)}


def test_engine_modules_pass_under_the_plugin() -> None:
    ran = _pytest(*ENGINE_MODULES)
    tail = _tail(ran)
    counts = _counts(ran)
    assert ran.returncode == 0, tail
    assert counts.get("passed", 0) > 0 and set(counts) == {"passed"}, tail
    found = COUNT.search(ran.stdout)
    assert found is not None, tail
    compared, disagreements, breaks = int(found.group(1)), int(found.group(3)), int(found.group(4))
    kinds = dict(item.rsplit(" ", 1) for item in found.group(2).split(", "))
    assert compared > 0 and disagreements == 0 and breaks == 0, tail
    assert {"explain", "field", "state", "value", "viable", "change"} <= set(kinds), tail
