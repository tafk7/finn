# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The shadow run (``core.space._running_shadow``): today's public API answering every
question both ways, held to its own contract.

- **A planted disagreement is caught and named**: a running type's answer made wrong (or
  raising) is a disagreement naming the question, the member asked, the configuration and
  both answers, while the caller still gets today's answer; run as a plugin, the test it
  was met in fails and the session's count names it.
- **Transparent where both agree**: every question asked with the shadow on answers as it
  does with it off, a raise included, and nothing disagrees.
- **The count is right**: each question asked is counted once by its kind; a question a
  method asks of its receiver is left to today's engine, and so is one asked from another
  thread.
- **What the running type is asked**: a change's items as the caller gave them, snapshotted
  before today's run may change them; a clear in a bind, refused before any question, is
  not compared; nor is a test named in ``UNCOMPARED``.
- **Engine modules under it**: three of ``tests/core/space``'s modules that ask ``explain``
  and the bound accessors, run with the plugin, pass with 0 disagreements.
"""

from __future__ import annotations

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
from typing import cast

import pytest

from core.space._running_shadow import UNCOMPARED, Shadow, form
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
)
from finn.core.space.domains import Domain

TESTS = Path(__file__).parent.parent.parent

COUNT = re.compile(r"running shadow: (\d+) questions compared \((.*)\); disagreements: (\d+)")
SUMMARY = re.compile(r"(\d+) (passed|failed|errors?|xfailed|xpassed|skipped|deselected)")


@contextmanager
def shadowed(**running: Callable[..., object]) -> Iterator[Shadow]:
    """A shadow applied for the block (its running type's answers replaced by kind where
    given), then restored."""
    shadow = Shadow(running=running)
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
    """The running type's answer to ``query`` made wrong: the caller gets today's answer, and
    the disagreement names the question, the member, the configuration and both answers."""
    expected = _opened().query(House.total)
    with shadowed(query=lambda session, at, point, arguments: Available(-1)) as shadow:
        point = _opened()
        assert point.query(House.total) == expected
        assert point.budget == 100
    assert shadow.compared == Counter({"design_space": 1, "query": 1, "value": 1})
    [found] = shadow.disagreements
    assert shadow.take_pending() == [found] and shadow.take_pending() == []
    assert (found.question, found.asked) == ("query", "total")
    assert found.configuration.startswith("House {")
    differences = "\n".join(found.differences)
    assert 'kind: "unresolved" | "value"' in differences, differences
    assert 'value: {"missing": true} | -1' in differences, differences
    text = str(found)
    assert text.startswith("query total of House {}: today's engine and the running type")
    assert all(line in text for line in found.differences)
    assert "disagreements: 1" in shadow.count()


def test_a_raise_on_one_side_is_a_disagreement() -> None:
    """The running type raising where today's answers, and answering where today's raises:
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
    the tests beside it pass (the purity check on, unless a test turns it off), and the
    summary counts the tests' questions."""
    module = tmp_path / "test_planted.py"
    module.write_text(
        textwrap.dedent(
            """
            from support.purity import purity_unchecked

            from finn.core.space import Decision, Param, Space, _world, design_space
            from finn.core.space._running import surface


            class Example(Space):
                width: int = Param()
                pe: int = Decision(values=(1, 2))


            def test_agreeing():
                assert design_space(Example(width=4)).width == 4


            def test_planted(monkeypatch):
                monkeypatch.setattr(surface, "value", lambda at, reference, session: 5)
                assert design_space(Example(width=4)).width == 4


            def test_the_purity_check_is_on():
                assert _world._PURITY


            def test_a_test_may_turn_it_off(purity_unchecked):
                assert not _world._PURITY
            """
        )
    )
    ran = _pytest(str(module))
    tail = _tail(ran)
    assert ran.returncode == 1, tail
    assert _counts(ran) == {"failed": 1, "passed": 3}, tail
    assert "FAILED " in ran.stdout and "test_planted.py::test_planted" in ran.stdout, tail
    assert "ShadowDisagreement: value width of Example {}" in ran.stdout, tail
    assert "running shadow: 4 questions compared (design_space 2, value 2); disagreements: 1" in (
        ran.stdout
    ), tail


# -- transparent where both agree ---------------------------------------------------------


def test_the_shadow_is_transparent_where_both_agree() -> None:
    """Every question answers with the shadow on as with it off (a raise included), and the
    configurations it returns are today's own."""
    without = _answers(_opened())
    with shadowed() as shadow:
        point = _opened()
        within = _answers(point)
        unchanged = point.try_with_choices({House.want_garage: False})
        assert unchanged.accepted
        assert point.try_with_choices().instance is point
    assert within == without
    assert cast(tuple[object, ...], within["selected"])[0] == "raises"  # the selector is open
    assert not shadow.disagreements, "\n".join(map(str, shadow.disagreements))


# -- the count ------------------------------------------------------------------------------


class Doubled(Space):
    width: int = Param()
    pe: int = Decision(values=(1, 2))

    @derived
    def doubled(self) -> int:
        return 2 * self.width  # a receiver's read, left to today's engine


def test_the_count_is_right() -> None:
    """Each question asked once is counted once, by its kind; the house's questions once
    each, and a design space's opening; a receiver's reads and another thread's questions
    are left to today's engine, by reason."""
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
        f"running shadow: {sum(expected.values())} questions compared ("
        + ", ".join(f"{kind} {number}" for kind, number in sorted(expected.items()))
        + "); disagreements: 0"
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


# -- what the running type is asked -----------------------------------------------------------


def test_a_planted_mutation_of_the_callers_input_is_not_a_disagreement() -> None:
    """A membership callback appends to the list the caller set another choice to, while
    today's change runs: today's had snapshotted it first, and the running type is asked
    the snapshot taken before today's run, so the two agree."""
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
    """Today's refuses a clear in a bind as it reads the bind's items: refused before any
    question, left to today's engine, and nothing disagrees."""
    with shadowed() as shadow:
        point = design_space(Doubled(width=3)).with_choices(pe=2)
        with pytest.raises(RequestError, match="clears none"):
            bind(point, point.field(Doubled.pe).clear())
    assert shadow.compared["bind"] == 0
    assert shadow.left["refused before any question"] == 1
    assert not shadow.disagreements, "\n".join(map(str, shadow.disagreements))


def test_a_test_making_the_running_type_wrong_on_purpose_is_not_compared() -> None:
    """Each test in ``UNCOMPARED`` exists, and its questions, even a planted wrong answer,
    are left to today's engine."""
    for test in UNCOMPARED:
        module, name = test.split("::")
        assert f"def {name}(" in (TESTS.parent / module).read_text(), test
    with shadowed(query=lambda session, at, point, arguments: Available(-1)) as shadow:
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
"""Engine modules that ask ``explain``, the bound accessors and a configuration's own
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
        "core.space._running_shadow",
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
    compared, disagreements = int(found.group(1)), int(found.group(3))
    kinds = dict(item.rsplit(" ", 1) for item in found.group(2).split(", "))
    assert compared > 0 and disagreements == 0, tail
    assert {"explain", "field", "state", "value", "viable", "change"} <= set(kinds), tail
