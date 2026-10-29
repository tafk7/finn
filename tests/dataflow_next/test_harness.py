# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The comparison harness checked on itself: its translation round-trips, its recorder
takes FINN's calls and only those, and a planted difference fails it."""

from __future__ import annotations

import pytest

import dataclasses
import subprocess
import sys
from dataflow_next.harness import __main__ as entry
from dataflow_next.harness import translate
from dataflow_next.harness.checks import Comparison, Priced
from dataflow_next.harness.record import Call, Raised, Recorder
from dataflow_next.harness.replay import (
    CHECKERS,
    COMPARATORS,
    FAILING,
    PRICERS,
    VERDICTS,
    judge,
    round_trip,
    table,
)
from pathlib import Path
from typing import Any

import finn.dataflow.steps as line_steps
import finn.dataflow_next.steps as next_steps
import finn.dataflow_next.traversal as next_traversal
from finn.dataflow.gemm import Form, k, m, n
from finn.dataflow.schedule import Schedule
from finn.dataflow.steps import steps
from finn.dataflow.traversal import BeatSequence, Repetition, Traversal, vector_major


def ask() -> None:
    """A few questions, as a kernel asks them."""
    schedule = Schedule({m: 2, n: 4, k: 4}, {n: 2, k: 2}, (m, n, k))
    port = schedule.present((2, 4), Form.DENSE.x, lanes=(k,))
    steps(
        BeatSequence(vector_major((2, 4), 2)), BeatSequence(port, markers=(schedule.closing((k,)),))
    )
    steps(BeatSequence(vector_major((2, 4), 4), Repetition.CYCLIC), BeatSequence(port))
    _ = port.lanes


def test_a_value_crosses_and_comes_back_equal() -> None:
    port = Schedule({m: 2, k: 4}, {k: 2}, (m, k)).present((2, 4), (m, k), lanes=(k,))
    crossed = translate.forward(BeatSequence(port, Repetition.CYCLIC))
    assert type(crossed).__module__ == "finn.dataflow_next.traversal"
    assert type(crossed.form).__module__ == "finn.dataflow_next.traversal"
    assert crossed.repetition is next_traversal.Repetition.CYCLIC
    assert translate.back(crossed) == BeatSequence(port, Repetition.CYCLIC)
    dense: Any = translate.counterpart(Form)
    assert translate.forward(Form.DENSE) is dense.DENSE
    assert translate.forward((port, [port], {"a": port})) == (
        translate.forward(port),
        [translate.forward(port)],
        {"a": translate.forward(port)},
    )


def test_a_class_the_other_package_lacks_is_refused() -> None:
    lost = type("Lost", (), {"__module__": "finn.dataflow.traversal"})
    with pytest.raises(translate.Untranslatable):
        translate.counterpart(lost)


def test_the_recorder_takes_only_finns_calls_and_restores_the_line() -> None:
    original = line_steps.steps
    with Recorder() as recorder:
        recorder.source = "here"
        ask()
    assert line_steps.steps is original
    labels = {call.target.label for call in recorder.recorded()}
    assert {"schedule.Schedule.present", "schedule.Schedule.closing", "steps.steps"} <= labels
    assert "traversal.Traversal.lanes" in labels
    # steps asks classify inside: the package's own working, not recorded.
    assert "traversal.classify" not in labels
    assert all(call.sources == {"here"} for call in recorder.recorded())


def test_the_revision_answers_as_the_line_does() -> None:
    with Recorder() as recorder:
        ask()
    calls = list(recorder.recorded())
    found = table("here", calls)
    assert found.passed, found.failures
    assert round_trip(calls) == []


def test_a_planted_difference_fails_the_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    """A check that has never rejected anything is a check nobody has tested."""
    with Recorder() as recorder:
        ask()
    monkeypatch.setattr(next_steps, "steps", lambda source, sink: next_steps.Steps())
    found = table("here", list(recorder.recorded()))
    assert not found.passed
    assert {failure.target.label for failure in found.failures} == {"steps.steps"}
    assert all(failure.verdict == "differs" for failure in found.failures)


def test_an_exception_is_an_answer() -> None:
    with Recorder() as recorder, pytest.raises(ValueError):
        Traversal((2,), (), ()).replayed(3, inner_beats=4)
    (call,) = [call for call in recorder.recorded() if call.target.name == "replayed"]
    assert table("here", [call]).passed


# -- the verdict classes ------------------------------------------------------------------


def _steps_call(outcome: Any) -> Call:
    """A recorded ``steps`` call with the line's outcome replaced by ``outcome``."""
    with Recorder() as recorder:
        ask()
    (call, *_) = [c for c in recorder.recorded() if c.target.label == "steps.steps"]
    return dataclasses.replace(call, outcome=outcome)


def _revision(monkeypatch: pytest.MonkeyPatch, answer: Any) -> None:
    """The revision's ``steps`` made to answer ``answer``, or to raise it if an exception."""

    def steps(*args: Any, **kwargs: Any) -> Any:
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(next_steps, "steps", steps)


def _refused() -> Raised:
    return Raised("builtins.ValueError", "no chain")


def _accepts(call: Call, refusal: Raised, answer: Any) -> bool:
    return bool(answer == (1, 2))


def _builds_at(call: Call, answer: Any) -> Priced | None:
    return Priced(len(answer), "two steps")


def _prices(call: Call, line: Any, revision: Any) -> Comparison | None:
    return Comparison("smaller", line[0] == revision[0], len(line), len(revision))


def _latency(call: Call, line: Any, revision: Any) -> Comparison | None:
    return Comparison("larger_latency_counted", True, 1, 1, latency="two beats")


def _silent_latency(call: Call, line: Any, revision: Any) -> Comparison | None:
    return Comparison("larger_latency_counted", True, 1, 1)


def _unrelated(call: Call, line: Any, revision: Any) -> Comparison | None:
    return None


def test_the_registries_hold_the_steps_and_closing_checks() -> None:
    assert set(CHECKERS) == {"steps.steps", "steps.presented", "schedule.Schedule.closing"}
    assert set(PRICERS) == {"steps.steps"}
    assert set(COMPARATORS) == {"steps.steps", "steps.presented"}


def test_equal_answers_are_same_with_or_without_a_comparator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(COMPARATORS, "steps.steps", _prices)
    _revision(monkeypatch, (1, 2))
    assert judge(_steps_call((1, 2))).verdict == "same"
    _revision(monkeypatch, ValueError("no chain"))
    assert judge(_steps_call(_refused())).verdict == "same"


def test_a_refusal_where_the_line_builds_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _revision(monkeypatch, ValueError("no chain"))
    monkeypatch.setitem(CHECKERS, "steps.steps", _accepts)
    monkeypatch.setitem(COMPARATORS, "steps.steps", _prices)
    assert judge(_steps_call((1, 2))).verdict == "refuses_where_today_builds"
    # A different message or class between two refusals is a difference, not a refusal.
    assert judge(_steps_call(Raised("builtins.ValueError", "other"))).verdict == "differs"


def test_a_build_where_the_line_refuses_passes_only_through_a_checker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = _steps_call(_refused())
    _revision(monkeypatch, (1, 2))
    monkeypatch.delitem(CHECKERS, "steps.steps")
    none = judge(call)
    assert (none.verdict, "no checker" in none.detail) == ("differs", True)
    monkeypatch.setitem(CHECKERS, "steps.steps", _accepts)
    assert judge(call).verdict == "builds_where_today_refuses"
    _revision(monkeypatch, (3, 4))
    rejected = judge(call)
    assert (rejected.verdict, "checker rejects" in rejected.detail) == ("differs", True)
    # A comparator does not stand in for a checker.
    monkeypatch.delitem(CHECKERS, "steps.steps")
    monkeypatch.setitem(COMPARATORS, "steps.steps", _prices)
    assert judge(call).verdict == "differs"


def test_a_build_where_the_line_refuses_carries_the_cost_a_pricer_gives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = _steps_call(_refused())
    _revision(monkeypatch, (1, 2))
    monkeypatch.setitem(CHECKERS, "steps.steps", _accepts)
    unpriced = judge(call)
    assert (unpriced.verdict, unpriced.line_cost, unpriced.next_cost) == (
        "builds_where_today_refuses",
        None,
        None,
    )
    monkeypatch.setitem(PRICERS, "steps.steps", _builds_at)
    built = judge(call)
    assert built.verdict == "builds_where_today_refuses"
    assert (built.line_cost, built.next_cost, built.note) == (None, 2, "two steps")
    found = table("here", [call])
    assert [(v.verdict, v.line_cost, v.next_cost) for v in found.priced] == [
        ("builds_where_today_refuses", None, 2)
    ]
    # A rejected answer is no build, and is not priced.
    _revision(monkeypatch, (3, 4))
    assert table("here", [call]).priced == ()


def test_smaller_passes_only_through_a_comparator_that_confirms_the_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = _steps_call((1, 2, 3))
    _revision(monkeypatch, (1, 2))
    assert judge(call).verdict == "differs"
    monkeypatch.setitem(COMPARATORS, "steps.steps", _prices)
    smaller = judge(call)
    assert (smaller.verdict, smaller.line_cost, smaller.next_cost) == ("smaller", 3, 2)
    _revision(monkeypatch, (9, 2))
    assert judge(call).verdict == "smaller_changes_positions"
    monkeypatch.setitem(COMPARATORS, "steps.steps", _unrelated)
    assert judge(call).verdict == "differs"


def test_a_larger_latency_passes_only_when_a_comparator_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = _steps_call((1, 2, 3))
    _revision(monkeypatch, (1, 2))
    assert judge(call).verdict == "differs"
    monkeypatch.setitem(COMPARATORS, "steps.steps", _latency)
    counted = judge(call)
    assert (counted.verdict, counted.latency) == ("larger_latency_counted", "two beats")
    monkeypatch.setitem(COMPARATORS, "steps.steps", _silent_latency)
    assert judge(call).verdict == "differs"


def test_an_untranslatable_call_fails() -> None:
    lost = type("Lost", (), {"__module__": "finn.dataflow.traversal"})
    call = dataclasses.replace(_steps_call((1, 2)), args=(lost(),), kwargs={})
    assert judge(call).verdict == "untranslatable"


def test_the_table_counts_every_class_and_a_failing_one_fails_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(CHECKERS, "steps.steps", _accepts)
    monkeypatch.setitem(COMPARATORS, "steps.steps", _prices)
    _revision(monkeypatch, (1, 2))
    calls = [_steps_call(outcome) for outcome in ((1, 2), _refused(), (1, 2, 3))]
    found = table("here", calls)
    assert list(found.totals) == list(VERDICTS)
    assert found.totals == dict.fromkeys(VERDICTS, 0) | {
        "same": 1,
        "builds_where_today_refuses": 1,
        "smaller": 1,
    }
    assert found.passed
    assert [(v.verdict, v.line_cost, v.next_cost) for v in found.priced] == [("smaller", 3, 2)]
    assert found.counts == (
        ("steps.steps", "builds_where_today_refuses", 1),
        ("steps.steps", "same", 1),
        ("steps.steps", "smaller", 1),
    )

    for failing in FAILING[:-1]:  # untranslatable is covered above
        outcome, answer = {
            "refuses_where_today_builds": ((1, 2), ValueError("no chain")),
            "smaller_changes_positions": ((9, 2, 3), (1, 2)),
            "differs": ((5,), (6,)),
        }[failing]
        _revision(monkeypatch, answer)
        monkeypatch.setitem(
            COMPARATORS, "steps.steps", _unrelated if failing == "differs" else _prices
        )
        found = table("here", [_steps_call(outcome)])
        assert not found.passed
        assert found.totals[failing] == 1
        assert [v.verdict for v in found.failures] == [failing]


# -- the header ---------------------------------------------------------------------------


def test_the_header_names_the_package_the_line_and_the_sources() -> None:
    import finn.dataflow  # noqa: PLC0415
    import finn.dataflow_next  # noqa: PLC0415

    line = entry.header(["chain-8", "geometries"])
    assert "\n" not in line
    package = Path(finn.dataflow_next.__file__).resolve().parent
    assert f"finn.dataflow_next {package};" in line
    assert f"finn.dataflow {Path(finn.dataflow.__file__).resolve().parent};" in line
    assert "sources chain-8, geometries;" in line
    assert line.endswith(f"git {entry.git_state(package)}")
    assert "tree" not in line.split("sources")[1]  # no label unless one is given


def test_the_header_names_the_tree_it_is_given_beside_the_git_state() -> None:
    plain = entry.header(["chain-8"])
    line = entry.header(["chain-8"], "9467164f3")
    assert line == plain.replace("; git ", "; tree 9467164f3; git ")
    assert "sources chain-8; tree 9467164f3; git " in line


def _git(directory: Path, *arguments: str) -> str:
    done = subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "-C",
            str(directory),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def test_the_header_reads_git_only_where_a_git_directory_is_found(tmp_path: Path) -> None:
    archive = tmp_path / "archive" / "package"
    archive.mkdir(parents=True)
    assert not any((each / ".git").exists() for each in (archive, *archive.parents))
    assert entry.git_state(archive) == "(no git metadata)"

    tree = tmp_path / "tree"
    package = tree / "src" / "package"
    package.mkdir(parents=True)
    _git(tree, "init", "-q", "-b", "somewhere")
    (package / "a.py").write_text("x = 1\n")
    _git(tree, "add", "src/package/a.py")
    _git(tree, "commit", "-q", "-m", "one")
    commit = _git(tree, "rev-parse", "--short=9", "HEAD")
    assert entry.git_state(package) == f"{commit} on somewhere, clean"
    (package / "a.py").write_text("x = 2\n")
    assert entry.git_state(package) == f"{commit} on somewhere, dirty, changed paths: 1"


def test_the_header_comes_before_any_source_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[str] = []
    real = subprocess.run

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "git":  # the header's own reading of the tree
            return real(command, **kwargs)
        seen.append(capsys.readouterr().out)  # what has been printed when a source starts
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["harness", "chain-8", "geometries"])
    assert entry.main() == 0
    assert len(seen) == 2
    first, second = seen
    assert first == entry.header(["chain-8", "geometries"]) + "\n"
    assert second == ""  # printed once, not for each source


def test_the_tree_argument_reaches_the_header(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = subprocess.run

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "git":
            return real(command, **kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["harness", "chain-8", "--tree", "9467164f3"])
    assert entry.main() == 0
    assert capsys.readouterr().out == entry.header(["chain-8"], "9467164f3") + "\n"
