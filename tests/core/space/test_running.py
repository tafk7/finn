# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The running type (``finn.core.space._running``) against today's engine and the reference
model, through the class-based oracle (``support.space_oracle``), as the implementation
``core.space._running_type:RUNNING``.

- **The toys' walks**: each of FINN's toys walked from seeds 0 and 1, 20 steps, the running
  type against today's engine, with no disagreement.
- **The reference walks**: seeds 0-23, 40 steps, strict, with and without selectors picked
  by a derived value, the running type against the reference model, every invariant and
  every permanent check holding.
- **The reference harness's hand cases** under the running type, in a process of their own,
  the harness's default implementation named by its option (``--space-implementation``);
  the option itself, and the matrix's expected failures by the milestones an implementation
  has reached.
- **Hand cases**: a ``when`` read past its guard through a formal of its scope; a chosen
  selector under a refused ``when``, one piece; a refused ``when`` below a guard that waits,
  unknown until all above it holds, and two nested ones, one piece (each as today's
  engine answers it, and the model); an answer refused through its scope turning
  inapplicable; a reader beside a formal's scope; a formal read from outside its scope;
  models dropped and their ids reused, the node cache's tables with them; a line deeper
  than the recursion limit allows without the depth guard; a membership reading that reads
  a decision, for a guarded choice and for a choice with no guard; a choice in a case of a
  selector picked by a choice, whose membership reads its case's guard as holding, and one
  outside the case, whose reading reaches the choice; a selector given as a binding's
  argument, reading its case's name (a ``None`` case's too).
- **The node cache** (``_running.cache``): the toys walked with a cache dropping half its
  entries at random, with no disagreement; the purity check over it (each hit evaluated
  anyway, and each node the depth guard abandoned evaluated again, and compared): a planted
  impure function caught with it on and not with it off, and the toys' and the reference
  walks with it on, with no disagreement and no impure function; a cache wrong on purpose
  (a read left out of each key) caught; each type the cache keys by ``==`` checked against
  its value semantics, the rule keying a value class field by field checked on sample
  classes, and nothing else keyed: no value under semantics of its own (two design spaces
  differing only in two values ``==`` holds equal and such semantics tell apart answer as
  today's by that check alone), nor a value class with a field annotated so.

The implementations are loaded by name, as ``--implementation`` names them
(``core.space._running_type``, a private module of these tests).
"""

from __future__ import annotations

import gc
import itertools
import math
import os
import re
import subprocess
import sys
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Annotated, Any, cast

import pytest
from support.space_oracle import Answers, Implementation, compare, forms, load, toys
from support.space_oracle.today import TODAY

from core.space._running_shapes import (
    IN_AN_OPEN_CASE,
    ByArgument,
    Capped,
    PickedByAChoice,
    capped,
    even,
)
from core.space._running_type import LeavesOneOut, Running
from core.space._toys_support import House
from core.space.reference import (
    Change,
    Choice,
    Constraint,
    Derived,
    Refuse,
    Scope,
    Selector,
    When,
    listed,
)
from core.space.reference import Space as Description
from core.space.reference_harness import MILESTONES, ROWS, Case, Spelling, expected, reached
from core.space.reference_harness import implementation as harness_implementation
from core.space.reference_harness import walks as reference_walks
from core.space.reference_harness.engine import definition
from core.space.reference_harness.forms import ValidityForm
from core.space.reference_harness.implementation import OUTSIDE, Option
from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    Inapplicable,
    Param,
    Rejected,
    Select,
    Space,
    Unresolved,
    ValueSemantics,
    _compose,
    bind,
    constraint,
    derived,
    design_space,
    divisors_of,
    inspection,
    reject,
    view,
)
from finn.core.space._failure import NativeEvaluationError
from finn.core.space._running import cache as cache_module
from finn.core.space._running import evaluate, questions, structure
from finn.core.space._running.cache import CHECKED, PLAIN, NodeCache, plain, value_class
from finn.core.space._running.cache import key as cache_key
from finn.core.space._running.evaluate import MembershipReadsChoice, Session
from finn.core.space._running.value import DesignSpace, FactsLayer
from finn.core.space._structure import Structure
from finn.core.space._world import _same_value, check_purity
from finn.core.space.ir import Node
from finn.core.space.occurrence import state
from finn.core.space.semantics import default_semantics

RUNNING: Implementation = load("core.space._running_type:RUNNING")
RUNNING_DROPPING: Implementation = load("core.space._running_type:RUNNING_DROPPING")
RUNNING_PLANTED: Implementation = load("core.space._running_type:RUNNING_PLANTED")
RUNNING_VERIFIED: Implementation = load("core.space._running_type:RUNNING_VERIFIED")
RUNNING_CACHED: Implementation = load("core.space._running_type:RUNNING_CACHED")

TOYS = toys.TOYS


def fresh() -> Implementation:
    """The running type with a session of its own (its counters this test's alone)."""
    made: Implementation = type(RUNNING)()
    return made


def counters(implementation: Implementation) -> evaluate.Counters:
    return cast(Session, cast(Any, implementation).session).counters


def cached() -> Running:
    """The running type with a node cache and a session of their own."""
    return Running(NodeCache(), "running, cached")


# -- the walks ------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("toy", sorted(TOYS))
def test_the_toys_walk_as_today(toy: str, seed: int) -> None:
    found = compare.walk(TOYS[toy].make, TOYS[toy].facts, seed, 20, TODAY, RUNNING, forms.ALL)
    assert found.lines() == []
    assert found.tally["changes"] > 0 and found.tally["cold builds"] > 0
    asked = found.tally["changes"] + found.tally["binds"]
    assert found.tally["reports"] + found.tally["requests raising alike"] == asked


@pytest.mark.parametrize("by", [False, True], ids=["chosen", "by"])
@pytest.mark.parametrize("seed", range(24))
def test_the_reference_walks_hold_strictly(seed: int, by: bool) -> None:
    tally = reference_walks.walk(seed, 40, strict=True, implementation=RUNNING, by=by)
    assert tally.failure is None, tally.failure
    assert tally.implementation == "running"
    assert tally.steps == 40 and not tally.by_milestone and tally.engine_errors == 0
    assert tally.engine_checks["cold"] == tally.engine_checks["bind"] == tally.compared
    assert tally.engine_checks["cold change"] == tally.steps


# -- the reference harness's hand cases under the running type ---------------------------------
#
# ``--space-implementation`` makes the running type the harness's default implementation. The
# gate's pytest names none, so the hand cases run here, in a process of their own.

TESTS = Path(__file__).parent.parent.parent
HAND_CASES = tuple(
    f"core/space/{name}.py"
    for name in (
        "test_reference_cases",
        "test_reference_harness",
        "test_reference_misuse",
        "test_reference_model",
        "test_reference_rules",
        "test_reference_scopes",
        "test_reference_selectors",
    )
)
"""The modules of hand cases run under the running type, each named (from ``tests``)."""

_HARNESS = "tests/core/space/test_reference_harness.py"
TODAYS_SELF_TESTS = (
    *(
        f"{_HARNESS}::test_a_planted_fault_is_reported_by_its_question_in_a_hand_case[{case}]"
        for case in (
            "possible-the possible values of a",
            "standing-the change commit {'b': 2}",
            "standing-the possible values of b",
            "validity-validity",
            "value-the answer of double",
        )
    ),
    *(
        f"{_HARNESS}::test_a_planted_fault_is_reported_by_question_in_walks_0_to_3[{question}]"
        for question in ("possible", "standing", "validity", "value")
    ),
    f"{_HARNESS}::test_a_planted_value_on_a_stored_choice_is_seen",
    *(
        f"{_HARNESS}::test_history_is_reported_by_the_cold_build_in_walks_0_to_3[{seed}]"
        for seed in range(4)
    ),
    f"{_HARNESS}::test_the_check_against_a_cold_build_runs_for_every_implementation",
)
"""The harness's own tests that name today's engine (or a fault planted around it) on
purpose, deselected by node id (from the worktree's root, as pytest matches
``--deselect``)."""

_COUNTS = re.compile(r"(\d+) (passed|failed|errors?|xfailed|xpassed|skipped|deselected)")


def test_the_reference_hand_cases_pass_under_the_running_type() -> None:
    """Every hand case of the reference harness's seven modules of them, run with
    ``--space-implementation=core.space._running_type:RUNNING``: none fails or passes
    unexpectedly or is skipped, today's engine is never opened or made, exactly the harness's
    tests naming today's engine are deselected, and the expected failures are exactly the
    witnesses of the matrix's rows the running type has not reached (``matrix.ROWS``,
    ``matrix.reached``).

    The option is written ``--space-implementation=MODULE:ATTRIBUTE``, with ``=``, before
    the test paths: written apart from its value, pytest takes the value for a path.

    ``test_reference_walks`` is not run here: it pins today's engine by design (its walks'
    own default, and its tests' "today"), and runs under today's in the gate as ever; the
    running type's walks are this module's own."""
    stems = {PurePosixPath(module).stem for module in HAND_CASES}
    unimplemented = sorted(
        row.witness
        for row in ROWS
        if row.milestone in MILESTONES[1:]
        and row.milestone not in reached(RUNNING)
        and row.witness.partition("::")[0] in stems
    )
    assert unimplemented
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:xdist",
        "-q",
        f"--space-implementation={RUNNING_SPEC}",
        *(f"--deselect={node}" for node in TODAYS_SELF_TESTS),
        *HAND_CASES,
    ]
    environment = {key: value for key, value in os.environ.items() if key != "PYTEST_ADDOPTS"}
    ran = subprocess.run(
        command,
        cwd=str(TESTS),
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    lines = ran.stdout.splitlines()
    tail = "\n".join([*lines[-40:], *ran.stderr.splitlines()[-20:]])
    collected = re.search(r"collected (\d+) items", ran.stdout)
    summary = lines[-1] if lines else ""
    counts = {kind.rstrip("s"): int(number) for number, kind in _COUNTS.findall(summary)}
    assert ran.returncode == 0 and collected is not None, tail
    assert counts.get("failed", 0) == 0 and counts.get("error", 0) == 0, tail
    assert counts.get("xpassed", 0) == 0 and counts.get("skipped", 0) == 0, tail
    deselected = sorted(
        line.removeprefix("deselected: ") for line in lines if line.startswith("deselected: ")
    )
    assert deselected == sorted(TODAYS_SELF_TESTS), tail
    assert counts.get("deselected", 0) == len(TODAYS_SELF_TESTS), tail
    assert counts.get("xfailed", 0) == len(unimplemented), tail
    expected_passed = int(collected.group(1)) - len(unimplemented) - len(TODAYS_SELF_TESTS)
    assert counts.get("passed", 0) == expected_passed, tail
    assert "today's engine opened or made: 0" in lines, tail
    assert f"space implementation: running ({RUNNING_SPEC})" in ran.stdout, tail


RUNNING_SPEC = "core.space._running_type:RUNNING"
"""The running type, as an implementation is named (``support.space_oracle.load``)."""

TOY = Case("a toy", Description(("wide",), Scope("", choices=(Choice("x", listed(1, 2)),))))


@dataclass(frozen=True)
class _Recording:
    """The running type, recording what it opened and made."""

    name: str = "recording"
    calls: list[str] = field(default_factory=list)

    def open(self, configuration: Space) -> Answers:
        self.calls.append("open")
        return RUNNING.open(configuration)

    def make(
        self, space_class: type[Space], facts: Mapping[str, object], stored: Mapping[str, object]
    ) -> Answers:
        self.calls.append("make")
        return RUNNING.make(space_class, facts, stored)


@dataclass(frozen=True)
class _Reaching(_Recording):
    """An implementation declaring it has reached T4b."""

    milestones: frozenset[str] = frozenset({"T4b"})


@contextmanager
def _default(implementation: Implementation) -> Iterator[None]:
    previous = harness_implementation.use(implementation)
    try:
        yield
    finally:
        harness_implementation.use(previous)


def test_the_harness_asks_its_default_wherever_a_test_names_none() -> None:
    """``Engine.open``, ``Engine.make``, ``definition`` and ``Case.start`` ask the default
    implementation where they are given none, and the one named where one is."""
    assert harness_implementation.default() is TODAY
    recording = _Recording()
    with _default(recording):
        assert TOY.engine.open({"wide": 1}).implementation == "recording"
        assert TOY.engine.make({"wide": 1}, {"x": 2}).implementation == "recording"
        assert TOY.start({"wide": 1}, {"x": 1}).engine.implementation == "recording"
        assert definition(TOY.description, facts={"wide": 1}).error is None
        assert recording.calls == ["open", "make", "open", "open"]
        assert TOY.engine.open({"wide": 1}, TODAY).implementation == "today"
    assert harness_implementation.default() is TODAY
    assert TOY.start({"wide": 1}).engine.implementation == "today"


@pytest.mark.parametrize(
    "spec",
    ["core.space._running_type:NOWHERE", "core.space._nowhere:RUNNING", "running"],
    ids=["missing-attribute", "missing-module", "not-registered"],
)
def test_the_option_refuses_a_name_that_names_nothing(spec: str) -> None:
    """A name that loads nothing is a usage error, and the default stays today's engine."""
    with pytest.raises(pytest.UsageError, match="--space-implementation "):
        Option.start(spec)
    assert harness_implementation.default() is TODAY


def test_the_option_makes_its_implementation_the_default_and_counts_todays() -> None:
    """Started, the running type is the default and each ``open`` and ``make`` of today's
    engine is counted by the test it ran in; stopped, both are as before. Named today's, it
    counts nothing and adds nothing to the summary."""
    methods = (type(TODAY).open, type(TODAY).make)
    option = Option.start(RUNNING_SPEC)
    try:
        assert harness_implementation.default() is RUNNING and option.opens is not None
        assert option.opens.count() == "today's engine opened or made: 0" and not option.failed()
        with option.running("here"):
            TODAY.make(Ordered, {}, {"a": 2, "b": 1})
            TODAY.open(design_space(Ordered()))
        TODAY.open(design_space(Ordered()))
        assert option.opens.opened == {"here": 2, OUTSIDE: 1} and option.failed()
        option.deselected.append("a::test")
        assert option.summary() == [
            f"space implementation: running ({RUNNING_SPEC}), the harness's default "
            "(--space-implementation)",
            "deselected: a::test",
            f"today's engine opened or made: 3 ({OUTSIDE}: 1; here: 2)",
        ]
    finally:
        option.stop()
    assert harness_implementation.default() is TODAY
    assert (type(TODAY).open, type(TODAY).make) == methods
    today = Option.start("today")
    try:
        assert today.opens is None and today.summary() == [] and not today.failed()
        assert (type(TODAY).open, type(TODAY).make) == methods
    finally:
        today.stop()
    assert harness_implementation.default() is TODAY


def test_a_witness_is_expected_to_fail_for_a_default_short_of_its_milestone() -> None:
    """A row's witness is a strict expected failure for a default that has not reached the
    row's milestone (today's engine, the running type: neither declares any), and a plain
    test for one that has."""

    def witness() -> None: ...

    def marks(test: Callable[[], None]) -> list[str]:
        return [mark.name for mark in getattr(test, "pytestmark", [])]

    assert reached(TODAY) == reached(RUNNING) == frozenset()
    assert reached(_Reaching()) == {"T4b"}
    with pytest.raises(ValueError, match="no later milestones"):
        reached(_Reaching(milestones=frozenset({"T9"})))

    def other() -> None: ...

    assert marks(expected("view-read-in")(witness)) == ["xfail"]
    with _default(_Reaching()):
        assert marks(expected("view-read-in")(witness)) == ["xfail"]
        assert marks(expected("view-read-in")(other)) == []
        assert marks(expected("optional-fact-none")(other)) == ["xfail"]


# -- a ``when`` read past its guard, through a formal of its scope ----------------------------
#
# A scope ``g`` with ``when f0 >= 2`` inside case ``a`` of the open selector ``s``: its
# condition, written in ``s.a``, reads the fact through ``s.a``'s formal, bound under the
# case's guard. Read past that guard, the formal is read at its source (its reader, ``g``'s
# guard, is in ``s.a``): with ``f0 = 1`` the condition is false and ``g`` is absent, the
# open selector notwithstanding. Read as a member of ``s.a``, it would wait on ``s``.

WHEN_THROUGH_FORMAL = Case(
    "a when read past its guard through a formal",
    Description(
        ("f0",),
        Scope(
            "",
            selectors=(
                Selector(
                    "s",
                    (
                        Scope(
                            "a",
                            children=(
                                Scope(
                                    "g",
                                    when=When(("f0",), lambda read: read["f0"] >= 2),
                                    choices=(Choice("c1", listed(1, 2)),),
                                    constraints=(
                                        Constraint(
                                            "r", ("s.a.g.c1",), lambda r: r["s.a.g.c1"] < 2, "big"
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        Scope("b", choices=(Choice("c3", listed(0, 1)),)),
                    ),
                ),
            ),
        ),
    ),
)


def test_a_when_read_past_its_guard_reads_its_scopes_formal_at_its_source() -> None:
    pair = WHEN_THROUGH_FORMAL.start({"f0": 1}, implementation=RUNNING)
    assert dict(pair.engine.answers.states())["s.a.g.c1"] == inspection.Absent()
    pair.check_state("s.a.g.c1")
    pair.check_validity()
    pair.check_possible("s")
    configuration = design_space(WHEN_THROUGH_FORMAL.engine.root(f0=1))  # type: ignore[call-arg]
    assert compare.everything(TODAY.open(configuration), RUNNING.open(configuration)).lines() == []


# -- a chosen selector under a refused ``when``: one piece ------------------------------------
#
# A chosen selector whose ``when`` reads a refused derived value: the selector's piece and
# each case's would each count the refused guard. It counts once, at the highest refused
# guard of the chain, in the selector's name, as today's engine counts it. Setting the other
# case under the refused guard is then possible: it adds no pair. The reference harness
# spells no selector with a ``when`` of its own; there, the selector sits in a scope the
# ``when`` places, and the model's one piece is the running type's.


class Boiler(Space):
    kw: int = Param()
    stages: int = Decision(values=(1, 2, 3))


class HeatPump(Space):
    kw: int = Param()
    cop: int = Decision(values=(2, 3, 4))


class Guarded(Space):
    gate: int = Decision(values=(0, 1))

    @derived
    def heated(self) -> bool | Rejected:
        return reject("gate-refused", "gate 1 refuses") if self.gate == 1 else True

    heating = Select(boiler=Boiler, heat_pump=HeatPump, kw=8, when=heated)


ONE_PIECE = ("heating", [("gate-refused", "heated"), ("guard-refused", "heating")])


def _pieces(validity: inspection.Validity) -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        (piece.owner, sorted((finding.code, finding.owner) for finding in piece.result.findings))
        for piece in validity.pieces
    ]


@pytest.mark.parametrize("case", [None, "boiler", "heat_pump"])
def test_a_chosen_selector_under_a_refused_when_is_one_piece(case: str | None) -> None:
    stored: dict[object, object] = {Guarded.gate: 1}
    if case is not None:
        stored[Guarded.heating] = case
    point = bind(design_space(Guarded()), stored).instance
    found = RUNNING.open(point).validity()
    assert found.status == "refused" and _pieces(found) == [ONE_PIECE]
    if case is not None:
        (viable,) = RUNNING.open(point).viable(["heating"])
        assert all(isinstance(item, inspection.Possible) for item in viable.values or ())
    assert compare.everything(TODAY.open(point), RUNNING.open(point)).lines() == []


def test_a_selector_in_a_scope_under_a_refused_when_is_its_scopes_one_piece_as_the_model_says() -> (
    None
):
    cases = (
        Scope("boiler", choices=(Choice("stages", listed(1, 2)),)),
        Scope("pump", choices=(Choice("cop", listed(2, 3)),)),
    )
    shape = Case(
        "a selector under a refused when",
        Description(
            ("gate",),
            Scope(
                "",
                derived=(
                    Derived(
                        "heated",
                        ("gate",),
                        lambda r: Refuse("gate-refused") if r["gate"] == 1 else 1,
                    ),
                ),
                children=(
                    Scope(
                        "h",
                        when=When(("heated",), lambda r: r["heated"] == 1),
                        selectors=(Selector("heating", cases),),
                    ),
                ),
            ),
        ),
    )
    pair = shape.start({"gate": 1}, implementation=RUNNING)
    pair.check_validity()
    for stored in ({"h.heating": "boiler"}, {"h.heating": "pump", "h.heating.pump.cop": 3}):
        bound = pair.check_bind(stored)
        bound.check_validity()
        bound.check_possible("h.heating")


# -- a refused ``when`` below a guard that waits: unknown, until all above it holds ------------
#
# ``g``'s ``when`` reads ``d``, refused for an odd ``f``. Below a guard that waits (``g`` in
# case ``a`` of the open selector ``s``; or in ``p``, whose ``when`` reads the open choice
# ``c``), ``g`` waits with it: its piece is unknown, waiting on what that guard waits on, and
# validity is open. The refusal takes effect once all above it holds: ``s`` bound to ``a``
# (or ``c`` to 1), ``g``'s guard is one false piece, and setting it is refused. Today's
# engine answers the same, as the model does. The same with the refused ``when`` a
# selector's own (``plant``, written as ``heating``'s guard), in a case of an open
# selector: there the piece is the selector's.


def _heated(read: Mapping[str, Any]) -> object:
    return Refuse("gate-refused") if read["gate"] == 1 else True


UNDER_A_WAITING_PARENT = Case(
    "a refused when under a parent whose when waits",
    Description(
        ("f",),
        Scope(
            "",
            choices=(Choice("c", listed(0, 1)),),
            derived=(Derived("d", ("f",), even),),
            children=(
                Scope(
                    "p",
                    when=When(("c",), lambda read: read["c"] == 1),
                    choices=(Choice("x", listed(1, 2)),),
                    children=(
                        Scope(
                            "g",
                            when=When(("d",), lambda read: read["d"] >= 1),
                            choices=(Choice("q", listed(1, 2)),),
                        ),
                    ),
                ),
            ),
        ),
    ),
)

GUARDED_SELECTOR_IN_AN_OPEN_CASE = Case(
    "a selector's own refused when in a case of an open selector",
    Description(
        ("gate",),
        Scope(
            "",
            derived=(Derived("heated", ("gate",), _heated),),
            selectors=(
                Selector(
                    "s",
                    (
                        Scope(
                            "a",
                            children=(
                                Scope(
                                    "plant",
                                    when=When(("heated",), lambda read: bool(read["heated"])),
                                    selectors=(
                                        Selector(
                                            "heating",
                                            (
                                                Scope(
                                                    "boiler",
                                                    choices=(Choice("stages", listed(1, 2)),),
                                                ),
                                                Scope(
                                                    "pump", choices=(Choice("cop", listed(2, 3)),)
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        Scope("b", choices=(Choice("r", listed(1, 2)),)),
                    ),
                ),
            ),
        ),
    ),
    Spelling(types={"heated": bool}, guarded_selectors=frozenset({"s.a.plant"})),
)

WAITING = (
    (IN_AN_OPEN_CASE, {"f": 1}, "s", {"s": "a"}),
    (UNDER_A_WAITING_PARENT, {"f": 1}, "c", {"c": 1}),
    (GUARDED_SELECTOR_IN_AN_OPEN_CASE, {"gate": 1}, "s", {"s": "a"}),
)


@pytest.mark.parametrize("cached_", [False, True], ids=["running", "cached"])
@pytest.mark.parametrize("number", range(len(WAITING)))
def test_a_refused_when_below_a_guard_that_waits_is_unknown_until_all_above_it_holds(
    number: int, cached_: bool
) -> None:
    case, facts, waits_on, holding = WAITING[number]
    implementation = cached() if cached_ else RUNNING
    pair = case.start(facts, implementation=implementation)
    pair.check_validity()
    assert pair.engine.validity() == ValidityForm("open", blockers=frozenset({waits_on}))
    today = case.start(facts)  # today's engine, against the model too
    today.check_validity()
    assert today.engine.answers.validity() == pair.engine.answers.validity()
    pair.check_possible(waits_on)
    assert pair.check_change(Change(commit=holding)) is None  # refused on both sides
    bound = pair.check_bind(holding)
    bound.check_validity()
    assert bound.engine.validity().verdict == "refused"
    # an even fact: nothing refused, valid on both sides, the wait notwithstanding
    other = case.start({name: 2 for name in facts}, implementation=implementation)
    other.check_validity()
    assert other.engine.validity() == ValidityForm("valid")


# Two refused ``when``s, one inside the other, in case ``a`` of the open selector ``s``:
# ``outer``'s reads ``d``, ``inner``'s reads ``e``, both refused for an odd ``f``. They count
# once, at the highest refused ``when`` of the chain: one piece, unknown, waiting on ``s``,
# keyed by ``outer``. With ``s`` bound to ``a``, that piece is refused, still one. Today's
# engine answers the same.


def _odd(read: Mapping[str, Any]) -> object:
    return Refuse("e-odd") if read["f"] % 2 else read["f"]


NESTED_IN_AN_OPEN_CASE = Case(
    "two nested refused whens in a case of an open selector",
    Description(
        ("f",),
        Scope(
            "",
            derived=(Derived("d", ("f",), even), Derived("e", ("f",), _odd)),
            selectors=(
                Selector(
                    "s",
                    (
                        Scope(
                            "a",
                            choices=(Choice("p", listed(1, 2)),),
                            children=(
                                Scope(
                                    "outer",
                                    when=When(("d",), lambda read: read["d"] >= 1),
                                    choices=(Choice("x", listed(1, 2)),),
                                    children=(
                                        Scope(
                                            "inner",
                                            when=When(("e",), lambda read: read["e"] >= 1),
                                            choices=(Choice("q", listed(1, 2)),),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        Scope("b", choices=(Choice("r", listed(1, 2)),)),
                    ),
                ),
            ),
        ),
    ),
)


@pytest.mark.parametrize("cached_", [False, True], ids=["running", "cached"])
def test_two_nested_refused_whens_in_a_case_of_an_open_selector_are_one_unknown_piece(
    cached_: bool,
) -> None:
    implementation = cached() if cached_ else RUNNING
    pair = NESTED_IN_AN_OPEN_CASE.start({"f": 1}, implementation=implementation)
    pair.check_validity()
    assert pair.engine.validity() == ValidityForm("open", blockers=frozenset({"s"}))
    found = pair.engine.answers.validity()
    assert [(piece.owner, type(piece.result)) for piece in found.pieces] == [
        ("s.a.outer", Unresolved)
    ]
    assert {finding.owner for finding in found.pieces[0].result.findings} == {"s"}
    bound = pair.check_bind({"s": "a"})
    bound.check_validity()
    refused = bound.engine.answers.validity()
    assert refused.status == "refused"
    assert [piece.owner for piece in refused.pieces] == ["s.a.outer"]
    today = NESTED_IN_AN_OPEN_CASE.start({"f": 1})
    today.check_validity()
    assert today.engine.answers.validity() == found
    assert today.check_bind({"s": "a"}).engine.answers.validity() == refused


# -- an answer refused through its scope may turn inapplicable ------------------------------
#
# A scope ``h`` placed under a ``when`` refused through an input holds the open selector
# ``s``. A value in case ``b`` answers refused through its scope's presence; setting ``s`` to
# ``a`` makes ``b`` absent, and the value inapplicable: a move up, and the change is
# accepted.

REFUSED_THROUGH_SCOPE = Case(
    "an answer refused through its scope turns inapplicable",
    Description(
        ("gate", "n"),
        Scope(
            "",
            derived=(
                Derived(
                    "heated", ("gate",), lambda r: Refuse("gate-refused") if r["gate"] == 1 else 1
                ),
            ),
            children=(
                Scope(
                    "h",
                    when=When(("heated",), lambda r: r["heated"] == 1),
                    selectors=(
                        Selector(
                            "s",
                            (
                                Scope("a", choices=(Choice("p", listed(1, 2)),)),
                                Scope("b", derived=(Derived("d", ("n",), lambda r: r["n"] + 1),)),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)


@pytest.mark.parametrize("name", ["today", "running"])
def test_an_answer_refused_through_its_scope_turns_inapplicable_and_the_change_is_accepted(
    name: str,
) -> None:
    implementation = TODAY if name == "today" else RUNNING
    pair = REFUSED_THROUGH_SCOPE.start({"gate": 1, "n": 3}, implementation=implementation)
    pair.check_answer("h.s.b.d")
    assert isinstance(pair.engine.answers.query("h.s.b.d"), Rejected)
    after = pair.check_change(Change(commit={"h.s": "a"}))
    assert after is not None
    after.check_answer("h.s.b.d")
    assert isinstance(after.engine.answers.query("h.s.b.d"), Inapplicable)
    configuration = design_space(REFUSED_THROUGH_SCOPE.engine.root(gate=1, n=3))  # type: ignore[call-arg]
    found, _, _ = compare.change(
        TODAY.open(configuration), RUNNING.open(configuration), compare.Request({"h.s": "a"})
    )
    assert found.lines() == []


# -- a reader beside a formal's scope ---------------------------------------------------------
#
# ``part.leaf``'s formal ``w`` is bound to ``part.y``; ``leaf`` has no ``when`` of its own, so
# it is present wherever ``part`` is. Two readers beside it, under the same guard: ``part``'s
# constraint ``fits`` and the sibling scope ``part.side``, whose formal is bound to
# ``leaf.w``. Neither is in ``leaf``, so each reads ``leaf.w`` as a member of ``leaf``: its
# guard (the reader's own) and then its source. Every answer is today's, standing included.


class Leaf(Space):
    w: int = Param()
    lanes: int = Decision(values=(1, 2))

    @view
    def bits(self) -> int:
        return self.w * self.lanes


class Side(Space):
    v: int = Param()
    k: int = Decision(values=(1, 2, 3))

    @constraint
    def within(self) -> bool | Rejected:
        return True if self.v * self.k <= 6 else reject("side-wide", "v times k past 6")


class Part(Space):
    y: int = Decision(values=(1, 2, 3))
    leaf = Leaf(w=y)
    side = Side(v=leaf.w)

    @constraint
    def fits(self) -> bool | Rejected:
        return True if self.leaf.w + self.y <= 4 else reject("too-wide", "w and y past 4")


class Sibling(Space):
    gate: int = Decision(values=(0, 1, 2))

    @derived
    def on(self) -> bool | Rejected:
        return reject("gate-refused", "gate 2 refuses") if self.gate == 2 else self.gate == 1

    part = Part(when=on)


@pytest.mark.parametrize("seed", range(6))
def test_a_reader_beside_a_formals_scope_answers_as_today(seed: int) -> None:
    found = compare.walk(lambda: design_space(Sibling()), {}, seed, 20, TODAY, RUNNING, forms.ALL)
    assert found.lines() == []


def test_a_refusal_through_a_formal_read_beside_its_scope_names_todays_standing_choices() -> None:
    point = bind(design_space(Sibling()), {Sibling.gate: 1}).instance
    found, base, other = compare.change(
        TODAY.open(point), RUNNING.open(point), compare.Request({"part.y": 1, "part.side.k": 3})
    )
    assert found.lines() == [] and base is not None and other is not None
    for request in (
        compare.Request({"part.y": 3}),
        compare.Request({"part.y": 2}),
        compare.Request({"part.y": 3}, ("part.side.k",)),
        compare.Request({"part.y": 3, "gate": 2}),
    ):
        found, _, _ = compare.change(base, other, request)
        assert found.lines() == [], request
    refused = other.change({"part.y": 3}, ())
    assert not refused.report.accepted
    standing = {rule.rule: rule.standing for rule in refused.report.broken}
    assert standing == {
        rule.rule: rule.standing for rule in base.change({"part.y": 3}, ()).report.broken
    }
    assert any(standing.values()), standing


# -- a formal read from outside its scope (``toys.Boundary``) --------------------------------


@pytest.mark.parametrize("enabled", [False, True])
def test_a_formal_read_from_outside_its_absent_scope_is_inapplicable(enabled: bool) -> None:
    boundary = toys.Boundary
    stored = {boundary.enabled: enabled, boundary.outside.lanes: 2}
    point = bind(design_space(boundary(width=3)), stored).instance
    answers = RUNNING.open(point)
    for key in ("outside.width", "outside.bits"):
        found = answers.query(key)
        if enabled:
            assert isinstance(found, Available), key
        else:
            assert isinstance(found, Inapplicable), key
        assert forms.answer(found) == forms.answer(TODAY.open(point).query(key)), key


# -- models dropped and their ids reused --------------------------------------------------------


def _shape(number: int) -> type[Space]:
    """A Space class of its own: a formal bound by reference to a choice, at one of two
    places by ``number``'s parity, so a record inherited from a dead model would read the
    wrong source."""

    class Inner(Space):
        w: int = Param()
        pe: int = Decision(values=(1, 2, 4))

        @constraint
        def fits(self) -> bool | Rejected:
            return True if self.pe <= self.w else reject("too-wide", "pe past w")

    if number % 2:

        class Odd(Space):
            a: int = Decision(values=(1, 2, 4))
            b: int = Decision(values=(1, 2, 4))
            inner = Inner(w=a)

        return Odd

    class Even(Space):
        b: int = Decision(values=(1, 2, 4))
        a: int = Decision(values=(1, 2, 4))
        inner = Inner(w=b)

    return Even


def test_a_dropped_model_takes_its_record_layers_and_tables_with_it_when_its_id_is_reused() -> None:
    """Each model is asked by the running type with no cache and with a node cache of its
    own; once dropped, its record, its facts layers and its cache tables are gone, and the
    new model at its id starts with none of them."""
    gc.collect()
    alive = structure.records()
    running = cached()
    cache = cast(NodeCache, running.cache)
    dead: dict[int, weakref.ref[Any]] = {}
    layers: list[weakref.ref[FactsLayer]] = []
    reused = None
    for number in range(400):
        space_class = _shape(number)
        configuration = design_space(space_class())
        model = state(configuration).model
        identity = id(model.linked)
        if identity in dead:
            assert dead[identity]() is None
            # a new model at a dead one's id: no record and no table until it is first read
            assert structure._RECORDS.get(model.linked) is None
            assert cache.models() == 0
            reused = (space_class, configuration)
            break
        for implementation in (RUNNING, running):
            answers = implementation.make(space_class, {}, {"a": 2, "b": 1, "inner.pe": 2})
            assert answers.validity().status in ("valid", "refused")
            layers.append(weakref.ref(cast(Any, answers).ds.layer))
        assert cache.models() == 1
        dead[identity] = weakref.ref(model.linked)
        del space_class, configuration, model, answers
        gc.collect()
    assert reused is not None, "no id was reused in 400 models"
    assert all(layer() is None for layer in layers)
    assert structure.records() == alive  # every dead model's record gone, the new one unread
    space_class, configuration = reused
    hits = counters(running).hits
    first = running.make(space_class, {}, {"a": 4, "b": 2, "inner.pe": 4})
    assert (
        first.validity() == TODAY.make(space_class, {}, {"a": 4, "b": 2, "inner.pe": 4}).validity()
    )
    assert counters(running).hits == hits  # nothing of a dead model's found
    assert cache.models() == 1
    for stored in ({"a": 2, "b": 1, "inner.pe": 2}, {"a": 1, "b": 4, "inner.pe": 4}):
        today = forms.space(TODAY.make(space_class, {}, stored))
        for implementation in (RUNNING, running):
            made = implementation.make(space_class, {}, stored)
            assert forms.differences(today, forms.space(made)) == []
    for implementation in (RUNNING, running):
        found = compare.walk(lambda: design_space(space_class()), {}, 0, 12, TODAY, implementation)
        assert found.lines() == []
    assert counters(running).hits > hits


def test_a_facts_layer_is_interned_while_a_design_space_holds_it() -> None:
    configuration = toys.TOYS["house"].make()
    model = state(configuration).model
    facts = dict(state(configuration).world.facts)
    first = FactsLayer.of(model, facts)
    assert FactsLayer.of(model, dict(facts)) is first
    held = DesignSpace(first, {})
    assert DesignSpace(FactsLayer.of(model, facts), {}) == held
    assert hash(DesignSpace(FactsLayer.of(model, facts), {})) == hash(held)
    probe = weakref.ref(first)
    del first, held
    gc.collect()
    assert probe() is None


# -- depth ---------------------------------------------------------------------------------------


def _line(depth: int) -> type[Space]:
    """``value_<n>`` reads ``value_<n-1>`` through its receiver, ``depth`` deep."""
    members: dict[str, object] = {"__annotations__": {"value_0": int}, "value_0": Param()}

    def step(previous: str) -> Callable[[Space], int]:
        def compute(self: Space) -> int:
            return cast(int, getattr(self, previous)) + 1

        return compute

    for number in range(1, depth + 1):
        members[f"value_{number}"] = derived(step(f"value_{number - 1}"))
    return cast(type[Space], type("Line", (Space,), members))


LINE = _line(300)


def test_a_line_deeper_than_the_stack_allows_answers_as_today_with_the_limit_unchanged() -> None:
    limit = sys.getrecursionlimit()
    configuration = design_space(LINE(value_0=0))  # type: ignore[call-arg]
    running = fresh()
    answers = running.open(configuration)
    assert answers.query("value_300") == Available(300)
    assert forms.space(answers, []) == forms.space(TODAY.open(configuration), [])
    assert sys.getrecursionlimit() == limit
    assert counters(running).deepest == evaluate.DEPTH and counters(running).restarts > 0


def test_without_the_depth_guard_the_line_overflows_the_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluate, "DEPTH", 10**9)
    configuration = design_space(LINE(value_0=0))  # type: ignore[call-arg]
    with pytest.raises(NativeEvaluationError, match="maximum recursion depth exceeded"):
        fresh().open(configuration).query("value_300")


@pytest.mark.usefixtures("purity_unchecked")
def test_with_a_cache_the_depth_guard_keeps_nothing_for_an_abandoned_node() -> None:
    """Every answer the cache kept on the line reads ``value_0`` alone: no node abandoned by
    the depth guard was kept with the reads it had made so far. The line answers anew for
    another ``value_0``, and from the cache for the first again."""
    running = cached()
    cache = cast(NodeCache, running.cache)
    first = running.open(design_space(LINE(value_0=0)))  # type: ignore[call-arg]
    assert first.query("value_300") == Available(300)
    assert counters(running).restarts > 0
    origin = cast(Any, first).ds.node("value_0")
    tables = list(cache._tables.values())
    read_sets = {read for table in tables for sets in table.values() for read in sets}
    assert read_sets == {(origin,)}
    assert running.open(design_space(LINE(value_0=1))).query("value_300") == Available(301)  # type: ignore[call-arg]
    evaluated = counters(running).evaluated
    assert running.open(design_space(LINE(value_0=0))).query("value_300") == Available(300)  # type: ignore[call-arg]
    assert counters(running).evaluated == evaluated


# -- the membership assertion ---------------------------------------------------------------------


def test_a_membership_reading_that_reads_a_decision_is_an_error_naming_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``g.pe``'s domain reads the choice ``w`` through ``g``'s formal: the compile check
    refuses that (``domain-reads-choice``), so it is taken out to build the case. With ``on``
    open, ``g.pe``'s type check is a membership reading, and it reads ``w``."""
    monkeypatch.setattr(_compose, "check_domains", lambda template: None)

    class Inside(Space):
        width: int = Param()
        pe: int = Decision(domain=divisors_of(width))

    class Reads(Space):
        on: bool = Decision(values=(False, True))
        w: int = Decision(values=(4, 6))
        g = Inside(width=w, when=on)

    model = cast(Any, state(design_space(Reads())).model)
    ds = DesignSpace(FactsLayer.of(model, {}), {model.linked.keys["w"]: 4})
    ev = evaluate.evaluation(ds, Session())
    with pytest.raises(NativeEvaluationError) as raised:
        questions.check(ev, model.linked.keys["g.pe"], 2)
    error = raised.value.primary
    assert isinstance(error, MembershipReadsChoice)
    assert (error.choice, error.decision) == ("g.pe", "w")


def test_a_membership_reading_of_a_choice_with_no_guard_that_reads_a_decision_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pe`` has no guard, and its domain reads the choice ``w``: built past the compile
    check as above. Its membership is read on the facts layer with nothing stored, as a
    guarded choice's is, so the reading meets ``w`` and raises."""
    monkeypatch.setattr(_compose, "check_domains", lambda template: None)

    class Unguarded(Space):
        w: int = Decision(values=(4, 6))
        pe: int = Decision(domain=divisors_of(w))

    model = cast(Any, state(design_space(Unguarded())).model)
    keys = model.linked.keys
    ds = DesignSpace(FactsLayer.of(model, {}), {keys["w"]: 4})
    ev = evaluate.evaluation(ds, Session())
    reading = ev.membership(keys["pe"])
    assert reading.ds is ds.layer.bare and reading.assumed == frozenset()
    for value in (2, 3):
        with pytest.raises(NativeEvaluationError) as raised:
            questions.check(ev, keys["pe"], value)
        error = raised.value.primary
        assert isinstance(error, MembershipReadsChoice)
        assert (error.choice, error.decision) == ("pe", "w")


# -- a choice in a case of a selector picked by a choice: the case's guard is its own ------------
#
# ``sel.a.z``'s domain reads ``sel.a.cap`` in its own case, and ``sel`` is picked by ``pick``,
# which reads the choice ``x`` (``_running_shapes.PickedByAChoice``). Reading ``z``'s
# membership as if it existed reads the case's guard, ``z``'s own existence guard, taken as
# holding: it never reaches ``x``, and every answer is today's. The same read from a choice
# outside the case (``w`` at the root) reads that guard as any other: it reaches ``x``, which
# the compile check refuses (``domain-reads-choice``), and, built past it, the membership
# assertion names ``w`` and ``x``.


@pytest.mark.parametrize("f", [1, 2])
def test_a_choice_in_a_case_picked_by_a_choice_reads_its_cases_guard_as_holding(f: int) -> None:
    point = design_space(PickedByAChoice(f=f))
    model = cast(Any, state(point).model)
    keys = model.linked.keys
    ds = DesignSpace(FactsLayer.of(model, {keys["f"]: f}), {})
    ev = evaluate.evaluation(ds, Session())
    reading = ev.membership(keys["sel.a.z"])
    assert reading.assumed == frozenset({keys["sel.$known"], keys["sel.a.$selected"]})
    checks = [questions.check(ev, keys["sel.a.z"], value) for value in (1, 2, 3)]
    assert [isinstance(check, Available) for check in checks] == [
        value <= f + 1 for value in (1, 2, 3)
    ]
    z = PickedByAChoice.sel["a"].z
    for stored in (
        {},
        {PickedByAChoice.x: 0},
        {PickedByAChoice.x: 1},
        {PickedByAChoice.x: 0, z: 2},
    ):
        bound = bind(point, stored).instance
        assert compare.everything(TODAY.open(bound), RUNNING.open(bound)).lines() == []


def test_a_choice_outside_the_case_reading_its_value_reads_the_picker_and_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def outside() -> type[Space]:
        class Outside(Space):
            f: int = Param()
            x: int = Decision(values=(0, 1))

            @derived
            def pick(self) -> str:
                return ("a", "b")[self.x]

            sel = Select(by=pick, a=Capped(f=f), b=None)
            w: int = Decision(domain=capped(sel["a"].cap))

        return Outside

    with pytest.raises(DefinitionError) as refused:
        design_space(outside()(f=2))  # type: ignore[call-arg]
    assert [finding.code for finding in refused.value.findings] == ["domain-reads-choice"]
    monkeypatch.setattr(_compose, "check_domains", lambda template: None)
    model = cast(Any, state(design_space(outside()(f=2))).model)  # type: ignore[call-arg]
    keys = model.linked.keys
    ds = DesignSpace(FactsLayer.of(model, {keys["f"]: 2}), {keys["x"]: 0})
    ev = evaluate.evaluation(ds, Session())
    assert ev.membership(keys["w"]).assumed == frozenset()
    assert questions.check(ev, keys["sel.a.z"], 3) == Available(True)
    with pytest.raises(NativeEvaluationError) as raised:
        questions.check(ev, keys["w"], 2)
    error = raised.value.primary
    assert isinstance(error, MembershipReadsChoice)
    assert (error.choice, error.decision) == ("w", "x")


# -- a selector given as a binding's argument reads its case name -------------------------------
#
# ``_running_shapes.ByArgument``: a chosen and a picked selector, each with the ``None`` case
# ``direct``, given as the argument of derived values, views, constraints and a guard's
# derived value. The argument reads the case's name, ``"direct"`` for the ``None`` case, as
# lowering resolves it: the running type follows with no frame of its own, and answers every
# question as today's.


@pytest.mark.parametrize(
    ("x", "chosen", "names"),
    [
        (None, None, (None, None)),
        (0, None, (None, "direct")),
        (0, "direct", ("direct", "direct")),
        (1, "lane", ("lane", "lane")),
        (1, "direct", ("direct", "lane")),
        (2, "stage", ("stage", "stage")),
    ],
)
def test_a_selector_given_as_a_bindings_argument_reads_its_case_name_as_today(
    x: int | None, chosen: str | None, names: tuple[str | None, str | None]
) -> None:
    stored: dict[object, object] = {}
    if x is not None:
        stored[ByArgument.x] = x
    if chosen is not None:
        stored[ByArgument.chosen] = chosen
    point = bind(design_space(ByArgument()), stored).instance
    answers = RUNNING.open(point)
    found = tuple(
        getattr(answers.query(key), "value", None) for key in ("chosen_name", "picked_name")
    )
    assert found == names
    assert compare.everything(TODAY.open(point), answers).lines() == []


# -- the node cache: a dropping cache, and a cache wrong on purpose -------------------------------
#
# The contract lets the cache drop any entry at any time: a miss changes nothing but the
# work. ``RUNNING_DROPPING`` drops half its hits and half its keeps, at random (seeded), and
# must answer as today's engine does. ``RUNNING_PLANTED`` keys each answer without the last
# read of its read set: the comparison must catch it, or the cache's proofs have no teeth.
# A test that counts what a hit saves, or that a wrong hit is answered, runs with the
# purity check off (``purity_unchecked``): the check evaluates every hit anyway.


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("toy", sorted(TOYS))
def test_the_toys_walk_as_today_with_a_cache_dropping_entries_at_random(
    toy: str, seed: int
) -> None:
    before = counters(RUNNING_DROPPING).hits
    found = compare.walk(
        TOYS[toy].make, TOYS[toy].facts, seed, 20, TODAY, RUNNING_DROPPING, forms.ALL
    )
    assert found.lines() == []
    assert found.tally["changes"] > 0 and found.tally["cold builds"] > 0
    assert counters(RUNNING_DROPPING).hits > before


class Ordered(Space):
    a: int = Decision(values=(1, 2, 3))
    b: int = Decision(values=(1, 2, 3))

    @constraint
    def ordered(self) -> bool | Rejected:
        return True if self.a <= self.b else reject("unordered", "a past b")


@pytest.mark.usefixtures("purity_unchecked")
def test_a_cache_leaving_a_read_out_of_its_key_answers_otherwise_and_is_caught() -> None:
    """``ordered`` reads ``a`` and ``b``; the planted cache keys it by ``a`` alone. Asked at
    (2, 3), then at (2, 1), it answers the second as the first: valid where today's
    engine refuses. A cache keyed by both answers the second as today's."""
    planted, honest = Running(LeavesOneOut(), "planted"), cached()
    for implementation in (planted, honest):
        first = implementation.make(Ordered, {}, {"a": 2, "b": 3})
        assert first.validity().status == "valid"
    today = TODAY.make(Ordered, {}, {"a": 2, "b": 1})
    assert today.validity().status == "refused"
    assert honest.make(Ordered, {}, {"a": 2, "b": 1}).validity() == today.validity()
    wrong = planted.make(Ordered, {}, {"a": 2, "b": 1})
    assert wrong.validity().status == "valid"
    found = compare.everything(today, wrong)
    assert {difference.kind for difference in found.disagreements} >= {"members", "validity"}
    assert counters(planted).hits > 0


@pytest.mark.usefixtures("purity_unchecked")
def test_a_hit_leaves_the_evaluation_as_evaluating_would() -> None:
    """Every node of a house, asked of a design space equal to one evaluated before with
    the same cache, is a hit; each answer is the one evaluating gives (its result, trace,
    assessment and read set), and it is in the evaluation's memo."""
    stored = {"kitchen.finish": 2, "want_garage": True}
    warm, cold = cached(), cached()
    before = cast(Any, warm.make(House, {"budget": 100}, stored)).ds
    nodes = sorted(set(cast(Structure, before.model.linked).keys.values()))
    for index in nodes:
        evaluate.evaluation(before, warm.session).get(index)
    again = cast(Any, warm.make(House, {"budget": 100}, stored)).ds
    other = cast(Any, cold.make(House, {"budget": 100}, stored)).ds
    assert again == before and again is not before
    hit, evaluated = evaluate.evaluation(again, warm.session), counters(warm).evaluated
    made = evaluate.evaluation(other, cold.session)
    for index in nodes:
        assert hit.get(index) == made.get(index), index
        assert index in hit.memo
    assert counters(warm).evaluated == evaluated  # every one a hit
    assert counters(cold).evaluated > 0


@pytest.mark.usefixtures("purity_unchecked")
def test_a_walk_catches_the_planted_cache() -> None:
    house = TOYS["house"]
    found = compare.walk(house.make, house.facts, 0, 5, TODAY, RUNNING_PLANTED)
    assert found.disagreements
    assert {"members", "validity"} <= set(found.by_kind), found.by_kind


# -- the purity check over the cache and the depth guard ------------------------------------------
#
# The node cache and the depth guard's restart rest on purity: an authored function answers
# from what it reads alone. The purity check (``evaluate.Session.verifying``: today's switch,
# ``_world.check_purity``, which this package's conftest turns on, or a session made with
# ``verify``) evaluates a cache hit anyway, and a node the depth guard abandoned again once it
# completes, and compares: an impure function is its ``EvaluationError``. A planted impure
# function is caught with the check on, and answers silently with it off.

STAMPS = itertools.count()


class Stamped(Space):
    """Impure: each answer reads a counter outside the Space."""

    width: int = Param()

    @derived
    def stamped(self) -> int:
        return self.width + next(STAMPS) % 2

    @derived
    def numbered(*, width: int) -> int:
        return width + next(STAMPS) % 2


SWITCHES = ("off", "the session's", "today's")


@contextmanager
def switched(switch: str) -> Iterator[Running]:
    """A cached running type with the purity check as ``switch`` says: off, on by its
    session's ``verify``, or on by today's switch alone."""
    previous = check_purity(switch == "today's")
    try:
        yield Running(
            NodeCache(), f"running, cached, check {switch}", verify=switch == "the session's"
        )
    finally:
        check_purity(previous)


@pytest.mark.parametrize("switch", SWITCHES)
@pytest.mark.parametrize("key", ["stamped", "numbered"])
def test_the_purity_check_evaluates_a_cache_hit_again_and_names_an_impure_function(
    key: str, switch: str
) -> None:
    with switched(switch) as running:
        first = running.make(Stamped, {"width": 4}, {}).query(key)
        again = running.make(Stamped, {"width": 4}, {})  # an equal design space: a hit
        if switch == "off":
            assert again.query(key) == first  # the kept answer, silently
            assert counters(running).hits == 1
            return
        with pytest.raises(NativeEvaluationError) as caught:
            again.query(key)
    assert counters(running).hits == 2  # the function's, and its fact's, read again
    assert caught.value.primary.owner == key
    assert "answered differently when run again" in str(caught.value.primary)


def _impure_line(depth: int) -> type[Space]:
    """``LINE``'s shape, its last value impure: past the depth guard, it is abandoned."""
    members: dict[str, object] = {"__annotations__": {"value_0": int}, "value_0": Param()}

    def step(previous: str, impure: bool) -> Callable[[Space], int]:
        def compute(self: Space) -> int:
            return cast(int, getattr(self, previous)) + 1 + (next(STAMPS) % 2 if impure else 0)

        return compute

    for number in range(1, depth + 1):
        members[f"value_{number}"] = derived(step(f"value_{number - 1}", number == depth))
    return cast(type[Space], type("ImpureLine", (Space,), members))


IMPURE_LINE = _impure_line(100)


@pytest.mark.parametrize("switch", SWITCHES)
def test_the_purity_check_evaluates_a_node_the_depth_guard_abandoned_again(switch: str) -> None:
    with switched(switch) as cached_:
        running = Running(verify=cached_.verify)  # no cache: the depth guard's alone
        answers = running.make(IMPURE_LINE, {"value_0": 0}, {})
        if switch == "off":
            assert isinstance(answers.query("value_100"), Available)
        else:
            with pytest.raises(NativeEvaluationError) as caught:
                answers.query("value_100")
            assert caught.value.primary.owner == "value_100"
            assert "answered differently when run again" in str(caught.value.primary)
    assert counters(running).restarts > 0


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("toy", sorted(TOYS))
def test_the_toys_walk_as_today_with_the_purity_check_over_the_cache(toy: str, seed: int) -> None:
    before = counters(RUNNING_VERIFIED).hits
    found = compare.walk(
        TOYS[toy].make, TOYS[toy].facts, seed, 20, TODAY, RUNNING_VERIFIED, forms.ALL
    )
    assert found.lines() == []
    assert found.tally["changes"] > 0 and found.tally["cold builds"] > 0
    assert counters(RUNNING_VERIFIED).hits > before  # each one evaluated again, the same


@pytest.mark.parametrize("seed", range(24))
def test_the_reference_walks_hold_strictly_with_the_purity_check_over_the_cache(
    seed: int,
) -> None:
    before = counters(RUNNING_VERIFIED).hits
    tally = reference_walks.walk(seed, 40, strict=True, implementation=RUNNING_VERIFIED)
    assert tally.failure is None, tally.failure
    assert tally.implementation == "running, cached, verified"
    assert tally.steps == 40 and not tally.by_milestone and tally.engine_errors == 0
    assert tally.engine_checks["cold"] == tally.engine_checks["bind"] == tally.compared
    assert tally.engine_checks["cold change"] == tally.steps
    assert counters(RUNNING_VERIFIED).hits > before


# -- the value types the cache keys -------------------------------------------------------------
#
# The cache matches values by type and ``==`` (``cache.key``): where a type's ``==`` is
# coarser than its value semantics, the cache is wrong, not slow. So it keys only a value
# whose choice or fact has the default semantics (``cache.plain``), and among those the
# plain types
# (``cache.PLAIN``) and the classes in ``cache.CHECKED``, each listed here with the value
# semantics in force and its ``==`` and hash checked against them (``_world._same_value``):
# values equal by the semantics are ``==`` and hash alike, and values the semantics tell
# apart are not ``==``, so their keys differ. Tuples and frozen sets of keyed values are
# keyed item by item; a value class (``cache.value_class``: a frozen dataclass whose ``==``
# and hash are the generated ones) field by field, each field's value keyed in turn. The
# rule is checked on this module's own sample classes: under the default semantics, a frozen
# dataclass's generated ``==`` is its semantics' equality, and its key is never coarser. A
# value under semantics of its own (``PAIR``, ``ENCODING``, ``WIDTH``) has no key, whatever
# its type, and neither has a value class with a field annotated with them (``Labelled``):
# the cache cannot read a semantics' equality, and ``WIDTH``'s is finer than ``==``. Any
# other type has no key, nor has an unhashable one, and nothing that read it is kept: an
# unlisted type is a miss, never a wrong answer.

NAN, OTHER_NAN = float("nan"), float("nan")


@dataclass(frozen=True)
class Spec:
    enabled: bool
    lanes: int


@dataclass(frozen=True)
class Encoding:
    bits: int
    integer: bool = True


@dataclass(frozen=True)
class Capabilities:
    fast: bool = True
    ports: int = 1


@dataclass(frozen=True)
class Stream:
    """A value class holding a value class, a tuple and a float."""

    dtype: Encoding
    shape: tuple[int, ...]
    rate: float


@dataclass(frozen=True)
class Tagged:
    """A field left out of ``==`` (``compare=False``) is left out of the key."""

    n: int
    note: str = field(default="", compare=False)


@dataclass(frozen=True)
class HandWritten:
    """Equal by ``lanes`` alone, by its own ``==``: not a value class."""

    lanes: int
    bits: int

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HandWritten) and other.lanes == self.lanes

    def __hash__(self) -> int:
        return hash(self.lanes)


@dataclass(frozen=True, eq=False)
class Identity:
    """Equal by identity: not a value class."""

    lanes: int


class Inherits(Spec):
    """A plain subclass of a value class: not one itself."""


@dataclass(frozen=True)
class HoldsList:
    """A value class whose field holds a list: no key."""

    cells: list[int]


@dataclass
class Bag:
    values: list[int]


BAG = ValueSemantics(
    Bag,
    "bag",
    lambda value: type(value) is Bag,
    lambda left, right: left.values == right.values,
    lambda value: Bag(list(value.values)),
)


class Grid:
    """A value like an array: ``==`` answers cell by cell, with no truth value."""

    def __init__(self, cells: tuple[int, ...]) -> None:
        self.cells = cells

    def __eq__(self, other: object) -> Cells:  # type: ignore[override]
        assert isinstance(other, Grid)
        return Cells(tuple(a == b for a, b in zip(self.cells, other.cells)))

    __hash__ = None  # type: ignore[assignment]


class Cells:
    def __init__(self, cells: tuple[bool, ...]) -> None:
        self.cells = cells

    def __bool__(self) -> bool:
        raise ValueError("the truth value of a grid comparison is ambiguous")


GRID = ValueSemantics(
    Grid,
    "grid",
    lambda value: type(value) is Grid,
    lambda left, right: left.cells == right.cells,
    lambda value: Grid(value.cells),
)

PAIR: ValueSemantics[tuple[int, int]] = ValueSemantics(
    tuple[int, int],
    "pair",
    lambda value: type(value) is tuple and len(value) == 2,
    lambda left, right: left == right,
    lambda value: value,
)

ENCODING = ValueSemantics.immutable_nominal(Encoding)


@dataclass(frozen=True)
class Width:
    """Equal by ``bits`` alone (``note`` is left out of ``==``), and told apart by its
    semantics (``WIDTH``) by both: semantics finer than ``==``."""

    bits: int
    note: str = field(default="", compare=False)


WIDTH: ValueSemantics[Width] = ValueSemantics(
    Width,
    "width",
    lambda value: type(value) is Width,
    lambda left, right: (left.bits, left.note) == (right.bits, right.note),
    lambda value: value,
)


@dataclass(frozen=True)
class Labelled:
    """A value class with a field annotated with semantics of its own: not keyed."""

    width: Annotated[Width, WIDTH]


@dataclass(frozen=True)
class Counted:
    """A value class with a field annotated with the default semantics: keyed."""

    count: Annotated[int, default_semantics(int)]


VALUE_TYPES: dict[str, tuple[ValueSemantics[Any], tuple[object, ...]]] = {
    "None": (default_semantics(type(None)), (None,)),
    "bool": (default_semantics(bool), (False, True)),
    "int": (default_semantics(int), (0, 1, 2, -1, 10**20)),
    "float": (default_semantics(float), (0.0, -0.0, 1.0, 0.1, math.inf, NAN, OTHER_NAN)),
    "complex": (default_semantics(complex), (0j, complex(-0.0, 0.0), 1 + 2j)),
    "str": (default_semantics(str), ("", "a", "A", "a ")),
    "bytes": (default_semantics(bytes), (b"", b"a")),
    "path": (default_semantics(PurePosixPath), (PurePosixPath("a"), PurePosixPath("a/b"))),
    "tuple": (
        default_semantics(tuple),
        ((), (1,), (True,), (1.0,), (0.0,), (-0.0,), (1, 2), (2, 1), ((1,),), ("a",)),
    ),
    "frozenset": (
        default_semantics(frozenset),
        (frozenset(), frozenset({1}), frozenset({True}), frozenset({1, 2})),
    ),
    "pair": (PAIR, ((1, 2), (2, 1), (1, 1), (True, 2))),
    "Encoding": (
        default_semantics(Encoding),
        (Encoding(8), Encoding(8, False), Encoding(16), Encoding(8, cast(bool, 1))),
    ),
    "Encoding (immutable nominal)": (
        ENCODING,
        (Encoding(8), Encoding(8, False), Encoding(16), Encoding(8, cast(bool, 1))),
    ),
    "Spec": (
        default_semantics(Spec),
        (Spec(True, 1), Spec(False, 1), Spec(True, 2), Spec(cast(bool, 1), 1)),
    ),
    "Capabilities": (
        default_semantics(Capabilities),
        (Capabilities(), Capabilities(False), Capabilities(True, 2)),
    ),
    "Stream": (
        default_semantics(Stream),
        (
            Stream(Encoding(8), (1, 2), 0.0),
            Stream(Encoding(8), (1, 2), -0.0),
            Stream(Encoding(8), (2, 1), 0.0),
            Stream(Encoding(16), (1, 2), 0.0),
            Stream(Encoding(8, cast(bool, 1)), (1, 2), 0.0),
            Stream(Encoding(8), (True, 2), 0.0),
            Stream(Encoding(8), (1, 2), NAN),
            Stream(Encoding(8), (1, 2), OTHER_NAN),
        ),
    ),
    "Tagged": (default_semantics(Tagged), (Tagged(1), Tagged(1, "a"), Tagged(2))),
    "Counted": (default_semantics(Counted), (Counted(1), Counted(cast(int, True)), Counted(2))),
}
"""Each hashable value type by name: its value semantics, and values to compare. Those
under semantics of their own (``PAIR``, ``ENCODING``) are left unkeyed."""

VALUE_CLASSES = (Spec, Encoding, Capabilities, Stream, Tagged, Counted)
"""The sample value classes above, each keyed by the rule."""

NOT_VALUE_CLASSES: dict[str, tuple[type, object]] = {
    "a hand-written ==": (HandWritten, HandWritten(1, 8)),
    "equal by identity": (Identity, Identity(1)),
    "mutable": (Bag, Bag([1])),
    "a plain subclass": (Inherits, Inherits(True, 1)),
    "a field with semantics of its own": (Labelled, Labelled(Width(8))),
}
"""Classes the rule does not key, each with a value."""

UNHASHABLE: dict[str, tuple[ValueSemantics[Any], tuple[object, ...]]] = {
    "list": (default_semantics(list), ([], [1], [1, 2])),
    "Bag": (BAG, (Bag([1]), Bag([1, 2]))),
    "Grid": (GRID, (Grid((1,)), Grid((1, 2)))),
}
"""Each unhashable value type by name: never keyed."""

HARNESS = (bool, int, str, tuple)
"""The value types the reference harness's builder gives a choice or a fact."""


def plain_key(value: object) -> object:
    """``value``'s key as a choice or fact of its type's default semantics holds it."""
    return cache_key(value, default_semantics(type(value)))


def _member(semantics: ValueSemantics[Any]) -> Node:
    """A member holding values of ``semantics``, as ``_same_value`` reads it."""
    return cast(Node, SimpleNamespace(semantics=semantics, owner="value"))


@pytest.mark.parametrize("name", sorted(VALUE_TYPES))
def test_a_value_types_equality_and_hash_agree_with_its_semantics(name: str) -> None:
    """Under the default semantics, each sample is keyed, and in the safe direction: a key
    hit is a value the semantics hold equal. Under semantics of their own (``PAIR``,
    ``ENCODING``), none is keyed."""
    semantics, values = VALUE_TYPES[name]
    member = _member(semantics)
    assert all(semantics.accepts(value) for value in values)
    if not plain(semantics):
        assert all(cache_key(value, semantics) is None for value in values)
    for left, right in itertools.product(values, repeat=2):
        same = _same_value(member, left, right)
        if left is not right:  # a value is its own (a NaN is not ``==`` to itself)
            assert (left == right) is same, (left, right)
        if same:
            assert hash(left) == hash(right), (left, right)
        if not plain(semantics):
            continue
        if not same:
            assert cache_key(left, semantics) != cache_key(right, semantics), (left, right)
        if cache_key(left, semantics) == cache_key(right, semantics):
            assert same, (left, right)  # a hit is never a value the semantics tell apart
    if plain(semantics):
        assert all(cache_key(value, semantics) is not None for value in values)


@pytest.mark.parametrize("name", sorted(UNHASHABLE))
def test_an_unhashable_value_type_is_never_keyed(name: str) -> None:
    semantics, values = UNHASHABLE[name]
    assert all(semantics.accepts(value) for value in values)
    assert all(cache_key(value, semantics) is None for value in values)
    assert cache_key((1, values[0]), default_semantics(tuple)) is None


def test_true_and_1_and_the_two_zeros_are_keyed_as_their_semantics_say() -> None:
    """``True`` and ``1``: no member holds both (each type recognizes its own), and their
    keys differ. Inside a tuple or a value class the semantics hold them equal, and the key
    still tells them apart: a miss, never a wrong hit. ``-0.0`` and ``0.0`` are equal by
    the semantics, and keyed alike, alone, inside a tuple and inside a value class."""
    boolean, integer, pair = (
        VALUE_TYPES["bool"][0],
        VALUE_TYPES["int"][0],
        VALUE_TYPES["tuple"][0],
    )
    assert not boolean.accepts(1) and not integer.accepts(True)
    assert plain_key(True) != plain_key(1) and plain_key(False) != plain_key(0)
    assert _same_value(_member(pair), (True,), (1,)) and plain_key((True,)) != plain_key((1,))
    assert _same_value(_member(pair), ((1, True),), ((1, 1),))
    assert plain_key(((1, True),)) != plain_key(((1, 1),))
    spec = VALUE_TYPES["Spec"][0]
    assert _same_value(_member(spec), Spec(True, 1), Spec(cast(bool, 1), 1))
    assert plain_key(Spec(True, 1)) != plain_key(Spec(cast(bool, 1), 1))
    real = VALUE_TYPES["float"][0]
    assert _same_value(_member(real), -0.0, 0.0) and plain_key(-0.0) == plain_key(0.0)
    assert _same_value(_member(pair), (-0.0, 1), (0.0, 1)) and plain_key((-0.0, 1)) == plain_key(
        (0.0, 1)
    )
    stream = Stream(Encoding(8), (1,), 0.0)
    assert plain_key(stream) == plain_key(Stream(Encoding(8), (1,), -0.0))
    assert not _same_value(_member(real), NAN, OTHER_NAN) and plain_key(NAN) != plain_key(OTHER_NAN)


def test_the_cache_keys_the_types_this_test_checks_and_value_classes_by_their_rule() -> None:
    """Every type the cache keys by ``==`` (``cache.PLAIN``, ``cache.CHECKED``), tuples and
    frozen sets are types whose semantics this test checks, and so are the harness's. The
    rule (``cache.value_class``) holds for each sample value class, and for none of the
    classes it must not key: a hand-written ``==``, equality by identity, a mutable
    dataclass, a plain subclass of a value class, one with a field annotated with
    semantics of its own. A value class holding a value with no key has none."""
    checked = {semantics.type_token for semantics, _ in VALUE_TYPES.values() if plain(semantics)}
    assert {*PLAIN, *CHECKED, tuple, frozenset, *HARNESS} <= checked
    assert set(VALUE_CLASSES) <= checked
    assert all(value_class(kind) for kind in VALUE_CLASSES)
    for name, (kind, value) in NOT_VALUE_CLASSES.items():
        assert not value_class(kind), name
        assert plain_key(value) is None, name
        assert plain_key((1, value)) is None, name
    assert value_class(HoldsList) and plain_key(HoldsList([1])) is None
    assert plain_key((1, "a", frozenset({2.0}), Spec(True, 2))) is not None


@pytest.mark.usefixtures("purity_unchecked")
def test_a_value_of_a_type_not_keyed_keeps_nothing_and_a_value_class_is_found_again() -> None:
    """A fact holding a ``HandWritten`` (a frozen dataclass with its own ``==``: no key)
    keeps nothing for what read it, every lookup there a miss; a fact holding a value class
    is keyed, and found again. Both answer as today's engine."""

    class Holds(Space):
        rate = Param(semantics=default_semantics(HandWritten))
        n: int = Param()

        @derived
        def total(self) -> int:
            return cast(int, cast(Any, self.rate).lanes) * 2

        @derived
        def double(self) -> int:
            return self.n * 2

    class Specified(Space):
        spec: Spec = Param()

        @derived
        def lanes(self) -> int:
            return self.spec.lanes * 2

    value = HandWritten(3, 8)
    running = cached()
    cache = cast(NodeCache, running.cache)
    for _ in range(2):
        made = running.make(Holds, {"rate": value, "n": 4}, {})
        assert (made.query("total"), made.query("double")) == (Available(6), Available(8))
    assert counters(running).hits == 1  # ``double`` alone, the second time
    assert cache.answers() == 2  # ``n`` and ``double``: nothing that read ``rate``
    today = TODAY.make(Holds, {"rate": value, "n": 4}, {})
    assert compare.everything(today, made).lines() == []
    hits = counters(running).hits
    for _ in range(2):
        found = running.make(Specified, {"spec": Spec(True, 3)}, {})
        assert found.query("lanes") == Available(6)
    assert counters(running).hits == hits + 1  # ``lanes``, the second time
    assert (
        compare.everything(TODAY.make(Specified, {"spec": Spec(True, 3)}, {}), found).lines() == []
    )


# -- semantics finer than ``==`` ------------------------------------------------------------
#
# ``Noted``'s fact is a ``Width`` under ``WIDTH``, which tells apart two widths ``==`` holds
# equal, and derives the note. Two design spaces differing only in the note never share an
# answer through the cache: the rule keys no value under semantics of its own
# (``cache.plain``). With that check taken out, the second design space takes the first's
# note. A ``Labelled``, a value class whose field is annotated with ``WIDTH``, is not keyed
# either (``cache.value_class``). Its own semantics are the default ones, its ``==`` their
# equality: two labels differing in the note alone are one value, to today's engine as to
# the running type, and a function that answers by that note is impure (the purity check
# says so). So no answer depends on the field check: it is checked on the key alone.


class Noted(Space):
    width: Width = Param(semantics=WIDTH)

    @derived
    def note(self) -> str:
        return self.width.note

    @derived
    def bits(self) -> int:
        return self.width.bits


NOTED = (Width(8, "a"), Width(8, "b"))


def _notes(
    implementation: Implementation, shape: type[Space], fact: str, values: tuple[object, ...]
) -> tuple[list[object], list[str]]:
    """Each value's design space, made by ``implementation`` one after the other: its note,
    and what tells it apart from today's."""
    notes: list[object] = []
    lines: list[str] = []
    for value in values:
        made = implementation.make(shape, {fact: value}, {})
        notes.append(made.query("note"))
        lines += compare.everything(TODAY.make(shape, {fact: value}, {}), made).lines()
    return notes, lines


def test_values_equal_but_told_apart_by_their_semantics_never_share_an_answer() -> None:
    assert NOTED[0] == NOTED[1] and hash(NOTED[0]) == hash(NOTED[1])
    assert not _same_value(_member(WIDTH), *NOTED)
    assert not plain(WIDTH) and all(cache_key(value, WIDTH) is None for value in NOTED)
    notes, lines = _notes(RUNNING_CACHED, Noted, "width", NOTED)
    assert notes == [Available("a"), Available("b")]
    assert lines == []
    running = cached()
    _notes(running, Noted, "width", NOTED)
    tables = cast(NodeCache, running.cache)._tables.values()
    assert all(not read for table in tables for sets in table.values() for read in sets)


@pytest.mark.usefixtures("purity_unchecked")
def test_without_the_semantics_check_the_cache_answers_with_another_design_spaces_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same shape with ``cache.plain`` taken out (every semantics held the default):
    the second design space hits the first's note, and answers unlike today's."""
    monkeypatch.setattr(cache_module, "plain", lambda semantics: True)
    running = cached()
    notes, lines = _notes(running, Noted, "width", NOTED)
    assert notes == [Available("a"), Available("a")]
    assert lines != [] and counters(running).hits > 0


def test_a_value_class_with_a_field_of_semantics_of_its_own_is_not_keyed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Labelled`` is not keyed, alone or in a tuple; with the field check taken out
    (``value_class``'s answers made again), it is keyed field by field, and two labels
    ``WIDTH`` tells apart share a key."""
    one, other = Labelled(Width(8, "a")), Labelled(Width(8, "b"))
    assert not value_class(Labelled) and plain_key(one) is None and plain_key((1, one)) is None
    assert value_class(Counted) and plain_key(Counted(1)) is not None
    with monkeypatch.context() as patched:
        patched.setattr(cache_module, "_own_semantics", lambda annotation: False)
        patched.setattr(cache_module, "_VALUE_CLASSES", weakref.WeakKeyDictionary())
        assert value_class(Labelled)
        assert plain_key(one) == plain_key(other) is not None
