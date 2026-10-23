# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Templates (``finn.core.space._template``) on the kernels: the scope records of every Space
class the kernels, the KernelOps and the test support define, of the Chain's members and of
stage 7's chain of eight kernels as one Space, read whole, alike by demand, permuted and
eager, by key (``core.space._templates_support``; the engine's own corpus and edge cases
are ``core.space.test_templates``); their domain rule, composed per class, equal to the
line's over every address (``core.space._domains_support``); and their answers to the questions
about the whole design space, from compiled forms, equal to the line's
(``core.space._whole_support``; the engine's are ``core.space.test_whole``)."""

from __future__ import annotations

import pkgutil
from pathlib import Path

import pytest
from core.space._domains_support import domain_differences
from core.space._templates_support import assert_reads_alike, defined, links, reached
from core.space._whole_support import differences
from support.chain import MEMBERS, chain
from support.one_space import chain_model, chain_space, declaration

import finn.custom_op
import finn.kernels
from finn.core.space import Space, inspection
from finn.core.space._running.evaluate import evaluation
from finn.core.space._scopes import EAGER, Address, ScopeRecord
from finn.core.space._template import link_template
from finn.core.space.compiler import compile_model
from finn.core.space.occurrence import ask, held
from finn.core.space.results import Available, QueryResult, Unresolved
from finn.kernels.composer import Composed, clear_classes, compose
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel

SUPPORT = Path(__file__).resolve().parents[1] / "support"


def test_the_test_supports_classes_read_alike_in_any_order_their_scopes_are_made_in() -> None:
    modules = [f"support.{path.stem}" for path in sorted(SUPPORT.glob("*.py"))]
    classes = [space_type for space_type in defined(modules) if links(space_type)]
    assert len(classes) > 10
    assert_reads_alike(classes)


def _kernels() -> list[type[Space]]:
    modules = [
        info.name
        for package in (finn.kernels, finn.custom_op)
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + ".")
    ]
    return list(dict.fromkeys(defined(modules)))


def test_the_kernels_read_alike_in_any_order_their_scopes_are_made_in() -> None:
    classes = _kernels()
    assert len(classes) > 15
    assert assert_reads_alike(classes) == len(classes)


def test_the_kernels_domains_compose_as_the_line_checked_them() -> None:
    modules = [f"support.{path.stem}" for path in sorted(SUPPORT.glob("*.py"))]
    classes = [
        *_kernels(),
        *(cls for cls in defined(modules) if links(cls)),
        chain_space(4)[0].space_type,
    ]
    differences = [found for found in map(domain_differences, classes) if found is not None]
    assert not differences, "\n".join(differences)


def test_the_chains_members_read_alike_in_any_order_their_scopes_are_made_in() -> None:
    net = chain()  # configured: its channels bound
    classes = {inspection.model(net.member(name)).space_type for name in MEMBERS}
    assert assert_reads_alike(classes) == 3  # the MatMul, the thresholding, the channel


def test_stage_7s_chain_of_eight_kernels_reads_alike_in_any_order_its_scopes_are_made_in() -> None:
    whole = chain_space(4)[0].space_type
    assert_reads_alike([whole])
    templates = reached(link_template(whole))
    # Fifteen classes, and the composite (a port is a declaration of its kernel, no class).
    assert len(templates) == 16
    # A channel's source and mover are placements: compiled where their guard's formal is
    # not supplied too (their members inapplicable there), so each of the 9 channels
    # without a value has its source's scope, and each of the 13 its mover's, where the
    # one-case selectors compiled none; a direct transport places none.
    assert len(compile_model(whole).scopes.ordered()) == 225


def _answers_as_the_line(classes: list[type[Space]]) -> int:
    """Each class's answers to the questions about its whole design space, from templates,
    equal the line's; no method gained an inferred read; the number of choices compared."""
    failures: list[str] = []
    decisions = 0
    for space_type in classes:
        compared = differences(space_type)
        failures.extend(compared.differences)
        if compared.gained:
            failures.append(f"{space_type.__qualname__}: reads gained {compared.gained}")
        decisions += compared.decisions
    assert not failures, "\n".join(failures)
    return decisions


def test_the_kernels_and_the_test_support_answer_as_the_line() -> None:
    modules = [f"support.{path.stem}" for path in sorted(SUPPORT.glob("*.py"))]
    classes = [*_kernels(), *(cls for cls in defined(modules) if links(cls))]
    net = chain()
    classes.extend(inspection.model(net.member(name)).space_type for name in MEMBERS)
    assert _answers_as_the_line(list(dict.fromkeys(classes))) > 50


def test_stage_7s_chain_of_eight_kernels_answers_as_the_line() -> None:
    whole = chain_space(4)[0].space_type
    # The choices compared: each MatMul's folding is its own, not each core's; each
    # channel's source's two (absent without a value) and its one adapter memory, and no
    # source, mover or output adapter selector, nor an input adapter's, which its steps
    # pick.
    assert _answers_as_the_line([whole]) == 130


def fresh_chain() -> tuple[Composed, dict[str, object]]:
    """Stage 7's chain at 8 kernels composed as one Space with a model of its own, and its
    stored values by path: the graph layer binds a partition of the same class as it reads
    the stored values (``support.one_space.declaration``), so the composer's classes are
    dropped before it is composed."""
    declared, stored = declaration(chain_model(4))
    clear_classes()
    composed = compose(declared)
    return composed, {composed.names.path(key): value for key, value in stored.items()}


def test_stage_7s_chain_read_for_one_value_makes_part_of_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(EAGER, raising=False)  # the eager test mode makes everything at once
    composed, _ = fresh_chain()
    point = composed.design_space()
    scopes = inspection.model(point).scopes
    assert isinstance(point.mm0.query(MatMulKernel.cycles), Unresolved)  # type: ignore[attr-defined]
    read = scopes.counts()
    assert read["scopes made"] < 225 // 8
    assert read["scopes in play"] < read["scopes made"]
    assert read["members evaluated"] < len(list(scopes.all_addresses())) // 100


# The members evaluated (and the constants among them) after the commit, and after
# ``viable`` and the change (the structure's nodes resolved were 1366 (96) and 1464 (96),
# 1371 and 1463 since T4c's piece 1 placed the required inputs' ports by construction).
# 1153 (96) and 1224 (96) before each channel side became one value: each port bound now
# evaluates its label and its side, and the channels' label constants are gone. 1180 (80)
# and 1259 (80) before the composer: each fact a member is given is a Param at the root
# now (59 constants of the members evaluated become 59 root Params evaluated), and each
# kernel's side a channel takes is a derived value at the root, the side labelled below its
# member (20 evaluated).
EVALUATED = ((1200, 21), (1279, 21))


def test_stage_7s_chain_committed_from_scratch_evaluates_what_the_change_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``commit`` of the chain's stored values on a design space made from scratch, before
    any trial or ``viable``. Listing its choices (``inspection.decisions``, in the flow
    order) makes every scope's record and evaluates no member; the change evaluates what it
    and its checks read, each member as its scope reads it. An absent scope's validity is
    read from its guard and its choices' presence from their guards, so no scope absent at
    the baseline has a member evaluated for either, and none of unknown presence in the
    design space before the change either: the scopes with a member evaluated (in play)
    are exactly the 38 present at the baseline (the line's present scopes). Each MatMul's
    ``compute.int8_dsp58`` case, of unknown presence before the change, has none evaluated.
    Then ``viable`` of the middle MatMul's choices and one change of its ``pe``: what trials
    evaluate more (the middle MatMul's ``int8_dsp58`` case, which a trial selects, and the
    output adapters of the channels beside it, ``a1`` and ``h2``, which read its ports'
    streams)."""
    monkeypatch.delenv(EAGER, raising=False)
    composed, stored = fresh_chain()
    point = composed.design_space()
    scopes = inspection.model(point).scopes
    committed = commit(point, stored)
    counted = scopes.counts()
    assert {key: counted[key] for key in ("scopes made", "scopes in play")} == {
        "scopes made": 225,
        "scopes in play": 38,
    }
    assert (counted["members evaluated"], counted["constants evaluated"]) == EVALUATED[0]
    in_play = set(scopes.in_play())

    def answer(of: Space, index: Address) -> QueryResult[object]:
        """``index``'s answer in the design space ``of`` holds, in its model's session."""
        ds = held(of).ds
        return ask(ds.model, lambda session: evaluation(ds, session).result(index))

    def holds(guard: Address | None, of: Space) -> bool:
        return guard is None or answer(of, guard) == Available(True)

    def present(record: ScopeRecord | None) -> bool:
        while record is not None:
            if not holds(record.guard, committed):
                return False
            record = record.parent
        return True

    records = scopes.ordered()
    present_scopes = {record for record in records if present(record)}
    assert len(present_scopes) == 38
    assert in_play == present_scopes
    unknown = {record for record in records if record.name.endswith(".compute.int8_dsp58")}
    assert {record.name for record in unknown} == {
        f"mm{layer}.compute.int8_dsp58" for layer in range(4)
    }
    assert not unknown & in_play
    for record in unknown:
        guard = record.guard
        assert guard is not None and isinstance(answer(point, guard), Unresolved)
    handles = {item.key: item.reference for item in inspection.decisions(committed)}
    middle = [key for key in handles if key.startswith("mm2.")]
    assert len(middle) == 7
    inspection.viable(committed, [handles[key] for key in middle])
    counted = scopes.counts()
    assert (counted["scopes made"], counted["scopes in play"]) == (225, 43)
    assert (counted["members evaluated"], counted["constants evaluated"]) == EVALUATED[1]
    assert {record.name for record in set(scopes.in_play()) - in_play} == {
        "mm2.compute.int8_dsp58",
        "a1.output_adapter",
        "a1.output_adapter.vpc",
        "h2.output_adapter",
        "h2.output_adapter.vpc",
    }
    report = committed.try_with_choices({handles["mm2.pe"]: 2})
    assert report.accepted
    counted = scopes.counts()
    assert (counted["scopes made"], counted["scopes in play"]) == (225, 43)
    assert (counted["members evaluated"], counted["constants evaluated"]) == EVALUATED[1]
