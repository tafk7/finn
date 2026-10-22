# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Templates (``finn.core.space._template``) on the kernels: the structure of every Space
class the kernels, the KernelOps and the test support define, of the Chain's members and of
stage 7's chain of eight kernels as one Space, fully resolved, equal to its eager form
``expand`` by key (``core.space._templates_support``; the engine's own corpus and edge
cases are ``core.space.test_templates``); their domain rule, composed per template, equal
to the line's on ``expand`` (``core.space._domains_support``); and their answers to the
questions about the whole design space, from templates, equal to the line's
(``core.space._whole_support``; the engine's are ``core.space.test_whole``)."""

from __future__ import annotations

import pkgutil
from pathlib import Path
from typing import Any

import pytest
from core.space._domains_support import domain_differences
from core.space._templates_support import assert_expands, defined, links, reached
from core.space._whole_support import differences
from support.chain import MEMBERS, chain
from support.one_space import chain_space

import finn.custom_op
import finn.kernels
from finn.core.space import Space, design_space, inspection
from finn.core.space._structure import EAGER, expand
from finn.core.space._template import link_template
from finn.core.space._running.evaluate import evaluation
from finn.core.space.occurrence import ask, held
from finn.core.space.results import Available, QueryResult, Unresolved
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel

SUPPORT = Path(__file__).resolve().parents[1] / "support"


def test_the_test_supports_classes_resolve_as_they_expand() -> None:
    modules = [f"support.{path.stem}" for path in sorted(SUPPORT.glob("*.py"))]
    classes = [space_type for space_type in defined(modules) if links(space_type)]
    assert len(classes) > 10
    assert_expands(classes)


def _kernels() -> list[type[Space]]:
    modules = [
        info.name
        for package in (finn.kernels, finn.custom_op)
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + ".")
    ]
    return list(dict.fromkeys(defined(modules)))


def test_the_kernels_resolve_as_they_expand() -> None:
    classes = _kernels()
    assert len(classes) > 15
    assert assert_expands(classes) == len(classes)


def test_the_kernels_domains_compose_as_the_line_checked_them() -> None:
    modules = [f"support.{path.stem}" for path in sorted(SUPPORT.glob("*.py"))]
    classes = [*_kernels(), *(cls for cls in defined(modules) if links(cls)), chain_space(4)[0]]
    differences = [found for found in map(domain_differences, classes) if found is not None]
    assert not differences, "\n".join(differences)


def test_the_chains_members_resolve_as_they_expand() -> None:
    net = chain()  # configured: its channels bound
    classes = {inspection.model(net.member(name)).space_type for name in MEMBERS}
    assert assert_expands(classes) == 3  # the MatMul, the thresholding, the channel


def test_stage_7s_chain_of_eight_kernels_resolves_as_it_expands() -> None:
    whole, _ = chain_space(4)
    assert_expands([whole])
    templates = reached(link_template(whole))
    # Fifteen classes, and the composite (a port is a declaration of its kernel, no class).
    assert len(templates) == 16
    # A channel's source and mover are placements: compiled where their guard's formal is
    # not supplied too (their members inapplicable there), so each of the 9 channels
    # without a value has its source's scope, and each of the 13 its mover's, where the
    # one-case selectors compiled none; a direct transport places none.
    assert len(expand(whole).scopes) == 225


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
    whole, _ = chain_space(4)
    # The choices compared: each MatMul's folding is its own, not each core's; each
    # channel's source's two (absent without a value) and its one adapter memory, and no
    # source, mover or output adapter selector, nor an input adapter's, which its steps
    # pick.
    assert _answers_as_the_line([whole]) == 130


def test_stage_7s_chain_read_for_one_value_makes_part_of_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(EAGER, raising=False)  # the eager test mode makes everything at once
    whole, stored = chain_space(4)
    point = design_space(whole())
    structure = inspection.model(point).structure
    assert structure is not None
    assert isinstance(point.mm0.query(MatMulKernel.cycles), Unresolved)  # type: ignore[attr-defined]
    read = structure.counts()
    table = expand(whole)
    assert read["scopes addressed"] < len(table.scopes) // 8
    assert read["scopes in play"] < read["scopes addressed"]
    assert read["nodes resolved"] < len(table.nodes) // 100


def test_stage_7s_chain_committed_from_scratch_resolves_what_the_change_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``commit`` of the chain's stored values on a design space made from scratch, before
    any trial or ``viable``. Listing its choices (``inspection.decisions``, in the flow
    order) makes every scope's record and resolves no node; the change resolves what it
    and its checks read, each node as its structure resolves it. An absent scope's validity
    is read from its guard and its choices' presence from their guards, so no scope absent
    at the baseline has a node resolved for either, and none of unknown presence in the
    design space before the change either: the scopes with a node resolved are exactly the
    38 present at the baseline (the line's present scopes). Each MatMul's
    ``compute.int8_dsp58`` case, of unknown presence before the change, has none resolved.
    Then ``viable`` of the middle MatMul's choices and one change of its ``pe``: what trials
    resolve more (the middle MatMul's ``int8_dsp58`` case, which a trial selects, and the
    output adapters of the channels beside it, ``a1`` and ``h2``, which read its ports'
    streams)."""
    monkeypatch.delenv(EAGER, raising=False)
    whole, stored = chain_space(4)
    point = design_space(whole())
    structure = inspection.model(point).structure
    assert structure is not None
    committed = commit(point, stored)
    assert structure.counts() == {
        "scopes addressed": 225,
        "scopes in play": 38,
        "nodes resolved": 1366,
        "const nodes resolved": 96,
    }
    in_play = set(structure._in_play)

    def answer(of: Space, index: int) -> QueryResult[object]:
        """``index``'s answer in the design space ``of`` holds, in its model's session."""
        ds = held(of).ds
        return ask(ds.model, lambda session: evaluation(ds, session).result(index))

    def holds(guard: int | None, of: Space) -> bool:
        return guard is None or answer(of, guard) == Available(True)

    def present(index: int) -> bool:
        record: Any = structure.records[index]
        while record is not None:
            if not holds(structure.guard_of(record), committed):
                return False
            record = record.parent
        return True

    present_scopes = {record.index for record in structure.records if present(record.index)}
    assert len(present_scopes) == 38
    assert in_play == present_scopes
    unknown = {
        record.index for record in structure.records if record.name.endswith(".compute.int8_dsp58")
    }
    assert {structure.records[index].name for index in unknown} == {
        f"mm{layer}.compute.int8_dsp58" for layer in range(4)
    }
    assert not unknown & in_play
    for index in unknown:
        guard = structure.guard_of(structure.records[index])
        assert guard is not None and isinstance(answer(point, guard), Unresolved)
    handles = {item.key: item.reference for item in inspection.decisions(committed)}
    middle = [key for key in handles if key.startswith("mm2.")]
    assert len(middle) == 7
    inspection.viable(committed, [handles[key] for key in middle])
    assert structure.counts() == {
        "scopes addressed": 225,
        "scopes in play": 43,
        "nodes resolved": 1464,
        "const nodes resolved": 96,
    }
    assert {structure.records[index].name for index in set(structure._in_play) - in_play} == {
        "mm2.compute.int8_dsp58",
        "a1.output_adapter",
        "a1.output_adapter.vpc",
        "h2.output_adapter",
        "h2.output_adapter.vpc",
    }
    report = committed.try_with_choices({handles["mm2.pe"]: 2})
    assert report.accepted
    assert structure.counts() == {
        "scopes addressed": 225,
        "scopes in play": 43,
        "nodes resolved": 1464,
        "const nodes resolved": 96,
    }
