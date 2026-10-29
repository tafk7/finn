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
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pytest
from core.space._domains_support import domain_differences
from core.space._templates_support import assert_reads_alike, defined, links, reached
from core.space._whole_support import differences
from support.chain import MEMBERS, chain
from support.models import chain_model

import finn.custom_op
import finn.kernels
from finn.core.graph.annotations.platform import read_target
from finn.core.space import Group, Space, inspection
from finn.core.space._running.evaluate import evaluation
from finn.core.space._scopes import EAGER, Address, ScopeRecord, Scopes
from finn.core.space._template import link_template
from finn.core.space.compiler import compile_model
from finn.core.space.occurrence import ask, held
from finn.core.space.results import Available, QueryResult, Unresolved
from finn.custom_op.kernels.network import declaration
from finn.custom_op.kernels.shell import in_shell
from finn.kernels.budgets import _ClockRequirement, _DoubledAny, _MoverSum, _PresentedSum, _Sum
from finn.kernels.composer import Composed, clear_classes, compose
from finn.kernels.configure import commit
from finn.kernels.control import ControlModel
from finn.kernels.dotp import Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.fifo import FifoKernel
from finn.kernels.input_generator import InputGeneratorKernel
from finn.kernels.matmul import MatMulKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.port_model import (
    IndexedPort,
    PortModel,
    ScheduledPort,
    SequencePort,
    port_surfaces,
)
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.vpc import VpcKernel
from finn.platform.shells import shell_row

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
        chain_class(),
    ]
    differences = [found for found in map(domain_differences, classes) if found is not None]
    assert not differences, "\n".join(differences)


def test_the_chains_members_read_alike_in_any_order_their_scopes_are_made_in() -> None:
    net = chain()  # configured
    classes = {type(net.member(name)) for name in MEMBERS}
    assert assert_reads_alike(classes) == 3  # the MatMul, the thresholding, the channel


def _port_scopes(records: Iterable[ScopeRecord]) -> set[ScopeRecord]:
    return {record for record in records if issubclass(record.space_type, PortModel)}


PORT_TYPES: dict[type[Space], dict[str, type[PortModel]]] = {
    FifoKernel: {"input": SequencePort, "output": SequencePort},
    InputGeneratorKernel: {"input": SequencePort, "output": SequencePort},
    VpcKernel: {"input": SequencePort, "output": SequencePort},
    ThresholdingAxiKernel: {"input": IndexedPort, "output": ScheduledPort, "set": SequencePort},
    PackedDotpKernel: {"x": IndexedPort, "w": IndexedPort, "y": ScheduledPort},
    Int8Dsp58DotpKernel: {"x": IndexedPort, "w": IndexedPort, "y": ScheduledPort},
    MemStreamKernel: {"set": SequencePort, "output": SequencePort},
}
GROUP_TYPES = {_Sum, _PresentedSum, _MoverSum, _DoubledAny, _ClockRequirement, ControlModel}


def _group_scopes(records: Iterable[ScopeRecord]) -> set[ScopeRecord]:
    groups = {record for record in records if record.compiled.validity is None}
    for record in groups:
        assert record.space_type in GROUP_TYPES
        assert record.owning_space is not record
        assert record.owning_space.compiled.validity is not None
    return groups


def _real_nonport_scopes(records: Iterable[ScopeRecord]) -> set[ScopeRecord]:
    return {
        record
        for record in records
        if record.compiled.validity is not None and not issubclass(record.space_type, PortModel)
    }


def _assert_native_scope_ownership(records: Iterable[ScopeRecord]) -> set[ScopeRecord]:
    """Every native Port subtype owns its scope, below the exact declared parent ports."""
    ports = _port_scopes(records)
    parents: set[ScopeRecord] = set()
    for record in ports:
        assert record.owning_space is record and not record.compiled.choices
        parent = record.parent
        assert parent is not None and parent.owning_space is parent
        assert parent.space_type in PORT_TYPES
        expected = PORT_TYPES[parent.space_type]
        assert record.space_type is expected[record.name.rsplit(".", 1)[-1]]
        assert set(port_surfaces(parent.space_type)) == set(expected)
        parents.add(parent)
    assert {record.name for record in ports} == {
        f"{parent.name}.{port}"
        for parent in parents
        for port in PORT_TYPES[cast(type[Space], parent.space_type)]
    }
    assert len(ports) == sum(
        len(PORT_TYPES[cast(type[Space], parent.space_type)]) for parent in parents
    )
    return ports


def test_stage_7s_chain_of_eight_kernels_reads_alike_in_any_order_its_scopes_are_made_in() -> None:
    whole = chain_class()
    assert_reads_alike([whole])
    templates = reached(link_template(whole))
    # Real Space classes, native Port classes and lexical Group classes are distinct.
    assert (
        len(
            [
                kind
                for kind in templates
                if issubclass(kind, Space) and not issubclass(kind, PortModel)
            ]
        )
        == 17
    )
    assert {kind for kind in templates if issubclass(kind, Group)} == GROUP_TYPES
    assert {kind for kind in templates if issubclass(kind, PortModel)} == {
        SequencePort,
        ScheduledPort,
        IndexedPort,
    }
    records = compile_model(whole).scopes.ordered()
    ports = _assert_native_scope_ownership(records)
    # All original226 real non-port scopes remain; the assembled native Ports and
    # Control/budget Group namespaces are accounted separately, with ownership checked.
    assert len(_real_nonport_scopes(records)) == 226
    assert len(ports) == 270
    assert len(_group_scopes(records)) == 55


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
    classes.extend(type(net.member(name)) for name in MEMBERS)
    assert _answers_as_the_line(list(dict.fromkeys(classes))) > 50


def test_stage_7s_chain_of_eight_kernels_answers_as_the_line() -> None:
    whole = chain_class()
    # The choices compared: each MatMul's folding is its own, not each core's; each
    # channel's source's two (absent without a value) and its one adapter memory, and no
    # source, mover or output adapter selector, nor an input adapter's, which its steps
    # pick.
    assert _answers_as_the_line([whole]) == 130


def fresh_chain() -> tuple[Composed, dict[str, object]]:
    """Stage 7's chain at 8 kernels composed as one Space with a model of its own (the
    graph layer's declaration, in its shell, composed), and its stored values by path:
    the graph layer binds a partition of the same class as it reads the stored values
    (``in_shell``), so the composer's classes are dropped before it is composed."""
    model = chain_model(4)
    nodes = list(model.graph.node)
    target = read_target(model)
    declared = declaration(model, nodes, movers=shell_row(target.shell, target.board).movers)
    stored = in_shell(model, nodes).chosen()
    clear_classes()
    composed = compose(declared)
    return composed, {composed.names.path(key): value for key, value in stored.items()}


def chain_class() -> type[Space]:
    """Stage 7's chain at 8 kernels: its partition's class, as the graph layer binds it in
    its shell."""
    model = chain_model(4)
    return in_shell(model, model.graph.node).composed.space_type


def test_stage_7s_chain_read_for_one_value_makes_part_of_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(EAGER, raising=False)  # the eager test mode makes everything at once
    composed, _ = fresh_chain()
    point = composed.design_space()
    scopes = inspection.model(point).scopes
    assert isinstance(point.mm0.query(MatMulKernel.cycles), Unresolved)  # type: ignore[attr-defined]
    read = scopes.counts()
    # Keep the original real-owner ratio, and bound unevaluated Port/Group allocation
    # separately before any whole-model address walk can make additional records.
    made = tuple(scopes.registry.records.values())
    assert len(_real_nonport_scopes(made)) < 225 // 8
    ports = _assert_native_scope_ownership(made)
    assert {record.name for record in ports} == {
        f"mm0.compute.{core}.{port}"
        for core in ("int8_dsp58", "packed")
        for port in ("x", "w", "y")
    }
    assert not _group_scopes(made)
    assert read["scopes in play"] < read["scopes made"]
    assert read["members evaluated"] < len(list(scopes.all_addresses())) // 100


# The members evaluated (and the constants among them) after the commit, and after
# ``viable`` and the change (the structure's nodes resolved were 1366 (96) and 1464 (96),
# 1371 and 1463 since the required inputs' ports are placed by construction).
# 1153 (96) and 1224 (96) before each channel side became one value: each port bound now
# evaluates its label and its side, and the channels' label constants are gone. 1180 (80)
# and 1259 (80) before the composer: each fact a member is given is a Param at the root
# now (59 constants of the members evaluated become 59 root Params evaluated), and each
# kernel's side a channel takes is a derived value at the root, the side labelled below its
# member (20 evaluated). Then 1200 (21) and 1279 (21) became these, with facts known and
# values compared by ``==``: 21 optional facts left out are constants (None) their readers
# now read as values (the 9 activation channels' ``contents``, the 4 weight channels'
# ``index_tensor``, MatMul's ``window``, a thresholding's ``set_tensor``); each weight
# channel's ``source.contents`` (4), a required formal bound to the channel's optional
# ``contents``, read as itself; each thresholding's ``table_dtype`` (4); and MatMul's
# weights one fact: per MatMul, its root Params ``w_contents`` and ``weights_dtype`` one
# (``weights``), and ``weighted``, ``carried_weights``' guard and its packed core's
# ``w_placed`` and its ``w_side`` and ``w_stream`` guards gone (-24; -27 after ``viable``,
# the trial's int8 core's three as well). Then 1205 (42) and 1281 (42) became these, with a
# computation's guard its body's early return: the guard nodes of guarded computations are
# gone (-33): each channel's ``adapter_uram_present`` guard (13), and per MatMul (4 each) its
# ``native_readable`` and ``stored_weights`` guards, its packed core's ``w_access`` and
# ``x_access`` guards and its thresholding's ``input_access`` guard; and
# ``adapter_uram_present``, reading the style it judges first, now evaluates
# ``adapter_ram_style`` where no adapter is placed (+9: the 4 ``h`` and 4 ``w`` channels and
# ``y``; the style answers inapplicable, and the rule with it). After ``viable``, -35 + 9:
# the trial's int8 core's ``w_access`` and ``x_access`` guards as well.
# Historical early-Port totals were ((1157,42),(1221,42)) for non-port reads,
# ((100,52),(140,70)) for Ports. The assembled native Dotp/Thresholding/MemStream
# models move reads to their actual owners; Group primitives use the owning Space.
# Exact counts observed on assembled edb7cbd7d, not a readiness or timing budget.
NONPORT_EVALUATED = ((1017, 42), (1061, 42))
PORT_EVALUATED = ((540, 304), (637, 356))
EVALUATED = ((1557, 346), (1698, 398))


def _assert_evaluated(scopes: Scopes, stage: int) -> None:
    """Exact functional demand accounting by real owner, without a readiness walk."""
    with scopes.lock:
        evaluated = tuple(scopes.registry.evaluated)
    port_reads = [
        address for address in evaluated if issubclass(address.scope.space_type, PortModel)
    ]
    nonport_reads = [
        address
        for address in evaluated
        if address.scope.compiled.validity is not None
        and not issubclass(address.scope.space_type, PortModel)
    ]
    # Group primitives are addresses of the owning Space, never independent reads.
    assert not [address for address in evaluated if address.scope.compiled.validity is None]
    assert (
        len(port_reads),
        sum(address.kind == "const" for address in port_reads),
    ) == PORT_EVALUATED[stage]
    assert (
        len(nonport_reads),
        sum(address.kind == "const" for address in nonport_reads),
    ) == NONPORT_EVALUATED[stage]
    counted = scopes.counts()
    assert (counted["members evaluated"], counted["constants evaluated"]) == EVALUATED[stage]


def test_stage_7s_chain_committed_from_scratch_evaluates_what_the_change_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``commit`` of the chain's stored values on a design space made from scratch, before
    any trial or ``viable``. Listing its model's choices (``inspection.decisions`` of the
    model, in the flow order, by which ``commit`` names them) makes every scope's record and
    evaluates no member; the change evaluates what it and its checks read, each member as
    its scope reads it. An absent scope's validity is read from its guard and its choices'
    presence from their guards, so no scope absent at the baseline has a member evaluated
    for either, and none of unknown presence in the design space before the change
    either: in-play scopes are the38 original non-port owners plus40 native Port children,
    all present at the baseline; Group namespaces have no independent validity. Each
    MatMul's ``compute.int8_dsp58`` case, of
    unknown presence before the change, has none evaluated.
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
    made_ports = _assert_native_scope_ownership(scopes.registry.records.values())
    in_play = set(scopes.in_play())
    active_ports = _port_scopes(in_play)
    assert len(_real_nonport_scopes(scopes.registry.records.values())) == 225
    assert len(_group_scopes(scopes.registry.records.values())) == 17
    assert counted["scopes made"] == 225 + 270 + 17
    assert counted["scopes in play"] - len(active_ports) == 38
    assert len(made_ports) == 270 and len(active_ports) == 40
    _assert_evaluated(scopes, 0)

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
    present_scopes = {
        record for record in records if record.compiled.validity is not None and present(record)
    }
    assert len(_real_nonport_scopes(present_scopes)) == 38
    assert len(_port_scopes(present_scopes)) == 40
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
    # the middle MatMul's choices listed: its ``realization`` (``when=depthwise``) is not,
    # its facts alone rule it out
    assert len(middle) == 6 and "mm2.realization" not in middle
    inspection.viable(committed, [handles[key] for key in middle])
    counted = scopes.counts()
    assert len(_real_nonport_scopes(scopes.registry.records.values())) == 225
    assert len(_group_scopes(scopes.registry.records.values())) == 17
    assert counted["scopes made"] == 225 + 270 + 17
    assert counted["scopes in play"] - len(_port_scopes(scopes.in_play())) == 43
    assert len(_port_scopes(scopes.in_play())) == 47
    _assert_evaluated(scopes, 1)
    additional = set(scopes.in_play()) - in_play
    assert {record.name for record in additional - _port_scopes(additional)} == {
        "mm2.compute.int8_dsp58",
        "a1.output_adapter",
        "a1.output_adapter.vpc",
        "h2.output_adapter",
        "h2.output_adapter.vpc",
    }
    assert {record.name for record in _port_scopes(additional)} == {
        *(f"mm2.compute.int8_dsp58.{port}" for port in ("x", "w", "y")),
        *(
            f"{parent}.{port}"
            for parent in ("a1.output_adapter.vpc", "h2.output_adapter.vpc")
            for port in ("input", "output")
        ),
    }
    assert not _group_scopes(additional)
    report = committed.try_with_choices({handles["mm2.pe"]: 2})
    assert report.accepted
    counted = scopes.counts()
    assert len(_real_nonport_scopes(scopes.registry.records.values())) == 225
    assert len(_group_scopes(scopes.registry.records.values())) == 17
    assert counted["scopes made"] == 225 + 270 + 17
    assert counted["scopes in play"] - len(_port_scopes(scopes.in_play())) == 43
    assert len(_port_scopes(scopes.in_play())) == 47
    _assert_evaluated(scopes, 1)
