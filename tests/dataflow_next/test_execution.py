# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Independent current-engine networks versus data requests; no self-oracle.

Fixtures construct engine stage values directly. Request programs independently
state masks/iteration words, logical IDs/counts and profiles. Full traces, not
just a predicate, must agree; hand checks exercise direct/pipeline/fork rules.
"""

from __future__ import annotations

import pytest

from dataclasses import replace
from dataflow_next.execution_cases import ENCODING, chain, fixture, variables
from typing import cast

from finn.dataflow_next import timing as engine
from finn.dataflow_next.execution import (
    Criterion,
    Execution,
    Flags,
    HostCriterion,
    IndexStream,
    Program,
    capabilities,
    events,
    execute,
    size,
    validate_result,
)
from finn.dataflow_next.logical import Available, Rejected, Source, Unsupported
from finn.dataflow_next.profiles import (
    Capacity,
    ChannelProfile,
    Evidence,
    Launch,
    TimedPlan,
    Unavailable,
    configuration_digest,
)
from finn.dataflow_next.tensor import Tensor
from finn.dataflow_next.timed_records import dump, load
from finn.dataflow_next.traversal import BeatSequence, LevelEnd, grouped, vector_major


@pytest.mark.parametrize("kind", ["kernel", "width", "reorder", "insert", "drop"])
@pytest.mark.parametrize("rules", [engine.FINNLIB, engine.ONE_CYCLE])
def test_complete_stage_family_traces_against_independently_constructed_engine(
    kind: str, rules: engine.Rules
) -> None:
    request, reference = chain(kind, rules=rules)
    result = execute(request)
    assert isinstance(result, Available), result
    assert result.value.run.to_run() == engine.run(reference, rules, request.execution.frames)
    assert load(dump(request)) == request
    assert load(dump(result.value)) == result.value
    assert result.value.words == reference.words
    assert result.value.met is None and not result.value.conditional


def test_memory_and_merge_source_ports_have_distinct_identity_and_trace() -> None:
    stages = (
        engine.Source("src", (0, 1), 2, 8),
        engine.Memory("mem"),
        engine.MergeStage("merge", (0, 1, 1), 6),
        engine.Sink("sink"),
    )
    channels = (
        engine.Channel("a", "src", "merge", None, 0, 0),
        engine.Channel("b", "mem", "merge", None, 0, 1),
        engine.Channel("c", "merge", "sink"),
    )
    programs = (
        Program("src", "source", (), ("src.out",), order=IndexStream((0,), 2, 1)),
        Program("mem", "memory", (), ("mem.out",)),
        Program("merge", "merge", ("merge.a", "merge.b"), ("merge.out",), pattern=(0, 1, 1)),
        Program("sink", "sink", ("sink.in",), ()),
    )
    request, reference = fixture(stages, channels, programs)
    found = execute(request)
    assert isinstance(found, Available)
    assert found.value.run.to_run() == engine.run(reference, engine.FINNLIB, 4)
    assert {s.id for s in request.timed.plan.sources} >= {"src.out.source", "mem.out.source"}


@pytest.mark.parametrize("capacity", [0, 1, None])
def test_atomic_fork_and_direct_deadlock_witness(capacity: int | None) -> None:
    stages = (
        engine.Source("src", (0, 1), 2, 8),
        engine.Fork("fork", 2),
        engine.Kernel("join", 3, ((0, 1), (1, 2)), (), 0),
    )
    channels = (
        engine.Channel("a", "src", "fork", capacity),
        engine.Channel("b", "fork", "join", capacity, 0, 0),
        engine.Channel("c", "fork", "join", capacity, 1, 1),
    )
    programs = (
        Program("src", "source", (), ("src.out",), order=IndexStream((0,), 2, 1)),
        Program("fork", "fork", ("fork.in",), ("fork.a", "fork.b")),
        Program(
            "join",
            "kernel",
            ("join.a", "join.b"),
            (),
            3,
            (IndexStream((0, 1)), IndexStream((1, 2))),
        ),
    )
    request, reference = fixture(stages, channels, programs)
    found = execute(request)
    assert isinstance(found, Available)
    assert found.value.run.to_run() == engine.run(reference, engine.FINNLIB, 4)
    if capacity == 0:
        assert found.value.run.deadlocked


def test_pipeline_direct_reservation_and_hand_written_cycles() -> None:
    request, reference = chain("kernel", 0)
    found = execute(request)
    assert isinstance(found, Available)
    assert found.value.run.to_run() == engine.run(reference, engine.FINNLIB, 4)
    assert found.value.run.to_run().written("b")[:4] == (0, 1, 3, 5)
    assert found.value.run.to_run().taken("b")[:4] == (2, 3, 5, 7)


def test_absence_and_unreliable_scenario_never_fabricate_a_guarantee() -> None:
    request, _ = chain("kernel")
    missing = replace(request.timed.attachment.profiles[1], latency=None)
    t = replace(
        request.timed,
        attachment=replace(
            request.timed.attachment,
            profiles=(
                request.timed.attachment.profiles[0],
                missing,
                request.timed.attachment.profiles[2],
            ),
        ),
    )
    found = execute(replace(request, timed=t))
    assert isinstance(found, Unavailable) and ("middle", "latency") in found.missing
    unreliable = Evidence(
        "python-event-1", "1", "unreliable", ("test guess",), ("pipeline estimate unverified",)
    )
    t = replace(
        request.timed,
        attachment=replace(
            request.timed.attachment,
            profiles=tuple(
                replace(p, evidence=unreliable) for p in request.timed.attachment.profiles
            ),
        ),
    )
    assert isinstance(execute(replace(request, timed=t)), Unavailable)
    found = execute(
        replace(request, timed=t, execution=replace(request.execution, conditional=True))
    )
    assert isinstance(found, Available) and found.value.conditional
    assert any("unverified" in s for s in found.value.assumptions)


def test_frozen_finite_reference_and_sizing_match_independent_default() -> None:
    request, reference = chain("kernel")
    request = replace(request, criterion=Criterion())
    found = execute(request)
    assert isinstance(found, Available) and found.value.reference is not None
    expected = engine.period_kept(reference, engine.FINNLIB, 4)
    assert found.value.met == expected(found.value.run.to_run())
    assert found.value.reference.deadlines == engine.due(reference, engine.FINNLIB, 4).by
    variable = variables(request)
    sized = size(variable, channels=("a", "b"))
    assert isinstance(sized, Available), sized
    independent = engine.minimal(reference, engine.FINNLIB, 4, channels=("a", "b"))
    assert {name: cap.words for name, cap in sized.value.capacities} == independent.found
    assert isinstance(validate_result(variable, sized.value, sizing_order=("a", "b")), Available)
    changed = replace(
        variable,
        capacities=(("a", Capacity("fixed", 1)), ("b", Capacity("fixed", 1))),
        reference=found.value.reference,
    )
    trial = execute(changed)
    assert isinstance(trial, Available)
    assert trial.value.reference == found.value.reference
    assert trial.value.met == expected(
        engine.run(reference.at({"a": 1, "b": 1}), engine.FINNLIB, 4)
    )
    bad = replace(found.value.reference, period=17)
    assert isinstance(execute(replace(changed, reference=bad)), Rejected)
    t = replace(
        request.timed,
        attachment=replace(
            request.timed.attachment,
            profiles=tuple(
                replace(p, latency=3, internal_words=3) if p.node == "middle" else p
                for p in request.timed.attachment.profiles
            ),
        ),
    )
    assert isinstance(execute(replace(request, timed=t, reference=found.value.reference)), Rejected)


def test_host_criterion_runs_after_whole_runs_and_exceptions_keep_ownership() -> None:
    request, reference = chain("kernel")
    criterion = Criterion("host", host_name="custom", host_version="1", monotone=True)
    request = replace(request, criterion=criterion)
    seen: list[engine.Run] = []

    def custom(run: engine.Run) -> bool:
        seen.append(run)
        return run.on_time

    found = execute(request, host=HostCriterion("custom", "1", custom))
    assert isinstance(found, Available) and len(seen) == 1
    assert seen[0] == engine.run(reference, engine.FINNLIB, 4)
    assert isinstance(execute(request), Unsupported)
    assert isinstance(
        size(
            replace(request, criterion=replace(criterion, monotone=False)),
            host=HostCriterion("custom", "1", custom),
        ),
        Unsupported,
    )

    def broken(run: engine.Run) -> bool:
        raise ValueError("author-owned custom failure")

    with pytest.raises(ValueError, match="author-owned"):
        execute(request, host=HostCriterion("custom", "1", broken))
    with pytest.raises(TypeError):
        dump(HostCriterion("custom", "1", custom))


def test_hardware_bounded_targets_leave_fixed_other_channels_and_reference_unchanged() -> None:
    request, reference = chain("kernel")
    policies = (
        ChannelProfile("a", "clk", Capacity("unspecified"), 2, 8),
        ChannelProfile("b", "clk", Capacity("fixed", 4), 1, 4),
    )
    request = replace(
        request,
        criterion=Criterion(),
        timed=replace(
            request.timed, attachment=replace(request.timed.attachment, channels=policies)
        ),
    )
    assert isinstance(execute(request), Unavailable)
    found = size(request, channels=("a",))
    assert isinstance(found, Available), found
    assert dict(found.value.capacities)["a"].words >= 2  # type: ignore[operator]
    assert dict(found.value.capacities)["b"] == Capacity("fixed", 4)
    assert isinstance(
        execute(replace(request, capacities=(("a", Capacity("fixed", 1)),))), Rejected
    )
    assert found.value.reference is not None
    assert found.value.reference.free.to_run() == engine.run(
        reference.unbounded(), engine.FINNLIB, 4
    )


@pytest.mark.parametrize(
    "fault", ["ids", "latency", "clock", "launch", "capacity", "mask", "budget", "internal"]
)
def test_refusal_and_unsupported_controls(fault: str) -> None:
    request, _ = chain("kernel")
    if fault == "ids":
        request = replace(
            request,
            programs=(replace(request.programs[0], outputs=("absent",)), *request.programs[1:]),
        )
    elif fault == "mask":
        request = replace(
            request,
            programs=(
                request.programs[0],
                replace(request.programs[1], reads=(IndexStream((0, 1, 2, 5)),)),
                request.programs[2],
            ),
        )
    elif fault == "budget":
        request = replace(request, execution=replace(request.execution, max_moves=1))
    elif fault == "capacity":
        assert request.timed.attachment.rules is not None
        t = replace(
            request.timed,
            attachment=replace(
                request.timed.attachment,
                rules=replace(request.timed.attachment.rules, direct="register"),
            ),
        )
        request = replace(request, timed=t, capacities=(("a", Capacity("fixed", 0)),))
    else:
        p = request.timed.attachment.profiles[1]
        p = (
            replace(p, latency=None)
            if fault == "latency"
            else replace(p, launch=Launch(2, 0))
            if fault == "launch"
            else replace(p, internal_words=None)
            if fault == "internal"
            else replace(p, clock="other")
        )
        a = replace(
            request.timed.attachment,
            profiles=(
                request.timed.attachment.profiles[0],
                p,
                request.timed.attachment.profiles[2],
            ),
        )
        if fault == "clock":
            with pytest.raises(ValueError, match="clock"):
                replace(request.timed, attachment=a)
            return
        request = replace(request, timed=replace(request.timed, attachment=a))
    assert not isinstance(execute(request), Available)


def test_scalar_integer_and_schema_boundaries_are_checked_without_native_objects() -> None:
    with pytest.raises(ValueError):
        IndexStream((2**63,))
    with pytest.raises(ValueError):
        Flags((1,))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Execution(True)
    request, _ = chain("kernel")
    with pytest.raises(TypeError):
        dump(
            engine.run(
                engine.Network(
                    (engine.Source("s", (0,), 1, 1), engine.Sink("t")),
                    (engine.Channel("e", "s", "t"),),
                ),
                engine.FINNLIB,
                2,
            )
        )
    assert not any(
        word in dump(request) for word in (b'"callback"', b'"_Service"', b'"Space"', b'"Kernel"')
    )


def test_criterion_horizon_and_reference_period_preconditions_remain_refusals() -> None:
    request, _ = chain("kernel")
    assert isinstance(
        execute(replace(request, criterion=Criterion(), execution=Execution(1))), Rejected
    )
    # A source of one word per tick, into a two-tick kernel, has no default
    # period reference even when a FIFO hides all backlog for these frames.
    stages = (engine.Source("src", (0,), 1, 1), engine.Kernel("sink", 2, ((0,),), (), 0))
    programs = (
        Program("src", "source", (), ("out",), order=IndexStream((0,))),
        Program("sink", "kernel", ("in",), (), 2, (IndexStream((0,)),)),
    )
    request, _ = fixture(stages, (engine.Channel("e", "src", "sink"),), programs)
    assert isinstance(execute(replace(request, criterion=Criterion())), Rejected)


def test_capabilities_and_result_decode_keep_exact_complete_trace_ids() -> None:
    request, _ = chain("kernel")
    found = execute(request)
    assert isinstance(found, Available)
    assert isinstance(validate_result(request, found.value), Available)
    assert load(dump(capabilities())) == capabilities()
    assert isinstance(
        validate_result(request, replace(found.value, plan_digest="0" * 64)), Rejected
    )
    wrong = replace(found.value.run, writes=(("absent", (0,)), *found.value.run.writes[1:]))
    assert isinstance(validate_result(request, replace(found.value, run=wrong)), Rejected)
    wrong = replace(
        found.value.run, writes=tuple((name, times[:-1]) for name, times in found.value.run.writes)
    )
    assert isinstance(validate_result(request, replace(found.value, run=wrong)), Rejected)


def test_tick_overflow_is_explicit_and_register_sizing_resolves_unknown_above_direct() -> None:
    request, reference = chain("kernel", rules=engine.ONE_CYCLE)
    a = replace(
        request.timed.attachment,
        channels=tuple(
            replace(c, capacity=Capacity("unspecified")) for c in request.timed.attachment.channels
        ),
    )
    sized = size(
        replace(request, timed=replace(request.timed, attachment=a), criterion=Criterion())
    )
    assert isinstance(sized, Available), sized
    independent = engine.minimal(reference, engine.ONE_CYCLE, 4)
    assert {name: cap.words for name, cap in sized.value.capacities} == independent.found
    p = replace(request.timed.attachment.profiles[0], period=2**63 - 1)
    a = replace(request.timed.attachment, profiles=(p, *request.timed.attachment.profiles[1:]))
    result = execute(replace(request, timed=replace(request.timed, attachment=a)))
    assert isinstance(result, Unsupported) and result.code == "tick-representation"


def test_sizing_cannot_erase_fixed_or_unbounded_attachment_choices() -> None:
    request, _ = chain("kernel")
    request = replace(request, criterion=Criterion())
    assert isinstance(size(request), Rejected)
    assert isinstance(
        execute(replace(request, capacities=(("a", Capacity("fixed", 7)),))), Rejected
    )


def test_grouped_marked_ports_retain_exact_descriptors_and_event_ids() -> None:
    request, reference = chain("width")
    plan = request.timed.plan
    ports = tuple(
        replace(
            p,
            sequence=BeatSequence(
                grouped(vector_major(p.tensor.shape, 1), 2),
                markers=cast(BeatSequence, p.sequence).markers,
            ),
        )
        if p.id in ("middle.out", "sink.in")
        else p
        for p in plan.ports
    )
    p = replace(plan, ports=ports)
    a = replace(
        request.timed.attachment,
        plan_digest=p.digest,
        profiles=tuple(
            replace(profile, configuration_digest=configuration_digest(p, profile.node))
            for profile in request.timed.attachment.profiles
        ),
    )
    request = replace(request, timed=TimedPlan(p, a))
    result = execute(request)
    assert isinstance(result, Available)
    assert result.value.run.to_run() == engine.run(reference, engine.FINNLIB, 4)
    trace = events(request, result.value, "b")
    assert isinstance(trace, Available)
    assert [(e.frame, e.word, e.source_pass) for e in trace.value] == [
        (f, w, 0) for f in range(4) for w in range(2)
    ]
    assert all(
        (e.producer_port, e.consumer_port, e.source)
        == ("middle.out", "sink.in", "middle.out.source")
        for e in trace.value
    )
    restored = load(dump(request))
    assert restored == request


def test_whole_pass_trace_occurrences_do_not_collapse_repeated_source_words() -> None:
    stages = (engine.Source("src", tuple(range(8)), 8, 16), engine.Sink("sink"))
    programs = (
        Program("src", "source", (), ("out",), order=IndexStream((0,), 8, 1)),
        Program("sink", "sink", ("in",), ()),
    )
    request, reference = fixture(stages, (engine.Channel("e", "src", "sink"),), programs)
    tensor = Tensor((4,), ENCODING)
    seq = BeatSequence(vector_major((4,), 1).repeated(2), markers=(LevelEnd(4),))
    p = replace(
        request.timed.plan,
        ports=tuple(
            replace(port, tensor=tensor, sequence=seq) for port in request.timed.plan.ports
        ),
        sources=(Source("out.source", tensor),),
    )
    a = replace(
        request.timed.attachment,
        plan_digest=p.digest,
        profiles=tuple(
            replace(profile, configuration_digest=configuration_digest(p, profile.node))
            for profile in request.timed.attachment.profiles
        ),
    )
    request = replace(request, timed=TimedPlan(p, a))
    found = execute(request)
    assert isinstance(found, Available)
    assert found.value.run.to_run() == engine.run(reference, engine.FINNLIB, 4)
    trace = events(request, found.value, "e")
    assert isinstance(trace, Available)
    assert [e.source_pass for e in trace.value[:8]] == [0] * 4 + [1] * 4
    assert [e.word for e in trace.value[:8]] == list(range(8))


@pytest.mark.parametrize("field", ["internal_words", "period"])
def test_supplied_storage_or_release_laws_are_not_silently_ignored(field: str) -> None:
    request, _ = chain("width")
    profile = replace(request.timed.attachment.profiles[1], **{field: 4})  # type: ignore[arg-type]
    a = replace(
        request.timed.attachment,
        profiles=(
            request.timed.attachment.profiles[0],
            profile,
            request.timed.attachment.profiles[2],
        ),
    )
    found = execute(replace(request, timed=replace(request.timed, attachment=a)))
    assert isinstance(found, Unsupported)
    assert found.code == ("internal-capacity" if field == "internal_words" else "stage-release")
