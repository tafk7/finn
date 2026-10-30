# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Negative data-boundary controls with independent current-engine fixtures."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from dataflow.execution_cases import chain, fixture, variables
from finn.dataflow import timing as engine
from finn.dataflow.execution import (
    Criterion,
    HostCriterion,
    IndexStream,
    Program,
    execute,
    size,
    validate_result,
)
from finn.dataflow.logical import Available, Rejected, Unsupported
from finn.dataflow.profiles import Capacity, TimedPlan, configuration_digest
from finn.dataflow.timed_records import dump, load
from finn.dataflow.traversal import BeatSequence, vector_major


@pytest.mark.parametrize("family", ["width", "drop", "insert"])
def test_materialized_counts_match_both_logical_endpoints_before_engine_run(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = chain(family)
    p = request.timed.plan
    sink = next(port for port in p.ports if port.id == "sink.in")
    assert isinstance(sink.sequence, BeatSequence)
    lanes = sink.sequence.form.lanes
    expected = 6 if family == "insert" else 2
    tensor = replace(sink.tensor, shape=((expected + 1) * lanes,))
    sequence = BeatSequence(vector_major(tensor.shape, lanes))
    p = replace(
        p,
        ports=tuple(
            replace(port, tensor=tensor, sequence=sequence)
            if port.id in ("middle.out", "sink.in")
            else port
            for port in p.ports
        ),
        sources=tuple(
            replace(source, tensor=tensor) if source.id == "middle.out.source" else source
            for source in p.sources
        ),
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

    def forbidden(*args: object, **kwargs: object) -> engine.Run:
        raise AssertionError("invalid endpoint counts reached the engine")

    monkeypatch.setattr(engine, "run", forbidden)
    result = execute(request)
    assert isinstance(result, Rejected)
    assert result.findings[0].code == "execution-frame-count"
    assert result.findings[0].owner == "b"


@pytest.mark.parametrize("family", ["width", "drop", "insert"])
def test_correct_stage_counts_keep_independent_trace_and_result_control(family: str) -> None:
    request, reference = chain(family)
    expected = engine.run(reference, engine.FINNLIB, request.execution.frames)
    result = execute(request)
    assert isinstance(result, Available)
    assert result.value.run.to_run() == expected
    assert isinstance(validate_result(request, result.value), Available)


@pytest.mark.parametrize("field", ["capacities", "criterion", "met", "lowering", "reference"])
def test_result_metadata_must_match_execution_request(field: str) -> None:
    request, _ = chain("kernel", 0)
    found = execute(request)
    assert isinstance(found, Available)
    result = found.value
    if field == "capacities":
        result = replace(result, capacities=(("a", Capacity("fixed", 7)), result.capacities[1]))
    elif field == "criterion":
        result = replace(result, criterion="unknown-criterion")
    elif field == "met":
        result = replace(result, met=True)
    elif field == "lowering":
        result = replace(result, lowering_order=("a",))
    else:
        periodic = execute(replace(request, criterion=Criterion()))
        assert isinstance(periodic, Available)
        result = replace(result, reference=periodic.value.reference)
    assert isinstance(validate_result(request, result), Rejected)


@pytest.mark.parametrize("field", ["met", "reference"])
def test_finite_criterion_results_require_typed_verdict_and_reference(field: str) -> None:
    request, _ = chain("kernel")
    request = replace(request, criterion=Criterion())
    result = execute(request)
    assert isinstance(result, Available)
    assert isinstance(validate_result(request, result.value), Available)
    changed = (
        replace(result.value, met=None) if field == "met" else replace(result.value, reference=None)
    )
    assert isinstance(validate_result(request, changed), Rejected)


def test_sizing_result_binds_explicit_order_variables_and_fixed_other_channel() -> None:
    request, _ = chain("kernel")
    request = variables(replace(request, criterion=Criterion()), ("a",))
    result = size(request, channels=("a",))
    assert isinstance(result, Available)
    assert isinstance(validate_result(request, result.value, sizing_order=("a",)), Available)
    assert isinstance(validate_result(request, result.value), Rejected)
    assert isinstance(validate_result(request, result.value, sizing_order=("b",)), Rejected)
    changed = replace(
        result.value,
        capacities=tuple(
            (name, Capacity("fixed", 7)) if name == "b" else (name, cap)
            for name, cap in result.value.capacities
        ),
    )
    assert isinstance(validate_result(request, changed, sizing_order=("a",)), Rejected)
    changed = replace(result.value, lowering_order=("a", "a"))
    assert isinstance(validate_result(request, changed, sizing_order=("a", "a")), Rejected)


@pytest.mark.parametrize("field", ["completions", "late", "writes", "reads"])
def test_codec_rejects_scalar_in_tick_series_before_result_validation(field: str) -> None:
    request, _ = chain("kernel", 0)
    result = execute(request)
    assert isinstance(result, Available)
    record = json.loads(dump(result.value))
    record["value"]["run"][field][0][1] = 0
    with pytest.raises(TypeError, match="tuples of ticks"):
        load(json.dumps(record).encode())


@pytest.mark.parametrize("field", ["idle", "peaks"])
def test_codec_rejects_tick_series_in_scalar_fields(field: str) -> None:
    request, _ = chain("kernel", 0)
    result = execute(request)
    assert isinstance(result, Available)
    record = json.loads(dump(result.value))
    record["value"]["run"][field][0][1] = [0]
    with pytest.raises(ValueError, match="integers"):
        load(json.dumps(record).encode())


def test_run_rows_are_immutable_and_correct_empty_partial_reload_is_retained() -> None:
    request, _ = chain("kernel", 0)
    result = execute(request)
    assert isinstance(result, Available)
    run = result.value.run
    with pytest.raises(TypeError, match="immutable"):
        replace(run, writes=(["a", (0,)], *run.writes[1:]))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuples of ticks"):
        replace(run, writes=(("a", [0]), *run.writes[1:]))  # type: ignore[arg-type]
    partial = replace(
        run,
        deadlocked=True,
        completions=tuple((name, ()) for name, _ in run.completions),
        late=tuple((name, ()) for name, _ in run.late),
        writes=tuple((name, ()) for name, _ in run.writes),
        reads=tuple((name, ()) for name, _ in run.reads),
    )
    decoded = load(dump(partial))
    assert decoded == partial
    assert isinstance(validate_result(request, replace(result.value, run=partial)), Available)
    assert load(dump(run)) == run
    assert isinstance(validate_result(request, result.value), Available)


def test_lateness_series_matches_completed_source_frames() -> None:
    request, _ = chain("kernel", 0)
    result = execute(request)
    assert isinstance(result, Available)
    wrong = replace(result.value.run, late=(("src", ()),))
    assert isinstance(validate_result(request, replace(result.value, run=wrong)), Rejected)


def test_legal_large_slack_has_representation_unavailability_not_criterion_rejection() -> None:
    request, _ = chain("kernel")
    found = execute(replace(request, criterion=Criterion(slack=2**63 - 1)))
    assert isinstance(found, Unsupported) and found.code == "tick-representation"
    control = execute(replace(request, criterion=Criterion(slack=0)))
    assert isinstance(control, Available) and control.value.reference is not None
    assert control.value.reference.deadlines == (("sink", 11),)
    assert isinstance(
        validate_result(replace(request, criterion=Criterion(slack=0)), control.value), Available
    )


def test_target_only_unbounded_sizing_bound_keeps_fixed_channel_and_due_reference() -> None:
    stages = (
        engine.Source("src", (0, 1, 2, 3), 4, 16),
        engine.Kernel("kernel", 12, ((0, 1, 2, 3),), ((8, 9, 10, 11),), 5),
        engine.Sink("sink"),
    )
    links = (engine.Channel("a", "src", "kernel"), engine.Channel("b", "kernel", "sink", 1))
    programs = (
        Program("src", "source", (), ("src.out",), order=IndexStream((0,), 4, 1)),
        Program(
            "kernel",
            "kernel",
            ("kernel.in",),
            ("kernel.out",),
            12,
            (IndexStream((0,), 4, 1),),
            (IndexStream((8,), 4, 1),),
        ),
        Program("sink", "sink", ("sink.in",), ()),
    )
    request, reference = fixture(stages, links, programs)
    request = variables(request, ("a",))
    request = replace(
        request, criterion=Criterion("host", host_name="on-time", host_version="1", monotone=True)
    )
    host = HostCriterion("on-time", "1", lambda found: found.on_time)
    all_free = engine.run(reference.unbounded(), engine.FINNLIB, 4)
    targets_free = engine.run(reference, engine.FINNLIB, 4)
    assert all_free.peak("a") == 3 and targets_free.peak("a") == 8
    assert engine.least(reference, "a", engine.FINNLIB, 4, keeps=host.keeps) == 8
    found = size(request, channels=("a",), host=host)
    assert isinstance(found, Available)
    assert dict(found.value.capacities) == {"a": Capacity("fixed", 8), "b": Capacity("fixed", 1)}
    assert found.value.reference is not None and found.value.reference.free.to_run() == all_free
    assert found.value.reference.deadlines == engine.due(reference, engine.FINNLIB, 4).by
    assert isinstance(validate_result(request, found.value, sizing_order=("a",)), Available)
    constrained = replace(
        request.timed.attachment,
        channels=tuple(
            replace(c, maximum=7) if c.edge == "a" else c for c in request.timed.attachment.channels
        ),
    )
    rejected = size(
        replace(request, timed=replace(request.timed, attachment=constrained)),
        channels=("a",),
        host=host,
    )
    assert isinstance(rejected, Rejected) and rejected.findings[0].code == "sizing-upper"


@pytest.mark.parametrize("field", ["completions", "late", "idle", "writes", "reads", "peaks"])
def test_reference_observations_require_known_stage_source_and_edge_ids(field: str) -> None:
    request, _ = chain("kernel")
    request = replace(request, criterion=Criterion())
    result = execute(request)
    assert isinstance(result, Available) and result.value.reference is not None
    ref = result.value.reference
    rows = getattr(ref.free, field)
    wrong = replace(ref.free, **{field: (("unknown", rows[0][1]), *rows[1:])})  # type: ignore[arg-type]
    changed = replace(result.value, reference=replace(ref, free=wrong))
    assert isinstance(validate_result(request, changed), Rejected)


def test_reference_observations_require_complete_counts_and_source_allowances() -> None:
    request, _ = chain("kernel")
    request = replace(request, criterion=Criterion())
    result = execute(request)
    assert isinstance(result, Available) and result.value.reference is not None
    ref = result.value.reference
    for field, rows in (
        ("writes", tuple((name, ticks[:-1]) for name, ticks in ref.free.writes)),
        ("idle", (("src", 0),)),
    ):
        wrong = replace(ref.free, **{field: rows})  # type: ignore[arg-type]
        assert isinstance(
            validate_result(request, replace(result.value, reference=replace(ref, free=wrong))),
            Rejected,
        )
    assert isinstance(validate_result(request, result.value), Available)


@pytest.mark.parametrize("field", ["completions", "late", "writes", "reads", "idle", "peaks"])
@pytest.mark.parametrize("value", [True, 0.25])
def test_tick_and_scalar_codec_fields_reject_booleans_and_fractional_numbers(
    field: str, value: object
) -> None:
    request, _ = chain("kernel", 0)
    result = execute(request)
    assert isinstance(result, Available)
    record = json.loads(dump(result.value))
    if field in ("idle", "peaks"):
        record["value"]["run"][field][0][1] = value
    else:
        record["value"]["run"][field][0][1][0] = value
    with pytest.raises(ValueError):
        load(json.dumps(record).encode())
