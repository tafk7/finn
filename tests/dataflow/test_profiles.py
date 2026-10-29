# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional timing absence/applicability matrix and independent logical controls."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, dataclass, replace

import pytest

from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.logical import (
    Available,
    Edge,
    LogicalPlan,
    Node,
    Port,
    Rejected,
    Source,
    Transfer,
    Unsupported,
    adapters,
    order_storage,
    packing,
)
from finn.dataflow.profiles import (
    Attachment,
    Capacity,
    ChannelProfile,
    ChannelRules,
    Clock,
    Evidence,
    Launch,
    Profile,
    TimedPlan,
    Unavailable,
    attach,
    configuration_digest,
    readiness,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.timed_records import dump, load
from finn.dataflow.traversal import BeatSequence, vector_major


def plan() -> LogicalPlan:
    tensor = Tensor((4,), ScalarEncoding(resolve_qonnx_datatype_name("INT8")))
    sequence = BeatSequence(vector_major((4,), 1))
    return LogicalPlan(
        "plan",
        (Node("src"), Node("sink")),
        (
            Port("out", "src", "write", "value", tensor, sequence),
            Port("in", "sink", "read", "value", tensor, sequence),
        ),
        (Edge("wire", "out", "in"),),
        (Source("value", tensor),),
        (Transfer("publish", "src", (), ("out",)), Transfer("consume", "sink", ("in",), ())),
    )


def evidence(reliability: str = "validated") -> Evidence:
    return Evidence(
        "python-event-1",
        "1",
        reliability,  # type: ignore[arg-type]
        ("analytic fixture, no measurement",),
        () if reliability in ("measured", "validated") else ("explicit local one-tick law",),
    )


def attachment(p: LogicalPlan, reliability: str = "validated") -> Attachment:
    e = evidence(reliability)
    return Attachment(
        p.digest,
        (
            Profile(
                "src",
                configuration_digest(p, "src"),
                e,
                "source",
                "clk",
                Launch(1, 0),
                0,
                span=4,
                period=8,
            ),
            Profile("sink", configuration_digest(p, "sink"), e, "sink", "clk", Launch(1, 0), 0),
        ),
        (ChannelProfile("wire", "clk", Capacity("fixed", 0)),),
        (Clock("clk", e, 1000),),
        ChannelRules(e, "handshake", 2, 1, 2),
    )


def test_absence_leaves_exact_logical_bytes_packing_route_and_order() -> None:
    p = plan()
    raw = p.record()
    t = attach(p, Attachment(p.digest))
    assert isinstance(t, Available)
    missing = readiness(t.value)
    assert isinstance(missing, Unavailable)
    assert set(missing.missing) == {
        ("src", "profile"),
        ("sink", "profile"),
        ("wire", "channel-profile"),
        ("plan", "channel-rules"),
    }
    assert p.record() == raw
    assert isinstance(packing(p, "out"), Available)
    assert packing(p, "out").value.payload_bits == 8  # type: ignore[union-attr]
    assert isinstance(adapters(p, "wire"), Available)
    assert isinstance(order_storage(p, "wire"), Available)


@pytest.mark.parametrize(
    "reliability", ["measured", "validated", "assumed", "approximate", "unreliable"]
)
def test_reliability_matrix_has_explicit_conditional_assumptions(reliability: str) -> None:
    p = plan()
    t = TimedPlan(p, attachment(p, reliability))
    default = readiness(t)
    characterized = reliability in ("measured", "validated")
    assert isinstance(default, Available) == characterized
    scenario = readiness(t, conditional=True)
    assert isinstance(scenario, Available)
    assert scenario.value.conditional != characterized
    assert bool(scenario.value.assumptions) != characterized
    assert t.plan.digest == p.digest


@pytest.mark.parametrize("field", ["protocol", "clock", "launch", "latency", "span", "period"])
def test_missing_fields_never_inherit_zero_or_partial_guarantees(field: str) -> None:
    p = plan()
    a = attachment(p)
    missing = replace(a.profiles[0], **{field: None})  # type: ignore[arg-type]
    a = replace(a, profiles=(missing, a.profiles[1]))
    found = readiness(TimedPlan(p, a), conditional=True)
    assert isinstance(found, Unavailable) and ("src", field) in found.missing


def test_zero_latency_is_present_missing_capacity_is_distinct_from_handshake() -> None:
    p = plan()
    a = attachment(p)
    assert isinstance(readiness(TimedPlan(p, a)), Available)
    absent = replace(a, channels=(replace(a.channels[0], capacity=Capacity("unspecified")),))
    found = readiness(TimedPlan(p, absent))
    assert isinstance(found, Unavailable) and ("wire", "capacity") in found.missing
    assert isinstance(
        readiness(TimedPlan(p, absent), capacities=(("wire", Capacity("fixed", 0)),)), Available
    )


def test_invalid_applicability_ids_clocks_and_configuration_refuse() -> None:
    p = plan()
    a = attachment(p)
    for wrong in (
        replace(a, plan_digest="0" * 64),
        replace(a, profiles=(replace(a.profiles[0], node="other"),)),
        replace(a, profiles=(replace(a.profiles[0], configuration_digest="0" * 64),)),
        replace(a, profiles=(replace(a.profiles[0], clock="other"),)),
        replace(a, channels=(replace(a.channels[0], edge="other"),)),
    ):
        assert isinstance(attach(p, wrong), Rejected)
        with pytest.raises(ValueError):
            TimedPlan(p, wrong)


def test_bounds_precede_unreliable_scenario_and_slice_cannot_cut_dependencies() -> None:
    p = plan()
    a = attachment(p, "unreliable")
    channel = ChannelProfile("wire", "clk", Capacity("unspecified"), 2, 4)
    t = TimedPlan(p, replace(a, channels=(channel,)))
    assert isinstance(
        readiness(t, conditional=True, capacities=(("wire", Capacity("fixed", 1)),)), Rejected
    )
    assert isinstance(readiness(t, nodes=("src",), conditional=True), Rejected)
    assert isinstance(readiness(t, capacities=(("missing", Capacity("fixed", 1)),)), Rejected)
    with pytest.raises(ValueError, match="hardware"):
        replace(channel, capacity=Capacity("fixed", 5))


def test_record_identity_frozen_data_and_negative_tags() -> None:
    p = plan()
    t = TimedPlan(p, attachment(p))
    assert load(dump(t)) == t
    assert load(dump(t)).plan.record() == p.record()  # type: ignore[attr-defined]
    with pytest.raises(FrozenInstanceError):
        setattr(t.attachment, "plan_digest", "0" * 64)
    raw = json.loads(dump(t))
    raw["value"]["attachment"]["plan_digest"] = "0" * 64
    with pytest.raises(ValueError, match="mismatch"):
        load(json.dumps(raw).encode())
    for raw in (
        b'{"schema":"optional-timing-1","value":{"type":"_Engine"}}',
        b'{"schema":"optional-timing-1","value":1.5}',
        b'{"schema":"optional-timing-1","value":null,"value":null}',
    ):
        with pytest.raises(ValueError):
            load(raw)

    @dataclass(frozen=True)
    class Launch:
        spacing: int
        phase: int

    with pytest.raises(TypeError):
        dump(Launch(1, 0))
    with pytest.raises(TypeError):
        dump(lambda: None)


@pytest.mark.parametrize("state", [Capacity("fixed", 0), Capacity("unbounded")])
def test_attached_capacity_choices_are_immutable_under_execution_resolution(
    state: Capacity,
) -> None:
    p = plan()
    a = attachment(p)
    t = TimedPlan(p, replace(a, channels=(replace(a.channels[0], capacity=state),)))
    assert isinstance(readiness(t, capacities=(("wire", state),)), Available)
    for wrong in (
        Capacity("fixed", 7),
        Capacity("unspecified"),
        Capacity("fixed", 0),
        Capacity("unbounded"),
    ):
        if wrong != state:
            found = readiness(t, capacities=(("wire", wrong),))
            assert isinstance(found, Rejected) and found.findings[0].code == "timing-capacity-fixed"


@pytest.mark.parametrize("reliability", ["measured", "validated", "assumed"])
def test_all_participating_model_clock_and_rules_assumptions_survive(reliability: str) -> None:
    p = plan()
    a = attachment(p)
    e = replace(
        evidence(reliability), assumptions=("valid only under named external-stall scenario",)
    )
    assert a.rules is not None
    a = replace(
        a,
        profiles=tuple(replace(profile, evidence=e) for profile in a.profiles),
        clocks=(replace(a.clocks[0], evidence=e),),
        rules=replace(a.rules, evidence=e),
    )
    found = readiness(TimedPlan(p, a), conditional=True)
    assert isinstance(found, Available)
    assert {text.split(":", 1)[0] for text in found.value.assumptions} == {
        "src",
        "sink",
        "clk",
        "plan",
    }
    assert all("external-stall" in text for text in found.value.assumptions)
    default = readiness(TimedPlan(p, a))
    if reliability in ("measured", "validated"):
        assert (
            isinstance(default, Available) and default.value.assumptions == found.value.assumptions
        )
        assert not found.value.conditional
    else:
        assert isinstance(default, Unavailable) and found.value.conditional
    assert load(dump(a)) == a


def test_disconnected_complete_slice_does_not_inherit_missing_other_profiles() -> None:
    p = plan()
    p = replace(
        p,
        nodes=(*p.nodes, Node("other-src"), Node("other-sink")),
        ports=(
            *p.ports,
            replace(p.ports[0], id="other-out", node="other-src", source="other-value"),
            replace(p.ports[1], id="other-in", node="other-sink", source="other-value"),
        ),
        edges=(*p.edges, Edge("other-wire", "other-out", "other-in")),
        sources=(*p.sources, Source("other-value", p.sources[0].tensor)),
    )
    t = TimedPlan(p, attachment(p))
    assert isinstance(readiness(t), Unavailable)
    scoped = readiness(t, nodes=("src", "sink"))
    assert isinstance(scoped, Available) and scoped.value.edges == ("wire",)


def test_two_known_clock_domains_are_unsupported_without_guessing_a_conversion() -> None:
    p = plan()
    a = attachment(p)
    a = replace(
        a,
        clocks=(*a.clocks, Clock("other-clock", evidence(), 2000)),
        profiles=(a.profiles[0], replace(a.profiles[1], clock="other-clock")),
    )
    t = attach(p, a)
    assert isinstance(t, Available)
    found = readiness(t.value)
    assert isinstance(found, Unsupported) and found.code == "clock-domains"
