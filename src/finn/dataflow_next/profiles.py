# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional characterization on exact logical records, in explicitly named ticks.

Attaching a profile never changes a plan or supplies a missing estimate. Hardware
capacity bounds remain constraints even in an assumed scenario. Readiness is for
a closed connected query slice; cutting a fork/backpressure dependency refuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from finn.dataflow_next.descriptors import dump
from finn.dataflow_next.logical import (
    Available,
    LogicalPlan,
    Result,
    Unsupported,
    _reject,
)

Protocol = Literal[
    "source", "memory", "kernel", "fork", "sink", "width", "reorder", "insert", "drop", "merge"
]
PROTOCOLS = (
    "source",
    "memory",
    "kernel",
    "fork",
    "sink",
    "width",
    "reorder",
    "insert",
    "drop",
    "merge",
)


def _name(value: str) -> None:
    if type(value) is not str or not value:
        raise ValueError("names are nonempty strings")


def _number(value: int, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise ValueError(f"counts/ticks are integers in {minimum}..2**63-1")


def _digest(value: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("an exact-record digest is a lowercase SHA-256")


@dataclass(frozen=True)
class Evidence:
    """One model's provenance for every field of its enclosing profile/rules/clock.

    Numeric profile fields are local clock ticks; capacities are words. Characterized
    means measured or analytically validated, never just a cycles/frame guess."""

    model: str
    version: str
    reliability: Literal["measured", "validated", "assumed", "approximate", "unreliable"]
    references: tuple[str, ...]
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _name(self.model)
        _name(self.version)
        if self.reliability not in (
            "measured",
            "validated",
            "assumed",
            "approximate",
            "unreliable",
        ):
            raise ValueError("unknown reliability")
        for name in ("references", "assumptions"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            for text in getattr(self, name):
                _name(text)
        if not self.references:
            raise ValueError("provenance requires an evidence reference")
        if self.reliability not in ("measured", "validated") and not self.assumptions:
            raise ValueError("a non-characterized scenario states its assumptions")


@dataclass(frozen=True)
class Clock:
    id: str
    evidence: Evidence
    period_ps: int | None = None

    def __post_init__(self) -> None:
        _name(self.id)
        if type(self.evidence) is not Evidence:
            raise TypeError("clock provenance is Evidence")
        if self.period_ps is not None:
            _number(self.period_ps, 1)


@dataclass(frozen=True)
class Launch:
    """The stated model law launch(i)=phase+i*spacing local clock ticks.

    This is an assumption/characterization boundary, not measured Schedule time.
    The current executor supports spacing=1/phase=0 except on released sources."""

    spacing: int
    phase: int

    def __post_init__(self) -> None:
        _number(self.spacing, 1)
        _number(self.phase)


@dataclass(frozen=True)
class Capacity:
    state: Literal["fixed", "unbounded", "unspecified"]
    words: int | None = None

    def __post_init__(self) -> None:
        if self.state not in ("fixed", "unbounded", "unspecified"):
            raise ValueError("unknown capacity state")
        if self.state == "fixed":
            _number(self.words)  # type: ignore[arg-type]
        elif self.words is not None:
            raise ValueError("only a fixed capacity states words")


@dataclass(frozen=True)
class Profile:
    node: str
    configuration_digest: str
    evidence: Evidence
    protocol: Protocol | None = None
    clock: str | None = None
    launch: Launch | None = None
    latency: int | None = None
    internal_words: int | None = None
    span: int | None = None
    period: int | None = None

    def __post_init__(self) -> None:
        _name(self.node)
        _digest(self.configuration_digest)
        if type(self.evidence) is not Evidence:
            raise TypeError("profile provenance is Evidence")
        if self.protocol is not None and self.protocol not in PROTOCOLS:
            raise ValueError("unknown stage protocol")
        if self.clock is not None:
            _name(self.clock)
        if self.launch is not None and type(self.launch) is not Launch:
            raise TypeError("launch law is Launch")
        for name in ("latency", "internal_words", "span", "period"):
            value = getattr(self, name)
            if value is not None:
                _number(value, 1 if name in ("span", "period") else 0)


@dataclass(frozen=True)
class ChannelProfile:
    edge: str
    clock: str | None = None
    capacity: Capacity = Capacity("unspecified")
    minimum: int = 0
    maximum: int | None = None

    def __post_init__(self) -> None:
        _name(self.edge)
        if self.clock is not None:
            _name(self.clock)
        if type(self.capacity) is not Capacity:
            raise TypeError("channel capacity is Capacity")
        _number(self.minimum)
        if self.maximum is not None:
            _number(self.maximum, self.minimum)
        if not self.permits(self.capacity):
            raise ValueError("channel capacity violates independent hardware bounds")

    def permits(self, capacity: Capacity) -> bool:
        if capacity.state != "fixed":
            return capacity.state != "unbounded" or self.maximum is None
        assert capacity.words is not None
        return self.minimum <= capacity.words and (
            self.maximum is None or capacity.words <= self.maximum
        )


@dataclass(frozen=True)
class ChannelRules:
    evidence: Evidence
    direct: Literal["handshake", "register"]
    fifo_latency: int
    slot_free: int
    reorder_latency: int

    def __post_init__(self) -> None:
        if type(self.evidence) is not Evidence:
            raise TypeError("rules provenance is Evidence")
        if self.direct not in ("handshake", "register"):
            raise ValueError("direct transport is handshake or register")
        for value in (self.fifo_latency, self.slot_free, self.reorder_latency):
            _number(value)


@dataclass(frozen=True)
class Attachment:
    plan_digest: str
    profiles: tuple[Profile, ...] = ()
    channels: tuple[ChannelProfile, ...] = ()
    clocks: tuple[Clock, ...] = ()
    rules: ChannelRules | None = None

    def __post_init__(self) -> None:
        _digest(self.plan_digest)
        for name, kind, key in (
            ("profiles", Profile, "node"),
            ("channels", ChannelProfile, "edge"),
            ("clocks", Clock, "id"),
        ):
            values = tuple(getattr(self, name))
            if not all(type(v) is kind for v in values):
                raise TypeError(f"{name} contains registered {kind.__name__} values")
            if len({getattr(v, key) for v in values}) != len(values):
                raise ValueError(f"duplicate {name} IDs")
            object.__setattr__(self, name, values)
        if self.rules is not None and type(self.rules) is not ChannelRules:
            raise TypeError("channel rules are ChannelRules")


@dataclass(frozen=True)
class TimedPlan:
    plan: LogicalPlan
    attachment: Attachment

    def __post_init__(self) -> None:
        if type(self.plan) is not LogicalPlan or type(self.attachment) is not Attachment:
            raise TypeError("a timed plan has logical data and an attachment")
        error = _mismatch(self.plan, self.attachment)
        if error is not None:
            raise ValueError(error)


@dataclass(frozen=True)
class Unavailable:
    missing: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Ready:
    nodes: tuple[str, ...]
    edges: tuple[str, ...]
    clock: str
    assumptions: tuple[str, ...]
    conditional: bool


def configuration_digest(plan: LogicalPlan, node: str) -> str:
    value = next((v for v in plan.nodes if v.id == node), None)
    if value is None:
        raise ValueError("unknown node")
    return sha256(
        dump(
            (
                value,
                tuple(p for p in plan.ports if p.node == node),
                tuple(t for t in plan.transfers if t.node == node),
            )
        )
    ).hexdigest()


def _mismatch(plan: LogicalPlan, attachment: Attachment) -> str | None:
    if attachment.plan_digest != plan.digest:
        return "attachment logical-plan digest mismatch"
    nodes, edges, clocks = (
        {n.id for n in plan.nodes},
        {e.id for e in plan.edges},
        {c.id for c in attachment.clocks},
    )
    for profile in attachment.profiles:
        if profile.node not in nodes or profile.configuration_digest != configuration_digest(
            plan, profile.node
        ):
            return f"configuration/node mismatch: {profile.node}"
        if profile.clock is not None and profile.clock not in clocks:
            return f"unknown clock: {profile.clock}"
    for channel in attachment.channels:
        if channel.edge not in edges:
            return f"unknown edge: {channel.edge}"
        if channel.clock is not None and channel.clock not in clocks:
            return f"unknown clock: {channel.clock}"
    return None


def attach(plan: LogicalPlan, attachment: Attachment) -> Result[TimedPlan]:
    error = _mismatch(plan, attachment)
    return (
        _reject("timing-applicability", plan.id, error)
        if error
        else Available(TimedPlan(plan, attachment))
    )


def readiness(
    timed: TimedPlan,
    nodes: tuple[str, ...] | None = None,
    *,
    conditional: bool = False,
    capacities: tuple[tuple[str, Capacity], ...] = (),
) -> Result[Ready] | Unavailable:
    """Complete fields and reliability for a closed dependency slice, in local ticks.

    Unspecified execution capacities may be resolved explicitly; a known hardware
    bound is checked before reliability. No physical-time query needs period_ps to
    run in ticks, but conversion to physical time requires it explicitly."""
    plan, attachment = timed.plan, timed.attachment
    chosen = tuple(n.id for n in plan.nodes) if nodes is None else tuple(nodes)
    names = set(chosen)
    if not names or len(names) != len(chosen) or not names <= {n.id for n in plan.nodes}:
        return _reject("timing-slice", plan.id, "name known nodes once")
    ports = {p.id: p for p in plan.ports}
    edges = tuple(e.id for e in plan.edges if ports[e.producer].node in names)
    for edge in plan.edges:
        if (ports[edge.producer].node in names) != (ports[edge.consumer].node in names):
            return _reject("timing-slice", edge.id, "query includes fork/backpressure dependencies")
    overrides = dict(capacities)
    if (
        len(overrides) != len(capacities)
        or not set(overrides) <= set(edges)
        or not all(type(v) is Capacity for v in overrides.values())
    ):
        return _reject("timing-capacity", plan.id, "resolve named slice capacities once")
    profiles, channels = (
        {p.node: p for p in attachment.profiles},
        {c.edge: c for c in attachment.channels},
    )
    missing: list[tuple[str, str]] = []
    clocks: set[str] = set()
    evidences: list[tuple[str, Evidence]] = []
    for name in chosen:
        profile = profiles.get(name)
        if profile is None:
            missing.append((name, "profile"))
            continue
        for field in ("protocol", "clock", "launch", "latency"):
            if getattr(profile, field) is None:
                missing.append((name, field))
        if profile.protocol == "source":
            for field in ("span", "period"):
                if getattr(profile, field) is None:
                    missing.append((name, field))
        if profile.protocol in ("kernel", "reorder") and profile.internal_words is None:
            missing.append((name, "internal_words"))
        if profile.clock is not None:
            clocks.add(profile.clock)
        evidences.append((name, profile.evidence))
    for name in edges:
        channel = channels.get(name)
        if channel is None:
            missing.append((name, "channel-profile"))
            if name in overrides:
                return _reject("timing-capacity", name, "capacity resolution needs hardware bounds")
            continue
        capacity = overrides.get(name, channel.capacity)
        if (
            name in overrides
            and channel.capacity.state != "unspecified"
            and capacity != channel.capacity
        ):
            return _reject(
                "timing-capacity-fixed",
                name,
                "a stated fixed/unbounded channel choice cannot be overridden",
            )
        if not channel.permits(capacity):
            return _reject("hardware-capacity", name, "capacity violates hardware bounds")
        if capacity.state == "unspecified":
            missing.append((name, "capacity"))
        if channel.clock is None:
            missing.append((name, "clock"))
        else:
            clocks.add(channel.clock)
    if attachment.rules is None:
        missing.append((plan.id, "channel-rules"))
    else:
        evidences.append((plan.id, attachment.rules.evidence))
    evidences.extend((c.id, c.evidence) for c in attachment.clocks if c.id in clocks)
    if missing:
        return Unavailable(tuple(missing))
    if len(clocks) != 1:
        return Unsupported("clock-domains", plan.id, "current executor supports one clock, no CDC")
    uncertain = [
        (owner, e) for owner, e in evidences if e.reliability not in ("measured", "validated")
    ]
    if uncertain and not conditional:
        return Unavailable(tuple((owner, "characterized-profile") for owner, _ in uncertain))
    assumptions = tuple(
        f"{owner}: {e.reliability}; {text}" for owner, e in evidences for text in e.assumptions
    )
    return Available(Ready(chosen, edges, next(iter(clocks)), assumptions, bool(uncertain)))
