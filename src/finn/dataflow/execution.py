# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Data-only finite requests over the existing Python timing engine.

Programs describe an explicitly materialized stage graph, not arbitrary kernel
computation. Edges requiring an additional logical adapter are unsupported here.
Reservations, rendezvous, freeing and atomic moves remain the engine's. Profiles
explicitly supply each tick law; compact periodic streams expand only within the
stated budget. Host criteria are invoked after complete runs, never serialized.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from finn.dataflow import timing as engine
from finn.dataflow.logical import Available, Result, Unsupported, _reject, adapters
from finn.dataflow.profiles import (
    Capacity,
    Profile,
    Protocol,
    Ready,
    TimedPlan,
    Unavailable,
    _digest,
    _name,
    _number,
    readiness,
)
from finn.dataflow.reindex import Route
from finn.dataflow.steps import Steps
from finn.dataflow.timed_records import digest, dump
from finn.dataflow.traversal import Selected, one_pass

MODEL: Literal["python-event-1"] = "python-event-1"
MAX_MOVES = 1_000_000
MAX_PATTERN = 65_536


@dataclass(frozen=True)
class Capabilities:
    model: str = MODEL
    clock_domains: int = 1
    count_bits: int = 63
    max_moves: int = MAX_MOVES
    max_pattern: int = MAX_PATTERN
    initial_tokens: int = 0
    criteria: tuple[str, ...] = ("trace", "period-kept", "host")
    programs: tuple[str, ...] = (
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

    def __post_init__(self) -> None:
        for value in (
            self.clock_domains,
            self.count_bits,
            self.max_moves,
            self.max_pattern,
            self.initial_tokens,
        ):
            _number(value)
        if (
            self.model,
            self.clock_domains,
            self.count_bits,
            self.max_moves,
            self.max_pattern,
            self.initial_tokens,
        ) != (MODEL, 1, 63, MAX_MOVES, MAX_PATTERN, 0):
            raise ValueError("unsupported executor capabilities")
        object.__setattr__(self, "criteria", tuple(self.criteria))
        object.__setattr__(self, "programs", tuple(self.programs))
        if self.criteria != ("trace", "period-kept", "host") or self.programs != (
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
        ):
            raise ValueError("unsupported criterion/program capabilities")


def capabilities() -> Capabilities:
    return Capabilities()


@dataclass(frozen=True)
class IndexStream:
    """pattern[j] + repetition*shift, preserving replay versus whole-pass offsets."""

    pattern: tuple[int, ...]
    repeats: int = 1
    shift: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pattern", tuple(self.pattern))
        _number(self.repeats, 1)
        _number(self.shift)
        if not self.pattern or len(self.pattern) * self.repeats > MAX_PATTERN:
            raise ValueError("index stream exceeds finite pattern budget")
        for value in self.pattern:
            _number(value)
        _number(max(self.pattern) + (self.repeats - 1) * self.shift)

    def expand(self) -> tuple[int, ...]:
        return tuple(v + k * self.shift for k in range(self.repeats) for v in self.pattern)


@dataclass(frozen=True)
class Flags:
    pattern: tuple[bool, ...]
    repeats: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "pattern", tuple(self.pattern))
        _number(self.repeats, 1)
        if (
            not self.pattern
            or len(self.pattern) * self.repeats > MAX_PATTERN
            or not all(type(v) is bool for v in self.pattern)
        ):
            raise ValueError("masks are bounded boolean patterns")

    def expand(self) -> tuple[bool, ...]:
        return self.pattern * self.repeats


@dataclass(frozen=True)
class Program:
    node: str
    opcode: Protocol
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    iterations: int | None = None
    reads: tuple[IndexStream, ...] = ()
    writes: tuple[IndexStream, ...] = ()
    order: IndexStream | None = None
    mask: Flags | None = None
    pattern: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _name(self.node)
        for name in ("inputs", "outputs", "reads", "writes", "pattern"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in (*self.inputs, *self.outputs):
            _name(name)
        if len(set((*self.inputs, *self.outputs))) != len(self.inputs) + len(self.outputs):
            raise ValueError("program ports are named once")
        for index in (*self.reads, *self.writes):
            if type(index) is not IndexStream:
                raise TypeError("port iteration streams are IndexStream values")
        if self.order is not None and type(self.order) is not IndexStream:
            raise TypeError("word order is an IndexStream")
        if self.mask is not None and type(self.mask) is not Flags:
            raise TypeError("participation mask is Flags")
        for word in self.pattern:
            _number(word)
        if self.iterations is not None:
            _number(self.iterations, 1)
        used = {"iterations"} if self.opcode == "kernel" else set()
        used |= {"reads", "writes"} if self.opcode == "kernel" else set()
        used |= {"order"} if self.opcode in ("source", "reorder") else set()
        used |= {"mask"} if self.opcode in ("insert", "drop") else set()
        used |= {"pattern"} if self.opcode == "merge" else set()
        for name in ("iterations", "reads", "writes", "order", "mask", "pattern"):
            if name not in used and getattr(self, name) not in (None, ()):
                raise ValueError(f"{self.opcode} does not consume {name}")


@dataclass(frozen=True)
class Execution:
    frames: int
    nodes: tuple[str, ...] = ()
    conditional: bool = False
    max_moves: int = MAX_MOVES
    model: Literal["python-event-1"] = MODEL

    def __post_init__(self) -> None:
        _number(self.frames, 1)
        _number(self.max_moves, 1)
        if self.max_moves > MAX_MOVES or type(self.conditional) is not bool or self.model != MODEL:
            raise ValueError("unsupported execution budget/model")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        for node in self.nodes:
            _name(node)


@dataclass(frozen=True)
class Criterion:
    mode: Literal["trace", "period-kept", "host"] = "period-kept"
    slack: int | None = None
    host_name: str | None = None
    host_version: str | None = None
    monotone: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("trace", "period-kept", "host") or type(self.monotone) is not bool:
            raise ValueError("unknown criterion")
        if self.slack is not None:
            _number(self.slack)
        if self.mode == "host":
            _name(self.host_name)  # type: ignore[arg-type]
            _name(self.host_version)  # type: ignore[arg-type]
        elif self.host_name is not None or self.host_version is not None or self.monotone:
            raise ValueError("only host criteria state host identity/monotonicity")


@dataclass(frozen=True)
class RunData:
    """Complete current-engine observations in local ticks, keyed by stage/edge IDs.

    Channel beat k has frame=k//expected_words and beat=k%expected_words; logical
    port descriptors retain lane/pass/set and marker semantics. These are modeled
    token movements, not a payload or marker-bit simulator. Hidden internal actors'
    occupancy is not exposed by the current public Run API.
    """

    frames: int
    deadlocked: bool
    completions: tuple[tuple[str, tuple[int, ...]], ...]
    late: tuple[tuple[str, tuple[int, ...]], ...]
    idle: tuple[tuple[str, int], ...]
    writes: tuple[tuple[str, tuple[int, ...]], ...]
    reads: tuple[tuple[str, tuple[int, ...]], ...]
    peaks: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        _number(self.frames, 1)
        if type(self.deadlocked) is not bool:
            raise TypeError("deadlocked is boolean")
        for field in ("completions", "late", "idle", "writes", "reads", "peaks"):
            rows = getattr(self, field)
            if type(rows) is not tuple:
                raise TypeError(f"{field} is an immutable tuple of named rows")
            names = set()
            for row in rows:
                if type(row) is not tuple or len(row) != 2:
                    raise TypeError(f"{field} rows are immutable (name, value) pairs")
                name, values = row
                _name(name)
                if name in names:
                    raise ValueError("duplicate run IDs")
                names.add(name)
                if field in ("completions", "late", "writes", "reads"):
                    if type(values) is not tuple:
                        raise TypeError(f"{field} values are tuples of ticks")
                    for tick in values:
                        _number(tick)
                    if field in ("completions", "late") and len(values) > self.frames:
                        raise ValueError(f"{field} has at most one value per frame")
                else:
                    _number(values)
            object.__setattr__(self, field, rows)

    @classmethod
    def from_run(cls, run: engine.Run) -> RunData:
        return cls(
            run.frames,
            run.deadlocked,
            run.completions,
            run.late,
            run.idle,
            run.writes,
            run.reads,
            run.peaks,
        )

    def to_run(self) -> engine.Run:
        return engine.Run(
            self.frames,
            self.deadlocked,
            self.completions,
            self.late,
            self.idle,
            self.writes,
            self.reads,
            self.peaks,
        )


@dataclass(frozen=True)
class Reference:
    basis_digest: str
    free: RunData
    period: int
    deadlines: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        _digest(self.basis_digest)
        if type(self.free) is not RunData or self.free.frames < 2:
            raise ValueError("a finite reference has at least two frames")
        _number(self.period, 1)
        object.__setattr__(self, "deadlines", tuple(self.deadlines))
        for name, value in self.deadlines:
            _name(name)
            _number(value)


@dataclass(frozen=True)
class Request:
    timed: TimedPlan
    programs: tuple[Program, ...]
    execution: Execution
    criterion: Criterion = Criterion()
    capacities: tuple[tuple[str, Capacity], ...] = ()
    reference: Reference | None = None

    def __post_init__(self) -> None:
        if (
            type(self.timed) is not TimedPlan
            or type(self.execution) is not Execution
            or type(self.criterion) is not Criterion
        ):
            raise TypeError("requests contain registered timing data")
        object.__setattr__(self, "programs", tuple(self.programs))
        object.__setattr__(self, "capacities", tuple(self.capacities))
        if not all(type(p) is Program for p in self.programs):
            raise TypeError("request programs are Program values")
        for name, capacity in self.capacities:
            _name(name)
            if type(capacity) is not Capacity:
                raise TypeError("request capacities are Capacity values")
        if self.reference is not None and type(self.reference) is not Reference:
            raise TypeError("reference is data, not a predicate")


@dataclass(frozen=True)
class ResultData:
    request_digest: str
    plan_digest: str
    attachment_digest: str
    run: RunData
    clock: str
    conditional: bool
    assumptions: tuple[str, ...]
    criterion: str
    met: bool | None
    reference: Reference | None
    words: tuple[tuple[str, int], ...]
    capacities: tuple[tuple[str, Capacity], ...]
    lowering_order: tuple[str, ...] = ()
    limits: tuple[str, ...] = (
        "finite frames, no warm-up exemption or asymptotic proof",
        "local ticks, not measurements",
        "token timing only; no payload/marker-bit execution",
        "internal actor occupancy unavailable in public Run",
    )
    model: str = MODEL

    def __post_init__(self) -> None:
        for value in (self.request_digest, self.plan_digest, self.attachment_digest):
            _digest(value)
        if (
            type(self.run) is not RunData
            or type(self.conditional) is not bool
            or self.met is not None
            and type(self.met) is not bool
        ):
            raise TypeError("invalid result data")
        if self.reference is not None and type(self.reference) is not Reference:
            raise TypeError("invalid reference")
        _name(self.clock)
        if self.model != MODEL:
            raise ValueError("unknown result model")
        for name in ("assumptions", "words", "capacities", "lowering_order", "limits"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for text in (*self.assumptions, *self.lowering_order, *self.limits):
            _name(text)
        for name, words in self.words:
            _name(name)
            _number(words, 1)
        for name, capacity in self.capacities:
            _name(name)
            if type(capacity) is not Capacity:
                raise TypeError("result capacities are registered Capacity values")
        if len({n for n, _ in self.words}) != len(self.words) or len(
            {n for n, _ in self.capacities}
        ) != len(self.capacities):
            raise ValueError("duplicate result channel IDs")


@dataclass(frozen=True)
class HostCriterion:
    """Host-only contract. Neither this value nor its callback is serializable."""

    name: str
    version: str
    keeps: Callable[[engine.Run], bool]


@dataclass(frozen=True)
class TraceEvent:
    edge: str
    producer_port: str
    consumer_port: str
    source: str
    frame: int
    word: int
    source_pass: int
    sink_pass: int
    write_tick: int | None
    read_tick: int | None
    selector: str | None = None

    def __post_init__(self) -> None:
        for name in (self.edge, self.producer_port, self.consumer_port, self.source):
            _name(name)
        if self.selector is not None:
            _name(self.selector)
        for value in (self.frame, self.word, self.source_pass, self.sink_pass):
            _number(value)
        for tick in (self.write_tick, self.read_tick):
            if tick is not None:
                _number(tick)


def _beats(timed: TimedPlan, port: str) -> int:
    sequence = next(p.sequence for p in timed.plan.ports if p.id == port)
    return (sequence.sequence if isinstance(sequence, Selected) else sequence).form.beats


def _lanes(timed: TimedPlan, port: str) -> int:
    sequence = next(p.sequence for p in timed.plan.ports if p.id == port)
    return (sequence.sequence if isinstance(sequence, Selected) else sequence).form.lanes


def _stage(timed: TimedPlan, program: Program, profile: Profile) -> engine.Stage:
    name = program.node
    if program.opcode != profile.protocol:
        raise ValueError(f"{name}: program/profile protocol mismatch")
    ports = {p.id: p for p in timed.plan.ports if p.node == name}
    if set(program.inputs) != {p.id for p in ports.values() if p.direction == "read"} or set(
        program.outputs
    ) != {p.id for p in ports.values() if p.direction == "write"}:
        raise ValueError(f"{name}: program port ownership/direction mismatch")
    assert profile.launch is not None and profile.latency is not None
    groups = {(t.reads, t.writes) for t in timed.plan.transfers if t.node == name}
    if not groups:
        raise ValueError(f"{name}: explicit atomic participation groups required")
    counts = (len(program.inputs), len(program.outputs))
    opcode = program.opcode
    expected = {
        "source": (0, 1),
        "memory": (0, 1),
        "sink": (1, 0),
        "width": (1, 1),
        "reorder": (1, 1),
        "insert": (1, 1),
        "drop": (1, 1),
    }
    if opcode in expected and counts != expected[opcode]:
        raise ValueError(f"{name}: incorrect protocol port count")
    if opcode == "source":
        if groups != {(program.inputs, program.outputs)}:
            raise ValueError("source atomic participation mismatch")
        if program.order is None or profile.span is None or profile.period is None:
            raise ValueError(f"{name}: source word indices/span/period required")
        times = tuple(
            profile.launch.phase + v * profile.launch.spacing for v in program.order.expand()
        )
        if len(times) != _beats(timed, program.outputs[0]):
            raise ValueError(f"{name}: source expected frame count mismatch")
        for at in times:
            _number(at)
        return engine.Source(name, times, profile.span, profile.period)
    if opcode == "kernel":
        if (
            program.iterations is None
            or len(program.reads) != counts[0]
            or len(program.writes) != counts[1]
        ):
            raise ValueError(f"{name}: iteration program required for each port")
        schedule = next(n.schedule for n in timed.plan.nodes if n.id == name)
        if schedule is not None and program.iterations != schedule.iterations:
            raise ValueError(f"{name}: logical iteration count mismatch")
        reads, writes = (
            tuple(s.expand() for s in program.reads),
            tuple(s.expand() for s in program.writes),
        )
        for port, times in zip((*program.inputs, *program.outputs), (*reads, *writes)):
            if len(times) != _beats(timed, port):
                raise ValueError(f"{name}: expected port frame count mismatch")
        used = set()
        for at in {v for stream in (*reads, *writes) for v in stream}:
            used.add(
                (
                    tuple(p for p, stream in zip(program.inputs, reads) if at in stream),
                    tuple(p for p, stream in zip(program.outputs, writes) if at in stream),
                )
            )
        if used != groups:
            raise ValueError(f"{name}: firing atomic participation mismatch")
        return engine.Kernel(name, program.iterations, reads, writes, profile.latency)
    if opcode == "insert":
        if program.mask is None:
            raise ValueError("insertion needs a participation mask")
        expected_groups = (
            {(program.inputs, program.outputs), ((), program.outputs)}
            if False in program.mask.pattern
            else {(program.inputs, program.outputs)}
        )
        if groups != expected_groups:
            raise ValueError("insertion atomic participation mismatch")
        return engine.InsertionStage(name, program.mask.expand())
    if opcode == "drop":
        if program.mask is None:
            raise ValueError("drop needs a participation mask")
        expected_groups = (
            {(program.inputs, program.outputs), (program.inputs, ())}
            if False in program.mask.pattern
            else {(program.inputs, program.outputs)}
        )
        if groups != expected_groups:
            raise ValueError("drop atomic participation mismatch")
        return engine.DropStage(name, program.mask.expand())
    if opcode == "merge":
        if (
            counts[1] != 1
            or counts[0] < 2
            or groups != {((p,), program.outputs) for p in program.inputs}
        ):
            raise ValueError("merge atomic participation mismatch")
        return engine.MergeStage(name, program.pattern, _beats(timed, program.outputs[0]))
    if groups != {(program.inputs, program.outputs)}:
        raise ValueError(f"{name}: atomic participation mismatch")
    if opcode == "memory":
        return engine.Memory(name)
    if opcode == "sink":
        return engine.Sink(name)
    if opcode == "fork":
        if counts[0] != 1:
            raise ValueError("fork requires one input")
        return engine.Fork(name, counts[1])
    if opcode == "width":
        return engine.WidthStage(
            name,
            _lanes(timed, program.inputs[0]),
            _lanes(timed, program.outputs[0]),
        )
    if opcode == "reorder":
        if program.order is None or profile.internal_words is None:
            raise ValueError("reorder needs reads and explicit internal capacity")
        read_words = program.order.expand()
        if len(read_words) != _beats(timed, program.outputs[0]):
            raise ValueError("reorder frame output count mismatch")
        return engine.ReorderStage(
            name,
            read_words,
            engine.frees(read_words, _beats(timed, program.inputs[0])),
            profile.internal_words,
        )
    raise ValueError("unknown transfer opcode")


def _network(request: Request, ready: Ready) -> Result[engine.Network]:
    programs = {p.node: p for p in request.programs}
    if len(programs) != len(request.programs) or set(programs) != set(ready.nodes):
        return _reject("execution-programs", request.timed.plan.id, "name each slice node once")
    profiles = {p.node: p for p in request.timed.attachment.profiles}
    rules = request.timed.attachment.rules
    assert rules is not None
    for name in ready.nodes:
        p = profiles[name]
        if p.protocol not in ("kernel", "reorder") and p.internal_words is not None:
            return Unsupported(
                "internal-capacity",
                name,
                "this protocol has fixed storage; a supplied internal capacity is unsupported",
            )
        if p.protocol != "source" and (p.span is not None or p.period is not None):
            return Unsupported(
                "stage-release",
                name,
                "only released sources consume a span/period in the current engine",
            )
        if p.evidence.model != MODEL:
            return Unsupported("profile-model", name, "current adapter requires python-event-1")
        if (
            p.protocol != "source"
            and p.launch is not None
            and (p.launch.spacing != 1 or p.launch.phase != 0)
        ):
            return Unsupported(
                "launch-law", name, "current non-source launch law is phase0/spacing1"
            )
        if p.protocol != "kernel" and p.latency != 0:
            return Unsupported(
                "stage-latency",
                name,
                "this protocol has fixed engine latency plus explicit channel rules",
            )
        if p.protocol == "kernel" and p.internal_words != p.latency:
            return Unsupported(
                "pipeline-reservation",
                name,
                "current direct pipeline holds exactly latency registers per output",
            )
    ports = {p.id: p for p in request.timed.plan.ports}
    channels = {c.edge: c for c in request.timed.attachment.channels}
    caps = dict(request.capacities)
    stages = []
    links = []
    try:
        for name in ready.nodes:
            stages.append(_stage(request.timed, programs[name], profiles[name]))
        for edge in request.timed.plan.edges:
            if edge.id not in ready.edges:
                continue
            route = adapters(request.timed.plan, edge.id)
            if not isinstance(route, Available):
                return route
            if (
                isinstance(route.value, Route)
                or isinstance(route.value, Steps)
                and any(
                    hop.reorder is not None or hop.source.form.lanes != hop.sink.form.lanes
                    for hop in route.value.hops
                )
            ):
                return Unsupported(
                    "edge-adapter",
                    edge.id,
                    "materialize logical adapter stages with explicit profiles first",
                )
            capacity = caps.get(edge.id, channels[edge.id].capacity)
            producer, consumer = ports[edge.producer].node, ports[edge.consumer].node
            links.append(
                engine.Channel(
                    edge.id,
                    producer,
                    consumer,
                    capacity.words if capacity.state == "fixed" else None,
                    programs[producer].outputs.index(edge.producer),
                    programs[consumer].inputs.index(edge.consumer),
                )
            )
        network = engine.Network(tuple(stages), tuple(links))
        expected = {edge.id: edge for edge in request.timed.plan.edges}
        for name, words in network.words:
            edge = expected[name]
            if words != _beats(request.timed, edge.producer) or words != _beats(
                request.timed, edge.consumer
            ):
                return _reject(
                    "execution-frame-count",
                    name,
                    "materialized channel words must match both logical endpoint counts",
                )
        count = sum(n for _, n in network.words) * request.execution.frames
        if count > request.execution.max_moves or any(
            isinstance(s, engine.Kernel)
            and s.iterations * request.execution.frames > request.execution.max_moves
            for s in stages
        ):
            return Unsupported(
                "execution-budget",
                request.timed.plan.id,
                "expanded finite token program exceeds move budget",
            )
        # A conservative acyclic event-bound also protects the durable int64
        # tick representation. Python arithmetic itself must never be its policy.
        per_frame = sum(
            s.period + max(s.times)
            if isinstance(s, engine.Source)
            else s.iterations
            if isinstance(s, engine.Kernel)
            else 0
            for s in stages
        )
        delay = (
            rules.fifo_latency
            + rules.slot_free
            + rules.reorder_latency
            + 2
            + sum(s.latency for s in stages if isinstance(s, engine.Kernel))
        )
        if request.execution.frames * per_frame + count * delay > 2**63 - 1:
            return Unsupported(
                "tick-representation",
                request.timed.plan.id,
                "finite event bound exceeds signed 63-bit tick schema",
            )
        for link in links:
            engine.Rules(
                rules.fifo_latency,
                engine.Direct(rules.direct),
                rules.slot_free,
                rules.reorder_latency,
            ).latency(link.capacity)
        return Available(network, "current-python-engine")
    except ValueError as error:
        return _reject("execution-invalid", request.timed.plan.id, str(error))


def _rules(request: Request) -> engine.Rules:
    rules = request.timed.attachment.rules
    assert rules is not None
    return engine.Rules(
        rules.fifo_latency, engine.Direct(rules.direct), rules.slot_free, rules.reorder_latency
    )


def _basis(request: Request) -> str:
    # The fixed all-channel-unbounded witness depends on rules, clock/profile,
    # internal storage and horizon, not a candidate channel capacity choice.
    a = request.timed.attachment
    reference_attachment = replace(
        a, channels=tuple(replace(c, capacity=Capacity("unspecified")) for c in a.channels)
    )
    return digest(
        (
            TimedPlan(request.timed.plan, reference_attachment),
            request.programs,
            request.execution,
            request.criterion.slack,
        )
    )


def _reference(request: Request, network: engine.Network) -> Result[Reference]:
    if request.execution.frames < 2:
        return _reject(
            "criterion-horizon",
            request.timed.plan.id,
            "finite period criterion requires at least two frames",
        )
    try:
        free = engine._checked(network, _rules(request), request.execution.frames)
        due = engine._due(network, free, request.criterion.slack)
        if due.period > 2**63 - 1 or any(deadline > 2**63 - 1 for _, deadline in due.by):
            return Unsupported(
                "tick-representation",
                request.timed.plan.id,
                "derived finite period/deadlines exceed the signed tick schema",
            )
        reference = Reference(_basis(request), RunData.from_run(free), due.period, due.by)
        if request.reference is not None and request.reference != reference:
            return _reject(
                "reference-mismatch",
                request.timed.plan.id,
                "reference must match profiles/storage/rules/horizon and unbounded witness",
            )
        return Available(reference)
    except ValueError as error:
        return _reject("criterion-reference", request.timed.plan.id, str(error))


def _judge(
    request: Request, reference: Reference | None, host: HostCriterion | None
) -> Result[Callable[[engine.Run], bool]]:
    if request.criterion.mode == "host":
        if (
            host is None
            or host.name != request.criterion.host_name
            or host.version != request.criterion.host_version
        ):
            return Unsupported(
                "host-criterion",
                request.timed.plan.id,
                "named/versioned host criterion required outside serialized data",
            )
        return Available(host.keeps)
    if reference is None:
        return Available(lambda found: not found.deadlocked)
    due = engine.Due(reference.period, reference.free.frames, reference.deadlines)
    return Available(lambda found: found.on_time and due.met(found))


def execute(
    request: Request, *, host: HostCriterion | None = None
) -> Result[ResultData] | Unavailable:
    """Execute a finite scenario. False criterion is an available modeled failure."""
    ready = readiness(
        request.timed,
        request.execution.nodes or None,
        conditional=request.execution.conditional,
        capacities=request.capacities,
    )
    if not isinstance(ready, Available):
        return ready
    network = _network(request, ready.value)
    if not isinstance(network, Available):
        return network
    reference = None
    if request.criterion.mode != "trace" or request.reference is not None:
        ref = _reference(request, network.value)
        if not isinstance(ref, Available):
            return ref
        reference = ref.value
    judge = _judge(request, reference, host)
    if not isinstance(judge, Available):
        return judge
    found = engine.run(network.value, _rules(request), request.execution.frames)
    met = None if request.criterion.mode == "trace" else judge.value(found)
    result = _result(request, ready.value, network.value, found, met, reference)
    return _bounded_result(result)


def _bounded_result(result: ResultData) -> Result[ResultData]:
    try:
        dump(result)
    except ValueError as error:
        return Unsupported("result-byte-budget", result.plan_digest, str(error))
    return Available(result)


def _result(
    request: Request,
    ready: Ready,
    network: engine.Network,
    found: engine.Run,
    met: bool | None,
    reference: Reference | None,
    lowering: tuple[str, ...] = (),
) -> ResultData:
    return ResultData(
        digest(request),
        request.timed.plan.digest,
        digest(request.timed.attachment),
        RunData.from_run(found),
        ready.clock,
        ready.conditional,
        ready.assumptions,
        request.criterion.mode,
        met,
        reference,
        network.words,
        tuple(
            (c.name, Capacity("unbounded") if c.capacity is None else Capacity("fixed", c.capacity))
            for c in network.channels
        ),
        lowering,
    )


def validate_result(
    request: Request,
    result: ResultData,
    *,
    sizing_order: tuple[str, ...] | None = None,
) -> Result[engine.Run] | Unavailable:
    """Check exact identity, complete trace IDs/counts and monotone tick records.

    This validates data integrity/association, not independently certifying a timing
    calculation. Differential reference execution supplies that separate evidence.
    """
    if (
        result.request_digest != digest(request)
        or result.plan_digest != request.timed.plan.digest
        or result.attachment_digest != digest(request.timed.attachment)
    ):
        return _reject(
            "result-identity", request.timed.plan.id, "result belongs to another exact request"
        )
    if result.criterion != request.criterion.mode or (
        result.met is not None
        if request.criterion.mode == "trace"
        else type(result.met) is not bool
    ):
        return _reject("result-criterion", request.timed.plan.id, "criterion/result shape mismatch")
    order = () if sizing_order is None else tuple(sizing_order)
    if result.lowering_order != order or len(set(order)) != len(order):
        return _reject("result-lowering", request.timed.plan.id, "explicit sizing order mismatch")
    resolved = dict(request.capacities)
    reported = dict(result.capacities)
    policies = {c.edge: c for c in request.timed.attachment.channels}
    if len(resolved) != len(request.capacities):
        return _reject("result-capacities", request.timed.plan.id, "duplicate request capacities")
    if order:
        if request.criterion.mode == "trace" or (
            request.criterion.mode == "host" and not request.criterion.monotone
        ):
            return _reject(
                "result-lowering", request.timed.plan.id, "criterion cannot size channels"
            )
        for name in order:
            if (
                name not in policies
                or policies[name].capacity.state != "unspecified"
                or name not in reported
                or reported[name].state != "fixed"
            ):
                return _reject("result-lowering", name, "sizing requires declared finite variables")
            resolved[name] = reported[name]
    prepared = replace(request, capacities=tuple(resolved.items()))
    ready = readiness(
        prepared.timed,
        prepared.execution.nodes or None,
        conditional=prepared.execution.conditional,
        capacities=prepared.capacities,
    )
    if not isinstance(ready, Available):
        return ready
    network = _network(prepared, ready.value)
    if not isinstance(network, Available):
        return network
    found = result.run
    nodes, edges = set(ready.value.nodes), set(ready.value.edges)
    expected_capacities = {
        c.name: Capacity("unbounded") if c.capacity is None else Capacity("fixed", c.capacity)
        for c in network.value.channels
    }
    if reported != expected_capacities or not set(order) <= edges:
        return _reject(
            "result-capacities", request.timed.plan.id, "resolved capacity choices mismatch"
        )
    source_names = {s.name for s in network.value.stages if isinstance(s, engine.Source)}
    required_reference = request.criterion.mode != "trace" or request.reference is not None
    if required_reference != (result.reference is not None):
        return _reject(
            "result-reference", request.timed.plan.id, "criterion reference presence mismatch"
        )
    if result.reference is not None:
        ref = result.reference
        try:
            period = network.value.period
        except ValueError as error:
            return _reject("result-reference", request.timed.plan.id, str(error))
        if (
            request.reference is not None
            and ref != request.reference
            or ref.basis_digest != _basis(request)
            or ref.free.frames != request.execution.frames
            or ref.free.deadlocked
            or ref.period != period
            or len(ref.deadlines) != len(set(name for name, _ in ref.deadlines))
            or {name for name, _ in ref.deadlines} != set(engine.ends(network.value))
        ):
            return _reject(
                "result-reference", request.timed.plan.id, "reference scope/identity mismatch"
            )
        for field, names in (
            ("completions", nodes),
            ("late", source_names),
            ("idle", source_names),
            ("writes", edges),
            ("reads", edges),
            ("peaks", edges),
        ):
            if {name for name, _ in getattr(ref.free, field)} != names:
                return _reject(
                    "result-reference", request.timed.plan.id, f"reference {field} IDs mismatch"
                )
        words = dict(network.value.words)
        for field in ("completions", "late", "writes", "reads"):
            for name, ticks in getattr(ref.free, field):
                count = (
                    ref.free.frames
                    if field in ("completions", "late")
                    else words[name] * ref.free.frames
                )
                if (
                    len(ticks) != count
                    or field != "late"
                    and any(b < a for a, b in zip(ticks, ticks[1:]))
                ):
                    return _reject(
                        "result-reference",
                        request.timed.plan.id,
                        "reference observations must be complete and ordered",
                    )
        expected_idle = {
            s.name: s.idle for s in network.value.stages if isinstance(s, engine.Source)
        }
        if dict(ref.free.idle) != expected_idle:
            return _reject(
                "result-reference", request.timed.plan.id, "reference source allowances mismatch"
            )
    if (
        found.frames != request.execution.frames
        or result.words != network.value.words
        or result.clock != ready.value.clock
        or result.conditional != ready.value.conditional
        or result.assumptions != ready.value.assumptions
    ):
        return _reject(
            "result-scope", request.timed.plan.id, "result horizon/words/reliability/clock mismatch"
        )
    for field, names in (
        ("completions", nodes),
        ("late", source_names),
        ("idle", source_names),
        ("writes", edges),
        ("reads", edges),
        ("peaks", edges),
    ):
        if {name for name, _ in getattr(found, field)} != names:
            return _reject("result-ids", request.timed.plan.id, f"{field} IDs mismatch")
    for field in ("completions", "writes", "reads"):
        for name, ticks in getattr(found, field):
            limit = (
                found.frames if field == "completions" else dict(result.words)[name] * found.frames
            )
            if (
                len(ticks) > limit
                or not found.deadlocked
                and len(ticks) != limit
                or any(b < a for a, b in zip(ticks, ticks[1:]))
            ):
                return _reject(
                    "result-counts", name, "trace must retain all completed observations in order"
                )
    completed = dict(found.completions)
    if any(len(ticks) != len(completed[name]) for name, ticks in found.late):
        return _reject(
            "result-counts", request.timed.plan.id, "lateness needs each source completion"
        )
    return Available(found.to_run())


def events(
    request: Request,
    result: ResultData,
    edge_id: str,
    *,
    sizing_order: tuple[str, ...] | None = None,
) -> Result[tuple[TraceEvent, ...]] | Unavailable:
    """Decode ordinal token observations with stable port/source/pass references.

    Selected records retain their selector name; no selector value or new whole-set
    association is inferred. This timing-only view makes no payload delivery claim.
    """
    validated = validate_result(request, result, sizing_order=sizing_order)
    if not isinstance(validated, Available):
        return validated
    edge = next((e for e in request.timed.plan.edges if e.id == edge_id), None)
    if edge is None or edge_id not in dict(result.words):
        return _reject("trace-edge", edge_id, "unknown query edge")
    ports = {p.id: p for p in request.timed.plan.ports}
    source, sink = ports[edge.producer], ports[edge.consumer]
    produced = (
        source.sequence.sequence if isinstance(source.sequence, Selected) else source.sequence
    )
    consumed = sink.sequence.sequence if isinstance(sink.sequence, Selected) else sink.sequence
    writes, reads = validated.value.written(edge_id), validated.value.taken(edge_id)
    words = dict(result.words)[edge_id]
    return Available(
        tuple(
            TraceEvent(
                edge_id,
                source.id,
                sink.id,
                source.source,
                ordinal // words,
                ordinal % words,
                (ordinal % words) // one_pass(produced.form).beats,
                (ordinal % words) // one_pass(consumed.form).beats,
                writes[ordinal] if ordinal < len(writes) else None,
                reads[ordinal] if ordinal < len(reads) else None,
                source.sequence.selector if isinstance(source.sequence, Selected) else None,
            )
            for ordinal in range(max(len(writes), len(reads)))
        )
    )


def size(
    request: Request, *, channels: tuple[str, ...] | None = None, host: HostCriterion | None = None
) -> Result[ResultData] | Unavailable:
    """Coordinatewise lowering; direct tested separately from monotone FIFO bisection.

    Only requested variables move. Fixed others, hardware minima/maxima and the
    all-channel-unbounded finite reference remain fixed across every whole-run call.
    Custom predicates require an explicit monotonicity contract; exceptions escape.
    """
    if (
        request.criterion.mode == "trace"
        or request.criterion.mode == "host"
        and not request.criterion.monotone
    ):
        return Unsupported(
            "sizing-criterion",
            request.timed.plan.id,
            "sizing requires a finite criterion or host monotonicity contract",
        )
    ports = {p.id: p for p in request.timed.plan.ports}
    names = (
        tuple(
            e.id
            for e in request.timed.plan.edges
            if not request.execution.nodes or ports[e.producer].node in request.execution.nodes
        )
        if channels is None
        else tuple(channels)
    )
    if (
        not names
        or len(set(names)) != len(names)
        or not set(names) <= {e.id for e in request.timed.plan.edges}
    ):
        return _reject("sizing-order", request.timed.plan.id, "name target channels once")
    resolved = dict(request.capacities)
    policies = {c.edge: c for c in request.timed.attachment.channels}
    # Targets' unknown capacity is an explicitly requested sizing variable.
    for name in names:
        policy = policies.get(name)
        if policy is not None and policy.capacity.state != "unspecified":
            return _reject(
                "sizing-fixed",
                name,
                "only explicitly unspecified channel capacities are sizing variables",
            )
        if policy is not None and name not in resolved and policy.capacity.state == "unspecified":
            direct = _rules(request).direct_capacity if request.timed.attachment.rules else 0
            resolved[name] = Capacity("fixed", max(policy.minimum, direct))
    prepared = replace(request, capacities=tuple(resolved.items()))
    ready = readiness(
        prepared.timed,
        prepared.execution.nodes or None,
        conditional=prepared.execution.conditional,
        capacities=prepared.capacities,
    )
    if not isinstance(ready, Available):
        return ready
    if not set(names) <= set(ready.value.edges):
        return _reject(
            "sizing-order", request.timed.plan.id, "sizing variables belong to the query slice"
        )
    base = _network(prepared, ready.value)
    if not isinstance(base, Available):
        return base
    ref = _reference(request, base.value)
    if not isinstance(ref, Available):
        return ref
    judge = _judge(request, ref.value, host)
    if not isinstance(judge, Available):
        return judge
    rules = _rules(request)
    target_free = engine.run(
        base.value.at({name: None for name in names}), rules, request.execution.frames
    )
    values = {}
    for name in names:
        upper = max(target_free.peak(name), rules.direct_capacity, policies[name].minimum)
        maximum = policies[name].maximum
        if maximum is not None:
            upper = min(upper, maximum)
        values[name] = upper
    network = base.value.at(values)

    def kept(candidate: engine.Network) -> bool:
        return judge.value(engine.run(candidate, rules, request.execution.frames))

    if not kept(network):
        return _reject(
            "sizing-upper",
            request.timed.plan.id,
            "bounded upper capacities do not satisfy the criterion",
        )
    moved = True
    while moved:
        moved = False
        for name in names:
            upper = values[name]
            low = max(rules.direct_capacity, policies[name].minimum)
            if low == rules.direct_capacity and kept(network.at({name: low})):
                best = low
            else:
                lower = max(low, rules.direct_capacity + 1)
                if lower > upper:
                    best = upper
                else:
                    candidates = range(lower, upper + 1)
                    index = bisect_left(
                        candidates,
                        True,
                        hi=len(candidates) - 1,
                        key=lambda cap: kept(network.at({name: cap})),
                    )
                    best = candidates[index]
            if best < upper:
                moved = True
                values[name] = best
                network = network.at({name: best})
    found = engine.run(network, rules, request.execution.frames)
    return _bounded_result(
        _result(request, ready.value, network, found, judge.value(found), ref.value, names)
    )
