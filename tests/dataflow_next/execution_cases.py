# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared logical request and independently stated current-engine case fixtures."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, cast

from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow_next import timing as engine
from finn.dataflow_next.execution import (
    Criterion,
    Execution,
    Flags,
    IndexStream,
    Program,
    Request,
)
from finn.dataflow_next.logical import Edge, LogicalPlan, Node, Port, Source, Transfer
from finn.dataflow_next.profiles import (
    Attachment,
    Capacity,
    ChannelProfile,
    ChannelRules,
    Clock,
    Evidence,
    Launch,
    Profile,
    TimedPlan,
    configuration_digest,
)
from finn.dataflow_next.tensor import ScalarEncoding, Tensor
from finn.dataflow_next.traversal import BeatSequence, LevelEnd, vector_major

ENCODING = ScalarEncoding(resolve_qonnx_datatype_name("INT8"))


EVIDENCE = Evidence("python-event-1", "1", "validated", ("independent arithmetic fixture",))


def fixture(
    stages: tuple[engine.Stage, ...],
    channels: tuple[engine.Channel, ...],
    programs: tuple[Program, ...],
    *,
    rules: engine.Rules = engine.FINNLIB,
    frames: int = 4,
    counts: dict[str, tuple[int, int]] | None = None,
) -> tuple[Request, engine.Network]:
    """Logical fixture construction uses declared port counts, not adapter outputs.

    Each materialized stage's output is a distinct logical source. Timing asserts
    token movements, not the computation/value relation between those sources.
    """
    reference = engine.Network(stages, channels)
    words = dict(reference.words)
    ports: list[Port] = []
    sources: list[Source] = []
    transfers: list[Transfer] = []
    for program in programs:
        for direction, names in (("write", program.outputs), ("read", program.inputs)):
            for index, name in enumerate(names):
                link = next(
                    c
                    for c in channels
                    if (
                        c.producer == program.node and c.output == index
                        if direction == "write"
                        else c.consumer == program.node and c.input == index
                    )
                )
                producer = next(p for p in programs if p.node == link.producer)
                source_id = producer.outputs[link.output] + ".source"
                lanes = (counts or {}).get(link.producer, (1, 1))[1]
                tensor = Tensor((words[link.name] * lanes,), ENCODING)
                sequence = BeatSequence(
                    vector_major(tensor.shape, lanes), markers=(LevelEnd(words[link.name]),)
                )
                ports.append(
                    Port(
                        name,
                        program.node,
                        cast(Literal["read", "write"], direction),
                        source_id,
                        tensor,
                        sequence,
                    )
                )
                if direction == "write":
                    sources.append(Source(source_id, tensor))
        if program.opcode == "kernel":
            groups = set()
            for at in {v for s in (*program.reads, *program.writes) for v in s.expand()}:
                groups.add(
                    (
                        tuple(p for p, s in zip(program.inputs, program.reads) if at in s.expand()),
                        tuple(
                            p for p, s in zip(program.outputs, program.writes) if at in s.expand()
                        ),
                    )
                )
        elif program.opcode == "insert":
            groups = {(program.inputs, program.outputs), ((), program.outputs)}
        elif program.opcode == "drop":
            groups = {(program.inputs, program.outputs), (program.inputs, ())}
        elif program.opcode == "merge":
            groups = {((p,), program.outputs) for p in program.inputs}
        else:
            groups = {(program.inputs, program.outputs)}
        for index, (reads, writes) in enumerate(sorted(groups)):
            transfers.append(Transfer(f"{program.node}.fire{index}", program.node, reads, writes))
    p = LogicalPlan(
        "test-plan",
        tuple(Node(s.name) for s in stages),
        tuple(ports),
        tuple(
            Edge(
                c.name,
                next(s for s in programs if s.node == c.producer).outputs[c.output],
                next(s for s in programs if s.node == c.consumer).inputs[c.input],
            )
            for c in channels
        ),
        tuple(sources),
        tuple(transfers),
    )
    profiles = []
    for stage, program in zip(stages, programs):
        profiles.append(
            Profile(
                stage.name,
                configuration_digest(p, stage.name),
                EVIDENCE,
                program.opcode,
                "clk",
                Launch(1, 0),
                stage.latency if isinstance(stage, engine.Kernel) else 0,
                stage.capacity
                if isinstance(stage, engine.ReorderStage)
                else (stage.latency if isinstance(stage, engine.Kernel) else None),
                stage.span if isinstance(stage, engine.Source) else None,
                stage.period if isinstance(stage, engine.Source) else None,
            )
        )
    a = Attachment(
        p.digest,
        tuple(profiles),
        tuple(
            ChannelProfile(
                c.name,
                "clk",
                Capacity("unbounded") if c.capacity is None else Capacity("fixed", c.capacity),
            )
            for c in channels
        ),
        (Clock("clk", EVIDENCE, 1000),),
        ChannelRules(
            EVIDENCE, rules.direct.value, rules.fifo_latency, rules.slot_free, rules.reorder_latency
        ),
    )
    return Request(TimedPlan(p, a), programs, Execution(frames), Criterion("trace")), reference


def chain(
    kind: str, capacity: int | None = None, *, rules: engine.Rules = engine.FINNLIB
) -> tuple[Request, engine.Network]:
    source = engine.Source("src", (0, 1, 2, 3), 4, 16)
    start = Program("src", "source", (), ("src.out",), order=IndexStream((0,), 4, 1))
    finish = Program("sink", "sink", ("sink.in",), ())
    if kind == "kernel":
        middle: engine.Stage = engine.Kernel("middle", 6, ((0, 1, 3, 5),), ((0, 1, 3, 5),), 2)
        program = Program(
            "middle",
            "kernel",
            ("middle.in",),
            ("middle.out",),
            6,
            (IndexStream((0, 1, 3, 5)),),
            (IndexStream((0, 1, 3, 5)),),
        )
    elif kind == "width":
        middle = engine.WidthStage("middle", 1, 2)
        program = Program("middle", "width", ("middle.in",), ("middle.out",))
    elif kind == "reorder":
        middle = engine.ReorderStage("middle", (0, 2, 1, 3, 3), (1, 1, 3, 3, 4), 3)
        program = Program(
            "middle", "reorder", ("middle.in",), ("middle.out",), order=IndexStream((0, 2, 1, 3, 3))
        )
    elif kind == "insert":
        middle = engine.InsertionStage("middle", (True, False, True, True, False, True))
        program = Program(
            "middle",
            "insert",
            ("middle.in",),
            ("middle.out",),
            mask=Flags((True, False, True, True, False, True)),
        )
    elif kind == "drop":
        middle = engine.DropStage("middle", (True, False, False, True))
        program = Program(
            "middle",
            "drop",
            ("middle.in",),
            ("middle.out",),
            mask=Flags((True, False, False, True)),
        )
    else:
        raise AssertionError(kind)
    return fixture(
        (source, middle, engine.Sink("sink")),
        (
            engine.Channel("a", "src", "middle", capacity),
            engine.Channel("b", "middle", "sink", capacity),
        ),
        (start, program, finish),
        rules=rules,
        counts={"middle": (1, 2)} if kind == "width" else None,
    )


def variables(request: Request, names: tuple[str, ...] = ("a", "b")) -> Request:
    a = request.timed.attachment
    a = replace(
        a,
        channels=tuple(
            replace(c, capacity=Capacity("unspecified")) if c.edge in names else c
            for c in a.channels
        ),
    )
    return replace(request, timed=replace(request.timed, attachment=a))
