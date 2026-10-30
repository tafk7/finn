# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Real partition construction and source emission shared by the untimed examples.

This is example code, not a production extractor or executor. It copies the actual
channel presentations into a logical plan for per-edge queries. It does not infer
atomic firing groups, timing profiles, clock rates, or runtime input mechanisms.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from finn.core.search import Key
from finn.core.space import Available, Inapplicable, inspection
from finn.dataflow.descriptors import dump
from finn.dataflow.logical import Available as LogicalAvailable
from finn.dataflow.logical import (
    Edge,
    LogicalPlan,
    Node,
    Port,
    Source,
    adapters,
    packing,
)
from finn.dataflow.steps import Steps
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import Tensor
from finn.kernels.artifacts.build import emit_module
from finn.kernels.artifacts.module import ProducerIdentity
from finn.kernels.composer import PartitionDeclaration
from finn.kernels.fifo import FifoKernel
from finn.kernels.partition import Partition, partition
from finn.kernels.sides import Side, SideKind
from finn.platform.capabilities import DspBlock, Fabric, Platform
from finn.resources import finnlib_root

PLATFORM = Platform(
    period_ns=5.0,
    dsp=DspBlock.DSP48E2,
    fabric=Fabric.ULTRASCALE,
    uram=True,
    uram_init=True,
    resources=None,
)
# This is a required physical clock target, not a cycle estimate or achieved frequency.


def configured(declaration: PartitionDeclaration, choices: dict[Key, object]) -> Partition:
    """Commit explicit physical choices; apply existing policies only to channels."""
    network = partition(declaration).with_choices(choices)
    done = network.defaulted(members=network.channels)
    if done.unset or done.refused or done.fallbacks:
        raise ValueError(f"channel policy did not resolve cleanly: {done}")
    return network.at(done.point)


def logical_plan(network: Partition) -> LogicalPlan:
    """Per-channel topology from real streams, with distinct source identities."""
    nodes: dict[str, Node] = {}
    ports: list[Port] = []
    edges: list[Edge] = []
    sources: list[Source] = []
    for name in network.channels:
        channel = network.member(name)
        sides = channel.sides
        value_id = f"value/{name}"
        sources.append(Source(value_id, channel.tensor))
        ends: tuple[tuple[Literal["write", "read"], Side, str, Stream], ...] = (
            ("write", sides.producer, sides.source_owner, sides.source),
            ("read", sides.consumer, sides.sink_owner, sides.sink),
        )
        ids = []
        for direction, side, owner, stream in ends:
            if side.kind is SideKind.BOUNDARY:
                node_id = f"boundary/{owner}"
            elif side.kind is SideKind.SOURCE:
                node_id = f"source/{name}/{owner}"
            else:
                node_id = f"kernel/{owner.rpartition('.')[0]}"
            nodes.setdefault(node_id, Node(node_id))
            port_id = f"port/{name}/{direction}"
            ids.append(port_id)
            tensor = Tensor(stream.sequence.form.shape, stream.element)
            ports.append(Port(port_id, node_id, direction, value_id, tensor, stream.sequence))
        edges.append(Edge(f"edge/{name}", ids[0], ids[1]))
    return LogicalPlan("example", tuple(nodes.values()), tuple(ports), tuple(edges), tuple(sources))


def emit(network: Partition, name: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp") / f"finn-untimed-{name}")
    args = parser.parse_args()
    validity = inspection.validity(network.point)
    if validity.result != Available(True):
        raise ValueError(f"design is not completely valid: {validity}")
    waiting = [
        key
        for key, state in inspection.states(network.point)
        if isinstance(state, (inspection.Open, inspection.Waiting))
    ]
    unsettled = [
        owner
        for owner, result in inspection.presence(network.point)
        if not isinstance(result, (Available, Inapplicable))
    ]
    if waiting or unsettled:
        raise ValueError(f"unresolved design: {waiting}, {unsettled}")
    found = network.module(ProducerIdentity("example.untimed", "1"), f"finn_{name}")
    if not isinstance(found, Available):
        raise ValueError(f"module refused or unresolved: {found}")
    plan = logical_plan(network)
    requirements = {}
    capacities = {}
    for name in network.channels:
        result = adapters(plan, f"edge/{name}")
        if not isinstance(result, LogicalAvailable) or not isinstance(result.value, Steps):
            raise ValueError(f"logical requirement not available: {result}")
        channel = network.member(name)
        assert result.value == channel.steps
        requirements[name] = {
            "steps": [kind.value for kind in result.value.kinds],
            "stages": [stage.label for stage in channel.stages],
        }
        fifo = channel.transport
        if fifo is not None:
            capacities[name] = {
                "requested_depth": fifo.depth,
                "realized_words": fifo.buffer.planned_storage.capacity,
            }
    packed = {}
    for port in plan.ports:
        packed_port = packing(plan, port.id)
        assert isinstance(packed_port, LogicalAvailable)
        packed[port.id] = asdict(packed_port.value)
    emitted = emit_module(found.value, args.out / "rtl", roots={"finnlib": Path(finnlib_root())})
    (args.out / "logical-plan.json").write_bytes(dump(plan))
    native_capacities = {
        member: {
            "requested_depth": fifo.depth,
            "realized_words": fifo.planned_storage.capacity,
        }
        for member in network.kernels
        if isinstance(fifo := network.member(member), FifoKernel)
    }
    summary = {
        "module": emitted.entry_point,
        "sources": list(emitted.sources),
        "data": list(emitted.data),
        "logical_digest": plan.digest,
        "packing": packed,
        "requirements": requirements,
        "channel_fifo_capacities": capacities,
        "native_fifo_capacities": native_capacities,
        "timing": "not requested; emission is not a rate or FIFO-sufficiency proof",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
