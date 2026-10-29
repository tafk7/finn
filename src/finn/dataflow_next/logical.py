# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable logical plans and queries, without clocks or latency prerequisites.

IDs identify graph objects, never payload equality or physical pin names. Transfers
state atomic port participation; they do not assert a firing rate or execute a
kernel. Existing descriptors own selection, replay, markers and adapter policy.
An order bound is explicitly scoped to one reorder in source words, not a timed
channel capacity. A plan's digest identifies its exact durable record.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Generic, Literal, TypeAlias, TypeVar

from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.dataflow_next._isl import _Service
from finn.dataflow_next.reindex import Reader, Reindex, Route, route
from finn.dataflow_next.schedule import Schedule
from finn.dataflow_next.steps import Step, Steps, steps
from finn.dataflow_next.tensor import Tensor
from finn.dataflow_next.traversal import BeatSequence, Selected

T = TypeVar("T")


@dataclass(frozen=True)
class Available(Generic[T]):
    value: T
    method: str = "retained-descriptor"


@dataclass(frozen=True)
class Unsupported:
    code: str
    owner: str
    message: str


@dataclass(frozen=True)
class Rejected:
    findings: tuple[Finding, ...]
    message: str


Result: TypeAlias = Available[T] | Unsupported | Rejected


class InvalidPlan(Refusal):
    def __init__(self, code: str, owner: str, message: str) -> None:
        super().__init__((Finding(FindingKind.REJECTION, code, owner, message),), message)


def _id(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise InvalidPlan("id-invalid", "LogicalPlan", "IDs are nonempty strings")


@dataclass(frozen=True)
class Source:
    id: str
    tensor: Tensor

    def __post_init__(self) -> None:
        _id(self.id)
        if not isinstance(self.tensor, Tensor):
            raise TypeError("a source carries a Tensor")


@dataclass(frozen=True)
class Node:
    id: str
    schedule: Schedule | None = None

    def __post_init__(self) -> None:
        _id(self.id)
        if self.schedule is not None and not isinstance(self.schedule, Schedule):
            raise TypeError("a logical schedule is a Schedule")


@dataclass(frozen=True)
class Port:
    id: str
    node: str
    direction: Literal["read", "write"]
    source: str
    tensor: Tensor
    sequence: BeatSequence | Selected

    def __post_init__(self) -> None:
        for value in (self.id, self.node, self.source):
            _id(value)
        if self.direction not in ("read", "write"):
            raise InvalidPlan("direction-invalid", self.id, "a port reads or writes")
        if not isinstance(self.tensor, Tensor) or not isinstance(
            self.sequence, (BeatSequence, Selected)
        ):
            raise TypeError("a port carries a Tensor and a BeatSequence or Selected")
        shape = (
            self.sequence.shape if isinstance(self.sequence, Selected) else self.sequence.form.shape
        )
        if shape != self.tensor.shape:
            raise InvalidPlan("shape-mismatch", self.id, "the sequence must walk its tensor")


@dataclass(frozen=True)
class Edge:
    id: str
    producer: str
    consumer: str
    reader: Reader | None = None

    def __post_init__(self) -> None:
        for value in (self.id, self.producer, self.consumer):
            _id(value)
        if self.reader is not None and not isinstance(self.reader, Reader):
            raise TypeError("an edge's reader is a Reader")


@dataclass(frozen=True)
class Transfer:
    """One atomic participation group; a fork names all write ports together.

    Multiple groups express separate firings. Schedule/port descriptors retain
    iteration ordering. This record neither schedules nor measures cycles.
    """

    id: str
    node: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]

    def __post_init__(self) -> None:
        _id(self.id)
        _id(self.node)
        object.__setattr__(self, "reads", tuple(self.reads))
        object.__setattr__(self, "writes", tuple(self.writes))
        names = (*self.reads, *self.writes)
        if not names or len(set(names)) != len(names):
            raise InvalidPlan("participation-invalid", self.id, "name participating ports once")
        for name in names:
            _id(name)


@dataclass(frozen=True)
class LogicalPlan:
    id: str
    nodes: tuple[Node, ...]
    ports: tuple[Port, ...]
    edges: tuple[Edge, ...]
    sources: tuple[Source, ...]
    transfers: tuple[Transfer, ...] = ()
    occurrence_policy: Literal["first-in-pass"] = "first-in-pass"

    def __post_init__(self) -> None:
        _id(self.id)
        if self.occurrence_policy != "first-in-pass":
            raise InvalidPlan(
                "occurrence-policy", self.id, "only first-in-pass association is supported"
            )
        for name, kind in (
            ("nodes", Node),
            ("ports", Port),
            ("edges", Edge),
            ("sources", Source),
            ("transfers", Transfer),
        ):
            values = tuple(getattr(self, name))
            if not all(isinstance(value, kind) for value in values):
                raise TypeError(f"{name} contains {kind.__name__} values")
            object.__setattr__(self, name, values)
        # One namespace avoids cross-kind graph-path assignment collisions.
        ids = [self.id]
        ids.extend(v.id for v in self.nodes)
        ids.extend(v.id for v in self.ports)
        ids.extend(v.id for v in self.edges)
        ids.extend(v.id for v in self.sources)
        ids.extend(v.id for v in self.transfers)
        if len(set(ids)) != len(ids):
            raise InvalidPlan("id-collision", self.id, "graph IDs must be unique")
        nodes, ports, sources = (
            {v.id: v for v in self.nodes},
            {v.id: v for v in self.ports},
            {v.id: v for v in self.sources},
        )
        producers: dict[str, str] = {}
        for port in self.ports:
            if port.node not in nodes or port.source not in sources:
                raise InvalidPlan("reference-unknown", port.id, "unknown node or source")
            if port.direction == "write":
                tensor = sources[port.source].tensor
                if port.tensor.shape != tensor.shape or not port.tensor.element.fits(
                    tensor.element
                ):
                    raise InvalidPlan(
                        "source-mismatch", port.id, "writer values must fit their source tensor"
                    )
                if port.source in producers and producers[port.source] != port.node:
                    raise InvalidPlan(
                        "multiple-producers", port.id, "one source has one producing node"
                    )
                producers[port.source] = port.node
        consumers: set[str] = set()
        for edge in self.edges:
            if edge.producer not in ports or edge.consumer not in ports:
                raise InvalidPlan("reference-unknown", edge.id, "unknown port")
            source, sink = ports[edge.producer], ports[edge.consumer]
            if source.direction != "write" or sink.direction != "read":
                raise InvalidPlan("direction-invalid", edge.id, "an edge joins write to read")
            if sink.id in consumers:
                raise InvalidPlan("multiple-producers", edge.id, "a reader has one producer")
            consumers.add(sink.id)
            if source.source != sink.source:
                raise InvalidPlan("source-mismatch", edge.id, "an edge retains its source ID")
            if not sources[source.source].tensor.element.fits(sink.tensor.element):
                raise InvalidPlan(
                    "encoding-mismatch", edge.id, "produced values must fit the reader"
                )
            if edge.reader is None:
                if source.tensor.shape != sink.tensor.shape:
                    raise InvalidPlan("shape-mismatch", edge.id, "an unmapped edge retains shape")
            elif isinstance(source.sequence, Selected) or isinstance(sink.sequence, Selected):
                raise InvalidPlan(
                    "selection-not-shared", edge.id, "mapped selected edges are unsupported"
                )
            elif edge.reader.tensor != source.tensor.shape or edge.reader.stream != sink.sequence:
                raise InvalidPlan(
                    "reader-mismatch", edge.id, "reader descriptor must match both ports"
                )
            if edge.reader is not None:
                reader = edge.reader
                encoding = sink.tensor.element
                span = encoding.value_range
                for fill in _Service().presented_fills(
                    reader.map, reader.stream.form, reader.at_producer, reader.source
                ):
                    if not encoding.dtype.allowed(fill) or not span.least <= fill <= span.greatest:
                        raise InvalidPlan(
                            "encoding-mismatch",
                            edge.id,
                            f"reader-presented fill {fill} does not fit {encoding}",
                        )
        for transfer in self.transfers:
            if transfer.node not in nodes:
                raise InvalidPlan("reference-unknown", transfer.id, "unknown node")
            for direction, names in (("read", transfer.reads), ("write", transfer.writes)):
                for name in names:
                    if name not in ports:
                        raise InvalidPlan("reference-unknown", transfer.id, "unknown port")
                    port = ports[name]
                    if port.node != transfer.node or port.direction != direction:
                        raise InvalidPlan(
                            "participation-invalid", transfer.id, "wrong owner or direction"
                        )

    def record(self) -> bytes:
        from finn.dataflow_next.descriptors import (  # noqa: PLC0415 -- codec closes over plan types
            dump,
        )

        return dump(self)

    @property
    def digest(self) -> str:
        return sha256(self.record()).hexdigest()


@dataclass(frozen=True)
class Packing:
    port: str
    element_bits: int
    lanes: int
    payload_bits: int
    beats_per_pass: int
    selected_sets: int | None


@dataclass(frozen=True)
class OrderBound:
    value: int
    source: str
    edge: str
    segment: str
    hop: int
    unit: Literal["source-words"] = "source-words"
    scope: Literal["finite-reorder-frame"] = "finite-reorder-frame"


def packing(plan: LogicalPlan, port_id: str) -> Result[Packing]:
    port = next((v for v in plan.ports if v.id == port_id), None)
    if port is None:
        return _reject("reference-unknown", port_id, "unknown port")
    selected = port.sequence if isinstance(port.sequence, Selected) else None
    sequence = selected.sequence if selected else port.sequence
    assert isinstance(sequence, BeatSequence)
    form = sequence.form
    return Available(
        Packing(
            port.id,
            port.tensor.element.bits,
            form.lanes,
            port.tensor.element.bits * form.lanes,
            form.beats,
            selected.sets if selected else None,
        )
    )


def _reject(code: str, owner: str, message: str) -> Rejected:
    return Rejected((Finding(FindingKind.REJECTION, code, owner, message),), message)


def adapters(plan: LogicalPlan, edge_id: str) -> Result[Steps | Route]:
    """Derive this reader's route. Fanout stays separate per-reader edges.

    Retained extraction/factoring is reported, not claimed to be native algebra.
    No hardware realization or timed liveness certificate is implied.
    """
    edge = next((v for v in plan.edges if v.id == edge_id), None)
    if edge is None:
        return _reject("reference-unknown", edge_id, "unknown edge")
    ports = {v.id: v for v in plan.ports}
    source, sink = ports[edge.producer], ports[edge.consumer]
    try:
        if edge.reader is None:
            return Available(steps(source.sequence, sink.sequence))
        assert isinstance(source.sequence, BeatSequence)
        if len(edge.reader.map.sources) != 1:
            return Unsupported("merge-query", edge.id, "query a joined Merge descriptor separately")
        found = route(Reindex(source.sequence, (edge.reader,)))
        assert isinstance(found, Route)
        return Available(found, "retained-factoring")
    except Refusal as error:
        return Rejected(tuple(error.findings), str(error))
    except ValueError as error:
        return _reject("adapter-unrealizable", edge.id, str(error))


def order_storage(plan: LogicalPlan, edge_id: str) -> Result[tuple[OrderBound, ...]]:
    """Each reorder's source-word bound separately; widths and stages are not added."""
    derived = adapters(plan, edge_id)
    if not isinstance(derived, Available):
        return derived
    edge = next(v for v in plan.edges if v.id == edge_id)
    source = next(v.source for v in plan.ports if v.id == edge.producer)
    chains = (
        (("before", derived.value.before), ("after", derived.value.after))
        if isinstance(derived.value, Route)
        else (("direct", derived.value),)
    )
    bounds = []
    service = _Service()
    for segment, chain in chains:
        for number, hop in enumerate(chain.hops):
            if hop.step is Step.REORDER:
                assert hop.reorder is not None
                bounds.append(
                    OrderBound(service.reorder_bound(hop.reorder), source, edge.id, segment, number)
                )
    return Available(tuple(bounds), "native-bound;retained-route-factoring")
