# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""What every world of one model shares: a bit per choice and fact, and a memo of
authored functions by what they read.

An evaluation records which choices and facts it read as a **read set**, a bit set
over the model's choices (``decision`` nodes) and facts (root ``param`` nodes). A bit
is allocated the first time a node is read or changed (``BitTable``), so a model pays
only for the choices and facts it reads or changes.

An authored function is pure: it answers from what it reads alone. So its answer is
kept across every world of the model (``Memo``): an explicit function by the answers
of its arguments, a ``self`` method by its **trace** (each read it made, in order, with
the answer it got). Answers are compared by identity: a world shares the answer
objects of what it did not change, and early cutoff keeps an answer object when what
it read answers the same, so a key built of identities keeps finding its entry.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar

from .ir import NodeId
from .results import QueryResult

# A bit set over a ``BitTable``'s choices and facts.
ReadSet: TypeAlias = int

# What an evaluation read, in order, with the answer each read got.
Trace: TypeAlias = tuple[tuple[NodeId, QueryResult[object]], ...]

T = TypeVar("T")

# How many traces a method's memo keeps, newest last.
TRACES_KEPT = 8


class BitTable:
    """A bit per choice or fact of one model, allocated on first use."""

    __slots__ = ("_bits", "_nodes")

    def __init__(self) -> None:
        self._bits: dict[NodeId, int] = {}
        self._nodes: list[NodeId] = []

    def bit(self, node: NodeId) -> int:
        """The bit of ``node`` (a choice or a fact); the first call allocates it."""
        found = self._bits.get(node)
        if found is None:
            found = self._bits[node] = 1 << len(self._nodes)
            self._nodes.append(node)
        return found

    def bits(self, nodes: Iterable[NodeId]) -> ReadSet:
        found = 0
        for node in nodes:
            found |= self.bit(node)
        return found

    def nodes(self, reads: ReadSet) -> tuple[NodeId, ...]:
        """The choices and facts a read set holds, in the order their bits were allocated."""
        found: list[NodeId] = []
        while reads:
            lowest = reads & -reads
            found.append(self._nodes[lowest.bit_length() - 1])
            reads ^= lowest
        return tuple(found)

    def __len__(self) -> int:
        return len(self._nodes)


@dataclass(slots=True)
class Counters:
    """The work of one model's evaluations: nodes evaluated, entries taken from a parent
    world, entries kept by early cutoff (and how the answers were compared), and calls
    of authored functions and memo hits, by the kind of what was called (a node's kind,
    or a domain's membership or listing)."""

    evaluated: int = 0
    inherited: int = 0
    cut: int = 0
    compared: int = 0
    identical: int = 0
    semantic: int = 0
    calls: dict[str, int] = field(default_factory=dict)
    hits: dict[str, int] = field(default_factory=dict)

    def call(self, kind: str) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + 1

    def hit(self, kind: str) -> None:
        self.hits[kind] = self.hits.get(kind, 0) + 1

    def reset(self) -> None:
        self.evaluated = self.inherited = self.cut = 0
        self.compared = self.identical = self.semantic = 0
        self.calls.clear()
        self.hits.clear()


@dataclass(frozen=True, slots=True)
class _Held(Generic[T]):
    """A memo entry: the answers it was found from (held, so an identity reused after
    they are collected never matches) and what was found."""

    answers: tuple[object, ...]
    outcome: T


class Memo(Generic[T]):
    """Authored functions' outcomes across every world of one model, by what they read.

    ``explicit``: a function of its arguments, by the identities of their answers, per
    node and purpose (its call; a Decision's domain membership or listing). ``methods``:
    a ``self`` method's traces, newest last, each with its outcome; a caller replays a
    trace in its world, read by read, and takes the outcome where every read answers the
    same."""

    __slots__ = ("_explicit", "_methods")

    def __init__(self) -> None:
        self._explicit: dict[tuple[NodeId, str], dict[tuple[int, ...], _Held[T]]] = {}
        self._methods: dict[NodeId, list[tuple[Trace, T]]] = {}

    def explicit(self, node: NodeId, purpose: str, answers: tuple[object, ...]) -> _Held[T] | None:
        """What ``node``'s function found for these very answers, if it was called on them."""
        table = self._explicit.get((node, purpose))
        if table is None:
            return None
        held = table.get(tuple(map(id, answers)))
        if held is None or any(a is not b for a, b in zip(held.answers, answers)):
            return None
        return held

    def keep_explicit(
        self, node: NodeId, purpose: str, answers: tuple[object, ...], outcome: T
    ) -> None:
        table = self._explicit.setdefault((node, purpose), {})
        table[tuple(map(id, answers))] = _Held(answers, outcome)

    def traces(self, node: NodeId) -> tuple[tuple[Trace, T], ...]:
        """A method's kept traces and outcomes, newest first."""
        return tuple(reversed(self._methods.get(node, ())))

    def keep_method(self, node: NodeId, trace: Trace, outcome: T) -> None:
        traces = self._methods.setdefault(node, [])
        traces.append((trace, outcome))
        if len(traces) > TRACES_KEPT:
            del traces[0]


__all__ = ["TRACES_KEPT", "BitTable", "Counters", "Memo", "ReadSet", "Trace"]
