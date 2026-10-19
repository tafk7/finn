# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""What every world of one model shares: a bit per choice and fact, and a memo of
authored functions by what they read.

An evaluation records which choices and facts it read as a **read set**, a bit set
over the model's choices (``decision`` nodes) and facts (root ``param`` nodes). A bit
is allocated the first time a node is read or changed (``BitTable``), so a model pays
only for the choices and facts it reads or changes.

An authored function is pure: it answers from what it reads alone. So its outcome is
kept across every world of the model (``Memo``), with the **trace** it was found from:
each read it made, in order, with the answer it got. One rule serves every authored
function: an explicit function's trace is its arguments' answers (a domain's, its
inputs'), a ``self`` method's the reads it made. A caller finds a kept outcome where
every read of its trace answers the same in its world: the same answer object (the fast
path: a world shares the answer objects of what it did not change, and early cutoff and
the memo keep an answer object where what it read answers the same), else equal by the
read node's value semantics. So two configurations bound apart share what they compute
alike. Each function keeps a few traces (``TRACES_KEPT``), and the model's memo a budget
of them in all (``BUDGET``), the least recently used dropped first; a trace holds its
answers, so an identity reused after they are collected never matches.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar

from .ir import NodeId
from .results import QueryResult

# A bit set over a ``BitTable``'s choices and facts.
ReadSet: TypeAlias = int

# What an evaluation read, in order, with the answer each read got.
Trace: TypeAlias = tuple[tuple[NodeId, QueryResult[object]], ...]

# What a memo keeps outcomes of: a node, and what its function is called for (its call;
# a Decision's domain membership, listing, enumeration or order).
Key: TypeAlias = tuple[NodeId, str]

T = TypeVar("T")

# How many traces one function keeps, newest first.
TRACES_KEPT = 8

# How many traces a model's memo keeps in all, the least recently used dropped first.
# Measured on a chain of 64 kernels and their channels composed as one Space: configuring
# it keeps about 12,800 traces, about 5 MB (some 400 bytes a trace, its answers shared
# with the worlds); 40 changes add about 1,500, and the possible values of one kernel's
# choices about 200. Below what configuring keeps, a later binding of the same Space finds
# little: those possible values, configured again without the kernel's choices, took 13 s
# with a budget of 5,000 traces (as with none) against 5 s. A budget of 100,000 keeps such
# a configuration and a few thousand changes of exploration whole, so nothing measured is
# dropped, and bounds what a model keeps for the life of the process at about 40 MB.
BUDGET = 100_000


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
    or a domain's membership, listing, enumeration or order) and by node (``calls_at``,
    ``hits_at``: a report tells apart where the nodes sit)."""

    evaluated: int = 0
    inherited: int = 0
    cut: int = 0
    compared: int = 0
    identical: int = 0
    semantic: int = 0
    calls: dict[str, int] = field(default_factory=dict)
    hits: dict[str, int] = field(default_factory=dict)
    calls_at: dict[NodeId, int] = field(default_factory=dict)
    hits_at: dict[NodeId, int] = field(default_factory=dict)

    def call(self, kind: str, node: NodeId) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + 1
        self.calls_at[node] = self.calls_at.get(node, 0) + 1

    def hit(self, kind: str, node: NodeId) -> None:
        self.hits[kind] = self.hits.get(kind, 0) + 1
        self.hits_at[node] = self.hits_at.get(node, 0) + 1

    def reset(self) -> None:
        self.evaluated = self.inherited = self.cut = 0
        self.compared = self.identical = self.semantic = 0
        self.calls.clear()
        self.hits.clear()
        self.calls_at.clear()
        self.hits_at.clear()


@dataclass(slots=True, eq=False)
class Kept(Generic[T]):
    """One outcome a memo keeps: the function's key, the trace it was found from (its
    answers held), and the outcome."""

    key: Key
    trace: Trace
    outcome: T


class Memo(Generic[T]):
    """Authored functions' outcomes across every world of one model, by the traces they
    were found from: per function (``Key``), its newest ``TRACES_KEPT``; in all, at most
    ``budget``, the least recently used (kept or found) dropped first."""

    __slots__ = ("_kept", "_recent", "_budget")

    def __init__(self, budget: int = BUDGET) -> None:
        self._kept: dict[Key, list[Kept[T]]] = {}
        self._recent: OrderedDict[Kept[T], None] = OrderedDict()
        self._budget = budget

    @property
    def budget(self) -> int:
        """How many traces the memo keeps in all; lowering it drops the least recently
        used at once."""
        return self._budget

    @budget.setter
    def budget(self, budget: int) -> None:
        self._budget = budget
        self._trim()

    def __len__(self) -> int:
        return len(self._recent)

    def kept(self, key: Key) -> tuple[Kept[T], ...]:
        """A function's kept traces and outcomes, newest first."""
        return tuple(reversed(self._kept.get(key, ())))

    def used(self, kept: Kept[T]) -> None:
        """``kept`` was found: the most recently used (unless it was dropped meanwhile)."""
        if kept in self._recent:
            self._recent.move_to_end(kept)

    def keep(self, key: Key, trace: Trace, outcome: T) -> None:
        """Keep an outcome and its trace: the function's newest, the most recently used."""
        kept = Kept(key, trace, outcome)
        traces = self._kept.setdefault(key, [])
        traces.append(kept)
        self._recent[kept] = None
        if len(traces) > TRACES_KEPT:
            del self._recent[traces.pop(0)]
        self._trim()

    def _trim(self) -> None:
        while len(self._recent) > self._budget:
            oldest, _ = self._recent.popitem(last=False)
            traces = self._kept[oldest.key]
            traces.remove(oldest)
            if not traces:
                del self._kept[oldest.key]


__all__ = [
    "BUDGET",
    "TRACES_KEPT",
    "BitTable",
    "Counters",
    "Key",
    "Kept",
    "Memo",
    "ReadSet",
    "Trace",
]
