# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The bind cache: node-root points keyed by the facts they were bound from, by value.

qonnx builds a fresh op instance for every query, and binding a node root reads
the weights into integers (hundreds of milliseconds for 512 x 512). So an op
binds through one process-wide cache. A base point is keyed by its node-root
class and its facts, an initializer by its value summary's ``content_digest``;
a replayed point by the same and its choices, because replay re-derives the
weights' values for every new configuration and costs as much as a bind.

Points are immutable (replay returns successors), so sharing one across op
instances and models is sound, and keys are values, so nothing is ever
invalidated: changed facts are another key. The bound is memory: a
least-recently-used cache of entries, base and replayed points alike.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from finn.core.space import design_space
from finn.kernels.base import Kernel


@dataclass(frozen=True)
class Facts:
    """What binding a node reads: its node-root class, the key that identifies its facts by
    value, its formals (a thunk: the weights become integers only on a miss), and the
    channels the node owns beside its outputs (a stored parameter's, by port name)."""

    root: type[Kernel]
    key: tuple[Hashable, ...]
    formals: Callable[[], dict[str, object]]
    owned: tuple[str, ...] = ()


class BindCache:
    """Base points by facts, replayed points by facts and choices; a bounded LRU."""

    def __init__(self, size: int = 128) -> None:
        self.size = size
        self.entries: OrderedDict[tuple[Hashable, ...], Kernel] = OrderedDict()
        self.hits = self.misses = 0

    def _get(self, key: tuple[Hashable, ...], build: Callable[[], Kernel]) -> Kernel:
        found = self.entries.get(key)
        if found is not None:
            self.entries.move_to_end(key)
            self.hits += 1
            return found
        self.misses += 1
        point = build()
        self.entries[key] = point
        while len(self.entries) > self.size:
            self.entries.popitem(last=False)
        return point

    def point(self, facts: Facts) -> Kernel:
        """The node root bound from ``facts``, nothing chosen."""
        return self._get(
            (facts.root, *facts.key), lambda: design_space(cast(Any, facts.root)(**facts.formals()))
        )

    def configured(
        self, facts: Facts, choices: Mapping[str, object], build: Callable[[Kernel], Kernel]
    ) -> Kernel:
        """The point ``build`` replays ``choices`` onto from the base point; a refusal
        raises out of ``build`` and caches nothing."""
        chosen = tuple(sorted((key, type(value).__name__, value) for key, value in choices.items()))
        return self._get((facts.root, *facts.key, chosen), lambda: build(self.point(facts)))

    def clear(self) -> None:
        self.entries.clear()
        self.hits = self.misses = 0


BIND_CACHE = BindCache()
"""The process's cache, which every KernelOp binds through."""


__all__ = ["BIND_CACHE", "BindCache", "Facts"]
