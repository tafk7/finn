# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The bind cache: each node's kernel, bound, keyed by the facts it was bound from.

qonnx creates a new op instance for every call, so without a cache a KernelOp would
bind its kernel again each time, and evaluate again what the bound point had already
evaluated. A node's kernel is bound once from its facts, by value: its formals, its
input tensors and the values of the initializers it holds (an initializer by its
content digest). Bound points never change, so one can be shared between instances,
models and networks, and a cached point never goes stale: changed facts make a new key.
Its outputs are the kernel's statement, not a fact, so a node has one binding. The cache
is a bounded LRU (``LeastRecentlyUsed``).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from finn.core.space import design_space
from finn.kernels.base import Kernel

V = TypeVar("V")


@dataclass(frozen=True)
class Facts:
    """What binding a node's kernel reads: its kernel class, a key that identifies the
    facts by value, and a function that gives the kernel's arguments by name (called only
    on a cache miss): its formals, each input port's tensor and each held initializer's
    values. ``owned`` lists the ports whose initializer the node holds."""

    kernel: type[Kernel]
    key: tuple[Hashable, ...]
    arguments: Callable[[], dict[str, object]]
    owned: tuple[str, ...] = ()


class LeastRecentlyUsed(Generic[V]):
    """Values by key, at most ``size``: a hit makes its entry the most recent, a miss
    builds the value and evicts the least recently used past the bound. A build that
    raises caches nothing."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.entries: OrderedDict[Hashable, V] = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key: Hashable, build: Callable[[], V]) -> V:
        found = self.entries.get(key)
        if found is not None:
            self.entries.move_to_end(key)
            self.hits += 1
            return found
        self.misses += 1
        value = build()
        self.entries[key] = value
        while len(self.entries) > self.size:
            self.entries.popitem(last=False)
        return value

    def clear(self) -> None:
        self.entries.clear()
        self.hits = self.misses = 0


class BindCache(LeastRecentlyUsed[Kernel]):
    """Each node's kernel bound from its facts, nothing chosen; a bounded LRU."""

    def __init__(self, size: int = 128) -> None:
        super().__init__(size)

    def bound(self, facts: Facts) -> Kernel:
        """The kernel bound from ``facts``, nothing chosen: one binding per set of facts."""

        def bind() -> Kernel:
            point: Kernel = design_space(cast(Any, facts.kernel)(**facts.arguments()))
            return point

        return self.get((facts.kernel, *facts.key), bind)


BIND_CACHE = BindCache()
"""The process's cache, which every KernelOp binds through."""


__all__ = ["BIND_CACHE", "BindCache", "Facts", "LeastRecentlyUsed"]
