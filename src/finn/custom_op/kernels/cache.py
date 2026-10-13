# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The bind cache: bound node roots, keyed by the facts they were bound from.

qonnx creates a new op instance for every call, so without a cache a KernelOp would
bind its kernel again each time, and evaluate again what the bound point had already
evaluated. Bound points never change and are keyed by value (an initializer by its
content digest), so one can be shared between instances and models, and a cached point
never goes stale: changed facts make a new key. The cache is a bounded LRU
(``LeastRecentlyUsed``, which the shell roots use too).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from finn.core.space import design_space
from finn.dataflow.tensor import Tensor
from finn.kernels.base import Kernel
from finn.kernels.values.semantics import IntegerTensorValue

V = TypeVar("V")


@dataclass(frozen=True)
class Facts:
    """What binding a node reads: the node root class, a key that identifies the facts
    by value, and functions that give the kernel arguments and the held initializers'
    values (called only on a cache miss) and the channels' tensors by port (an output's
    is known only once inference has written it). ``owned`` lists the ports whose
    initializer the node holds."""

    root: type[Kernel]
    key: tuple[Hashable, ...]
    formals: Callable[[], dict[str, object]]
    inputs: Callable[[], dict[str, Tensor]]
    outputs: Callable[[], dict[str, Tensor]]
    values: Callable[[], dict[str, IntegerTensorValue]] = dict
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
    """Points by facts, replayed points by facts and choices; a bounded LRU."""

    def __init__(self, size: int = 128) -> None:
        super().__init__(size)

    def _bound(
        self, facts: Facts, outputs: bool
    ) -> tuple[tuple[Hashable, ...], Callable[[], Kernel]]:
        """The key of the root bound on its inputs, and its outputs too when ``outputs``;
        and how to bind it."""
        edges = facts.inputs() | (facts.outputs() if outputs else {})
        key = (facts.root, *facts.key, outputs, *sorted(edges.items()))

        def bind() -> Kernel:
            tensors = {f"{port}_tensor": tensor for port, tensor in edges.items()}
            values = {f"{port}_contents": value for port, value in facts.values().items()}
            bound: Kernel = design_space(
                cast(Any, facts.root)(**facts.formals(), **tensors, **values)
            )
            return bound

        return key, bind

    def inputs(self, facts: Facts) -> Kernel:
        """The node root bound on its inputs from ``facts``, its outputs absent."""
        return self.get(*self._bound(facts, outputs=False))

    def point(self, facts: Facts) -> Kernel:
        """The node root bound from ``facts``, nothing chosen."""
        return self.get(*self._bound(facts, outputs=True))

    def configured(
        self, facts: Facts, choices: Mapping[str, object], build: Callable[[Kernel], Kernel]
    ) -> Kernel:
        """The point with ``choices`` applied, made by ``build`` from the base point and
        cached by facts and choices. A refusal raises out of ``build`` and caches
        nothing."""
        chosen = tuple(sorted((key, type(value).__name__, value) for key, value in choices.items()))
        key, _ = self._bound(facts, outputs=True)
        return self.get((*key, chosen), lambda: build(self.point(facts)))


BIND_CACHE = BindCache()
"""The process's cache, which every KernelOp binds through."""


__all__ = ["BIND_CACHE", "BindCache", "Facts", "LeastRecentlyUsed"]
