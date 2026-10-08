# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The bind cache: node-root points keyed by the facts they were bound from, by value.

qonnx builds a fresh op instance for every query, and a point answers each query
from what it has evaluated (its views, its forced Decisions), which a fresh
binding would derive again. So an op binds through one process-wide cache,
three kinds of point, all of its node root:

- **on its inputs**, from the kernel's formals, the tensor of each input channel
  the graph states (``x_tensor``, ``w_tensor``) and the value of each
  initializer the node owns (``w_contents``), its outputs absent: what
  inference reads (``result_tensor``, which may derive from an input channel's
  value), before any output of the node is known;
- **whole**, nothing chosen: the same and each output's tensor (``y_tensor``);
- whole with **choices** replayed, because replay evaluates a new
  configuration from its facts again.

A key is the facts by value, an initializer by its value summary's
``content_digest``, beside the class it binds, and the tensors bound.

Points are immutable (replay returns successors), so sharing one across op
instances and models is sound, and keys are values, so nothing is ever
invalidated: changed facts are another key. The bound is memory: a
least-recently-used cache of entries, base and replayed points alike
(``LeastRecentlyUsed``, which also keeps the composite classes of the Partitions,
``finn.custom_op.kernels.partition``, and of the shell roots, ``shell``).
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
    """What binding a node reads: its op's node root class; the key that identifies the
    kernel's formals and the node's values by value; the formals (a thunk, called only
    on a miss); the tensor of each input channel and of each output channel, by port, as
    the graph states them (thunks: an output's is known only once inference wrote it);
    the value of each parameter port the node owns (an initializer's), by port (a thunk);
    and those ports."""

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
        """The point ``build`` replays ``choices`` onto from the base point; a refusal
        raises out of ``build`` and caches nothing."""
        chosen = tuple(sorted((key, type(value).__name__, value) for key, value in choices.items()))
        key, _ = self._bound(facts, outputs=True)
        return self.get((*key, chosen), lambda: build(self.point(facts)))


BIND_CACHE = BindCache()
"""The process's cache, which every KernelOp binds through."""


__all__ = ["BIND_CACHE", "BindCache", "Facts", "LeastRecentlyUsed"]
