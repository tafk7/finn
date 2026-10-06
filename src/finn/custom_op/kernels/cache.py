# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The bind cache: kernel and node-root points keyed by the facts they were bound from, by value.

qonnx builds a fresh op instance for every query, and a point answers each query
from what it has evaluated (its views, its forced Decisions), which a fresh
binding would derive again. So an op binds through one process-wide cache,
three kinds of point:

- its **kernel** alone, from the kernel's formals: what inference reads (the
  fact-level views, ``result_tensor``), before any output of the node is known;
- its **node root**, nothing chosen, from the same formals and the tensor of
  each edge the graph states (``x_tensor``, ``y_tensor``);
- a node root with **choices** replayed, because replay evaluates a new
  configuration from its facts again.

A key is the facts by value, an initializer by its value summary's
``content_digest``, beside the class it binds.

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
from finn.dataflow.tensor import Tensor
from finn.kernels.base import Kernel


@dataclass(frozen=True)
class Facts:
    """What binding a node reads: its op's node root and kernel classes; the key that
    identifies the kernel's formals by value; the formals (a thunk, called only on a miss);
    the tensor of each edge, by port, as the graph states it (a thunk: an output's is known
    only once inference wrote it); and the parameter ports whose value the node owns (an
    initializer's)."""

    root: type[Kernel]
    kernel: type[Kernel]
    key: tuple[Hashable, ...]
    formals: Callable[[], dict[str, object]]
    edges: Callable[[], dict[str, Tensor]]
    owned: tuple[str, ...] = ()


class BindCache:
    """Kernels and base points by facts, replayed points by facts and choices; a bounded
    LRU."""

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

    def kernel(self, facts: Facts) -> Kernel:
        """The op's kernel alone, bound from the formals in ``facts``."""
        return self._get(
            (facts.kernel, *facts.key),
            lambda: design_space(cast(Any, facts.kernel)(**facts.formals())),
        )

    def _root_key(self, facts: Facts) -> tuple[tuple[Hashable, ...], dict[str, Tensor]]:
        edges = facts.edges()
        return (facts.root, *facts.key, *sorted(edges.items())), edges

    def point(self, facts: Facts) -> Kernel:
        """The node root bound from ``facts``, nothing chosen."""
        key, edges = self._root_key(facts)
        tensors = {f"{port}_tensor": tensor for port, tensor in edges.items()}
        return self._get(
            key, lambda: design_space(cast(Any, facts.root)(**facts.formals(), **tensors))
        )

    def configured(
        self, facts: Facts, choices: Mapping[str, object], build: Callable[[Kernel], Kernel]
    ) -> Kernel:
        """The point ``build`` replays ``choices`` onto from the base point; a refusal
        raises out of ``build`` and caches nothing."""
        chosen = tuple(sorted((key, type(value).__name__, value) for key, value in choices.items()))
        key, _ = self._root_key(facts)
        return self._get((*key, chosen), lambda: build(self.point(facts)))

    def clear(self) -> None:
        self.entries.clear()
        self.hits = self.misses = 0


BIND_CACHE = BindCache()
"""The process's cache, which every KernelOp binds through."""


__all__ = ["BIND_CACHE", "BindCache", "Facts"]
