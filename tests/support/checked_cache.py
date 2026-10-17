# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The cache the tests install, which refuses what the engine's rule forbids.

A snapshot's cache is only added to (``finn.core.space._runtime.Snapshot``): an evaluation
once held is never replaced, or the readers index (``Readers``) would go on holding what the
replaced evaluation read, and a change's reach would miss it. Production keeps its caches
exact dicts and pays nothing for the rule; the test suite makes every cache a
``CheckedCache`` (``tests/conftest.py`` sets ``_runtime.cache_factory``), so every suite, the
engine's and the kernels', checks it. A test of the engine's source (``core.space.
test_cache_writes``) checks that no code writes a cache but where the rule allows it.

``setdefault`` is how an evaluation is added, and how one given twice is met: it keeps the
held one, which is sound only if the one dropped answers the same and read at least what the
held one read. It refuses otherwise. Where the reads differ (the held evaluation's a strict
subset of the dropped one's: a trial's evaluation of a candidate counts the reads of its
admission, which the base's, evaluated as the successor's frame does, does not), it counts
the case in ``DIFFERING_READS``, by the function that gave the evaluation.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Mapping

from finn.core.space._runtime import Evaluation

# (calling function, "held reads a strict subset of the dropped one's") -> times
DIFFERING_READS: Counter[tuple[str, str]] = Counter()


class CheckedCache(dict[int, Evaluation]):
    """A cache that refuses an evaluation replaced in place (``[]``, ``update``, ``|=``) and a
    ``setdefault`` that drops an evaluation answering differently or reading less."""

    __slots__ = ()

    def __setitem__(self, index: int, entry: Evaluation) -> None:
        if index in self:
            raise RuntimeError(
                f"the cache already holds an evaluation at index {index}: one is only added, "
                "or the readers would go on holding what the replaced one read"
            )
        super().__setitem__(index, entry)

    def update(self, other: Mapping[int, Evaluation]) -> None:  # type: ignore[override]
        for index, entry in other.items():
            self[index] = entry

    def __ior__(self, other: Mapping[int, Evaluation]) -> CheckedCache:  # type: ignore[override,misc]
        self.update(other)
        return self

    def setdefault(self, index: int, entry: Evaluation) -> Evaluation:
        held = self.get(index)
        if held is None:
            super().__setitem__(index, entry)
            return entry
        if held is entry:
            return held
        if held.result != entry.result:
            raise RuntimeError(
                f"the cache holds an evaluation at index {index} and is given another that "
                "answers differently: the first stays, so the two must answer alike"
            )
        if not held.decisions <= entry.decisions:
            raise RuntimeError(
                f"the cache holds an evaluation at index {index} that read "
                f"{sorted(held.decisions - entry.decisions)} which the one given in its place "
                "did not: the kept one must not have read more"
            )
        if held.decisions != entry.decisions:
            DIFFERING_READS[sys._getframe(1).f_code.co_name, "kept reads a strict subset"] += 1
        return held
