#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Bounded native stack reuse and thread-boundary observations."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import runpy
from weakref import ref

import greenlet

from finn.kernels.space import _self_prototype as candidate

benchmark = runpy.run_path("scripts/benchmark-self-space.py")
work = Counter()
bind, query, expected = benchmark["self_fixture"]("chain", 20000, work)
rows = []
with candidate.using_scheduler("greenlet"):
    for generation in range(3):
        point = bind()
        snapshot = ref(point._snapshot)
        assert query(point) == expected
        after_query = benchmark["memory"]()
        del point
        gc.collect()
        rows.append(
            {
                "generation": generation,
                "snapshot_reclaimed": snapshot() is None,
                "after_query": after_query,
                "after_discard": benchmark["memory"](),
            }
        )
        assert snapshot() is None
foreign = greenlet.greenlet(lambda: 1)


def foreign_switch():
    try:
        foreign.switch()
    except greenlet.error as error:
        return type(error).__name__, str(error)
    raise AssertionError("cross-thread continuation unexpectedly succeeded")


with ThreadPoolExecutor(max_workers=1) as executor:
    thread_boundary = executor.submit(foreign_switch).result()
print(
    json.dumps(
        {
            "greenlet_version": greenlet.__version__,
            "generations": rows,
            "work": dict(work),
            "cross_thread_switch": thread_boundary,
        },
        sort_keys=True,
    )
)
