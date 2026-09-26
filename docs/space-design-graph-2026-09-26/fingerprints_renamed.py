# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Fingerprints with the cyclic node's instance renamed from u_implementation to u_weights.

Shows that instance naming is the only difference from the base revision.
Run from the FINN checkout on spike/space-design-graph.
"""

from dataclasses import replace

import finn.kernels.streams as streams

original = streams._wire


def _renamed(placed, connections, module, producer):  # type: ignore[no-untyped-def]
    def fix(name):  # type: ignore[no-untyped-def]
        return "u_weights" if name == "u_implementation" else name

    placed = {fix(name): value for name, value in placed.items()}
    connections = [
        (n, replace(c, source_owner=fix(c.source_owner), sink_owner=fix(c.sink_owner)))
        for n, c in connections
    ]
    return original(placed, connections, module, producer)


streams._wire = _renamed

import runpy  # noqa: E402

runpy.run_path("docs/space-graph-composition-2026-09-25/fingerprints.py")
