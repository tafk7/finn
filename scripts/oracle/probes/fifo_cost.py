# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The HWCustomOp flow's FIFO model (``finn.util.resource_models``) over the grid
``tests/kernels/test_fifo.py`` compares with: the storage style ``fifo.sv`` elaborates
for ``auto`` (``_resolve``) and that style's cost (``_fifo_cost``, UltraScale+)."""

from _probe import arguments, write

from finn.util.resource_models import _fifo_cost, _resolve

GRID = [
    *((depth, width) for depth in (2, 5, 17, 33) for width in (1, 8, 32, 100)),
    (257, 32),
    (1500, 32),
    (2028, 32),
    (100_000, 72),
]
"""(depth, width): shift FIFOs at every width, then one of each memory, the cases the
test names."""

raw, _ = arguments()
values = []
for depth, width in GRID:
    style = _resolve(depth, width, "auto")
    cost = _fifo_cost(depth, width, style)
    values.append({"depth": depth, "width": width, "style": style, **cost._asdict()})
write(raw, values)
