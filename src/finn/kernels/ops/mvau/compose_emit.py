############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""``emit_composed`` — emit BOTH halves of a composed kernel and stitch them.

The compute pool and the parameters pool each emit an independent, byte-equivalent
cell (proven in the Docker differential test). This helper closes the composition:
dispatch both emits, gather their artifacts, and run the op-agnostic ``stitch``
resolver over their declared ports to WIRE them into one block design.

It is deliberately thin and op-agnostic in the part that matters: it knows only the
two pool ROOTS (``implementation`` and ``parameters.topology``) and the cell instance/
module naming convention; the wiring itself is entirely the resolver's, driven by the
ports each emit declares. A topology with no streamer (``embedded``: ``emit=None``)
yields a single compute cell, and its weight sink simply exports as a boundary pin —
the "no delivery sibling" case handled structurally, with no branch here.
"""

from __future__ import annotations

from finn.kernels.space import Artifacts
from finn.kernels.space.stitch import Cell, stitch
from finn.kernels.ops.parameters import parameters_pool
from finn.kernels.ops.parameters.names import TOPOLOGY

from . import mvau_pool


def emit_composed(point, context, module_name: str = "mvau_top") -> Artifacts:
    """Emit the composed kernel: compute cell + (optional) parameter-delivery cell,
    wired by the stitch resolver. Returns the merged Artifacts with ``ipi`` REPLACED
    by the resolver's block-design commands (instantiation + wiring both derive from
    the declared ports, superseding the per-emit hand-built instantiation lines)."""
    compute_arts = _emit_compute(point, context, module_name)
    cells = [Cell(instance=module_name, module=module_name, ports=compute_arts.ports)]
    merged = compute_arts

    delivery_arts = _emit_delivery(point, context, module_name)
    if delivery_arts is not None:
        wstrm = f"{module_name}_wstrm"
        cells.append(
            Cell(
                instance=wstrm,
                module=f"{module_name}_memstream_wrapper",
                ports=delivery_arts.ports,
            )
        )
        merged = merged.merge(delivery_arts)

    commands = stitch(tuple(cells), region_name=module_name)
    # Replace the merged (per-emit, hand-built) IPI with the resolver's output — the
    # stitch is now the single source of instantiation + wiring truth.
    return Artifacts(
        generated=merged.generated,
        data_files=merged.data_files,
        static_files=merged.static_files,
        ports=merged.ports,
        ipi=commands,
    )


def _emit_compute(point, context, module_name):
    """Dispatch the selected compute bundle's emit, threading ``module_name`` into the
    wrapper name. Looked up by the compute pool's root axis (``implementation``)."""
    impl = point["implementation"]
    bundle = {b.name: b for b in mvau_pool()}[impl]
    return bundle.emit(point, context, module_name)


def _emit_delivery(point, context, module_name):
    """Dispatch the selected parameters topology's emit, or None when the topology has
    no streamer (embedded, ``emit=None``). Looks the bundle up by the pool's own root
    axis so a new topology needs no change here. Calls the emit directly (rather than
    ``emit_point``) to thread ``module_name`` into the wrapper name."""
    topo = point[TOPOLOGY]
    bundle = {b.name: b for b in parameters_pool()}[topo]
    if bundle.emit is None:
        return None
    return bundle.emit(point, context, module_name)
