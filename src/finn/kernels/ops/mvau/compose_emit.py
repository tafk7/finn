############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
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
from finn.kernels.space.backend import BACKEND_AXIS
from finn.kernels.emit.stitch import Cell, stitch
from finn.kernels.space.param_names import topology_key

from . import mvau_pool
from .op import mvau_kernel


def emit_composed(point, context, module_name: str = "mvau_top") -> Artifacts:
    """Emit the composed kernel: compute cell + (optional) parameter-delivery cell,
    wired by the stitch resolver. Returns the merged Artifacts with ``ipi`` REPLACED
    by the resolver's block-design commands (instantiation + wiring both derive from
    the declared ports, superseding the per-emit hand-built instantiation lines)."""
    compute_arts = _emit_compute(point, context, module_name)
    cells = [Cell(instance=module_name, module=module_name, ports=compute_arts.ports)]
    merged = compute_arts

    # One delivery cell per DELIVERED PARAMETER whose selected topology streams (constant-mode
    # topologies have emit=None → no cell, e.g. embedded weights or the always-constant fused
    # thresholds). Iterate the Kernel's declared delivered_parameters so a second interface
    # needs no change here — symmetric with the generic resolve-side wiring (space/delivery.py).
    for dp in mvau_kernel().delivered_parameters:
        delivery_arts = _emit_delivery(point, context, module_name, dp)
        if delivery_arts is not None:
            strm = f"{module_name}_{dp.iface}strm"
            cells.append(
                Cell(
                    instance=strm,
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
    wrapper name. Looked up by the compute pool's root axis (``backend``)."""
    impl = point[BACKEND_AXIS]
    bundle = {b.name: b for b in mvau_pool()}[impl]
    return bundle.emit(point, context, module_name)


def _emit_delivery(point, context, module_name, dp):
    """Dispatch the selected delivery topology's emit for one delivered parameter, or None
    when the topology has no streamer (``constant`` mode / ``emit=None`` — embedded weights or
    the always-constant fused thresholds). Looks the bundle up by the interface-keyed root axis
    from the DeliveredParam's own pool, so a new topology or interface needs no change here.
    Calls the emit directly (not ``emit_point``) to thread ``module_name`` + the interface."""
    topo = point[topology_key(dp.iface)]
    bundle = {b.name: b for b in dp.pool}[topo]
    if bundle.emit is None:
        return None
    return bundle.emit(point, context, module_name, dp.iface)
