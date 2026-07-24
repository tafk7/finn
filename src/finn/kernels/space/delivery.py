############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The op-facing parameter-DELIVERY declaration (:class:`DeliveredParam`) + the seam
COMPUTATION bodies a :class:`~finn.kernels.space.backend_interface.BackendInterface`
composes.

A :class:`~finn.kernels.space.kernel.Kernel` that delivers parameters (weights,
thresholds, …) declares a :class:`DeliveredParam` per interface — the WHAT (interface +
cadence + concrete delivery pool). The HOW — the ``(DEMAND stage, guarded delivery
sub-schema)`` pair, in supply-waterfall order (COMPUTE → DEMAND → MEMORY) — is OWNED by
``BackendInterface`` (design pitch §2), which reuses the two seam computations kept here:

* :func:`_demand_for` — the DEMAND closure: a realization-free
  :class:`~finn.kernels.space.demand.ParamDemand` sized from the RESOLVED interface
  geometry (the tiling engine's ``stream_width.<iface>`` + block extents), read by the
  selected delivery topology to size its own memstream geometry. ``None`` in two emergent
  cases — no initializer (a live activation, not a stored parameter) or constant-mode
  consumption (baked into the core, nothing to stream).
* :func:`_topology_domain` / :func:`_topology_default` — the topology-mode GUARD: keeps
  only topologies whose :attr:`Backend.mode` the selected compute backend ``consumes`` for
  this interface. Permissive when the backend declares nothing.

Both are interface-generic: they read a compute backend's :attr:`Backend.consumes` and a
delivery topology's :attr:`Backend.mode` — both plain ``Backend`` fields — with no
knowledge of any op or any topology identity string.
"""

from __future__ import annotations

from typing import Any, Callable

from dataclasses import dataclass

from .demand import ParamDemand
from .param_names import ALL_MODES, topology_key
from .tiling import stream_width_key


@dataclass(frozen=True)
class DeliveredParam:
    """One parameter interface an op delivers through a delivery (parameters) pool.

    The op DECLARES the WHAT; the generic ``BackendInterface`` wiring owns the HOW. Fields:

    * ``iface`` — the parameter interface name (also the Context tensor key).
    * ``cadence`` — ``(point, context) -> int``: how often the core consumes one word (the
      one op-specific residue — weights ``->1``; thresholds ``->prod(folded_in[:-1])``).
    * ``pool`` — the CONCRETE delivery pool (a tuple of storage-topology ``Backend``\\ s),
      built in ops for this interface. Handed to the Kernel so the wiring reads it directly
      instead of hardcoding a lookup.
    """

    iface: str
    cadence: Callable[[Any, Any], int]
    pool: tuple = ()


# --- DEMAND stage -------------------------------------------------------------


def _demand_for(dp: DeliveredParam):
    """The compute core's DEMAND on ``dp.iface`` as a resolve closure, sized from RESOLVED
    interface geometry (the tiling engine's ``stream_width.<iface>`` + block extents), not
    named backend dials — so a tiled/packed backend that folds the interface in its own
    terms works automatically. Returns ``None`` (nothing to deliver) when the interface has
    no initializer (a live activation, streams like any dataflow edge) or is consumed in
    ``constant`` mode (baked into the core)."""
    iface = dp.iface
    width_key = stream_width_key(iface)
    topo_key = topology_key(iface)
    topo_modes = {b.name: b.mode for b in dp.pool}  # topology value -> its consumption mode

    def compute(p, ctx):
        if ctx.initializer(iface) is None:
            return None
        if topo_modes.get(p[topo_key]) == "constant":
            return None
        width_bits = int(p[width_key])  # resolved stream width (PE*SIMD*wbits)
        elem_bits = ctx.tensor_datatype(iface).bitwidth()
        parallelism = width_bits // elem_bits  # elements/cycle, in the backend's own fold
        block = ctx.tensor_shape(iface)  # the block extents (MW, MH for weights)
        depth = _prod(block) // parallelism  # words/set = WMEM, from geometry not p.WMEM
        return ParamDemand(
            parallelism=parallelism,
            elem_bits=elem_bits,
            depth=depth,
            cadence=int(dp.cadence(p, ctx)),
        )

    return compute


# --- topology-mode GUARD ------------------------------------------------------


def _topology_domain(compute_pool, dp: DeliveredParam):
    """A domain override for ``parameters.<iface>.topology`` keeping only topologies whose
    ``mode`` the selected compute backend consumes for this interface. The compute backend
    is master: an interface it says nothing about is PERMISSIVE (both modes), so nothing
    regresses. Returns ``(domain_closure, legal)`` — ``legal(p) -> tuple[str, ...]``."""
    iface = dp.iface
    by_name = {b.name: b for b in compute_pool}
    topo_modes = {b.name: b.mode for b in dp.pool}  # topology identity -> its consumption mode

    def legal(p):
        # Defensive read: during real resolve `implementation` is fixed before this axis;
        # under a bare probe point (nodeattr typing) it is absent → permissive (all modes).
        impl = p.get("implementation") if hasattr(p, "get") else None
        backend = by_name.get(impl)
        modes = (backend.consumes.get(iface) if backend else None) or ALL_MODES
        return tuple(name for name, mode in topo_modes.items() if mode in modes)

    return lambda p, ctx: frozenset(legal(p)), legal


def _topology_default(base_default, legal):
    """The topology axis default, guarded to the consumable modes: keep the pool's own
    first-registered default when the selected backend can consume it, else fall to the
    first in-domain topology (a stream-only backend defaults to a streamer). Prevents an
    out-of-domain default from making an unpinned topology illegal."""

    def default(p, ctx):
        allowed = legal(p)
        d = base_default(p, ctx)
        return d if d in allowed else (allowed[0] if allowed else d)

    return default


# --- helpers ------------------------------------------------------------------


def _prod(shape) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return out
