############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Storage topology: ``embedded`` — parameters compiled into the compute core.

The trivial topology: params are baked in (HLS ``params.h`` / RTL literal), so there
is NO streamer, NO param file, NO free choice, and NO stream port. This bundle owns only
its identity plus the one consequence of having no stream — its
``parameters.<iface>.stream_width`` is ``0`` regardless of the compute demand. (This used
to be an op-level derived branching on the topology; it is a per-topology fact, so it
belongs in the topology bundle, dispatched on selection.) It is the pool default (first
registered → root-axis default), so a kernel with no delivery choice still resolves to a
legal parameters point.
"""

from __future__ import annotations

from finn.kernels.engine.datatype_spec import value_optimized
from finn.kernels.engine.derived import Derived
from finn.kernels.engine.storage_descriptor import StorageDescriptor
from finn.kernels.model.param_names import EMBEDDED as EMBEDDED_MODE
from finn.kernels.model.param_names import param_stream_width_key, storage_datatype_key

from .names import EMBEDDED
from .registry import register
from finn.kernels.model.source_backend import source_backend


def _no_stream_width(p, ctx):
    # Embedded params are baked in — there is no delivery port, so its width is 0
    # regardless of the compute demand.
    return 0


def _storage_descriptor(iface):
    def _descriptor(p, ctx):
        # embedded is ALWAYS value-visible (params baked into the core), so the owner
        # narrows and authorizes narrowing (values_trusted=True). value_optimized falls
        # back to the graph dtype when the tensor is dynamic. None when the interface is
        # unwired (standalone resolve, no compute context) — mirrors the geometry deriveds.
        if not ctx.has_tensor(iface):
            return None
        dtype = value_optimized(iface)(p, ctx)
        return StorageDescriptor(dtype=dtype, values_trusted=True)

    return _descriptor


@register
def embedded_topology(iface):
    # embedded owns no interface-namespaced axes/geometry (it is the ``constant`` mode —
    # nothing is delivered). It contributes the per-interface compute-facing stream-width
    # (0 — no port) and, as the storage OWNER, the published storage-datatype authority
    # (always trusted — baked-in params are always value-visible). Both namespaced per
    # interface, so emitted unconditionally for every delivered interface with no collision.
    return source_backend(
        EMBEDDED,
        mem_mode=EMBEDDED_MODE,  # baked into the compute core — no streamer, no port
        derived=(
            Derived(param_stream_width_key(iface), _no_stream_width),
            Derived(storage_datatype_key(iface), _storage_descriptor(iface)),
        ),
    )
