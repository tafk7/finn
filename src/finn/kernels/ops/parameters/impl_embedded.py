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

from finn.kernels.space import Derived
from finn.kernels.model.param_names import CONSTANT, param_stream_width_key

from .names import EMBEDDED
from .registry import register
from finn.kernels.model.memory_backend import memory_backend


def _no_stream_width(p, ctx):
    # Embedded params are baked in — there is no delivery port, so its width is 0
    # regardless of the compute demand.
    return 0


@register
def embedded_topology(iface):
    # embedded owns no interface-namespaced axes/geometry (it is the ``constant`` mode —
    # nothing is delivered). The one thing it contributes is the per-interface compute-facing
    # stream-width (0 — no port), which the COMPUTE side reads; namespaced per interface, so
    # it is emitted unconditionally for every delivered interface with no collision.
    return memory_backend(
        EMBEDDED,
        mode=CONSTANT,  # baked into the compute core — no streamer, no port
        derived=(Derived(param_stream_width_key(iface), _no_stream_width),),
    )
