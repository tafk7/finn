############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Storage topology: ``embedded`` — parameters compiled into the compute core.

The trivial topology: weights are baked in (HLS ``params.h`` / RTL literal), so there
is NO streamer, NO weight file, NO free choice, and NO weight stream port. This bundle
owns only its identity plus the one consequence of having no stream — its
``weight_stream_width`` is ``0`` regardless of the compute demand. (This used to be an
op-level derived branching on the topology; it is a per-topology fact, so it belongs in
the topology bundle, dispatched on selection.) It is the pool default (first registered
→ root-axis default), so a kernel with no delivery choice still resolves to a legal
parameters point.
"""

from __future__ import annotations

from finn.kernels.space import Derived
from finn.kernels.space.param_names import CONSTANT, WEIGHT_STREAM_WIDTH

from .names import EMBEDDED, WEIGHTS
from .registry import register
from .topology import storage_topology


def _no_stream_width(p, ctx):
    # Embedded weights are baked in — there is no weight-delivery port, so its width
    # is 0 regardless of the compute demand.
    return 0


@register
def embedded_topology(iface):
    # embedded owns no interface-namespaced axes/geometry (it is the ``constant`` mode —
    # nothing is delivered). The one thing it contributes is the weights-only global
    # WEIGHT_STREAM_WIDTH alias (0 — no port), which the COMPUTE side reads; it is emitted
    # ONLY for the ``weights`` interface, since it is un-namespaced and a second interface
    # (thresholds) composing after weights would otherwise overwrite the weights value.
    derived = (Derived(WEIGHT_STREAM_WIDTH, _no_stream_width),) if iface == WEIGHTS else ()
    return storage_topology(
        EMBEDDED,
        mode=CONSTANT,  # baked into the compute core — no streamer, no port
        derived=derived,
    )
