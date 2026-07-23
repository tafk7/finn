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

from .names import EMBEDDED, WEIGHT_STREAM_WIDTH
from .registry import register
from .topology import storage_topology


def _no_stream_width(p, ctx):
    # Embedded weights are baked in — there is no weight-delivery port, so its width
    # is 0 regardless of the compute demand.
    return 0


@register
def embedded_topology(iface):
    # ``iface`` (the parameter interface this topology delivers) is accepted for a uniform
    # bundle-factory signature; embedded owns no interface-namespaced axes/geometry (it is
    # the ``constant`` mode — nothing is delivered), so it uses only the weights-only global
    # WEIGHT_STREAM_WIDTH alias (see names.py). A future thresholds increment that
    # generalizes that width will key it off ``iface``.
    return storage_topology(
        EMBEDDED,
        derived=(Derived(WEIGHT_STREAM_WIDTH, _no_stream_width),),
    )
