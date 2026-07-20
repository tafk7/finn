############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""Storage topology: ``embedded`` — parameters compiled into the compute core.

The trivial topology: weights are baked in (HLS ``params.h`` / RTL literal), so there
is NO streamer, NO weight file, NO free choice, and NO weight stream port. The
compute-core emit already produces the embedded artifact ([[emit-phase]]); the
"no weight stream" consequence (``weight_stream_width = 0``) is a CROSS-COORDINATE
fact that reads the compute fold, so it lives at the op level as a parent derived over
the lifted ``topology`` axis — not here. This bundle therefore owns nothing but its
identity. It is the pool default (first registered → root-axis default), so a kernel
with no delivery choice still resolves to a legal parameters point.
"""

from __future__ import annotations

from .names import EMBEDDED
from .registry import register
from .topology import storage_topology


@register
def embedded_topology():
    return storage_topology(EMBEDDED)
