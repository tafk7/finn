############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""T1.4 (F1) — the folded-shape / stream-width projections are STREAM-only.

A shape/width query is defined over the ``Stream`` interfaces only; a ``MemoryMapped``/
``Config`` interface is outside the projection's domain, so it is not counted in the
stream-port index. Asking for a stream port index a non-Stream interface would occupy
raises a clean "no stream port at index N", not a tiling error (IODMA's forcing function).
The all-Stream common case (MVAU/Thresholding) is unchanged.
"""

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.engine.context import Context
from finn.kernels.model.backend import Backend, ports_from
from finn.kernels.model.kernel import InterfaceSchema, Kernel, KernelError
from finn.kernels.model.ports import Direction, Protocol
from finn.kernels.model.tiling import FULL


def _kernel(ifaces, stream):
    backend = Backend(name="core", ports=ports_from(stream=stream))
    return Kernel(name="K", interfaces=ifaces, pool=(backend,))


def _ctx():
    return Context(
        shapes={"inp": (1, 8), "mem": (8, 4), "out": (1, 4)},
        datatypes={"inp": DataType["INT8"], "mem": DataType["INT8"], "out": DataType["INT16"]},
        initializers={"mem": np.ones((8, 4), dtype=np.float32)},
        fpgapart="xcvc1902-vsva2197-2MP-e-S",
    )


def test_memorymapped_input_absent_from_stream_projection():
    # inp (Stream, index 0), mem (MemoryMapped, index 1), out (Stream, index 0).
    k = _kernel(
        (
            InterfaceSchema("inp", Direction.IN, block=[1, FULL]),
            InterfaceSchema("mem", Direction.IN, block=[FULL, FULL], protocol=Protocol.MemoryMapped),
            InterfaceSchema("out", Direction.OUT, block=[1, FULL]),
        ),
        stream={"inp": [1, "PE"], "out": [1, "PE"]},
    )
    ctx = _ctx()
    point = k.configure(ctx, {"backend": "core"})
    from finn.kernels.engine.point import Illegal

    assert not isinstance(point, Illegal), getattr(point, "reasons", None)

    # Stream input index 0 is 'inp' — the MemoryMapped 'mem' does NOT occupy a stream index,
    # so there is exactly ONE stream input. Index 1 is outside the projection's domain.
    assert k.get_instream_width(point, ctx, 0) > 0
    with pytest.raises(KernelError, match="no stream input port at index 1"):
        k.get_instream_width(point, ctx, 1)


def test_all_stream_kernel_indexing_is_unchanged():
    # Every input is Stream: the stream-port index equals the ONNX input index (MVAU shape).
    k = _kernel(
        (
            InterfaceSchema("inp", Direction.IN, block=[1, FULL]),
            InterfaceSchema("out", Direction.OUT, block=[1, FULL]),
        ),
        stream={"inp": [1, "PE"], "out": [1, "PE"]},
    )
    ctx = _ctx()
    point = k.configure(ctx, {"backend": "core"})
    assert k.get_instream_width(point, ctx, 0) > 0
    assert k.get_outstream_width(point, ctx, 0) > 0
