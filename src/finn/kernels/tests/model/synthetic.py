############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``synthetic_op`` — a throwaway :class:`DataflowOp` subclass for testing op mechanisms.

Several tests exercise op-CLASS machinery — interface index resolution, per-port direction
validation, delivered-parameter derivation, the tiling projections — on a made-up op rather
than on MVAU, so the mechanism is tested in isolation from any real design space.

Before the container merged into the op class (F6), that meant constructing a
``DataflowKernel(name=..., interfaces=..., pool=...)`` value. The op class declares that
design space in its own body now, so the equivalent is a subclass — which is also what production does, making these tests
exercise the real construction path instead of a parallel one.

``type()`` rather than a ``class`` statement because the interfaces and pool vary per test;
the two are identical to Python, and ``DataflowOp.__init_subclass__`` fires either way. That
is the point: index resolution and port-direction validation happen HERE, at class creation,
which is why a test asserting :class:`KernelError` wraps this call.
"""

from __future__ import annotations

from finn.kernels.ir import DataflowOp


def synthetic_op(interfaces, pool, *, name="K", **extra):
    """A fresh ``DataflowOp`` subclass with this design space and nothing else.

    Args:
        interfaces: the op's ``InterfaceSchema`` tuple.
        pool: its ``Backend`` tuple.
        name: the op_type name (rarely matters; these ops are never in a graph).
        **extra: any further class attributes (``op_axes``, ``kernel_attrs``, …).

    Raises whatever ``__init_subclass__`` raises — notably
    :class:`~finn.kernels.model.kernel.KernelError` for a port-direction violation, which is
    now a class-DEFINITION error rather than a construction one.
    """
    return type(
        f"_Synthetic{name}",
        (DataflowOp,),
        {"name": name, "interfaces": tuple(interfaces), "pool": tuple(pool), **extra},
    )
