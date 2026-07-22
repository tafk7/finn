############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``MvauKernelOp`` — the concrete FINN adapter for the MVAU design space.

Binds :func:`~finn.kernels.ops.mvau.mvau_kernel` to a FINN node: three ports
(activation in, weights in, activation out) mapping the graph's tensor slots to the
kernel's ``inp``/``weights``/``out`` interfaces. Registered as ``MVAUKernel_hls`` in
the ``finn.custom_op.fpgadataflow.hls`` domain so ``is_hls_node`` sees it and the
estimate analyses run — while the *real* compute impl is chosen by the ``implementation``
nodeattr, not the domain (consumer-surface-model.md R11).
"""

from __future__ import annotations

from finn.kernels.ops.mvau import mvau_kernel
from finn.kernels.space import Role
from .kernel_op import KernelOp, PortSpec

_PORTS = (
    PortSpec(iface="inp", direction="in", index=0, role=Role.DATA_IN),
    PortSpec(iface="weights", direction="in", index=1, role=Role.WEIGHT_SINK),
    PortSpec(iface="out", direction="out", index=0, role=Role.DATA_OUT),
)


class MvauKernelOp(KernelOp):
    """MVAU (matrix-vector activation) as a Kernel-backed FINN op."""

    def kernel(self):
        return mvau_kernel()

    def ports(self) -> tuple[PortSpec, ...]:
        return _PORTS

    def _output_datatype_from_point(self, kernel, ctx, point, index):
        # MVAU's outputDataType is a resolved derived: the graph dtype when forwarding,
        # or the weight-derived accumulator type under noActivation. Read it off the
        # point so infer propagates the exact (possibly narrowed) type.
        if index == 0 and "outputDataType" in point:
            return point["outputDataType"]
        return super()._output_datatype_from_point(kernel, ctx, point, index)

    def get_folding_axes(self):
        """The folding dials this op exposes, each mapped to its resolved max value —
        the capability SetFolding queries instead of op_type prefix-matching
        (consumer-surface-model.md R1). SIMD folds the reduction dim MW, PE the output
        dim MH; both are context-derived (from the weights shape), so we read them off a
        resolved point rather than from a stored nodeattr."""
        _, _, point = self._point()
        return {"SIMD": int(point.MW), "PE": int(point.MH)}
