############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Single-source invariant: the stitch vocabulary (``PortSpec.role``) and the resolver
vocabulary (the derived ``delivered_parameters``) must AGREE on which interfaces are
delivered parameters.

Parameter-ness is authored ONCE — a backend declares a param port in its ``mem_modes``,
from which ``Kernel`` derives ``delivered_parameters``. ``PortSpec.role=WEIGHT_SINK`` is a
SECOND vocabulary (the stitch/port-taxonomy side) that names the same fact. Until role is
fully derived from the per-node initializer (an ir-layer follow-up), this test is the guard
that the two cannot silently disagree: every ``WEIGHT_SINK`` port is a derived delivered
interface, and every derived delivered interface has a ``WEIGHT_SINK`` port.
"""

import pytest

from finn.kernels.model.ports import Role
from finn.kernels.compute.mvau.op import MvauKernelOp, _PORTS as MVAU_PORTS
from finn.kernels.compute.thresholding.op import (
    ThresholdingKernelOp,
    _PORTS as THRESHOLDING_PORTS,
)

_OPS = [
    (MvauKernelOp, MVAU_PORTS),
    (ThresholdingKernelOp, THRESHOLDING_PORTS),
]


@pytest.mark.parametrize("op_cls,ports", _OPS, ids=lambda x: getattr(x, "__name__", ""))
def test_weight_sink_roles_match_derived_delivered_set(op_cls, ports):
    delivered = {dp.iface for dp in op_cls.kernel().delivered_parameters}
    weight_sinks = {p.iface for p in ports if p.role is Role.WEIGHT_SINK}
    assert weight_sinks == delivered, (
        f"{op_cls.__name__}: WEIGHT_SINK ports {weight_sinks} disagree with the "
        f"derived delivered-set {delivered} — the stitch and resolver vocabularies drifted"
    )
