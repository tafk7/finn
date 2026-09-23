# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Model specializations consume the canonical generic kernel foundation."""

import finn.dataflow.model as model
import finn.kernels as kernels
from finn.kernels.base import Kernel
from finn.kernels.space import Space


def test_every_layer_names_its_own_specialization() -> None:
    for name in (
        "Kernel",
        "LogicalView",
        "PhysicalView",
        "PublicOperandDeclaration",
        "ModuleParameter",
        "RegionDeclaration",
        "kernel_physical",
        "NetworkBoundary",
        "NetworkEdge",
        "KernelChoice",
        "EdgeSink",
    ):
        assert name in model.__all__
    assert not {
        "KernelChoice",
        "LogicalView",
        "ModuleParameter",
        "RegionDeclaration",
    } & set(kernels.__all__)
    assert issubclass(model.Kernel, Kernel)
    assert issubclass(Kernel, Space)
    assert Kernel.__module__ == "finn.kernels.base"
