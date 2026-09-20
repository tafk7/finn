# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Static API fixture for the lazy Kernel-domain facade."""

from finn.dataflow.model import Kernel, RegionDeclaration


class FacadeKernel(Kernel):
    id = "typing_facade"


def accepts_region_declaration(value: RegionDeclaration) -> RegionDeclaration:
    return value
