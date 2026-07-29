############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Thresholding's implementation registry — a thin binding of the op-agnostic
:func:`finn.kernels.compute._shared._registry.make_registry` factory.
"""

from __future__ import annotations

from finn.kernels.compute._shared._registry import make_registry

register, build_pool, registered_names, unregister = make_registry("thresholding")
