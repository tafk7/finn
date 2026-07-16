############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""Generic design-space model (resolve core).

`primitives/` holds proven domain value objects — ordered parameters, datatype
range-builders, template resolution, interface shape/width — reused by the engine.
`space/` holds the engine: Context, Axis, Derived, Predicate, and resolve.
"""
