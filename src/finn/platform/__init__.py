# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Parts, boards and shells, and the one function that resolves a build's target from
them (``finn.platform.resolve.resolve_target``).

Kernels import only ``finn.platform.capabilities``, the plain data a target gives them
(``Platform``); the rest of the package sits above the kernels. Every refusal is a
``finn.platform.refusal.TargetRefused`` carrying its finding, by code.

The package re-exports nothing: each name is imported from the module that owns it.
"""
