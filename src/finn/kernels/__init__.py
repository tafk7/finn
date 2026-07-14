############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""FINN-native dataflow-kernel backend.

An op's *identity* (schema, golden reference, design-space derivation) is
separated from its *implementation* (a composed backend value object selected
per device context). A single generic ``KernelCustomOp`` adapter is the only
boundary to the FINN/qonnx graph.
"""
