############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Shared string constants for the ONNX-level Pool op (MaxPool / AveragePool)."""

from __future__ import annotations

# Context tensor names this Kernel resolves against. Layout is NHWC at the hardware
# boundary (channels last) — ONNX pooling is NCHW, but the RTL requires channels-last, so
# a layout transform precedes (declared as required_layout, mirroring LayerNorm). The
# windowing geometry derivation is layout-independent regardless.
INPUT = "inp"
OUTPUT = "out"
INDICES = "indices"  # MaxPool's optional second output (int64 argmax positions)

# Pool-member identity.
POOL_HLS = "pool_hls"

# Function kernel_param values (ONNX op_type distinction, carried as a structural param).
MAXPOOL = "MaxPool"
AVGPOOL = "AveragePool"
