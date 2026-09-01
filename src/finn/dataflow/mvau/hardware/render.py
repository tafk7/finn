# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU wrapper and stitch-shim rendering boundary."""

from finn.dataflow.mvau.hardware.composition import (
    WRAPPER_MODULE,
    render_decomposed_wrapper,
    render_stitch_shim,
)

__all__ = ["WRAPPER_MODULE", "render_decomposed_wrapper", "render_stitch_shim"]
