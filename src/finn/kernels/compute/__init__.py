############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The compute kernels — the per-op bounded contexts (``mvau``, ``thresholding``,
``pool``), each a folder owning its Kernel/Schema definition, backends, and emit.

The op-agnostic registry factory lives in ``model/registry.py``; MVAU's DSP RTL
constants live in ``mvau/_dsp_rtl.py``.
"""
