############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The compute kernels — the per-op bounded contexts (``mvau``, ``thresholding``,
``pool``), each a folder owning its Kernel/Schema definition, backends, and emit.

``_shared/`` holds the op-authoring infrastructure (registry factory, DSP RTL
constants) reused across them.
"""
