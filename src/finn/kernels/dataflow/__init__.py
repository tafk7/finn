############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################
"""Non-compute (dataflow-infrastructure) kernels.

Holds the parameter-feed subsystem under ``memory/`` (the storage-topology
REALIZATIONS — ``embedded``/``decoupled`` and their memstream emit; the abstraction
lives in ``model/memory_backend.py``). As the classic ``custom_op/fpgadataflow/``
infra ops migrate, ``fifo/``/``dwc/``/``iodma/`` join as siblings of ``memory/``.
"""
