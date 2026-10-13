# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's view of an ONNX model: the graph metadata it writes and reads
(``annotations``), what it reads of a tensor (``facts``), the partition of KernelOps
(``partition``), walks over the dataflow (``walk``), and the errors the readers raise
(``errors``).
"""
