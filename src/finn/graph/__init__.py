# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Graph adapters: ONNX models as Designs of kernels.

The layer above ``finn.kernels``, which never reads a graph: a model is read
once, here, into a ``Design`` of kernels built from its facts, and rewritten
for FINN from the configured kernels (``finn.graph.adapter``).
"""

from finn.graph.adapter import FINN_DOMAIN, GraphDesign, GraphError, finn_model, graph_design

__all__ = ["FINN_DOMAIN", "GraphDesign", "GraphError", "finn_model", "graph_design"]
