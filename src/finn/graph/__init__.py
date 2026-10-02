# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Graph adapters: ONNX models as kernels.

The layer above ``finn.kernels``, which never reads a graph: a model is read
once, here, into a root ``Kernel`` of kernels built from its facts and the
streams between them, and rewritten for FINN from the configured kernels
(``finn.graph.adapter``).
"""

from finn.graph.adapter import (
    FINN_DOMAIN,
    Graph,
    GraphDesign,
    GraphError,
    finn_model,
    graph_design,
)

__all__ = ["FINN_DOMAIN", "Graph", "GraphDesign", "GraphError", "finn_model", "graph_design"]
