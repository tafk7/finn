# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""SetFolding on TFC_W2A2 (``_tfc.folded``: Ultra96, 5 ns, 1,000,000 frames a second):
each layer of the dataflow partition, its folding and its ``get_exp_cycles``, and the
target in cycles a frame."""

from _probe import arguments, write
from _tfc import folded
from pathlib import Path
from qonnx.custom_op.registry import getCustomOp

raw, captures = arguments()
model, cfg = folded(captures, Path.cwd())
layers = []
for node in model.graph.node:
    op = getCustomOp(node)
    folding = {
        name: op.get_nodeattr(name) for name in ("PE", "SIMD") if name in op.get_nodeattr_types()
    }
    layers.append(
        {"name": node.name, "op_type": node.op_type, **folding, "cycles": op.get_exp_cycles()}
    )
write(raw, {"target_cycles": cfg._resolve_cycles_per_frame(), "layers": layers})
