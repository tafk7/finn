# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU: its canonical semantics, its Designs, and its operation.

```text
regions.py   the one authority for MVAU Region construction
networks.py  the decomposed Network they form
designs/     shared base, dot_product, batch_interleaved
op.py        the source node and its two Design alternatives
```

``regions`` and ``networks`` survived the legacy reset because they were never
an experiment: they are the reference semantics every later MVAU implementation
is measured against, and they depend on nothing but ``model.region`` and
``model.network``.

``regions`` is now the sole production authority, not merely the reference one.
Each Kernel used to carry its own copy of the constructor its
``RegionDeclaration`` named -- value-identical, and a second place for one
semantic contract to live.  A Kernel declares candidates and physics; the
Region family it realizes comes from here.

This namespace performs no eager import; ``finn.custom_op.dataflow`` is where
the operation is registered for QONNX to find.
"""

__all__: list[str] = []
