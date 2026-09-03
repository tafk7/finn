# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU: its canonical semantics, its Designs, and its operation.

```text
regions.py   the normalized Region declarations
networks.py  the decomposed Network they form
designs/     base, dot_product, supplied_dot_product
op.py        the source node and its two Design alternatives
```

``regions`` and ``networks`` survived the legacy reset because they were never
an experiment: they are the reference semantics every later MVAU implementation
is measured against, and they depend on nothing but ``region`` and ``network``.

This namespace performs no eager import; ``finn.custom_op.dataflow`` is where
the operation is registered for QONNX to find.
"""

__all__: list[str] = []
