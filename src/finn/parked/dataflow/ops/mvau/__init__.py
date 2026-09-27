# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU source interpretation and compiler-facing lifecycle.

```text
op.py           source declarations and the choice of reusable implementation
computation.py  source execution against the shared mathematical profile
numerics.py     source numerical-support evidence using shared pure rules
```

Reusable profiles, Region/Network recipes, implementations and physical
assembly live under ``finn.parked.dataflow.kernels.matmul``. This source adapter binds
node facts to that library and owns native hydration, persistence and graph
effects; the implementation library never imports this package back.

This namespace performs no eager import; ``finn.parked.custom_op.dataflow`` is where
the operation is registered for QONNX to find.
"""

__all__: list[str] = []
