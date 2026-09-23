# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Logical dataflow modeling and compiler integration.

These consumers use the shared physical kernel package:

```text
finn.kernels             physical components, Space, artifacts and shared values
        ^
        |
finn.dataflow.model      logical models and physical binding adapters
finn.dataflow.kernels    modeling experiments and compiler-facing kernel adapters
finn.dataflow.ops        source interpretation and compiler integration
```

Logical values and public operand exports live under ``model.logical``. Region
bindings and conventional physical View adapters live under ``model.physical``.
Detached physical structures, lowering and portable build services are owned by
``finn.kernels``. Its private engine and generic Space remain independent of
concrete components and dataflow models.

This module deliberately re-exports nothing.  A value with two importable paths
looks like a value with two owners, and the whole point of the model/space split
is that every concept has one.  Import from the package that owns it.
"""
