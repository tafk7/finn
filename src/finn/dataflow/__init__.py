# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The dataflow stack, as a namespace and nothing more.

The packages form a one-way responsibility stack:

```text
finn.dataflow.space      generic declarations, compilation and occurrences
finn.dataflow.model      Kernel-domain framework and detached value subpackages
finn.dataflow.kernels    concrete reusable implementations and resources
finn.dataflow.ops        source interpretation and compiler integration
```

Logical values live under ``model.logical``; generic physical structures and
lowering under ``model.physical``; and logical-to-physical correspondence under
``model.relations``. Portable build schemas and services remain in
``artifacts``. The private ``_engine`` and generic ``space`` package do not
depend on the Kernel domain, concrete library or source adapters.

This module deliberately re-exports nothing.  A value with two importable paths
looks like a value with two owners, and the whole point of the model/space split
is that every concept has one.  Import from the package that owns it.
"""
