# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The dataflow stack, as a namespace and nothing more.

Two packages carry the vocabulary, and each owns exactly what its name says:

```text
finn.dataflow.model    the detached DataflowRegion and DataflowNetwork model
finn.dataflow.space    the generic Space declaration language, compiler and
                       occurrence runtime
```

Above them sit ``kernels``, ``designs``, ``ops`` and ``parameters``; below and
to the side, ``artifacts`` and the private ``_engine``.

This module deliberately re-exports nothing.  A value with two importable paths
looks like a value with two owners, and the whole point of the model/space split
is that every concept has one.  Import from the package that owns it.
"""
