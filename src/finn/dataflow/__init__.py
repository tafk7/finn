# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical logical dataflow values: Regions, Networks, maps and their validation.

The dependency direction is fixed:

```text
finn.core.space                    generic Space engine and value semantics
        ^
        |
finn.dataflow                      canonical logical values (this package)
        ^
        |
finn.kernels                       physical components built on those values
        ^
        |
finn.parked, graph integration     retired dataflow implementation
```

``finn.dataflow`` imports only ``finn.core.space``, ``qonnx.core.datatype`` and
the standard library. It never imports ``finn.kernels`` or ``finn.parked``.

- ``datatypes``: the QONNX scalar datatype value boundary.
- ``model.logical``: Regions, Networks, coordinate maps, validation,
  presentation, references, composition and their value semantics.

This module deliberately re-exports nothing.  A value with two importable paths
looks like a value with two owners.  Import from the module that owns it.
"""
