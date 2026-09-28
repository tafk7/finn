# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical logical dataflow values: tensors, their traversals and presentations.

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
- ``tensor``: a stream's fact, a ``Tensor`` of one ``ScalarEncoding``.
- ``traversal``: how one end presents a tensor (``Traversal``,
  ``Presentation``), marker rules, and ``classify``, which names the adapter
  between two traversals of one tensor.

The earlier Region/Network model was retired to
``finn.parked.dataflow.logical_values`` (D10, G0.1): its concepts live on in
the values above, its code does not.

This module deliberately re-exports nothing.  A value with two importable paths
looks like a value with two owners.  Import from the module that owns it.
"""
