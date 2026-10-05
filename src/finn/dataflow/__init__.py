# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical logical dataflow values: tensors, schedules, traversals and beat sequences.

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
  ``BeatSequence``), marker rules (``LevelEnd``), and ``classify``, which names
  the adapter between two traversals of one tensor.
- ``schedule``: named indices (``Index``, with affine arithmetic) and a
  kernel's ``Schedule`` over them, whose ``present`` derives each port's
  traversal; ``bind_extents`` takes the indices' extents from the tensors the
  ports read (``Access``).
- ``gemm``: matrix multiplication's canonical indices ``m``, ``n``, ``k`` and
  its operand ``Form``s.
- ``plan``: the canonical steps between two beat sequences of one tensor.
- ``ends``: what a channel's two ends present (``End``, ``Ends``) and the rule
  that they traverse its tensor (``misfit``); the channel itself, which finds
  its ends among the kernels that reference it, is ``finn.kernels.channels``.

The earlier Region/Network model was retired to
``finn.parked.dataflow.logical_values`` (D10, G0.1): its concepts live on in
the values above, its code does not.

This module deliberately re-exports nothing.  A value with two importable paths
looks like a value with two owners.  Import from the module that owns it.
"""
