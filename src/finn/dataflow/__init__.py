# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical logical dataflow values: tensors, schedules, traversals and beat sequences.

The package sits below the physical kernels (``finn.kernels``) built on its
values, beside the generic Space engine (``finn.core.space``) and independent
of it: it states values, and a refusal in the engine's terms is the kernels'
(``finn.kernels.values.domains``). The layer table, ``tests/layering.py``,
states the whole order. ``finn.dataflow`` imports only ``qonnx.core.datatype``,
numpy and the standard library.

- ``datatypes``: the QONNX scalar datatype value boundary.
- ``tensor``: the fact a channel carries, a ``Tensor`` of one ``ScalarEncoding``.
- ``traversal``: how one end presents a tensor (``Traversal``,
  ``BeatSequence``), marker rules (``LevelEnd``), and ``classify``, which names
  the adaptation between two traversals of one tensor.
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

This module deliberately re-exports nothing.  A value with two importable paths
looks like a value with two owners.  Import from the module that owns it.
"""
