# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical logical dataflow values: tensors, schedules, traversals and beat sequences.

A revision of ``finn.dataflow``, built beside it and compared with it. It takes ``datatypes`` from
the line. Its vocabulary uses ``Schedule.iterations``,
``Schedule.fold``, ``present(reshape=)`` and ``traversal.one_pass``; line importers
will migrate to these names when the revision replaces the line.
Nothing outside it and its tests (``tests/dataflow_next``) imports it; it replaces
``finn.dataflow`` under that name once its comparison holds. Until then, the docstrings
below name its modules by the package's final name.

The package sits below the physical kernels (``finn.kernels``) built on its
values, beside the generic Space engine (``finn.core.space``) and independent
of it: it states values, and a refusal in the engine's terms is the kernels'
(``finn.kernels.values.domains``). The package's own layering guard,
``tests/dataflow_next/test_guards.py``, states its allowed dependencies. The
private ``_isl`` service alone imports islpy. The remaining dependency boundary
is QONNX datatype/range values, NumPy, the standard library and findings
(``finn.core.reporting``).

- ``datatypes``: the QONNX scalar datatype value boundary.
- ``tensor``: the fact a channel carries, a ``Tensor`` of one ``ScalarEncoding``.
- ``traversal``: how one port presents a tensor (``Traversal``, or a lane group,
  ``Grouped``, where its lanes cut its order; ``BeatSequence``), marker rules
  (``LevelEnd``), ``classify``, which names the adaptation between two traversals
  of one tensor, and what a cyclic source stores (``stored``).
- ``schedule``: named indices (``Index``, with affine arithmetic) and a
  kernel's ``Schedule`` over them, whose ``present`` derives each port's
  traversal; ``bind_extents`` takes the indices' extents from the tensors the
  ports read (``Access``).
- ``gemm``: matrix multiplication's canonical indices ``m``, ``n``, ``k`` and
  its operand ``Form``s.
- ``steps``: the canonical steps between two beat sequences of one tensor.
- ``reindex``: one value from its producer's port to its readers' ports (``Reindex``):
  each reader's position map and stream, and the route derived for it (a fork's, each
  reader's); Concat's merge.
- ``streams``: what a port presents of a tensor (``Stream``: its element and
  beat sequence), a channel's two (``Streams``) and the rule that they traverse
  its tensor (``misfit``); the channel itself, which finds its streams among the
  kernels that reference it, is ``finn.kernels.channels``.
- ``logical``: immutable graph IDs, tensor/stream/source and atomic-participation
  descriptors, untimed packing/per-reader adapter queries and finite order bounds.
- ``queries``: descriptor-only native map/equality/through/word queries; certified
  export and explicitly bounded legacy extraction are separate outcomes.
- ``descriptors``: a closed, versioned durable schema, with source/occurrence
  identity; exact-record digests are not semantic keys.
- ``_isl``: the private context-owning service, confined to its creating thread.
- ``timing``: a whole network's time, per word (``Network``: sources, kernels paced by
  their schedules, forks, sinks and the channel's steps, under stated channel rules), and
  the capacities that keep its period (``minimal``, ``least``).

This module deliberately re-exports nothing.  A value with two importable paths
looks like a value with two owners.  Import from the module that owns it.
"""
