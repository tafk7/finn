# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The running type, beside today's engine (private; nothing public reads it yet): a
design space as an immutable value, and one evaluator answering every question from it,
with no worlds and no early cutoff; answers are kept across design spaces only by the
node cache a session may hold.

- ``structure``: the per-model record (each formal bound by reference and its source,
  each scope's ancestors, the flow order's ranks, the interned facts layers), held weakly
  by the model's structure;
- ``value``: ``FactsLayer`` (a model with its facts, and what reads facts alone) and
  ``DesignSpace`` (a layer with its stored choices), each equal and hashed by value;
- ``evaluate``: ``Evaluation``, every answer of one design space, memoized within it
  only; each node kind's frame; the receiver a ``self`` method reads through; the depth
  guard; the ``Session`` (counters, and the seam for a node cache);
- ``questions``: presence, validity, states, pairs and their origins, standing, a change
  of choices or facts, a bind, the possible values;
- ``cache``: the node cache, the one reuse of answers across design spaces, behind its
  one contract; on, bounded or dropping, fixed when it is made.

It reads today's per-class compiled form through the model's structure (each node as its
template resolves it, never collapsed), and today's static questions (a method's inferred
reads, the existence guards, the flow order, the presence readers).

Single-threaded, as today's engine is: nothing here takes a lock, and a design space, its
evaluations and a session are used by one thread at a time.
"""
