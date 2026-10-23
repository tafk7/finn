# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The running type, the engine the public API answers from (private: ``occurrence`` and
``inspection`` ask it): a design space as an immutable value, and one evaluator answering
every question from it, with no worlds and no early cutoff; answers are kept across design
spaces only by the node cache a session may hold.

- ``structure``: the per-model record (each formal bound by reference and its source,
  the flow order's ranks, the interned facts layers), held by the model's scope records,
  as long as the model lives;
- ``value``: ``FactsLayer`` (a model with its facts, and what reads facts alone) and
  ``DesignSpace`` (a layer with its stored choices), each equal and hashed by value;
- ``evaluate``: ``Evaluation``, every answer of one design space, memoized within it
  only; each member kind's frame; the receiver a ``self`` method reads through; the depth
  guard; failure completion; the ``Session`` (each question's run, the counters, and the
  seam for a node cache);
- ``questions``: presence, validity, states, pairs and their origins, standing, a change
  of choices or facts, a bind, the possible values;
- ``surface``: the rest of the public questions (a member's answer and value, presence,
  an assessment, a Decision's state and candidates, a bound accessor, the evidence of an
  answer, the configurations below), each in today's public result type;
- ``cache``: the node cache, the one reuse of answers across design spaces, behind its
  one contract; on, bounded or dropping, fixed when it is made;
- ``per_class``: what a model answers once for every design space of it (a method's
  inferred reads, the existence guards, the flow order, the presence readers) and the
  model's one session, in which every public question runs;
- ``answers``, ``receiver``, ``depth``, ``purity``, ``facts``, ``requests``: how an
  answer is formed and detached, the method reader, the depth guard, the purity check's
  switch, the facts' preparation and a design space opened on them, and the public
  request forms as addresses.

It reads each class's compiled form through the model's scope records, by address (each
member record read through its scope, its edges mapped when read, never collapsed:
``_scopes``), and the per-class record (``per_class``).

One run of a model at a time: the public layer asks each question under the model's lock
(``occurrence.ask``), and the model's node cache is written under the same lock; different
models run at once. Nothing here takes a lock itself: a design space, its evaluations and
a session are used by one thread at a time.
"""
