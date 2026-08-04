############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``ParamDatatype`` — a delivered parameter's datatype authority, published by its owner.

A storage topology (``embedded``/``decoupled``/…) OWNS the parameters it stores, and so owns
their at-rest datatype. It publishes that authority — not the raw values — as a
``ParamDatatype`` under ``parameters.<iface>.datatype`` (see
:func:`~finn.kernels.model.param_names.param_datatype_key`). It is the TYPE-facts sibling of
:class:`~finn.kernels.model.demand.ParamDemand` (the geometry facts): demand sizes the memory,
this types the values. A consumer that depends on a stored parameter (thresholds → the dtype;
MVAU's accumulator → the value regime) reads this instead of re-deriving datatype authority by
hand or peeking at storage it does not own.

The object carries **authority, not data**:

* ``dtype`` — the value-optimized at-rest datatype. Every dtype consumer (thresholds, weight
  serialization) reads this and is self-sufficient — no Context, no branch.
* ``values_visible`` — whether the owner can SEE the values at build time: ``True`` for
  ``embedded`` and decoupled-static (``runtime_writeable_weights==0``); ``False`` when the
  owner is blind (runtime-writable now; external/DMA/MLO later). A blind owner publishes the
  declared graph dtype (the envelope), not a narrowed one.

The object exists whenever the interface is WIRED (present-but-``None`` only on an unwired
standalone resolve) — including the runtime-writable case, where it carries the envelope dtype
with ``values_visible=False`` so the dtype consumers stay branch-free. Visibility is a property
of an existing object, not its existence condition.

``values_visible`` is a permission, not the payload. A consumer that needs the actual values —
the accumulator's per-column mixed-sign matvec sum, which no compact range can reproduce — reads
the matrix from :class:`~finn.kernels.engine.context.Context` (the shared source of values) but
ONLY when ``values_visible`` authorizes it. Presence and visibility are orthogonal: baseline
FINN keeps a runtime-writable weight initializer present yet gates narrowing on the flag
(``matrixvectoractivation.py:506/517``). So the consumer acts on published permission rather
than reaching around the seam. (Publishing the range itself — so the accumulator needs no
Context read — is a deferred graph-scale storage-optimization pass; today it reaches in.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParamDatatype:
    dtype: Any  # the value-optimized at-rest QONNX DataType (or graph dtype when blind)
    values_visible: bool  # owner can see the values at build time -> narrowing is authorized
