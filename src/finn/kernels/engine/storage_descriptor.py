############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``StorageDescriptor`` — what a storage owner publishes across the datatype-authority seam.

A storage topology (``embedded``/``decoupled``/…) OWNS the parameters it stores, and so owns
their at-rest datatype. It publishes that authority — not the raw values — as a
``StorageDescriptor`` under ``parameters.<iface>.storageDataType`` (see
:func:`~finn.kernels.model.param_names.storage_datatype_key`). A non-owning compute core reads
the descriptor to size derivations that depend on the stored parameters (e.g. MVAU's
accumulator), instead of re-deriving datatype authority by hand or peeking at storage it does
not own.

The descriptor carries **authority, not data**:

* ``dtype`` — the value-optimized at-rest datatype (what ``weightDataType`` becomes).
* ``values_trusted`` — whether the owner has BUILD-TIME value visibility: ``True`` for
  ``embedded`` and decoupled-static (``runtime_writeable_weights==0``); ``False`` when the
  owner is blind (runtime-writable now; external/DMA/MLO later). A blind owner publishes the
  declared graph dtype (the envelope), not a narrowed one.

The bit is a permission, not the payload. A consumer that needs the actual value range (the
accumulator's per-column mixed-sign sum, which a scalar range cannot reproduce) reads the
matrix from :class:`~finn.kernels.engine.context.Context` — the shared source of values — but
ONLY when ``values_trusted`` authorizes it. Presence and trust are orthogonal: baseline FINN
keeps a runtime-writable weight initializer present yet gates narrowing on the flag
(``matrixvectoractivation.py:506/517``). So the consumer acts on published permission rather
than reaching around the seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StorageDescriptor:
    dtype: Any  # the value-optimized at-rest QONNX DataType (or graph dtype when blind)
    values_trusted: bool  # owner has build-time value visibility -> narrowing is authorized
