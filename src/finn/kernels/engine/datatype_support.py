############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``DatatypeSupport`` — a declarative per-port datatype-support gate.

A backend declares which datatypes it can build for a port. The pool's UNION of these
declarations is what a frontend claim (``can_infer_from`` via ``has_feasible_point``)
accepts: a new backend widens the accepted set with ZERO edits to the op — an all-integer
pool rejects a float MatMul FOR THE RIGHT REASON (no backend supports it), and adding an
FP16 backend makes the same MatMul claimable.

Support is a category (:class:`DatatypeKind`) + a bitwidth range, mirroring brainsmith's
``DatatypeInRange`` (``../brainsmith/brainsmith/dataflow/constraints.py``). It is DATA, not a
closure — introspectable by a validator or a build-manifest tool. A backend that needs a
gate this vocabulary cannot express declares a custom callable on the port instead (see
``Interface.accepted_dtypes``); this class is the common case.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DatatypeKind(Enum):
    """The datatype CATEGORY a port supports. ``ANY`` matches every kind (bitwidth still
    gates). Mirrors the base-type axis of brainsmith's ``DatatypeInRange``."""

    INTEGER = "integer"  # any integer type (signed or unsigned; includes BINARY)
    FLOAT = "float"
    FIXED = "fixed"
    ANY = "any"


@dataclass(frozen=True)
class DatatypeSupport:
    """A backend's declared datatype support for one port: a :class:`DatatypeKind` plus an
    inclusive bitwidth range. :meth:`accepts` is the gate ``pool_schema`` compiles into a
    guarded feasibility predicate (fires only when this backend is selected)."""

    kind: DatatypeKind = DatatypeKind.ANY
    min_bits: int = 1
    max_bits: int = 64

    def __post_init__(self):
        if self.min_bits <= 0:
            raise ValueError(f"min_bits must be positive, got {self.min_bits}")
        if self.max_bits < self.min_bits:
            raise ValueError(
                f"max_bits ({self.max_bits}) must be >= min_bits ({self.min_bits})"
            )

    def accepts(self, dt) -> str | None:
        """``None`` if ``dt`` is supported, else a reason string. Checks bitwidth range
        first, then category (``ANY`` skips the category check). ``dt`` is a QONNX
        ``BaseDataType``; category is read from ``is_integer()`` / the canonical name, as in
        brainsmith's proven ``DatatypeInRange.check``."""
        bits = dt.bitwidth()
        if not (self.min_bits <= bits <= self.max_bits):
            return (
                f"datatype {dt} bitwidth {bits} not in "
                f"[{self.min_bits}, {self.max_bits}]"
            )
        if self.kind is DatatypeKind.ANY:
            return None
        canonical = dt.get_canonical_name()
        ok = {
            DatatypeKind.INTEGER: dt.is_integer(),
            DatatypeKind.FLOAT: canonical.startswith("FLOAT"),
            DatatypeKind.FIXED: canonical.startswith("FIXED<"),
        }[self.kind]
        if not ok:
            return f"datatype {dt} is not {self.kind.value}"
        return None
