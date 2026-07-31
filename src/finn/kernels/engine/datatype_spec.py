############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``DatatypeSpec`` — the one vocabulary for DECLARING a produced datatype.

A backend declares HOW a datatype it OWNS is produced — an output port's stream-width
dtype, or an internal register (accumulator, narrowed weight) with no port. Every such
declaration is a ``DatatypeSpec``, resolved by :func:`resolve_datatype_spec` against a
resolved point + frozen Context. This is the DERIVATION (endogenous) half of the datatype
model — its mirror is :class:`~finn.kernels.engine.datatype_support.DatatypeSupport`, the
GATE (exogenous) half that says which INPUT dtypes a port ACCEPTS.

Modeled on brainsmith's ``DatatypeSpec`` union + ``_resolve_datatype_spec``
(``../brainsmith/brainsmith/dataflow/types.py`` + ``spec_helpers.py``), adapted to the
engine's ``(point, context)`` closure signature. The union variants:

* ``None`` — no declared derivation; the resolver falls back to the graph tensor dtype
  (``context.tensor_datatype(iface)``). The unspecialized/absent case.
* a QONNX ``DataType`` — a FIXED datatype, used as-is (e.g. Softmax always FLOAT32).
* ``str`` — COPY another interface's graph dtype (the string names that interface).
* :data:`VALUE_OPTIMIZED` — narrow from a static initializer's actual VALUES, falling back
  to the graph dtype when the tensor is dynamic (no initializer). The acc/weight narrowing
  pattern.
* a ``Callable(point, context) -> DataType`` — a custom derivation (the escape hatch for a
  rule this vocabulary cannot express, e.g. MVAU's accumulator-or-graph output rule).
"""

from __future__ import annotations

from typing import Any, Callable


class _ValueOptimizedType:
    """Sentinel for VALUE_OPTIMIZED — narrow a datatype from a static tensor's actual
    values, falling back to the graph dtype when the tensor is dynamic. Mirrors brainsmith's
    ``_ValueOptimizedType``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "VALUE_OPTIMIZED"

    __str__ = __repr__


VALUE_OPTIMIZED = _ValueOptimizedType()


def value_optimized(iface: str) -> Callable[[Any, Any], Any]:
    """A ``DatatypeSpec`` callable that narrows ``iface``'s dtype from its static initializer
    VALUES, falling back to the graph dtype when the tensor is dynamic (no initializer).

    The narrowing body is the acc/weight pattern in MVAU's ``mvau_register_dtypes`` (base
    FINN matrixvectoractivation.py:529-549): a signed range uses the more-negative extreme,
    an unsigned range the max — then ``DataType.get_smallest_possible``. :data:`VALUE_OPTIMIZED`
    is sugar for ``value_optimized(<the port's own interface>)``; this factory is used
    directly when the narrowed tensor differs from the declaring port."""
    from qonnx.core.datatype import DataType

    def compute(point, context, _iface=iface):
        values = context.initializer(_iface)
        if values is None:
            return context.tensor_datatype(_iface)
        v_min = float(values.min())
        v_max = float(values.max())
        if v_min < 0:
            extreme = v_min if abs(v_min) > v_max else -v_max - 1
            return DataType.get_smallest_possible(extreme)
        return DataType.get_smallest_possible(v_max)

    return compute


def resolve_datatype_spec(spec: Any, *, iface: str, point, context):
    """Resolve a ``DatatypeSpec`` union value to a concrete QONNX ``DataType``.

    ``iface`` is the DECLARING interface — the fallback tensor for ``None``/``VALUE_OPTIMIZED``
    (an internal register with no port passes its register name; the resolver only reads
    Context through it). ``point``/``context`` are the resolved design point + frozen givens.

    Dispatch mirrors brainsmith's ``_resolve_datatype_spec``:

    * ``None`` → ``context.tensor_datatype(iface)`` (graph fallback).
    * ``DataType`` → as-is.
    * ``str`` → ``context.tensor_datatype(spec)`` (copy that interface's graph dtype).
    * :data:`VALUE_OPTIMIZED` → :func:`value_optimized` on ``iface``.
    * ``Callable`` → ``spec(point, context)``.
    """
    from qonnx.core.datatype import BaseDataType

    if spec is None:
        return context.tensor_datatype(iface)
    if isinstance(spec, BaseDataType):
        return spec
    if isinstance(spec, str):
        return context.tensor_datatype(spec)
    if spec is VALUE_OPTIMIZED:
        return value_optimized(iface)(point, context)
    if callable(spec):
        return spec(point, context)
    raise ValueError(
        f"Invalid DatatypeSpec {spec!r} (type {type(spec).__name__}). Must be None, "
        f"a DataType, a str interface name, VALUE_OPTIMIZED, or a callable."
    )
