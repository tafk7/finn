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

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class DependentSpec:
    """A ``DatatypeSpec`` that ALSO declares the deriveds its derivation reads.

    A bare spec has no ordering constraint (it reads only axes + Context). When a derivation
    reads ANOTHER derived — e.g. MVAU's ``accDataType`` reads the storage owner's published
    ``parameters.<iface>.datatype`` — it wraps its spec here, carrying the dep names. Both
    consumers unwrap identically via :func:`spec_and_deps`: the inner ``spec`` resolves exactly
    as a bare one, and the ``deps`` flow onto the synthesized
    :class:`~finn.kernels.engine.derived.Derived` so the topo-sort orders it correctly.

    WHY A WRAPPER AND NOT JUST A ``Derived``: a ``DatatypeSpec`` is a UNION (None / DataType /
    str / VALUE_OPTIMIZED / Callable), not a closure — that union is the vocabulary's whole
    point, and it is load-bearing across both ops and both topologies. The ``Derived`` that
    ultimately carries these deps is SYNTHESIZED later (by ``_merge_derived_dtypes`` for a
    register, ``_width_derived`` for a port), so at declaration time there is no ``deps`` field
    to write into. Pairing the spec with its deps is therefore the minimum needed, not
    redundancy.

    Formerly ``RegisterSpec`` — a misnomer, since ``Interface.derived_dtype`` (a PORT, not a
    register) uses it too."""

    spec: Any  # the wrapped DatatypeSpec (None, DataType, str, VALUE_OPTIMIZED, or Callable)
    deps: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        if not isinstance(self.deps, frozenset):
            object.__setattr__(self, "deps", frozenset(self.deps))


def spec_and_deps(spec) -> tuple[Any, frozenset[str]]:
    """Normalize any declared dtype spec to ``(inner_spec, deps)``.

    The ONE place the optional wrapper is unwrapped, so a port and a register cannot drift
    into handling it differently — they previously each open-coded the same isinstance check."""
    if isinstance(spec, DependentSpec):
        return spec.spec, spec.deps
    return spec, frozenset()


def datatype_derived(name: str, spec, *, origin: str = "", extra_deps=()):
    """Lift one declared dtype spec into the :class:`~finn.kernels.engine.derived.Derived`
    that resolves it onto the point under ``name``.

    A ``DatatypeSpec`` is a UNION, not a closure, so something must synthesize the ``Derived``
    that carries it — and every caller needs the same two things: resolve the spec against
    ``(point, context)``, and hoist the spec's declared ``deps`` onto the node so the
    topo-sort orders it after whatever the derivation reads. Both call sites derived that
    pair independently before this existed.

    ``name`` doubles as the fallback tensor: an internal register has no port, so
    :func:`resolve_datatype_spec` reads Context through the register name only if the spec
    (``None``/``VALUE_OPTIMIZED``) asks it to. ``extra_deps`` is for a caller with an
    additional ordering constraint the spec itself does not express.
    """
    from .derived import Derived

    inner, spec_deps = spec_and_deps(spec)

    def compute(point, context, _spec=inner, _name=name):
        return resolve_datatype_spec(_spec, iface=_name, point=point, context=context)

    return Derived(
        name,
        compute,
        deps=frozenset(spec_deps) | frozenset(extra_deps),
        origin=origin,
    )


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

    spec, _deps = spec_and_deps(spec)  # deps are consumed at merge; resolution reads the spec
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
