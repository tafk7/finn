############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``Constraint`` — declarative structural legality that compiles to a ``Predicate``.

A constraint is DATA (a small frozen dataclass), not a hand-written closure: it names a
structural-legality rule that tiling cannot derive — a tensor rank, a value-sign rule, a
static-initializer requirement, a datatype-support gate. :func:`compile_constraint` lowers
one to a :class:`~finn.kernels.engine.predicate.Predicate` the resolver runs unchanged.

Two wins over the raw predicate closures these replace (design pitch, Finding 3):

* **Optionality is structural.** Every constraint names the port it constrains (``iface``);
  the compiled predicate auto-noops when that port's tensor is ABSENT
  (``not context.has_tensor(iface)``). This bakes in once the ``if not
  ctx.has_tensor(THRESHOLDS): return None`` prelude that opened every optional-port rule.
  The skip fires on the CONSTRAINED port — a value rule gated on another tensor (unsigned
  input ⇒ thresholds ≥ 0) still skips on its own port (thresholds), not the gate's read
  port (input).
* **One vocabulary, one compile path.** ``DatatypeSupport`` (the per-port datatype gate) is
  just another member via :class:`DatatypeConstraint`; ``pool_space`` compiles it and every
  other per-port/kernel-level constraint through this one function instead of a special-cased
  branch.

The vocabulary is deliberately SMALL (rank, value-sign, static, datatype, custom escape
hatch): dimension/divisibility/shape-equal rules are tiling-engine-DERIVED here (not
hand-authored as in brainsmith), so importing a large catalog would regress the one place
we lead. ``CustomConstraint`` is the escape hatch for a one-off closure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .predicate import Predicate


@dataclass(frozen=True)
class ShapeRank:
    """The port's tensor must have exactly ``rank`` dims. The per-port structural rule that
    replaces a hand-written ``len(shape) != N`` predicate."""

    iface: str
    rank: int

    def check(self, point, context) -> str | None:
        shp = context.tensor_shape(self.iface)
        if len(shp) != self.rank:
            return f"{self.iface} tensor must be rank {self.rank} (got shape {shp})"
        return None

    def describe(self) -> str:
        return f"{self.iface} is rank {self.rank}"


@dataclass(frozen=True)
class ValueNonNeg:
    """The port's initializer values must all be >= 0, optionally gated by ``when`` — a
    ``(point, context) -> bool`` condition that may read ANOTHER tensor (the relational
    rule: unsigned INPUT ⇒ THRESHOLDS >= 0). The optional-port skip fires on ``iface`` (the
    constrained port), independent of what ``when`` reads."""

    iface: str
    when: Callable[[Any, Any], bool] | None = None
    deps: frozenset[str] = frozenset()  # point keys `when` reads, if any

    def check(self, point, context) -> str | None:
        if self.when is not None and not self.when(point, context):
            return None
        init = context.initializer(self.iface)
        if init is not None and (init < 0).any():
            return f"{self.iface} >= 0 required for all values (got a negative)"
        return None

    def describe(self) -> str:
        return f"{self.iface} >= 0" + ("" if self.when is None else " (when gated)")


@dataclass(frozen=True)
class IsStatic:
    """The port must carry a static initializer, optionally exempted by ``unless`` — a
    ``(point) -> bool`` reading the point (e.g. runtime-writable weights need no static
    initializer). Replaces the weight-present rule."""

    iface: str
    unless: Callable[[Any], bool] | None = None
    deps: frozenset[str] = frozenset()  # point keys `unless` reads, if any

    def check(self, point, context) -> str | None:
        if self.unless is not None and self.unless(point):
            return None
        if context.initializer(self.iface) is None:
            return f"{self.iface} initializer required (must be statically known)"
        return None

    def describe(self) -> str:
        return f"{self.iface} is static"


@dataclass(frozen=True)
class DatatypeConstraint:
    """Adapts a per-port datatype gate — a
    :class:`~finn.kernels.engine.datatype_support.DatatypeSupport` (its ``accepts``) or a raw
    ``(dt) -> reason | None`` callable — to the constraint protocol, binding the ``iface``
    whose Context dtype it reads. Lets ``pool_space`` compile datatype support through the
    SAME path as every other constraint (Finding 3b)."""

    iface: str
    support: Any  # DatatypeSupport or a (dt) -> reason|None callable

    def check(self, point, context) -> str | None:
        accepts = self.support.accepts if hasattr(self.support, "accepts") else self.support
        return accepts(context.tensor_datatype(self.iface))

    def describe(self) -> str:
        return f"{self.iface} datatype support"


@dataclass(frozen=True)
class CustomConstraint:
    """The escape hatch — a raw ``(point, context) -> reason | None`` closure for a rule the
    vocabulary above cannot express. ``iface`` (optional) enables the optional-port skip;
    omit it for a rule not bound to a single port."""

    fn: Callable[[Any, Any], str | None]
    desc: str = ""
    iface: str | None = None
    deps: frozenset[str] = frozenset()  # point keys `fn` reads, if any

    def check(self, point, context) -> str | None:
        return self.fn(point, context)

    def describe(self) -> str:
        return self.desc


def compile_constraint(constraint) -> Predicate:
    """Lower a constraint to a :class:`Predicate`, baking in the optional-port skip: the
    predicate noops when the constrained port's tensor is absent
    (``not context.has_tensor(iface)``). A constraint with no ``iface`` (a port-agnostic
    ``CustomConstraint``) always runs.

    ``ShapeRank`` and ``DatatypeConstraint`` read ONLY Context, so they compile to
    zero-dep predicates — decidable before any choice is pinned, which is what makes a
    cheap feasibility gate possible. The rest carry whatever their ``when``/``unless``/``fn``
    closure declares."""
    iface = getattr(constraint, "iface", None)

    def check(point, context, _c=constraint, _iface=iface):
        if _iface is not None and not context.has_tensor(_iface):
            return None
        return _c.check(point, context)

    return Predicate(
        check=check,
        description=constraint.describe(),
        deps=getattr(constraint, "deps", frozenset()),
    )
