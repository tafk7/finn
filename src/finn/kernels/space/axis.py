############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``Axis`` — a free design choice with guarded existence and a point-dependent
domain (design-space-model.md §1.2).

An axis is data: a name plus three callables. ``exists`` is the guard — when it
returns False the axis is *not in the point at all* (reading it errors, it is not
defaulted). ``domain`` returns the valid values (an ``OrderedParameter`` when
ordered/navigable, or a ``frozenset`` when discrete) and may read the point and
context. ``default`` supplies the value when the caller's assignment does not pin
the axis.

``deps`` names the axes this axis's guard/domain reads, so the schema can
topologically order resolution without introspecting closures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Union

from finn.kernels.primitives.ordered_parameter import OrderedParameter


class PredicateDomain:
    """A domain defined by a membership test rather than an enumerated set.

    For axes whose valid values cannot be enumerated as a finite set — an
    arbitrary integer bias (``ActVal``) or a list of vector counts
    (``numInputVectors``). ``val in dom`` delegates to the test.
    """

    __slots__ = ("label", "test")

    def __init__(self, label: str, test: Callable[[Any], bool]):
        self.label = label
        self.test = test

    def __contains__(self, value) -> bool:
        return bool(self.test(value))

    def __repr__(self) -> str:
        return f"PredicateDomain({self.label})"


# A Domain is ordered/navigable, a discrete set, or a membership predicate.
# Membership is `val in dom` for all three.
Domain = Union[OrderedParameter, frozenset, PredicateDomain]


def _always(_point) -> bool:
    return True


@dataclass(frozen=True)
class Axis:
    name: str
    domain: Callable[[Any, Any], Domain]
    default: Callable[[Any, Any], Any]
    exists: Callable[[Any], bool] = _always
    deps: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        if not isinstance(self.deps, frozenset):
            object.__setattr__(self, "deps", frozenset(self.deps))


# =============================================================================
# Factory helpers — cut the boilerplate of the common axis shapes.
# =============================================================================


def discrete_axis(
    name: str,
    values,
    default,
    *,
    guard: Callable[[Any], bool] | None = None,
    deps=(),
) -> Axis:
    """A choice over a fixed discrete set (a ``frozenset`` domain).

    ``default`` may be a plain value or a ``(point, context) -> value`` callable.
    ``guard`` is the existence predicate; omit it for an always-present axis.
    """
    dom = frozenset(values)
    default_fn = default if callable(default) else (lambda p, ctx, _d=default: _d)
    exists_fn = guard if guard is not None else _always
    return Axis(
        name=name,
        domain=lambda p, ctx, _dom=dom: _dom,
        default=default_fn,
        exists=exists_fn,
        deps=frozenset(deps),
    )


def divisor_axis(
    name: str,
    dim: str,
    default,
    *,
    guard: Callable[[Any], bool] | None = None,
    deps=(),
) -> Axis:
    """A folding-factor choice whose domain is the divisors of a context dim.

    ``dim`` is the name of an axis/derived carrying the dimension (e.g. "MH" for
    PE, "MW" for SIMD); its value is read from the point. The domain is built as
    an ordered ``OrderedParameter`` so the search can navigate it. Mirrors MVAU's
    ``PE.domain = divisors(MH)`` (base:351).
    """
    default_fn = default if callable(default) else (lambda p, ctx, _d=default: _d)
    exists_fn = guard if guard is not None else _always

    def domain(p, ctx, _dim=dim, _name=name):
        n = p[_dim]
        divisors = tuple(d for d in range(1, n + 1) if n % d == 0)
        return OrderedParameter(_name, divisors)

    return Axis(
        name=name,
        domain=domain,
        default=default_fn,
        exists=exists_fn,
        deps=frozenset(deps) | {dim},
    )


def predicate_axis(
    name: str,
    label: str,
    test: Callable[[Any], bool],
    default,
    *,
    guard: Callable[[Any], bool] | None = None,
    deps=(),
) -> Axis:
    """A choice whose domain is a membership test (non-enumerable values)."""
    dom = PredicateDomain(label, test)
    default_fn = default if callable(default) else (lambda p, ctx, _d=default: _d)
    exists_fn = guard if guard is not None else _always
    return Axis(
        name=name,
        domain=lambda p, ctx, _dom=dom: _dom,
        default=default_fn,
        exists=exists_fn,
        deps=frozenset(deps),
    )


def fixed_axis(name: str, default, *, deps=()) -> Axis:
    """An axis whose domain is the single given value (a context-fixed quantity
    such as MW/MH that comes from tensor shape but is addressed like an axis)."""
    default_fn = default if callable(default) else (lambda p, ctx, _d=default: _d)

    def domain(p, ctx):
        return frozenset({default_fn(p, ctx)})

    return Axis(name=name, domain=domain, default=default_fn, deps=frozenset(deps))
