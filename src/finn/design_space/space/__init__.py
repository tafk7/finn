############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""The resolve engine: Context, Axis, Derived, Predicate, Schema, and resolve.

A design space is data — a :class:`Schema` of guarded :class:`Axis` choices,
computed :class:`Derived` quantities, and :class:`Predicate` legality checks —
resolved against a :class:`Context` of givens into a :class:`Point` or an
:class:`Illegal`. See ``kernel-final-design/design-space-model.md``.
"""

from .axis import (
    Axis,
    Domain,
    PredicateDomain,
    discrete_axis,
    divisor_axis,
    fixed_axis,
    predicate_axis,
)
from .context import Context
from .derived import Derived
from .point import AbsentAxisError, Illegal, Point
from .predicate import Predicate, predicate
from .resolve import resolve
from .schema import Schema, SchemaError

__all__ = [
    "Context",
    "Axis",
    "Domain",
    "PredicateDomain",
    "discrete_axis",
    "divisor_axis",
    "fixed_axis",
    "predicate_axis",
    "Derived",
    "Predicate",
    "predicate",
    "Point",
    "Illegal",
    "AbsentAxisError",
    "Schema",
    "SchemaError",
    "resolve",
]
