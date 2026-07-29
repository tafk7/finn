############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################
"""The resolve engine: the generic design-space core.

A design space is data — a :class:`~finn.kernels.engine.schema.Schema` of guarded
:class:`~finn.kernels.engine.axis.Axis` choices, computed
:class:`~finn.kernels.engine.derived.Derived` quantities, and
:class:`~finn.kernels.engine.predicate.Predicate` legality checks — resolved against a
:class:`~finn.kernels.engine.context.Context` of givens into a
:class:`~finn.kernels.engine.point.Point` or an :class:`~finn.kernels.engine.point.Illegal`.

Also holds the domain value objects the engine reuses (``ordered_parameter``,
``spec_helpers``; formerly ``kernels/primitives/``).
"""
