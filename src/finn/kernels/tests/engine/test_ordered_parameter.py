############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""OrderedParameter invariants + DSE navigation (E9).

Vendored value object, but load-bearing: divisor_axis builds these as fold-dial domains
and the design-space explorer navigates them (step/percentage/min/max). The invariants
(sorted, unique, non-empty, default-in-values) are enforced at construction; navigation
clamps at the bounds and raises off-domain.
"""

import pytest

from finn.kernels.engine.ordered_parameter import OrderedParameter


# --- invariants (construction-time) ----------------------------------------


def test_empty_values_rejected():
    with pytest.raises(ValueError, match="empty"):
        OrderedParameter("p", ())


def test_unsorted_values_rejected():
    with pytest.raises(ValueError, match="sorted"):
        OrderedParameter("p", (4, 2, 1))


def test_duplicate_values_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        OrderedParameter("p", (1, 2, 2, 4))


def test_default_must_be_in_values():
    with pytest.raises(ValueError, match="Default"):
        OrderedParameter("p", (1, 2, 4), default=3)


def test_list_values_coerced_to_tuple():
    p = OrderedParameter("p", [1, 2, 4])
    assert p.values == (1, 2, 4)


# --- positional access ------------------------------------------------------


def test_min_max_default():
    p = OrderedParameter("SIMD", (1, 2, 4, 8, 16))
    assert p.min() == 1
    assert p.max() == 16
    assert p.get_default() == 1  # min when no explicit default
    assert OrderedParameter("SIMD", (1, 2, 4), default=2).get_default() == 2


def test_at_index_supports_negative_and_raises_out_of_range():
    p = OrderedParameter("PE", (1, 2, 4, 8, 16))
    assert p.at_index(0) == 1
    assert p.at_index(-1) == 16
    assert p.at_index(2) == 4
    with pytest.raises(IndexError):
        p.at_index(5)


def test_index_of_and_off_domain_raises():
    p = OrderedParameter("SIMD", (1, 2, 4, 8, 16))
    assert p.index_of(4) == 2
    with pytest.raises(ValueError):
        p.index_of(3)


# --- navigation -------------------------------------------------------------


def test_step_up_clamps_at_max():
    p = OrderedParameter("PE", (1, 2, 4, 8, 16, 32, 64))
    assert p.step_up(4, 1) == 8
    assert p.step_up(4, 2) == 16
    assert p.step_up(32, 10) == 64  # clamped


def test_step_down_clamps_at_min():
    p = OrderedParameter("SIMD", (1, 2, 4, 8, 16, 32, 64))
    assert p.step_down(16, 1) == 8
    assert p.step_down(16, 2) == 4
    assert p.step_down(4, 10) == 1  # clamped


def test_step_requires_nonnegative_n():
    p = OrderedParameter("PE", (1, 2, 4))
    with pytest.raises(ValueError):
        p.step_up(1, -1)
    with pytest.raises(ValueError):
        p.step_down(4, -1)


# --- percentage access ------------------------------------------------------


def test_at_percentage_endpoints_and_middle():
    p = OrderedParameter("PE", (1, 2, 4, 8, 16))
    assert p.at_percentage(0.0) == 1
    assert p.at_percentage(1.0) == 16
    assert p.at_percentage(0.5, rounding="natural") == 4


def test_at_percentage_rounding_modes():
    # 0.7 * 4 = 2.8 → floor=2 (values[2]=4), ceil=3 (values[3]=8).
    p = OrderedParameter("PE", (1, 2, 4, 8, 16))
    assert p.at_percentage(0.7, rounding="down") == 4
    assert p.at_percentage(0.7, rounding="up") == 8


def test_at_percentage_out_of_range_and_bad_mode():
    p = OrderedParameter("PE", (1, 2, 4))
    with pytest.raises(ValueError):
        p.at_percentage(1.5)
    with pytest.raises(ValueError):
        p.at_percentage(0.5, rounding="sideways")


# --- iteration/membership ---------------------------------------------------


def test_iteration_and_membership():
    p = OrderedParameter("PE", (1, 2, 4, 8))
    assert list(p) == [1, 2, 4, 8]
    assert len(p) == 4
    assert 4 in p
    assert 3 not in p
