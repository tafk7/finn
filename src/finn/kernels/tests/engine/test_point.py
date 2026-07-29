############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Point / Illegal value semantics (E2, E8).

Callers rely on these: ``if not result:`` branches on Illegal falsiness; guards use
``in``/``.get`` to test presence without raising; reading an absent name is a hard
AbsentAxisError, never a silent default. Points are immutable Mappings; Illegals compare
and hash by their reason tuple.
"""

import pytest

from finn.kernels.engine.point import AbsentAxisError, Illegal, Point


# --- Point (E2/E8) ---------------------------------------------------------


def test_point_absent_axis_raises_on_attr_and_key():
    p = Point({"a": 1})
    assert "a" in p
    assert p.a == 1
    assert p["a"] == 1
    with pytest.raises(AbsentAxisError):
        _ = p.b
    with pytest.raises(AbsentAxisError):
        _ = p["b"]


def test_point_presence_is_testable_without_raising():
    p = Point({"a": 1})
    assert "b" not in p
    assert p.get("b") is None
    assert p.get("a") == 1


def test_absent_axis_error_is_a_keyerror():
    # Callers narrowing on KeyError (resolve's _validate skip path) must catch it.
    assert issubclass(AbsentAxisError, KeyError)


def test_point_is_immutable():
    p = Point({"a": 1})
    with pytest.raises(AttributeError):
        p.a = 2


def test_point_is_a_mapping():
    p = Point({"a": 1, "b": 2})
    assert len(p) == 2
    assert set(p) == {"a", "b"}
    assert dict(p) == {"a": 1, "b": 2}


# --- Illegal (E8) ----------------------------------------------------------


def test_illegal_is_falsy():
    assert not Illegal(["nope"])


def test_illegal_eq_and_hash_by_reasons():
    a = Illegal(["r1", "r2"])
    b = Illegal(["r1", "r2"])
    c = Illegal(["r1"])
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert a != "not an illegal"


def test_illegal_is_immutable():
    ill = Illegal(["r"])
    with pytest.raises(AttributeError):
        ill.reasons = ("other",)
