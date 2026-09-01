# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A1: the projection, before anything is keyed by it.

The claim the whole package rests on is that a preimage of ordered pairs of
tagged scalars needs no canonical-encoding dependency.  Two properties carry
it, and they are tested rather than asserted: the preimage contains no mapping
and no bare number whose type is ambiguous, and it is byte-identical across
processes.

The look-alike cases come first because they are the ones that would produce a
wrong *hit* -- ``True`` and ``1`` keying alike is a build sharing another
build's output, and nothing downstream would notice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest

from finn.dataflow.artifacts.projection import (
    PROJECTION_VERSION,
    ProjectionError,
    digest,
    preimage,
    project,
)


class _Resource(Enum):
    LUT = "lut"
    DSP = "dsp"


class _OtherResource(Enum):
    """A second enum with a colliding member name, to pin the qualification."""

    LUT = "lut"


@dataclass(frozen=True)
class _Point:
    name: str
    width: int


# -- look-alikes stay distinct -------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (True, 1),
        (False, 0),
        (1, 1.0),
        (1, "1"),
        (1.0, "1.0"),
        (True, "true"),
        (None, ""),
        (None, "none"),
    ],
)
def test_values_that_look_alike_do_not_key_alike(left: object, right: object) -> None:
    """The tag is in the preimage, so the number model is not load-bearing."""

    assert digest(left) != digest(right)


def test_two_enums_with_one_member_name_do_not_key_alike() -> None:
    """The bare member name collides; the qualified one cannot.

    Two unrelated ``Mode`` enums in different modules would both project as
    ``Mode.X`` -- a wrong hit reached through the one field added specifically
    to keep distinct choices distinct.
    """

    assert digest(_Resource.LUT) != digest(_OtherResource.LUT)


def test_an_enum_is_keyed_by_member_name_not_by_value() -> None:
    """Two members sharing a value are two choices."""

    projected = project(_Resource.DSP)
    assert projected == (("", "enum", f"{__name__}._Resource.DSP"),)


# -- the two properties §6.1 names ---------------------------------------------


def test_the_preimage_of_a_nested_value_contains_no_mapping_and_no_bare_number() -> None:
    """Every leaf is text with a tag beside it.  There is nothing else to order."""

    value = {
        "sources": ({"root": "finnlib", "order": 2},),
        "clock": 5.0,
        "pumped": True,
    }
    for path, tag, text in project(value):
        assert isinstance(path, str) and isinstance(text, str)
        assert tag in ("none", "bool", "int", "float", "str", "bytes", "enum")


def test_mapping_order_does_not_reach_the_key() -> None:
    """A mapping is emitted as its items sorted by key, so it is a sequence.

    This is the property that makes the RFC 8949 §4.2.1-versus-§4.2.3 question
    -- bytewise ordering against length-first ordering -- not ours to answer.
    """

    assert digest({"b": 1, "a": 2}) == digest({"a": 2, "b": 1})


def test_sequence_order_does_reach_the_key() -> None:
    """Compile order is a fact about the build, and reordering it is a change."""

    assert digest(("dotp.sv", "dotp_axi.sv")) != digest(("dotp_axi.sv", "dotp.sv"))


def test_a_float_is_exact_and_not_confusable_with_its_repr() -> None:
    """``hex()`` has no precision policy and no locale to disagree about."""

    assert project(0.1) == (("", "float", (0.1).hex()),)
    assert digest(0.1) != digest("0.1")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_value_no_build_depends_on_is_refused(value: float) -> None:
    """``nan != nan``, so a key containing one would never match itself."""

    with pytest.raises(ProjectionError):
        project(value)


def test_a_type_with_no_canonical_text_is_refused_rather_than_stringified() -> None:
    """``str()`` on an arbitrary object is ``repr()`` by another route."""

    with pytest.raises(ProjectionError, match="no canonical text"):
        project(object())


# -- the framing that keeps two different values from sharing bytes ------------


def test_where_a_boundary_falls_cannot_change_the_key() -> None:
    """Length prefixes, or ``("ab", "c")`` and ``("a", "bc")`` concatenate alike."""

    assert digest(("ab", "c")) != digest(("a", "bc"))


def test_a_path_is_part_of_the_preimage_so_position_is_not_lost() -> None:
    """The same scalar in two places is two different facts."""

    assert digest({"pe": 2, "simd": 1}) != digest({"pe": 1, "simd": 2})


def test_the_projection_version_prefixes_every_preimage() -> None:
    """A shape change is an unsupported projection, not an unexplained miss."""

    assert preimage(project("anything")).startswith(
        f"{len(PROJECTION_VERSION)}:{PROJECTION_VERSION}".encode()
    )


# -- dataclasses project by declared field order -------------------------------


def test_a_frozen_dataclass_projects_field_by_field_under_its_own_names() -> None:
    assert project(_Point("in0", 16)) == (
        ("name", "str", "in0"),
        ("width", "int", "16"),
    )


def test_two_dataclasses_with_the_same_values_in_different_fields_differ() -> None:
    """The field name is in the path, so the shape is part of the identity."""

    assert digest(_Point("a", 1)) != digest(_Point("1", 1))
