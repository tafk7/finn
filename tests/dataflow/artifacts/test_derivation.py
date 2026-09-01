# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A1: ``Derivation`` invariants, one test per way a key could be wrong.

Two shapes, following the discipline the Phase 5 identity tests arrived at.
Positive: two things that are the same build key the same.  Negative, one per
declared input: change it and the key moves.  A missing negative is a build
input that could quietly leave the key without anyone noticing.
"""

from __future__ import annotations

from enum import Enum

import pytest

from finn.dataflow.artifacts.derivation import (
    ArtifactRef,
    ContentRef,
    Derivation,
    DerivationError,
    OutputLayout,
    ProducerIdentity,
    RequestSchema,
    ToolRequirement,
    build_key,
    tree_digest,
)

DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64

PRODUCER = ProducerIdentity("finn.kernel-source", "1")


class _Style(Enum):
    BLOCK = "block"
    DISTRIBUTED = "distributed"


def _derivation(**overrides: object) -> Derivation:
    defaults: dict[str, object] = {
        "kind": "kernel-source",
        "schema_version": "kernel-source-v1",
        "producer": PRODUCER,
        "templates": (ContentRef(DIGEST),),
        "inputs": (("dotp.sv", ContentRef(DIGEST)),),
        "options": (("SIMD", 2), ("PE", 2)),
        "outputs": OutputLayout(("dotp_axi.sv",)),
    }
    defaults.update(overrides)
    return Derivation(**defaults)  # type: ignore[arg-type]


# -- positive: equal declared inputs are one key -------------------------------


def test_two_derivations_with_equal_inputs_share_a_key() -> None:
    assert build_key(_derivation()) == build_key(_derivation())


def test_an_option_table_written_in_another_order_is_the_same_table() -> None:
    """Options are a mapping in everything but syntax, so order is not a fact."""

    assert build_key(_derivation(options=(("PE", 2), ("SIMD", 2)))) == build_key(_derivation())


# -- negative: one per declared input ------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "composed-source"),
        ("schema_version", "kernel-source-v2"),
        ("producer", ProducerIdentity("finn.kernel-source", "2")),
        ("producer", ProducerIdentity("finn.other-source", "1")),
        ("templates", (ContentRef(OTHER_DIGEST),)),
        ("templates", ()),
        ("inputs", (("dotp.sv", ContentRef(OTHER_DIGEST)),)),
        ("inputs", (("dotp_axi.sv", ContentRef(DIGEST)),)),
        ("options", (("SIMD", 4), ("PE", 2))),
        ("options", (("SIMD", 2),)),
        ("outputs", OutputLayout(("other.sv",))),
        ("request", RequestSchema("synth {top}")),
        ("tool", ToolRequirement("vitis-hls", ">=2023.2")),
    ],
)
def test_changing_a_declared_input_moves_the_key(field: str, value: object) -> None:
    assert build_key(_derivation(**{field: value})) != build_key(_derivation())


def test_input_order_moves_the_key_because_compile_order_is_a_fact() -> None:
    """``dotp_axi`` instantiates ``dotp``; a reordered manifest is another build.

    This is the reason ``inputs`` is an ordered tuple rather than the mapping
    §6 sketches.  A mapping would sort these two into one key.
    """

    forward = _derivation(inputs=(("dotp.sv", ContentRef(DIGEST)), ("axi.sv", ContentRef(DIGEST))))
    reversed_ = _derivation(
        inputs=(("axi.sv", ContentRef(DIGEST)), ("dotp.sv", ContentRef(DIGEST)))
    )
    assert build_key(forward) != build_key(reversed_)


def test_an_upstream_reference_and_a_blob_with_the_same_digest_differ() -> None:
    """A key and a content hash are different kinds of thing."""

    by_key = _derivation(inputs=(("upstream", ArtifactRef("kernel-source", DIGEST)),))
    by_content = _derivation(inputs=(("upstream", ContentRef(DIGEST)),))
    assert build_key(by_key) != build_key(by_content)


def test_an_enum_option_is_a_choice_and_not_its_value() -> None:
    assert build_key(_derivation(options=(("ram_style", _Style.BLOCK),))) != build_key(
        _derivation(options=(("ram_style", "block"),))
    )


# -- what a derivation refuses -------------------------------------------------


def test_a_naming_collision_is_refused_rather_than_silently_deduplicated() -> None:
    with pytest.raises(DerivationError, match="named twice"):
        _derivation(options=(("SIMD", 2), ("SIMD", 4)))
    with pytest.raises(DerivationError, match="named twice"):
        _derivation(inputs=(("a.sv", ContentRef(DIGEST)), ("a.sv", ContentRef(OTHER_DIGEST))))


def test_a_value_with_no_canonical_text_is_refused_where_it_was_supplied() -> None:
    """Not at the lookup that needed the key, which is a call too late."""

    with pytest.raises(DerivationError, match="cannot enter a key"):
        _derivation(options=(("bad", object()),))


def test_a_rendered_command_is_refused_as_a_request_schema() -> None:
    """A rendered command carries materialized paths; a shape cannot."""

    with pytest.raises(DerivationError, match="names no substitution"):
        RequestSchema("vivado -mode batch -source run.tcl")


def test_a_schema_that_hard_codes_an_absolute_path_is_refused() -> None:
    with pytest.raises(DerivationError, match="absolute path"):
        RequestSchema("synth {top} -dir /home/build/out")


def test_a_declared_layout_must_not_carry_a_materialized_path() -> None:
    for name in ("/abs/x.sv", "../escape.sv", ""):
        with pytest.raises(DerivationError):
            OutputLayout((name,))


def test_a_stage_must_say_what_it_produces() -> None:
    with pytest.raises(DerivationError, match="declare what it produces"):
        OutputLayout(())


def test_a_content_ref_refuses_anything_that_is_not_a_digest() -> None:
    for value in ("", "abc", "A" * 64, "g" * 64):
        with pytest.raises(DerivationError):
            ContentRef(value)


# -- tree_digest is the other identity, and it is not the lookup key -----------


def test_the_tree_digest_covers_names_as_well_as_contents() -> None:
    """The same bytes under another name are a different tree."""

    assert tree_digest({"a.sv": DIGEST}) != tree_digest({"b.sv": DIGEST})


def test_the_tree_digest_does_not_depend_on_the_order_it_was_collected_in() -> None:
    assert tree_digest({"b.sv": DIGEST, "a.sv": OTHER_DIGEST}) == tree_digest(
        {"a.sv": OTHER_DIGEST, "b.sv": DIGEST}
    )


def test_an_empty_tree_is_not_a_completed_one() -> None:
    with pytest.raises(DerivationError):
        tree_digest({})


def test_the_request_schema_reports_the_names_it_expects() -> None:
    assert RequestSchema("synth {top} -part {part}").substitutions == ("top", "part")
