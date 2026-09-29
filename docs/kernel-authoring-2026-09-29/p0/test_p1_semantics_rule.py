"""P0.1: `T | Rejected` infers the semantics of `T` (plan D2).

The rule is installed as a wrapper around the base `output_semantics`: a union
of one value type and the engine's result markers is presented to it as the
value type alone. That is exactly `p1_output_semantics.patch` (the same rewrite
inside the function), so the probe runs on the unmodified base.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests <venv python> -m pytest -q \
        docs/kernel-authoring-2026-09-29/p0/test_p1_semantics_rule.py
"""

from __future__ import annotations

import types
from collections.abc import Iterator, Mapping
from typing import Union, get_args, get_origin

import pytest

from finn.core.space import (
    Param,
    Rejected,
    Space,
    _signatures,
    collection,
    constraint,
    derived,
    design_space,
    reject,
)
from finn.core.space.collection import collect_space
from finn.core.space.errors import DefinitionError
from finn.core.space.results import Inapplicable, Unresolved
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.schedule import Index
from finn.dataflow.tensor import TENSOR, Tensor
from finn.dataflow.traversal import BEAT_SEQUENCE, BeatSequence, vector_major
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS

BASE = _signatures.output_semantics
MARKERS = (Inapplicable, Rejected, Unresolved)


def marked_value_type(annotation: object) -> object | None:
    if get_origin(annotation) not in (Union, types.UnionType):
        return None
    values = [arg for arg in get_args(annotation) if arg not in MARKERS]
    return values[0] if len(values) == 1 else None


def d2_output_semantics(declaration, hints: Mapping[str, object], owner: str):  # type: ignore[no-untyped-def]
    annotation = hints.get("return")
    if _signatures._answer_value_type(annotation) is None:
        value_type = marked_value_type(annotation)
        if value_type is not None:
            hints = {**hints, "return": value_type}
    return BASE(declaration, hints, owner)


@pytest.fixture
def d2(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(collection, "output_semantics", d2_output_semantics)
    yield


def semantics_of(space_type: type[Space], name: str):  # type: ignore[no-untyped-def]
    effective = collect_space(space_type)
    return effective.semantics[effective.members[name]]


# -- the rule -----------------------------------------------------------------------------


def make_marked() -> type[Space]:
    class Marked(Space):
        n: int = Param()

        @derived
        def sequence(self) -> BeatSequence | Rejected:
            if self.n % 2:
                return reject("odd", f"{self.n} is odd")
            return BeatSequence(vector_major((self.n,), 2))

        @derived
        def tensor(self) -> Tensor | Inapplicable | Rejected:
            return reject("no-tensor", "never built")

        @derived
        def index(self) -> tuple[Index, ...] | Rejected:
            return (Index("i"),)

        @constraint
        def even(self) -> bool | Rejected:
            return True

    return Marked


def test_base_rule_refuses_a_marked_union() -> None:
    if hasattr(_signatures, "_marked_value_type"):
        pytest.skip("src carries the D2 patch; the base rule is not in effect")
    with pytest.raises(DefinitionError, match="needs explicit semantics="):
        collect_space(make_marked())


def test_marked_union_infers_the_value_type(d2: None) -> None:
    Marked = make_marked()
    assert semantics_of(Marked, "sequence").type_token is BeatSequence
    assert semantics_of(Marked, "tensor").type_token is Tensor
    assert semantics_of(Marked, "index").type_token is tuple
    assert semantics_of(Marked, "even").type_token is bool  # constraints: unchanged
    even, odd = design_space(Marked(n=4)), design_space(Marked(n=3))
    assert even.sequence == BeatSequence(vector_major((4,), 2))
    refused = odd.query(Marked.sequence)
    assert isinstance(refused, Rejected) and refused.findings[0].code == "odd"


# -- explicit semantics= ------------------------------------------------------------------


def test_explicit_semantics_still_overrides(d2: None) -> None:
    class Explicit(Space):
        @derived(semantics=BEAT_SEQUENCE)
        def sequence(self) -> BeatSequence | Rejected:
            return BeatSequence(vector_major((2,), 2))

    chosen = semantics_of(Explicit, "sequence")
    assert chosen is BEAT_SEQUENCE and chosen.name == "beat_sequence"


def make_mismatched() -> type[Space]:
    class Mismatched(Space):
        @derived(semantics=TENSOR)
        def sequence(self) -> BeatSequence | Rejected:
            return BeatSequence(vector_major((2,), 2))

    return Mismatched


def test_explicit_semantics_is_checked_against_the_value_type(d2: None) -> None:
    with pytest.raises(DefinitionError, match="BeatSequence is incompatible with Tensor"):
        collect_space(make_mismatched())


def test_base_rule_does_not_check_an_explicit_override_on_a_union() -> None:
    if hasattr(_signatures, "_marked_value_type"):
        pytest.skip("src carries the D2 patch; the base rule is not in effect")
    # Today the union skips the check: a Tensor semantics on a BeatSequence output passes.
    assert semantics_of(make_mismatched(), "sequence") is TENSOR


# -- what still needs semantics= ----------------------------------------------------------


def test_protocol_still_needs_semantics(d2: None) -> None:
    class Proto(Space):
        @derived
        def dtype(self) -> QONNXDataType | Rejected:
            raise AssertionError("never run")

    with pytest.raises(DefinitionError, match="Protocol outputs require explicit semantics="):
        collect_space(Proto)

    class ProtoExplicit(Space):
        @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
        def dtype(self) -> QONNXDataType | Rejected:
            raise AssertionError("never run")

    assert semantics_of(ProtoExplicit, "dtype") is QONNX_DATATYPE_VALUE_SEMANTICS


@pytest.mark.parametrize("which", ["optional", "two_values"])
def test_genuine_union_still_needs_semantics(d2: None, which: str) -> None:
    class Optional(Space):
        @derived
        def policy(self) -> Integer | None:
            return None

    class TwoValues(Space):
        @derived
        def value(self) -> int | str | Rejected:
            return 1

    family = {"optional": Optional, "two_values": TwoValues}[which]
    with pytest.raises(DefinitionError, match="needs explicit semantics="):
        collect_space(family)
