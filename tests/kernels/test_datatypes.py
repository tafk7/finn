# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The datatype boundary: what may enter the stack, and as what.

Adopting QONNX datatypes removes a lossy conversion, and these pin the three
things that removal depends on -- that identity is by canonical name, that
recognition never promises more than the snapshot can deliver, and that a
datatype and its own name never get confused for one another.

The last is not a style point.  ``DataType["INT8"] == "INT8"`` is true and their
hashes agree, so if a string were admitted to this value domain a name and the
live value would be interchangeable, and the two would collide as mapping keys
without raising.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from qonnx.core.datatype import BaseDataType, DataType

from finn.core.space import ValueSemantics
from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    canonical_qonnx_datatype,
    is_qonnx_datatype,
    resolve_qonnx_datatype_name,
)
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS

#: The datatype domain at ``object``, so a test can offer it values of any type.
QONNX_DATATYPE_SEMANTICS = cast(ValueSemantics[object], QONNX_DATATYPE_VALUE_SEMANTICS)

#: Every datatype family the stack could be handed, including the ones the
#: previous representation could not express at all.
REPRESENTATIVE = (
    "BINARY",
    "BIPOLAR",
    "TERNARY",
    "INT1",
    "INT2",
    "INT8",
    "INT32",
    "UINT8",
    "FIXED<8,4>",
    "FIXED<8,3>",
    "SCALEDINT<8>",
    "FLOAT16",
    "FLOAT32",
    "FLOAT<5,10,15>",
    "FLOAT<5,10,7>",
)

#: Pairs that reduced to one value under ``NumericElementType`` and must not.
#: ``TERNARY``/``INT2`` is the one that was live: a ternary MatMul was admitted
#: and its artifact declared ``INT2``.
FORMERLY_COLLIDING = (
    ("TERNARY", "INT2"),
    ("FLOAT16", "FLOAT<5,10,7>"),
    ("FLOAT16", "FLOAT<5,10,15>"),
    ("FIXED<8,4>", "FIXED<8,3>"),
    ("BINARY", "BIPOLAR"),
    ("BIPOLAR", "INT1"),
)


# -- identity ----------------------------------------------------------------


@pytest.mark.parametrize(("left", "right"), FORMERLY_COLLIDING)
def test_equal_width_types_with_different_meanings_stay_different(left: str, right: str) -> None:
    assert DataType[left] != DataType[right]


def test_uint1_and_binary_are_deliberately_the_same_value() -> None:
    """The one merge adoption brings with it, pinned as accepted.

    QONNX canonicalizes ``UINT1`` to ``BINARY``, so they are one value, not two
    that happen to compare equal.  Adopting QONNX means adopting its
    canonicalization; a FINN-local exception here would re-establish the second
    authority this whole change removes.  Recorded so the merge reads as a
    decision rather than being rediscovered later as a bug.
    """

    assert DataType["UINT1"] == DataType["BINARY"]
    assert resolve_qonnx_datatype_name("UINT1").name == "BINARY"


# -- the string hazard -------------------------------------------------------


def test_a_canonical_name_is_not_a_datatype_value() -> None:
    assert is_qonnx_datatype("INT8") is False
    with pytest.raises(DatatypeError, match="resolve"):
        canonical_qonnx_datatype("INT8")


def test_value_semantics_do_not_admit_a_string_into_the_datatype_domain() -> None:
    assert QONNX_DATATYPE_SEMANTICS.accepts(DataType["INT8"]) is True
    assert QONNX_DATATYPE_SEMANTICS.accepts("INT8") is False


def test_a_string_and_its_datatype_would_otherwise_collide() -> None:
    """Why the rejection above is a correctness rule, not tidiness.

    Equality holds in both directions and the hashes agree, so the two really
    are one mapping key.  Nothing raises; the second write wins silently.  This
    documents the hazard the boundary exists to keep out of the stack.
    """

    assert DataType["INT8"] == "INT8"
    assert "INT8" == DataType["INT8"]
    assert hash(DataType["INT8"]) == hash("INT8")

    mixed: dict[Any, str] = {"INT8": "the name"}
    mixed[DataType["INT8"]] = "the datatype"
    assert len(mixed) == 1

    # The engine is protected because it compares only within the domain.
    assert QONNX_DATATYPE_SEMANTICS.values_equal(DataType["INT8"], "INT8") is False


# -- recognition implies the snapshot succeeds -------------------------------


@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_whatever_is_recognized_can_also_be_frozen(name: str) -> None:
    """The property the two hooks share a helper to guarantee."""

    datatype = DataType[name]
    assert QONNX_DATATYPE_SEMANTICS.accepts(datatype) is True
    assert QONNX_DATATYPE_SEMANTICS.freeze(datatype) == datatype


class _Unregistered(BaseDataType):  # type: ignore[misc]
    """A well-formed subclass QONNX has never heard of."""

    def get_canonical_name(self) -> str:
        return "NOTATYPE<3>"

    def bitwidth(self) -> int:
        return 3

    def min(self) -> int:
        return 0

    def max(self) -> int:
        return 7

    def allowed(self, value: float) -> bool:
        return 0 <= value <= 7

    def is_integer(self) -> bool:
        return True

    def is_fixed_point(self) -> bool:
        return False

    def to_numpy_dt(self) -> Any:
        return None

    def get_num_possible_values(self) -> int:
        return 8

    def get_hls_datatype_str(self) -> str:
        return "ap_uint<3>"


def test_an_unregistered_subclass_is_refused_at_recognition() -> None:
    """Not later, inside the snapshot: ``isinstance`` alone would admit it. A
    subclass defined outside qonnx is not one of its datatype values."""

    rogue = _Unregistered()
    assert isinstance(rogue, BaseDataType)
    assert is_qonnx_datatype(rogue) is False
    assert QONNX_DATATYPE_SEMANTICS.accepts(rogue) is False
    with pytest.raises(DatatypeError, match="not a QONNX datatype value"):
        canonical_qonnx_datatype(rogue)


@pytest.mark.parametrize(
    "name",
    [
        "NOTATYPE",  # no prefix QONNX dispatches on -- raises KeyError
        "INT8 but wrong",  # dispatches on "INT", then fails parsing -- ValueError
        "UINT",  # prefix with nothing after it
        "FIXED<8>",  # right family, wrong arity
        "",
    ],
)
def test_a_malformed_canonical_name_is_refused(name: str) -> None:
    """One failure type out: qonnx raises ``KeyError`` for every name that denotes
    no datatype, however far its parse got, and the boundary refuses it."""

    with pytest.raises(DatatypeError, match="no QONNX datatype is named"):
        resolve_qonnx_datatype_name(name)


class _ExplodingName(BaseDataType):  # type: ignore[misc]
    """A well-formed subclass that raises from ``get_canonical_name()``.

    Not a contrived case so much as a general one: the boundary calls a
    third-party method, and a third-party method may raise anything.
    """

    def get_canonical_name(self) -> str:
        raise RuntimeError("boom")

    def bitwidth(self) -> int:
        return 8

    def min(self) -> int:
        return 0

    def max(self) -> int:
        return 255

    def allowed(self, value: float) -> bool:
        return True

    def is_integer(self) -> bool:
        return True

    def is_fixed_point(self) -> bool:
        return False

    def to_numpy_dt(self) -> Any:
        return None

    def get_num_possible_values(self) -> int:
        return 256

    def get_hls_datatype_str(self) -> str:
        return "ap_uint<8>"


def test_recognition_is_total_over_arbitrary_failures() -> None:
    """``is_qonnx_datatype`` answers; it does not raise.

    qonnx's ``is_datatype`` runs no method of the value it is asked about, so a
    subclass whose methods raise is answered (False) without running them.
    """

    rogue = _ExplodingName()
    assert isinstance(rogue, BaseDataType)

    assert is_qonnx_datatype(rogue) is False
    assert QONNX_DATATYPE_SEMANTICS.accepts(rogue) is False
    with pytest.raises(DatatypeError, match="not a QONNX datatype value"):
        canonical_qonnx_datatype(rogue)


def test_a_jointly_invalid_fixed_point_name_is_refused_not_raised() -> None:
    """``FIXED<8,9>``: parts individually well formed, jointly invalid; a name
    qonnx cannot resolve, refused like any other."""

    with pytest.raises(DatatypeError, match="no QONNX datatype is named"):
        resolve_qonnx_datatype_name("FIXED<8,9>")


def test_source_facing_names_stay_permissive() -> None:
    """A source-facing name resolves as QONNX itself would read it.

    A legacy node attribute like ``accDataType`` is whatever a human or an older
    FINN wrote. Adopting QONNX means adopting its reading of those spellings,
    including the ones it normalizes into another name.
    """

    assert resolve_qonnx_datatype_name("UINT1") == DataType["BINARY"]
    assert resolve_qonnx_datatype_name("FLOAT<5,10>") == DataType["FLOAT<5,10,15>"]
    # An explicit zero exponent bias is a datatype of its own, not "unset".
    assert resolve_qonnx_datatype_name("FLOAT<5,10,0>").name == "FLOAT<5,10,0>"


# -- immutability ------------------------------------------------------------


def test_freezing_keeps_the_one_immutable_value() -> None:
    """A datatype is a value: one instance per canonical name, frozen once built.

    The snapshot is the caller's instance itself, and no field of it can be
    reassigned, so nothing can rename a datatype underneath a value holding it.
    """

    supplied = DataType["INT8"]
    frozen = cast(QONNXDataType, QONNX_DATATYPE_SEMANTICS.freeze(supplied))
    assert frozen is supplied and supplied is DataType["INT8"]
    with pytest.raises(AttributeError, match="immutable datatype value"):
        setattr(supplied, "_bitwidth", 9)
    assert frozen.name == "INT8"


def test_a_datatype_keeps_its_key_in_a_mapping() -> None:
    """The hazard the old defences guarded (a mutated key orphaned in a dict) is
    gone: the mutation is refused, so the entry stays reachable under its key."""

    held = DataType["INT8"]
    holder = {held: "eight"}
    with pytest.raises(AttributeError, match="immutable datatype value"):
        setattr(held, "_bitwidth", 9)
    assert holder[DataType["INT8"]] == "eight"


# -- one value domain --------------------------------------------------------


def test_every_datatype_field_shares_one_token() -> None:
    """``is_compatible_with`` is token identity, not subtyping.

    A second token -- the ``QONNXDataType`` protocol, say -- would partition the
    domain and make two datatype fields report that they cannot be compared.
    """

    assert QONNX_DATATYPE_SEMANTICS.type_token is BaseDataType
    assert QONNX_DATATYPE_SEMANTICS.is_compatible_with(QONNX_DATATYPE_SEMANTICS)


def test_equality_follows_canonical_identity() -> None:
    left = DataType["FIXED<8,4>"]
    right = DataType["FIXED<8,4>"]
    assert left is right  # one instance per canonical name
    assert QONNX_DATATYPE_SEMANTICS.values_equal(left, right) is True
    assert QONNX_DATATYPE_SEMANTICS.values_equal(left, DataType["FIXED<8,3>"]) is False
