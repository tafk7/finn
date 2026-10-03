# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The datatype boundary: what may enter the stack, and as what.

Adopting QONNX datatypes removes a lossy conversion, and these pin the three
things that removal depends on -- that identity is by canonical name, that
recognition never promises more than the snapshot can deliver, and that a
datatype and its own name never get confused for one another.

The last is not a style point.  ``DataType["INT8"] == "INT8"`` is true and their
hashes agree, so if a string were admitted to this value domain the persisted
form and the live value would be interchangeable, hydration could silently do
nothing, and the two would collide as mapping keys without raising.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from qonnx.core.datatype import BaseDataType, DataType

from finn.kernels.datatypes.semantics import QONNX_DATATYPE_SEMANTICS
from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    canonical_qonnx_datatype,
    resolve_qonnx_datatype_name,
    decode_datatype,
    encode_datatype,
    is_qonnx_datatype,
)

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
    assert encode_datatype(DataType[left]) != encode_datatype(DataType[right])


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


@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_a_datatype_round_trips_through_its_canonical_name(name: str) -> None:
    datatype = DataType[name]
    assert decode_datatype(encode_datatype(datatype)) == datatype


@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_the_encoded_form_is_the_canonical_name_and_nothing_else(name: str) -> None:
    """Not a ``repr``, not a class name, not a family and a width."""

    assert encode_datatype(DataType[name]) == {"qonnx_datatype": DataType[name].name}


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
    """Not later, inside the snapshot.

    ``isinstance`` alone would admit this and then raise ``KeyError`` while
    freezing it -- from a path whose contract says the value was already
    recognized.  Refusing here is what makes ``accepts`` implies ``snapshot``
    succeeds true rather than aspirational.
    """

    rogue = _Unregistered()
    assert isinstance(rogue, BaseDataType)
    assert is_qonnx_datatype(rogue) is False
    assert QONNX_DATATYPE_SEMANTICS.accepts(rogue) is False
    with pytest.raises(DatatypeError, match="cannot resolve"):
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
    """One failure type out, whatever QONNX raises going in.

    ``resolve_datatype`` dispatches on a prefix and then parses, so the
    exception depends on how far a bad name gets: ``NOTATYPE`` is a
    ``KeyError`` while ``INT8 but wrong`` reaches ``int()`` and is a
    ``ValueError``.  A boundary that caught only the documented one would let
    the others escape as unrelated exception types.
    """

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

    An earlier version caught an enumerated tuple of exception types, grown one
    at a time as each new QONNX failure mode was found -- ``KeyError``, then
    ``ValueError``, then ``AssertionError``. That is a fix for the case rather
    than for the class of case: a ``BaseDataType`` subclass can raise anything
    at all, and here it raises ``RuntimeError``, which no such tuple would have
    contained.

    Totality is the contract, so the boundary catches ``Exception``.
    """

    rogue = _ExplodingName()
    assert isinstance(rogue, BaseDataType)

    assert is_qonnx_datatype(rogue) is False
    assert QONNX_DATATYPE_SEMANTICS.accepts(rogue) is False
    with pytest.raises(DatatypeError, match="cannot name itself"):
        canonical_qonnx_datatype(rogue)


def test_an_interrupt_is_not_swallowed_as_a_refusal() -> None:
    """``BaseException`` still propagates.

    The counterweight to catching ``Exception``: an interrupt is not the
    datatype declining to be a datatype, and a boundary that swallowed one
    would make the process unkillable at exactly the wrong moment.
    """

    class _Interrupted(_ExplodingName):
        def get_canonical_name(self) -> str:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        is_qonnx_datatype(_Interrupted())


class _ExhaustedName(_ExplodingName):
    """Names itself once, then raises.

    The first ``get_canonical_name()`` is not the only one.  The boundary asks
    the value its name, resolves that name, and then compares the resolution
    back against the value -- and QONNX's own ``__eq__`` implements that
    comparison by calling ``other.get_canonical_name()``, which is this object's
    method again.  So a value can pass the guarded read and still run
    caller-controlled code afterwards.
    """

    def __init__(self) -> None:
        self.calls = 0

    def get_canonical_name(self) -> str:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("boom")
        return "INT8"


def test_recognition_survives_a_value_that_raises_on_the_second_naming() -> None:
    """The comparison inside recognition is guarded, not just the read.

    ``resolved != original`` looked like a comparison between two registered
    QONNX values.  It is not: ``original`` is the caller's object, and
    ``BaseDataType.__eq__`` reaches straight into its ``get_canonical_name()``.
    Guarding only the first read left the second one bare, so a value that
    names itself successfully and then fails escaped ``is_qonnx_datatype`` as a
    ``RuntimeError`` -- from a predicate that promises to answer rather than to
    raise.
    """

    rogue = _ExhaustedName()
    assert rogue.name == "INT8"  # the first call, spent deliberately
    assert rogue.calls == 1

    assert is_qonnx_datatype(_ExhaustedName()) is False
    assert QONNX_DATATYPE_SEMANTICS.accepts(_ExhaustedName()) is False
    with pytest.raises(DatatypeError, match="could not be compared"):
        canonical_qonnx_datatype(_ExhaustedName())


def test_a_jointly_invalid_fixed_point_name_is_refused_not_raised() -> None:
    """``FIXED<8,9>``: parts individually well formed, jointly invalid.

    QONNX validates fixed-point construction with ``assert``, so this raises
    ``AssertionError`` rather than a domain error.  That is not a name QONNX
    can resolve, so the boundary has to turn it into a refusal like any other
    -- otherwise a predicate whose whole contract is to *answer* the question
    raises it instead.
    """

    with pytest.raises(DatatypeError, match="no QONNX datatype is named"):
        resolve_qonnx_datatype_name("FIXED<8,9>")


@pytest.mark.parametrize(
    ("stored", "resolves_to"),
    [
        ("FLOAT<5,10>", "FLOAT<5,10,15>"),
        ("UINT1", "BINARY"),
    ],
)
def test_decoding_refuses_a_noncanonical_persisted_name(stored: str, resolves_to: str) -> None:
    """Persistence is strict: canonical spelling or nothing.

    ``FLOAT<5,10>`` omits its exponent bias, which QONNX fills with the default;
    a persisted name states it (``FLOAT<5,10,15>``), so the short spelling did
    not come from here.

    ``UINT1`` is benign in itself: it and ``BINARY`` are one value. It is
    refused anyway, because the rule is "canonical or nothing", and carving an
    exception for the alias that happens to be harmless is how the general case
    gets let back in.

    The justification for strictness here is that a persisted name is not a
    human's spelling: ``encode_datatype`` only ever writes canonical names, so a
    payload that resolves to a different name did not come from this stack.
    """

    assert resolve_qonnx_datatype_name(stored).name == resolves_to
    with pytest.raises(DatatypeError, match="not canonical"):
        decode_datatype({"qonnx_datatype": stored})


def test_source_facing_names_stay_permissive() -> None:
    """The other half of the split, so strictness does not leak into projection.

    A legacy node attribute like ``accDataType`` is whatever a human or an older
    FINN wrote. Adopting QONNX means adopting its reading of those spellings, so
    the resolver used there accepts what the decoder refuses.
    """

    assert resolve_qonnx_datatype_name("UINT1") == DataType["BINARY"]
    assert resolve_qonnx_datatype_name("FLOAT<5,10>") == DataType["FLOAT<5,10,15>"]
    # An explicit zero exponent bias is a datatype of its own, not "unset".
    assert resolve_qonnx_datatype_name("FLOAT<5,10,0>").name == "FLOAT<5,10,0>"


def test_everything_encode_writes_decodes_back(  # noqa: D401 - reads as a statement
) -> None:
    """Strictness must not refuse the stack's own output.

    The risk in tightening a decoder is refusing something the encoder emits.
    Asserted over the full representative list rather than a sample.
    """

    for name in REPRESENTATIVE:
        datatype = DataType[name]
        assert decode_datatype(encode_datatype(datatype)) == datatype


def test_the_previous_encoding_is_refused_rather_than_reinterpreted() -> None:
    """A stored ``numeric_element_type`` must not be read as a datatype."""

    with pytest.raises(DatatypeError, match="not an encoded datatype"):
        decode_datatype({"numeric_element_type": ["int", 8]})


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
