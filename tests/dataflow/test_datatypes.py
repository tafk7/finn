# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatype ingestion at the live value boundary, ``ScalarEncoding``.

Datatype Space semantics are covered in tests/kernels.
"""

import math
from fractions import Fraction
from typing import Any, cast

import pytest
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.ranges import TensorRange

from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    canonical_qonnx_datatype,
    is_ordinary_integer,
    is_qonnx_datatype,
    ordinary_integer_bounds,
    qonnx_datatype_width,
    resolve_qonnx_datatype_name,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor


class _LyingWidth(BaseDataType):
    """Names itself ``INT8`` truthfully and reports its width falsely."""

    def get_canonical_name(self) -> str:
        return "INT8"

    def bitwidth(self) -> int:
        raise RuntimeError("boom")

    def min(self) -> int:
        return 0

    def max(self) -> int:
        return 255

    def allowed(self, value: Any) -> bool:
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


class _ZeroWidth(_LyingWidth):
    def bitwidth(self) -> int:
        return 0


def test_a_subclass_naming_itself_int8_is_not_int8() -> None:
    """A ``BaseDataType`` subclass defined outside qonnx is not one of its datatype
    values, whatever it names itself: refused, and none of its methods run (one
    of these raises from ``bitwidth``, the other reports zero)."""

    for rogue in (_LyingWidth(), _ZeroWidth()):
        with pytest.raises(DatatypeError, match="not a QONNX datatype value"):
            qonnx_datatype_width(rogue)
        with pytest.raises(DatatypeError):
            ScalarEncoding(cast(QONNXDataType, rogue))


@pytest.mark.parametrize(
    ("name", "ordinary"),
    [
        ("INT1", True),
        ("INT33", True),
        ("UINT4", True),
        ("UINT1", True),  # resolves to BINARY
        ("BIPOLAR", False),
        ("TERNARY", False),
        ("SCALEDINT<8>", False),
        ("FIXED<8,4>", False),
        ("FLOAT32", False),
    ],
)
def test_an_ordinary_integer_is_an_int_or_uint_encoding(name: str, ordinary: bool) -> None:
    """``INT<n>``, ``UINT<n>`` and ``BINARY``; the other integer-valued encodings are
    distinct, and have no ordinary bounds."""

    dtype = DataType[name]
    assert is_ordinary_integer(dtype) is ordinary
    if not ordinary:
        with pytest.raises(DatatypeError, match="not an ordinary INT/UINT encoding"):
            ordinary_integer_bounds(dtype)


def test_a_degenerate_width_is_still_refused() -> None:
    """``INT0`` still resolves (qonnx warns until a later release refuses it), and
    its width really is zero, so it cannot describe a beat: FINN refuses it."""

    assert qonnx_datatype_width(DataType["INT0"]) == 0
    with pytest.raises(ValueError, match="positive width"):
        ScalarEncoding(DataType["INT0"])


def test_a_width_cannot_be_asked_of_a_non_datatype() -> None:
    """``qonnx_datatype_width`` canonicalizes first, so it refuses like the rest."""

    with pytest.raises(DatatypeError, match="not a QONNX datatype"):
        qonnx_datatype_width(8)
    with pytest.raises(DatatypeError, match="is not a datatype value"):
        qonnx_datatype_width("INT8")
    with pytest.raises(DatatypeError):
        ScalarEncoding(cast(QONNXDataType, "INT8"))


@pytest.mark.parametrize(
    ("refused", "code", "owner"),
    [
        (
            lambda: ordinary_integer_bounds(DataType["BIPOLAR"]),
            "integer-not-ordinary",
            "ordinary_integer_bounds",
        ),
        (
            lambda: resolve_qonnx_datatype_name(cast(str, 8)),
            "datatype-name-not-string",
            "resolve_qonnx_datatype_name",
        ),
        (
            lambda: resolve_qonnx_datatype_name("NOSUCHTYPE"),
            "datatype-name-unknown",
            "resolve_qonnx_datatype_name",
        ),
        (
            lambda: canonical_qonnx_datatype("INT8"),
            "datatype-name-not-value",
            "canonical_qonnx_datatype",
        ),
        (lambda: canonical_qonnx_datatype(8), "datatype-not-qonnx", "canonical_qonnx_datatype"),
    ],
)
def test_each_datatype_refusal_names_its_rule_and_its_function(
    refused: Any, code: str, owner: str
) -> None:
    with pytest.raises(DatatypeError) as error:
        refused()
    assert [(each.code, each.owner) for each in error.value.findings] == [(code, owner)]


def test_a_tensor_holds_the_one_immutable_datatype_value() -> None:
    """A datatype is a value: one instance per canonical name, frozen, so a tensor
    holding the caller's instance holds the datatype itself, and nothing can
    rename it underneath the tensor."""

    supplied = DataType["INT8"]
    tensor = Tensor((4,), ScalarEncoding(supplied))
    assert tensor.element.dtype is supplied
    with pytest.raises(AttributeError, match="immutable datatype value"):
        setattr(supplied, "name", "INT9")  # the frozen value refuses any write
    # The element holds the value itself, so its equality, hash and repr are the value's.
    assert tensor.element == ScalarEncoding(DataType["INT8"])
    assert repr(tensor.element) == (
        "ScalarEncoding(dtype=INT8, value_range=TensorRange(least=-128, greatest=127, nan=False))"
    )
    assert tensor.element.dtype.name == "INT8" and tensor.element.bits == 8


def test_an_element_is_a_datatype_and_the_range_of_its_values() -> None:
    plain = ScalarEncoding(DataType["INT8"])
    assert plain.value_range == TensorRange(-128, 127) and not plain.stated
    # Normalized: the datatype's own range stated is the plain element.
    full = ScalarEncoding(DataType["INT8"], TensorRange(-128, 127))
    assert full == plain and hash(full) == hash(plain)
    narrow = ScalarEncoding(DataType["INT8"], TensorRange(-127, 127))
    assert narrow != plain and narrow.stated
    assert (str(plain), str(narrow)) == ("INT8", "INT8 over [-127, 127]")
    assert ScalarEncoding(DataType["UINT2"]).value_range == TensorRange(0, 3)
    assert str(ScalarEncoding(DataType["BIPOLAR"])) == "BIPOLAR"


def test_any_numeric_element_carries_a_range() -> None:
    """A float's own range is unbounded, with NaN (``qonnx.core.ranges.full``); a range
    stated of it may be narrower, with dyadic ends, and without NaN."""
    floats = ScalarEncoding(DataType["FLOAT32"])
    assert floats.value_range == TensorRange(-math.inf, math.inf, nan=True)
    assert str(floats) == "FLOAT32"
    clipped = ScalarEncoding(DataType["FLOAT32"], TensorRange(0, Fraction(3, 4)))
    assert str(clipped) == "FLOAT32 over [0, 3*2^-2]"
    assert clipped.fits(floats) and not floats.fits(clipped)
    fixed = ScalarEncoding(DataType["FIXED<8,4>"], TensorRange(Fraction(-1, 2), 1))
    assert fixed.stated and fixed.fits(ScalarEncoding(DataType["FIXED<8,4>"]))


@pytest.mark.parametrize(
    ("name", "stated"),
    (
        ("INT3", TensorRange(-5, 3)),
        ("UINT4", TensorRange(0, 16)),
        ("INT8", TensorRange(0, Fraction(1, 2))),  # not integers
        ("INT8", TensorRange(0, 1, nan=True)),
        ("FIXED<8,4>", TensorRange(0, math.inf)),
    ),
)
def test_a_range_must_be_one_the_datatype_holds(name: str, stated: TensorRange) -> None:
    with pytest.raises(ValueError):
        ScalarEncoding(DataType[name], stated)


def test_a_range_is_a_tensor_range() -> None:
    with pytest.raises(TypeError):
        ScalarEncoding(DataType["INT3"], cast(TensorRange, (-1, 1)))


def test_an_element_fits_another_with_its_datatype_and_a_range_around_its_own() -> None:
    int4 = ScalarEncoding(DataType["INT4"])
    narrow = ScalarEncoding(DataType["INT4"], TensorRange(-7, 7))
    assert narrow.fits(int4) and not int4.fits(narrow)
    assert narrow.fits(narrow) and int4.fits(int4)
    assert not ScalarEncoding(DataType["INT3"]).fits(int4)  # another datatype, whatever its range
    floats = ScalarEncoding(DataType["FLOAT32"])
    assert floats.fits(floats) and not floats.fits(int4)


# -- names and identity ------------------------------------------------------

#: Pairs of one width and different meanings, which a (type, bit width) pair
#: would merge and must not. ``TERNARY``/``INT2`` is the one that matters most:
#: merged, a ternary MatMul would be admitted and its artifact declare ``INT2``.
FORMERLY_COLLIDING = (
    ("TERNARY", "INT2"),
    ("FLOAT16", "FLOAT<5,10,7>"),
    ("FLOAT16", "FLOAT<5,10,15>"),
    ("FIXED<8,4>", "FIXED<8,3>"),
    ("BINARY", "BIPOLAR"),
    ("BIPOLAR", "INT1"),
)


@pytest.mark.parametrize(("left", "right"), FORMERLY_COLLIDING)
def test_equal_width_types_with_different_meanings_stay_different(left: str, right: str) -> None:
    assert DataType[left] != DataType[right]


def test_uint1_and_binary_are_deliberately_the_same_value() -> None:
    """The one merge QONNX's canonicalization makes, pinned as accepted.

    QONNX canonicalizes ``UINT1`` to ``BINARY``, so they are one value, not two
    that happen to compare equal.  FINN takes QONNX's canonicalization as it is;
    a FINN-local exception here would be a second authority for a datatype's
    identity.  Pinned so the merge reads as a decision, not a bug.
    """

    assert DataType["UINT1"] == DataType["BINARY"]
    assert resolve_qonnx_datatype_name("UINT1").name == "BINARY"


def test_a_canonical_name_is_not_a_datatype_value() -> None:
    assert is_qonnx_datatype("INT8") is False
    with pytest.raises(DatatypeError, match="resolve"):
        canonical_qonnx_datatype("INT8")


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


def test_a_datatype_keeps_its_key_in_a_mapping() -> None:
    """A key mutated in place would be orphaned in a dict; the mutation is refused,
    so the entry stays reachable under its key."""

    held = DataType["INT8"]
    holder = {held: "eight"}
    with pytest.raises(AttributeError, match="immutable datatype value"):
        setattr(held, "_bitwidth", 9)
    assert holder[DataType["INT8"]] == "eight"


# -- encodings ---------------------------------------------------------------


@pytest.mark.parametrize("name", ("INT0", "UINT0"))
def test_zero_width_encodings_are_no_tensors_element(name: str) -> None:
    # A kernel's input encodings are its tensors': a zero-width one is refused before
    # any kernel reads it.
    with pytest.raises(ValueError, match="positive width"):
        ScalarEncoding(resolve_qonnx_datatype_name(name))
