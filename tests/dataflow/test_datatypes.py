# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatype ingestion at the live value boundary, ``ScalarEncoding``.

Datatype Space semantics are covered in tests/kernels.
"""

from typing import Any, cast

import pytest
from qonnx.core.datatype import BaseDataType, DataType

from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    qonnx_datatype_width,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor


class _LyingWidth(BaseDataType):  # type: ignore[misc]
    """Names itself ``INT8`` truthfully and reports its width falsely."""

    def get_canonical_name(self) -> str:
        return "INT8"

    def bitwidth(self) -> int:
        raise RuntimeError("boom")

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
    assert repr(tensor.element) == "ScalarEncoding(dtype=INT8, value_range=(-128, 127))"
    assert tensor.element.dtype.name == "INT8" and tensor.element.bits == 8


def test_an_element_is_a_datatype_and_the_range_of_its_values() -> None:
    plain = ScalarEncoding(DataType["INT8"])
    assert plain.value_range == (-128, 127)
    # Normalized: the datatype's own range stated is the plain element.
    full = ScalarEncoding(DataType["INT8"], (-128, 127))
    assert full == plain and hash(full) == hash(plain)
    narrow = ScalarEncoding(DataType["INT8"], (-127, 127))
    assert narrow != plain
    assert (str(plain), str(narrow)) == ("INT8", "INT8 over [-127, 127]")
    assert ScalarEncoding(DataType["UINT2"]).value_range == (0, 3)
    # A non-integer encoding carries its datatype alone.
    assert ScalarEncoding(DataType["FLOAT32"]).value_range is None
    assert str(ScalarEncoding(DataType["BIPOLAR"])) == "BIPOLAR"


@pytest.mark.parametrize(
    ("name", "bounds"),
    (("INT3", (-5, 3)), ("INT3", (2, 1)), ("UINT4", (0, 16)), ("FLOAT32", (0, 1))),
)
def test_a_range_must_be_one_the_datatype_holds(name: str, bounds: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        ScalarEncoding(DataType[name], bounds)
    refused = ScalarEncoding.admit(DataType[name], bounds)
    assert not isinstance(refused, ScalarEncoding)
    assert {finding.code for finding in refused.findings} == {"dtype-storage"}


def test_an_element_fits_another_with_its_datatype_and_a_range_around_its_own() -> None:
    int4, narrow = ScalarEncoding(DataType["INT4"]), ScalarEncoding(DataType["INT4"], (-7, 7))
    assert narrow.fits(int4) and not int4.fits(narrow)
    assert narrow.fits(narrow) and int4.fits(int4)
    assert not ScalarEncoding(DataType["INT3"]).fits(int4)  # another datatype, whatever its range
    floats = ScalarEncoding(DataType["FLOAT32"])
    assert floats.fits(floats) and not floats.fits(int4)
