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
    canonical_qonnx_datatype,
    QONNXDataType,
    is_qonnx_datatype,
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


def test_element_width_is_read_from_the_canonical_value_not_the_caller_s() -> None:
    """A ``ScalarEncoding`` measures what a tensor will hold.

    Both subclasses name themselves ``INT8``, so under QONNX's identity rule
    they *are* ``INT8``, and an encoding built from either holds the
    registered eight-bit value -- not the instance handed in.  Asking the
    caller's instance its width would answer a question about an object
    already discarded: one of these raises, and the other makes a genuine
    ``INT8`` look degenerate.

    Reading the width from the canonicalized value settles both the same way,
    and the way the tensor will actually behave.
    """

    for rogue in (_LyingWidth(), _ZeroWidth()):
        assert qonnx_datatype_width(rogue) == 8
        held = ScalarEncoding(cast(QONNXDataType, rogue))
        assert held.bits == 8 and held.dtype == DataType["INT8"]
        assert held.dtype is not rogue


def test_a_degenerate_width_is_still_refused() -> None:
    """The canonical read is not a way of ignoring the width test.

    ``INT0`` resolves, and its width really is zero, so it cannot describe a
    beat.  Measuring the canonical value keeps that refusal intact -- the change
    above is about *which* object is measured, not about relaxing the condition.
    """

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


def test_an_instance_mutated_into_an_invalid_state_is_refused_not_raised() -> None:
    """The same failure reached through a live object rather than a name.

    Mutating ``_intwidth`` past the total width renames the datatype to
    ``FIXED<8,9>``, which QONNX then refuses to reconstruct.  Recognition must
    stay total across that: ``is_qonnx_datatype`` returns ``False``, and the
    encoding refuses at its own documented boundary rather than propagating an
    ``AssertionError`` from three layers down.
    """

    mutated = DataType["FIXED<8,4>"]
    mutated._intwidth = 9
    assert mutated.name == "FIXED<8,9>"

    assert is_qonnx_datatype(mutated) is False
    with pytest.raises(DatatypeError):
        canonical_qonnx_datatype(mutated)

    with pytest.raises(DatatypeError):
        ScalarEncoding(mutated)


def test_a_tensor_does_not_retain_the_caller_s_datatype_instance() -> None:
    """The value half of the ingestion discipline, tested where it matters.

    The engine's snapshot protects the engine.  It does nothing for a tensor
    holding a caller's object, which would be mutable in place.  So an
    encoding keeps the canonical name only, and every read re-resolves it.
    """

    supplied = DataType["INT8"]
    tensor = Tensor((4,), ScalarEncoding(supplied))
    assert tensor.element.dtype is not supplied

    supplied._bitwidth = 9
    assert supplied.name == "INT9"
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
