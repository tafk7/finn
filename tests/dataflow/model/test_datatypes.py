# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Region datatype ingestion; scalar datatype coverage lives in tests/kernels."""

from typing import cast
import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from finn.kernels.datatypes.values import (
    DatatypeError,
    canonical_qonnx_datatype,
    QONNXDataType,
    is_qonnx_datatype,
    qonnx_datatype_width,
)
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_SEMANTICS
from finn.dataflow.model.logical.region import Operand, is_element_type
from kernels.test_datatypes import _LyingWidth, _ZeroWidth


def test_element_width_is_read_from_the_canonical_value_not_the_caller_s() -> None:
    """``is_element_type`` measures what a Region would hold.

    Both subclasses name themselves ``INT8``, so under QONNX's identity rule
    they *are* ``INT8``, and a Region constructed from either holds the
    registered eight-bit value -- not the instance handed in.  Asking the
    caller's instance its width therefore answered a question about an object
    already scheduled for discard: one of these raised out of the predicate
    entirely, and the other made a genuine ``INT8`` look degenerate.

    Reading the width from the canonicalized value settles both the same way,
    and the way the Region will actually behave.
    """

    for rogue in (_LyingWidth(), _ZeroWidth()):
        assert is_element_type(rogue) is True
        assert qonnx_datatype_width(rogue) == 8
        held = Operand("x", cast(QONNXDataType, rogue), (4,)).element_type
        assert held == DataType["INT8"]
        assert held is not rogue


def test_a_degenerate_width_is_still_refused() -> None:
    """The canonical read is not a way of ignoring the width test.

    ``INT0`` resolves, and its width really is zero, so it cannot describe a
    beat.  Measuring the canonical value keeps that refusal intact -- the change
    above is about *which* object is measured, not about relaxing the condition.
    """

    assert is_element_type(DataType["INT0"]) is False
    assert qonnx_datatype_width(DataType["INT0"]) == 0


def test_a_width_cannot_be_asked_of_a_non_datatype() -> None:
    """``qonnx_datatype_width`` canonicalizes first, so it refuses like the rest."""

    with pytest.raises(DatatypeError, match="not a QONNX datatype"):
        qonnx_datatype_width(8)
    with pytest.raises(DatatypeError, match="is not a datatype value"):
        qonnx_datatype_width("INT8")
    assert is_element_type("INT8") is False
    assert is_element_type(8) is False


def test_an_instance_mutated_into_an_invalid_state_is_refused_not_raised() -> None:
    """The same failure reached through a live object rather than a name.

    Mutating ``_intwidth`` past the total width renames the datatype to
    ``FIXED<8,9>``, which QONNX then refuses to reconstruct.  Recognition must
    stay total across that: ``accepts`` returns ``False``, and the Region
    constructor refuses at its own documented boundary rather than propagating
    an ``AssertionError`` from three layers down.
    """

    mutated = DataType["FIXED<8,4>"]
    mutated._intwidth = 9
    assert mutated.name == "FIXED<8,9>"

    assert is_qonnx_datatype(mutated) is False
    assert QONNX_DATATYPE_SEMANTICS.accepts(mutated) is False
    with pytest.raises(DatatypeError):
        canonical_qonnx_datatype(mutated)

    with pytest.raises(TypeError, match="QONNX datatype"):
        Operand("x", mutated, (4,))


def test_an_operand_does_not_retain_the_caller_s_datatype_instance() -> None:
    """The Region half of the ingestion discipline, tested where it matters.

    The engine's snapshot protects the engine.  It does nothing for a Region
    holding a caller's object, which is reachable as
    ``region.inputs[0].port.operand.element_type`` and mutable in place.  So
    ``Operand`` re-resolves rather than merely validating, and this is the test
    that tells the two apart: validation alone would leave the operand holding
    ``supplied`` and this would fail on the last line.
    """

    supplied = DataType["INT8"]
    operand = Operand("x", supplied, (4,))
    assert operand.element_type is not supplied

    supplied._bitwidth = 9
    assert supplied.name == "INT9"
    assert operand.element_type.name == "INT8"
