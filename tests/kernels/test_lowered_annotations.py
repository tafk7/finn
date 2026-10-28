# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Actual child annotations and clean removal of generated flat fields.

Original unsupported lowered-ClassVar annotation xfails remain at their immutable
historical source; current authoring uses typed real child declarations instead.
"""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Available, DefinitionError, Param, Space, design_space
from finn.core.space.compiler import compile_model
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import NATIVE_CLOCKING
from finn.kernels.port_model import SequencePort, native_port

TENSOR = Tensor((4,), ScalarEncoding(DataType["INT4"]))
SEQUENCE = BeatSequence(vector_major((4,), 1))


class Annotated(Space):
    input_tensor: Tensor = Param(default=TENSOR)
    input: SequencePort = native_port(
        "input", Endpoint.TARGET, input_tensor, SEQUENCE, clocking=NATIVE_CLOCKING, label="input"
    )


def test_an_annotation_matching_the_actual_child_is_accepted() -> None:
    point = design_space(Annotated())
    assert "input.width" in compile_model(Annotated).scopes.keys
    assert point.query(Annotated.input.width) == Available(4)
    assert not hasattr(Annotated, "input_data_width")


def test_a_wrong_typed_child_model_binding_is_refused() -> None:
    with pytest.raises(DefinitionError):

        class Mistyped(Space):
            input = SequencePort(name=7, label="input", tensor=TENSOR, sequence=SEQUENCE)

        design_space(Mistyped())


def test_no_flat_field_is_manufactured_from_an_annotation() -> None:
    assert not hasattr(Annotated, "input_width")
    assert "input_width" not in compile_model(Annotated).scopes.keys
