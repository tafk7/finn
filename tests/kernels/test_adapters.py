# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Stream adapters as ordinary nodes between two streams of one module.

Each adapter derives its output presentation from its input's, and ``classify``
names the adaptation between the two: a width conversion for ``vpc``, a lane
regroup for ``inner_shuffle``. A composite places the node between a boundary
stream and the stream its consumer reads.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Members, Rejected, Space, design_space, view
from finn.kernels.adapters import TransposeKernel, WidthConverterKernel
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.kernels.datatypes.scalar import ScalarEncoding
from finn.kernels.physical.forms import Adaptation, Traversal, classify, regrouped, vector_major
from finn.kernels.streams import (
    COMPOSED,
    CONNECTION,
    MODULE,
    TIEOFFS,
    Composed,
    Stream,
    StreamSpec,
    netlist,
)

ELEMENT = ScalarEncoding(DataType["INT4"])


def widths(before: int, after: int, shape=(3, 12), presented=None):
    source = vector_major(shape, before)
    presented = regrouped(source, after) if presented is None else presented

    class Widened(Space):
        a = Stream(spec=StreamSpec(ELEMENT, source), port="in0_V")
        b = Stream(spec=StreamSpec(ELEMENT, presented), port="out0_V")
        convert = WidthConverterKernel(lanes=after, input_stream=a, output_stream=b)
        modules = Members(MODULE)
        streams = Members(CONNECTION)
        tieoffs = Members(TIEOFFS)

        @view(semantics=COMPOSED, requires=(modules, streams, tieoffs))
        def structure(self) -> Composed | Rejected:
            return netlist(
                self.modules,
                self.streams,
                self.tieoffs,
                module="finn_widened",
                producer=ProducerIdentity("test.widened", "1"),
            )

    return design_space(Widened())


def transposed(rows: int, cols: int, simd: int, batches: int = 2):
    source = vector_major((batches, rows, cols), simd)

    class Transposed(Space):
        a = Stream(spec=StreamSpec(ELEMENT, source), port="in0_V")
        b = Stream(
            spec=StreamSpec(
                ELEMENT,
                Traversal.over(
                    (batches, rows, cols),
                    ((0, batches, 1), (2, cols, 1), (1, rows // simd, simd)),
                    ((1, simd, 1),),
                ),
            ),
            port="out0_V",
        )
        shuffle = TransposeKernel(input_stream=a, output_stream=b)
        modules = Members(MODULE)
        streams = Members(CONNECTION)
        tieoffs = Members(TIEOFFS)

        @view(semantics=COMPOSED, requires=(modules, streams, tieoffs))
        def structure(self) -> Composed | Rejected:
            return netlist(
                self.modules,
                self.streams,
                self.tieoffs,
                module="finn_transposed",
                producer=ProducerIdentity("test.transposed", "1"),
            )

    return design_space(Transposed()).with_choices({Transposed.shuffle.ram_style: "auto"})


@pytest.mark.parametrize("before,after,vector", [(2, 3, 6), (4, 2, 4), (3, 12, 12)])
def test_a_width_converter_regroups_the_same_sequence(before, after, vector):
    point = widths(before, after)
    assert classify(vector_major((3, 12), before), point.convert.output_form).adaptation is (
        Adaptation.WIDTH_CONVERSION
    )
    vpc = dict(point.convert.build_requirements.parameters)
    assert (vpc["PI"], vpc["PO"], vpc["N"]) == (before, after, vector)
    ports = {port.name for port in point.structure.structure.top_abi.ports}
    assert {"in0_V", "out0_V"} <= ports


def test_a_width_converter_refuses_partial_vectors():
    # Four elements in pairs make no whole six-element vector for three lanes.
    point = widths(2, 3, shape=(1, 4), presented=vector_major((1, 4), 2))
    refused = point.convert.query(WidthConverterKernel.build_requirements)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"vpc-geometry"}


def test_a_transpose_turns_rows_into_columns():
    point = transposed(4, 6, 2)
    source = vector_major((2, 4, 6), 2)
    assert classify(source, point.shuffle.output_form).adaptation is Adaptation.LANE_REGROUP
    shuffle = dict(point.shuffle.build_requirements.parameters)
    assert (shuffle["I"], shuffle["J"], shuffle["SIMD"]) == (4, 6, 2)
    first = next(point.shuffle.output_form.positions())
    assert first == ((0, 0, 0), (0, 1, 0))  # column 0, rows 0 and 1
    _ = point.structure


def test_a_transpose_needs_row_major_rows():
    column_major = Traversal.over((4, 6), ((1, 6, 1), (0, 2, 2)), ((0, 2, 1),))

    class Wrong(Space):
        a = Stream(spec=StreamSpec(ELEMENT, column_major), port="in0_V")
        b = Stream(spec=StreamSpec(ELEMENT, vector_major((4, 6), 2)), port="out0_V")
        shuffle = TransposeKernel(input_stream=a, output_stream=b)

    point = design_space(Wrong()).with_choices({Wrong.shuffle.ram_style: "auto"})
    refused = point.shuffle.query(TransposeKernel.build_requirements)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"transpose-form"}
