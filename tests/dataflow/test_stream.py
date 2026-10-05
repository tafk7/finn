# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The logical stream: its ends traverse its tensor, and it plans what joins them.

``Stream`` leaves ``ends`` required: how the ends are found is physical. A
test stream supplies them as inputs.
"""

from __future__ import annotations

import pytest

from finn.core.space import (
    Available,
    DefinitionError,
    Param,
    Rejected,
    default_semantics,
    derived,
    design_space,
)
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.plan import Step
from finn.dataflow.stream import End, Ends, Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd, vector_major

INT4 = ScalarEncoding(resolve_qonnx_datatype_name("INT4"))
ROWS = vector_major((3, 8), 2)
SEQUENCE = default_semantics(BeatSequence)


class Given(Stream):
    """A stream whose ends are given."""

    source: BeatSequence = Param(semantics=SEQUENCE)
    sink: BeatSequence = Param(semantics=SEQUENCE)

    @derived
    def ends(self) -> Ends:
        return Ends(End("producer", INT4, self.source), End("consumer", INT4, self.sink))


def given(source: BeatSequence, sink: BeatSequence, *, adaptable: bool = True) -> Given:
    return design_space(
        Given(tensor=Tensor((3, 8), INT4), source=source, sink=sink, adaptable=adaptable)
    )


def codes(result: object) -> set[str]:
    assert isinstance(result, Rejected)
    return {finding.code for finding in result.findings}


def test_the_logical_stream_cannot_be_placed_without_its_ends() -> None:
    with pytest.raises(DefinitionError, match=r"leaves required members unmet: Stream\.ends"):
        Stream(tensor=Tensor((3, 8), INT4))


def test_the_plan_joins_the_ends() -> None:
    direct = given(BeatSequence(ROWS), BeatSequence(ROWS))
    assert not direct.plan and not direct.adapting
    framed = BeatSequence(ROWS.replayed(2, inner_beats=4), markers=(LevelEnd(4),))
    replayed = given(BeatSequence(ROWS), framed)
    assert replayed.plan.steps == (Step.REORDER, Step.MARKERS) and replayed.adapting
    assert isinstance(replayed.inspect(Given.realizable).result, Available)


def test_a_stream_admitting_no_adapter_refuses_a_plan() -> None:
    framed = BeatSequence(ROWS, markers=(LevelEnd(4),))
    fixed = given(BeatSequence(ROWS), framed, adaptable=False)
    assert not fixed.adapting
    refused = fixed.inspect(Given.realizable).result
    assert codes(refused) == {"stream-plan"} and "markers" in str(refused)


def test_every_end_traverses_the_stream_s_tensor() -> None:
    other = BeatSequence(vector_major((4, 6), 2))
    refused = given(BeatSequence(ROWS), other).inspect(Given.well_formed).result
    assert codes(refused) == {"stream-tensor"} and "consumer traverses" in str(refused)


class Valued(Stream):
    """A stream whose ends carry the given elements."""

    produced: ScalarEncoding = Param()
    consumed: ScalarEncoding = Param()

    @derived
    def ends(self) -> Ends:
        rows = BeatSequence(ROWS)
        return Ends(End("producer", self.produced, rows), End("consumer", self.consumed, rows))


def test_a_producer_s_values_fit_the_tensor_and_the_tensor_s_the_consumer() -> None:
    narrow = ScalarEncoding(resolve_qonnx_datatype_name("INT4"), (-7, 7))

    def checked(produced: ScalarEncoding, carried: ScalarEncoding) -> object:
        stream = design_space(
            Valued(tensor=Tensor((3, 8), carried), produced=produced, consumed=INT4)
        )
        return stream.inspect(Valued.well_formed).result

    # A producer tighter than a plainly stated tensor is accepted.
    assert checked(narrow, INT4) == Available(True)
    # A tensor tighter than its producer is refused, the range printed only when tightened.
    refused = checked(INT4, narrow)
    assert codes(refused) == {"stream-tensor"}
    assert "producer carries INT4; the stream carries INT4 over [-7, 7]" in str(refused)
    # The consumer's side is the same check: a tighter tensor fits a plain consumer.
    plain = design_space(Valued(tensor=Tensor((3, 8), narrow), produced=narrow, consumed=INT4))
    assert plain.inspect(Valued.well_formed).result == Available(True)
