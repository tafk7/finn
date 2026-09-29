# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The logical stream: its ends traverse its tensor, and it plans what joins them.

``Stream`` leaves ``ends`` required: how the ends are found is physical. A
test stream supplies them as inputs.
"""

from __future__ import annotations

import pytest
from finn.dataflow.datatypes import resolve_qonnx_datatype_name

from finn.core.space import (
    Available,
    DefinitionError,
    Param,
    Rejected,
    default_semantics,
    derived,
    design_space,
)
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
