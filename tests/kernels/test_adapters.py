# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A stream plans what its ends need and its adapter carries the plan out.

A cyclic producer presents a tensor in any order; thresholding consumes it
PE channels a beat, channels innermost. The stream between them derives its
plan from the two presentations, and exactly one adapter candidate carries it
out: nothing for the same order, a ``vpc`` for other lanes, an ``input_gen``
for another beat order, and chains of both. A stream that admits no adapter
refuses a non-empty plan. FinnLib's ``inner_shuffle`` is not a candidate; a
kernel with children places it explicitly (``TransposeKernel``).
"""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Rejected, design_space, inspection
from finn.dataflow.plan import Step
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import Traversal, vector_major
from finn.kernels.configure import admission, commit
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.streams import Stream
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernels.helpers import Root, labels, with_adapter_memories
from finn.kernels.transpose import TransposeKernel

ELEMENT = ScalarEncoding(DataType["INT4"])
ROWS, CHANNELS = 3, 12
# Fifteen thresholds -7..7: an INT4 value v becomes the level v + 8, so every
# output identifies the element that produced it.
LEVELS = tuple(range(-7, 8))


def values(rows: int = ROWS, channels: int = CHANNELS) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple((5 * row + 3 * channel) % 16 - 8 for channel in range(channels))
        for row in range(rows)
    )


def adapted(source: Traversal, pe: int, *, adaptable: bool = True, commit_all: bool = True):
    """A cyclic producer presenting ``source``, thresholding ``pe`` channels a beat."""
    rows, channels = source.shape

    class Adapted(Root):
        x = Stream(tensor=Tensor(source.shape, ELEMENT), adaptable=adaptable)
        y = Stream(tensor=Tensor(source.shape, ScalarEncoding(DataType["UINT4"])), port="out0_V")
        producer = MemStreamKernel(
            dtype=DataType["INT4"], form=source, contents=values(rows, channels), output_stream=x
        )
        activate = ThresholdingAxiKernel(
            input_dtype=DataType["INT4"],
            threshold_dtype=DataType["INT4"],
            thresholds=(tuple(LEVELS for _ in range(channels)),),
            bias=0,
            pe=pe,
            ram_style="auto",
            ultra_stages=0,
            input_stream=x,
            output_stream=y,
        )

    point = commit(
        design_space(Adapted()),
        {
            "producer.ram_style": "distributed",
            "producer.pumped_memory": False,
            "activate.use_axilite": False,
            "activate.deep_pipeline": False,
        },
    )
    return with_adapter_memories(point) if commit_all else point


def columns_first(rows: int, channels: int, lanes: int) -> Traversal:
    """Channel folds outer, rows inner: another beat order at the same lanes."""
    return Traversal.over(
        (rows, channels), ((1, channels // lanes, lanes), (0, rows, 1)), ((1, lanes, 1),)
    )


def transposed(rows: int, cols: int, simd: int, batches: int = 2):
    """``inner_shuffle`` placed between two boundary streams: rows in, columns out."""
    shape = (batches, rows, cols)

    class Transposed(Root):
        a = Stream(tensor=Tensor(shape, ELEMENT), port="in0_V")
        b = Stream(tensor=Tensor(shape, ELEMENT), port="out0_V")
        shuffle = TransposeKernel(input_stream=a, output_stream=b)

    return commit(design_space(Transposed()), {"shuffle.ram_style": "auto", "shuffle.simd": simd})


def stage_parameters(point):
    """Each stage of ``x``, by its name below its chain, and its module's parameters."""
    return [
        (stage.label.rsplit(".", 1)[-1], dict(stage.module.parameters)) for stage in point.x.stages
    ]


def test_the_same_order_connects_directly():
    point = adapted(vector_major((ROWS, CHANNELS), 4), 4)
    assert not point.x.plan and not point.x.adapting
    assert point.x.stages == ()
    assert labels(point.module) == ["producer", "activate"]


@pytest.mark.parametrize("before,after,vector", [(2, 3, 6), (4, 2, 4), (3, 12, 12), (6, 4, 12)])
def test_other_lanes_are_a_vpc(before, after, vector):
    point = adapted(vector_major((ROWS, CHANNELS), before), after)
    assert point.x.plan.steps == (Step.WIDTH,)
    ((name, vpc),) = stage_parameters(point)
    assert name == "vpc" and (vpc["PI"], vpc["PO"], vpc["N"]) == (before, after, vector)
    # The stage sits below the stream, at its place in the root's netlist.
    assert labels(point.module) == ["x.adapter.vpc.vpc", "producer", "activate"]


def test_another_beat_order_is_an_input_gen():
    point = adapted(columns_first(ROWS, CHANNELS, 4), 4)
    assert point.x.plan.steps == (Step.REORDER,)
    ((name, generator),) = stage_parameters(point)
    # Per frame of every beat, each row's channel folds in order.
    assert name == "input_gen"
    assert (generator["FM_SIZE"], generator["DIMS"], generator["COEFS"]) == (
        9,
        "'{3, 3}",
        "'{1, 3}",
    )


def test_lanes_and_order_together_are_a_chain():
    point = adapted(columns_first(ROWS, CHANNELS, 4), 2)
    assert point.x.plan.steps == (Step.WIDTH, Step.REORDER)
    assert [name for name, _ in stage_parameters(point)] == ["vpc", "input_gen"]
    assert labels(point.module)[:2] == [
        "x.adapter.vpc_input_gen.vpc",
        "x.adapter.vpc_input_gen.input_gen",
    ]


def test_another_lane_axis_regroups_through_the_common_lane_count():
    # Three rows of one channel a beat: the lanes run along the rows.
    rows_as_lanes = Traversal.over((ROWS, CHANNELS), ((1, CHANNELS, 1),), ((0, ROWS, 1),))
    point = adapted(rows_as_lanes, 2)
    assert point.x.plan.steps == (Step.WIDTH, Step.REORDER, Step.WIDTH)
    assert [name for name, _ in stage_parameters(point)] == ["vpc", "input_gen", "vpc_1"]


def test_exactly_one_candidate_carries_out_each_plan():
    for source, pe, case in (
        (vector_major((ROWS, CHANNELS), 4), 2, "vpc"),
        (columns_first(ROWS, CHANNELS, 4), 4, "input_gen"),
        (columns_first(ROWS, CHANNELS, 4), 2, "vpc_input_gen"),
    ):
        point = adapted(source, pe, commit_all=False)
        # The one chain that carries out the plan is forced; every other refuses it.
        forced = {item.key: item for item in inspection.forced(point)}
        assert forced["x.adapter"].value == case
        assert all("adapter-plan" in why for why in forced["x.adapter"].refused.values())
        chosen = commit(point, {"x.adapter": "vpc_input_gen_vpc"})
        refused = admission(chosen.x.adapter)
        assert isinstance(refused, Rejected)
        assert {finding.code for finding in refused.findings} == {"adapter-plan"}


def test_a_stream_admitting_no_adapter_refuses_its_plan():
    point = adapted(vector_major((ROWS, CHANNELS), 4), 2, adaptable=False)
    refused = point.x.query(Stream.netlist)
    assert isinstance(refused, Rejected)
    plan = [finding for finding in refused.findings if finding.code == "stream-plan"]
    assert plan and "width_conversion" in plan[0].message
    # Its adapter Decision does not apply.
    assert not point.x.adapting


def test_a_transpose_turns_rows_into_columns():
    point = transposed(4, 6, 2)
    shuffle = dict(point.shuffle.module.parameters)
    assert (shuffle["I"], shuffle["J"], shuffle["SIMD"]) == (4, 6, 2)
    assert point.shuffle.input.presented.form == vector_major((2, 4, 6), 2)
    first = next(point.shuffle.output.presented.form.positions())
    assert first == ((0, 0, 0), (0, 1, 0))  # column 0, rows 0 and 1
    _ = point.module


def test_a_transposes_simd_divides_both_sides():
    """SIMD is a Decision over the common divisors of I and J (both bound from the ports)."""
    assert transposed(4, 6, 1).shuffle.field(TransposeKernel.simd).candidates().value == (1, 2)
    with pytest.raises(ValueError, match="domain-membership"):
        transposed(4, 6, 3)  # divides J, not I
