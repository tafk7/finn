# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Repetition and inner replay may contain equal loops with different multiplicities.

The reference enumerates tensor coordinates in Python, then looks up source beat IDs.
It uses neither the traversal offset helpers nor the reorder buffer helpers.
"""

import itertools
from math import prod

import pytest

from finn.dataflow.traversal import Adaptation, Traversal, classify


def positions(shape: tuple[int, ...], axes: tuple[int, ...]) -> list[int]:
    strides = [prod(shape[axis + 1 :]) for axis in range(len(shape))]
    return [
        sum(index * strides[axis] for axis, index in zip(axes, indices))
        for indices in itertools.product(*(range(shape[axis]) for axis in axes))
    ]


@pytest.mark.parametrize("shape", [(2, 2, 2), (2, 3, 2), (3, 2, 3)])
def test_reorder_coefficients_read_each_repeated_pass_in_its_own_frame(
    shape: tuple[int, ...],
) -> None:
    """288 pairs per shape: both axis orders, 2/3 inner runs, and 1/2/3/4 passes.

    Each source pass visits every position once, so the coordinate lookup gives the
    exact source beat ID, including the pass, independently of loop matching.
    """
    axes = list(itertools.permutations(range(len(shape))))
    for source_axes, sink_axes, runs, passes in itertools.product(axes, axes, (2, 3), (1, 2, 3, 4)):
        source_once = Traversal.over(shape, [(i, shape[i], 1) for i in source_axes], ())
        sink_once = Traversal.over(shape, [(i, shape[i], 1) for i in sink_axes], ())
        source = source_once.repeated(passes)
        sink = sink_once.replayed(runs, inner_beats=shape[sink_axes[-1]]).repeated(passes)
        found = classify(source, sink)
        assert found.adaptation is Adaptation.REORDER
        reorder = found.reorder
        assert reorder is not None
        local = [
            sum(digit * coef for digit, coef in zip(digits, reorder.coefs))
            for digits in itertools.product(*(range(dim) for dim in reorder.dims))
        ]
        assert source.beats % reorder.frame_beats == 0
        actual = [
            frame + beat for frame in range(0, source.beats, reorder.frame_beats) for beat in local
        ]
        produced = positions(shape, source_axes)
        wanted = positions(shape, sink_axes)
        inner = shape[sink_axes[-1]]
        replayed = [
            position
            for start in range(0, len(wanted), inner)
            for _ in range(runs)
            for position in wanted[start : start + inner]
        ]
        expected = [
            iteration * len(produced) + produced.index(position)
            for iteration in range(passes)
            for position in replayed
        ]
        assert actual == expected, (shape, source_axes, sink_axes, runs, passes, reorder)
        assert [(produced * passes)[beat] for beat in actual] == replayed * passes
