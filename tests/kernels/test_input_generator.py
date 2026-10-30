# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The input generator: its loop geometry, derived from the sequences it is given, its
replay and its end-of-loop markers."""

from __future__ import annotations

import pytest
from support.helpers import FULL_DSP48E2, generator, nested, stage_tensor

from finn.core.space import Rejected, design_space
from finn.dataflow.steps import Unrealizable
from finn.dataflow.traversal import BeatSequence, LevelEnd, vector_major
from finn.kernels.artifacts.abi import Signal
from finn.kernels.input_generator import Generate, InputGeneratorKernel, generated
from finn.kernels.library import library_module


def test_generator_states_zero_stride_replay_and_multibit_markers() -> None:
    point = generator()
    requirements = library_module(point)
    assert requirements.parameters == (
        ("COEFS", "'{0, 1}"),
        ("D", 2),
        ("DATA_WIDTH", 13),
        ("DIMS", "'{3, 6}"),
        ("FM_SIZE", 6),
        ("RAM_STYLE", '"auto"'),
    )
    widths = {port.name: port.width for port in requirements.abi.pins if isinstance(port, Signal)}
    assert widths["idat"] == widths["odat"] == 13
    assert widths["olst"] == 2
    assert all(isinstance(port, Signal) for port in requirements.abi.pins)
    ranked = generator(frame=56, dims=(3, 4, 2, 3), strides=(16, 1, 16, 2))
    ports = library_module(ranked).abi.pins
    assert (
        next(port.width for port in ports if isinstance(port, Signal) and port.name == "olst") == 4
    )


def test_its_nest_is_the_reorder_between_its_sequences_closing_each_marker_required() -> None:
    """The nest is ``classify``'s reorder between the two orders, split where a marker it
    must make closes no level; its output presents every level it closes there."""
    arriving, leaving = nested(6, (3, 6), (0, 1))
    assert generated(arriving, leaving) == Generate(6, (3, 6), (0, 1))
    # A marker every 2 beats: the inner loop split, its bit closing every second beat.
    marked = BeatSequence(leaving.form, markers=(LevelEnd(2),))
    assert generated(arriving, marked) == Generate(6, (3, 3, 2), (0, 2, 1))
    point = design_space(
        InputGeneratorKernel(
            input_tensor=stage_tensor(13),
            arriving=arriving,
            leaving=marked,
            platform=FULL_DSP48E2,
            ram_style="auto",
        )
    )
    assert point.output.stream.sequence.markers == (LevelEnd(18), LevelEnd(6), LevelEnd(2))
    assert dict(point.output.stream_pins.pins.markers) == {
        "olst[0]": LevelEnd(18),
        "olst[1]": LevelEnd(6),
        "olst[2]": LevelEnd(2),
    }
    # The same order with markers to make: identity frames as long as the widest.
    same = BeatSequence(vector_major((6, 1), 1))
    framed = BeatSequence(same.form, markers=(LevelEnd(6), LevelEnd(3)))
    assert generated(same, framed) == Generate(6, (2, 3), (3, 1))


def test_a_generator_with_nothing_to_do_is_refused() -> None:
    same = BeatSequence(vector_major((6, 1), 1))
    with pytest.raises(Unrealizable, match="nothing for an input_gen to do"):
        generated(same, same)
    point = design_space(
        InputGeneratorKernel(
            input_tensor=stage_tensor(13),
            arriving=same,
            leaving=same,
            platform=FULL_DSP48E2,
            ram_style="auto",
        )
    )
    refused = point.query(InputGeneratorKernel.nest)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"input-generator-nest"}
