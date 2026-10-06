# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The input generator: its loop geometry, its replay and its end-of-loop markers."""

from __future__ import annotations

from typing import cast

import pytest

from finn.core.space import DefinitionError, Rejected
from finn.kernels.artifacts.abi import Signal
from finn.kernels.input_generator import InputGeneratorKernel
from finn.kernels.values.semantics import IntegerVector
from kernels.helpers import generator


def test_generator_states_zero_stride_replay_and_multibit_markers() -> None:
    point = generator()
    requirements = point.module
    assert requirements.parameters == (
        ("COEFS", "'{0, 1}"),
        ("D", 2),
        ("DATA_WIDTH", 13),
        ("DIMS", "'{3, 6}"),
        ("FM_SIZE", 6),
        ("RAM_STYLE", '"auto"'),
    )
    widths = {port.name: port.width for port in requirements.pins.pins if isinstance(port, Signal)}
    assert widths["idat"] == widths["odat"] == 13
    assert widths["olst"] == 2
    assert all(isinstance(port, Signal) for port in requirements.pins.pins)
    ranked = generator(frame=56, dims=(3, 4, 2, 3), strides=(16, 1, 16, 2))
    ports = ranked.module.pins.pins
    assert (
        next(port.width for port in ports if isinstance(port, Signal) and port.name == "olst") == 4
    )


@pytest.mark.parametrize(
    ("dims", "strides"),
    (((), ()), ((3, 6), (1,)), ((2, 6), (1, 1)), ((3, 6), (-1, 1)), ((0, 6), (0, 1))),
)
def test_generator_refuses_invalid_loop_geometry(
    dims: IntegerVector, strides: IntegerVector
) -> None:
    assert isinstance(
        generator(dims=dims, strides=strides).inspect(InputGeneratorKernel.module).accepted_result,
        Rejected,
    )


@pytest.mark.parametrize("bad", ([3, 6], (3, True), (3, [6])))
def test_generator_requires_exact_immutable_integer_vectors(bad: object) -> None:
    # A bad literal is refused at the node call.
    with pytest.raises(DefinitionError, match="integer vector"):
        generator(dims=cast(IntegerVector, bad))
