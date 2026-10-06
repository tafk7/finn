# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The XSim harness refuses a stream the module does not present, before simulating."""

from __future__ import annotations

from pathlib import Path

import pytest

from kernels.chain import chain
from kernels.xsim import stream_through


@pytest.mark.parametrize(
    "inputs, outputs, complaint",
    [
        # A stale name (a partition root's before its ports took the shells' names).
        (
            {"in0_V": ([1], 6)},
            {"m_axis_0": ([1], 16)},
            "no input stream in0_V; its inputs: s_axis_0",
        ),
        # A name on the wrong side.
        (
            {"s_axis_0": ([1], 6)},
            {"s_axis_0": ([1], 6)},
            "no output stream s_axis_0; its outputs: m_axis_0",
        ),
    ],
)
def test_a_stream_the_module_lacks_is_refused_naming_its_ports(
    tmp_path: Path, inputs: dict, outputs: dict, complaint: str
) -> None:
    with pytest.raises(ValueError, match=complaint):
        stream_through(chain().module, tmp_path, inputs=inputs, outputs=outputs)
    assert not any(tmp_path.iterdir())  # refused before anything was built
