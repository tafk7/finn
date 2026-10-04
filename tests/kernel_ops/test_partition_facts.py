# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A partition's boundary facts (``finn.partition``).

PackagePartition writes the facts from the partition root's boundary.
test_design's Chain, its choices saved, as the partition; no Vivado.
"""

from __future__ import annotations

from typing import Any

import pytest

from finn.custom_op.kernels.base import PLATFORM_KEYS, KernelOpError
from finn.transformation.kernels import PackagePartition
from finn.transformation.kernels.package import (
    PARTITION_INPUTS,
    partition_facts,
    write_boundary_facts,
)
from kernel_ops.test_partition import configured, kernel_model


def facts(port: str, tensor: str, dims: list[int], dtype: str, *counts: int) -> dict[str, Any]:
    lanes, beats, bits, tdata = counts
    return {
        "port": port,
        "tensor": tensor,
        "shape": dims,
        "datatype": dtype,
        "lanes": lanes,
        "beats": beats,
        "element_bits": bits,
        "tdata": tdata,
    }


def test_the_facts_are_the_boundary_streams_at_the_partitions_end() -> None:
    model = kernel_model(second_weights=False)  # x and the streamed w2 cross the boundary
    configured(model)
    assert model.get(PARTITION_INPUTS) is None
    write_boundary_facts(model)
    inputs, outputs = partition_facts(model)
    assert inputs == [
        facts("s_axis_0", "x", [3, 4], "INT3", 2, 6, 3, 8),
        # The weights' repetition stays at the boundary: three rows, 4 beats each.
        facts("s_axis_1", "w2", [4, 4], "INT3", 4, 12, 3, 16),
    ]
    assert outputs == [facts("m_axis_0", "y", [3, 4], "INT7", 2, 6, 7, 16)]


def test_a_model_without_facts_is_refused() -> None:
    with pytest.raises(KernelOpError, match="no boundary facts.*run PackagePartition"):
        partition_facts(kernel_model())


def test_packaging_reads_the_part_and_period_from_the_target() -> None:
    model = kernel_model()
    configured(model)
    for key in PLATFORM_KEYS.values():
        model.delete(key)
    with pytest.raises(KernelOpError, match="states no target"):
        model.transform(PackagePartition("sdp_1"))
