# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Admitted INT3 stream bindings are authored by the selected Kernel."""

from dataclasses import replace

import pytest

from finn.dataflow._engine import Decided
from finn.dataflow.kernels.physical import (
    FieldPlacement,
    PackedBeatLayout,
    PeriodicLast,
    UnusedBitPolicy,
    UnusedBitRange,
    capture_kernel_realization,
    validate_kernel_stream_bindings,
)

from dataflow.ops.mvau.test_dot_product_design import _occurrence


def _children():
    design = _occurrence(
        repetitions=1,
        matrix_width=4,
        matrix_height=4,
        activation="INT3",
        weight="INT3",
        accumulator="INT16",
        pe=2,
        simd=2,
        pumping=True,
        narrow=True,
    )
    replay, compute = design.kernel("replay"), design.kernel("compute")
    assert isinstance(replay, Decided) and isinstance(compute, Decided)
    return replay.value, compute.value


def test_int3_simd2_authored_bindings_preserve_six_fields_bits_and_framing():
    replay, compute = _children()
    left, right = capture_kernel_realization(replay), capture_kernel_realization(compute)
    replay_streams = {b.region_port_id: b for b in left.streams}
    compute_streams = {b.region_port_id: b for b in right.streams}
    six_bits = (FieldPlacement(0, 0, 3), FieldPlacement(1, 3, 3))
    assert replay_streams["activation_out"].payload == PackedBeatLayout(six_bits)
    assert compute_streams["activation"].payload == PackedBeatLayout(
        six_bits,
        (UnusedBitRange(6, 2, UnusedBitPolicy.IGNORE_ON_RECEIVE),),
    )
    assert replay_streams["activation_out"].framing == PeriodicLast("tlast", 2, 1)
    assert compute_streams["activation"].framing == PeriodicLast("tlast", 2, 1)
    assert compute_streams["weight"].payload == PackedBeatLayout(
        tuple(FieldPlacement(i, i * 3, 3) for i in range(4)),
        (UnusedBitRange(12, 4, UnusedBitPolicy.IGNORE_ON_RECEIVE),),
    )
    assert compute_streams["output"].payload == PackedBeatLayout(
        (FieldPlacement(0, 0, 16), FieldPlacement(1, 16, 16)),
    )
    assert not hasattr(left.requirements, "region")


@pytest.mark.parametrize("case", ("missing", "duplicate", "overlap", "padding", "framing"))
def test_bad_local_stream_bindings_refuse(case):
    _, compute = _children()
    facts = capture_kernel_realization(compute)
    logical = compute.dataflow.accepted_answer
    assert isinstance(logical, Decided)
    binding, *rest = facts.streams
    if case == "missing":
        bindings = tuple(rest)
    elif case == "duplicate":
        bindings = (*facts.streams, binding)
    elif case == "overlap":
        bindings = (
            replace(
                binding,
                payload=PackedBeatLayout(
                    (FieldPlacement(0, 0, 3), FieldPlacement(1, 2, 3)),
                    binding.payload.unused,
                ),
            ),
            *rest,
        )
    elif case == "padding":
        bindings = (replace(binding, payload=PackedBeatLayout(binding.payload.fields)), *rest)
    else:
        bindings = (replace(binding, framing=None), *rest)
    with pytest.raises(ValueError):
        validate_kernel_stream_bindings(logical.value, facts.requirements.abi, bindings)
