# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A transpose kept as a position map, against FINN's channels-last graph.

FINN lowers each Conv to Im2Col and MatMul, bracketed by a transpose pair, and its
channels-last transformation moves those transposes away. Here every transpose stays a position
map, and a Thresholding kernel binds its channel index to the axis its operator states:
its schedule is ``(b0, h, w, c)`` with ``c`` in lanes (at its PE) whatever the layout, so
the stream order is the kernel's, not the tensor's. Each case is a ``Reindex`` with one
reader, and its steps are derived with ``route``.

CNV's five 3 x 3 layers, (H, W, C, N, SIMD) = (32, 32, 3, 64, 3), (30, 30, 64, 64, 32),
(14, 14, 64, 128, 32), (12, 12, 128, 128, 32) and (5, 5, 128, 256, 32); the MatMul has PE 4.

- **Into the Conv**, the threshold at PE = SIMD and at PE 1: kept, the producer is the
  threshold over ``(1, C, H, W)`` and the reader the MatMul's x port ``(m, k)`` at SIMD,
  its map the chain transpose ``(0, 2, 3, 1)``, ``im2col``, reshape to ``(OH * OW, 9 * C)``;
  channels-last, the threshold over ``(1, H, W, C)`` and the same reader without the
  transpose. Steps: ``reorder -> markers`` at PE = SIMD, ``width_conversion -> reorder ->
  markers`` at PE 1, on both sides.
- **Out of the Conv**: the producer is the MatMul's y port ``(m, n)`` at PE 4; kept, the
  reader is the threshold over ``(1, N, OH, OW)`` at PE 4 through Reshape ``(OH * OW, N)``
  to ``(1, OH, OW, N)`` and Transpose ``(0, 3, 1, 2)``; channels-last, through the Reshape
  alone. Steps: ``direct`` on both sides.
- **The same hardware**: the same step kinds and, for each reorder, the same frame and the
  same ``input_gen`` address sequence (the digits of ``dims`` times ``coefs``, outer
  first); the loop nests may group loops differently. 15 of 15 hold.
- **The control**, conv0 into the Conv: a threshold walking the NCHW tensor in its own
  row-major order ``(b0, c, h, w)``, one lane. Its reorder is the kept
  ``Reorder(3072, (30, 30, 16, 3, 3, 3), (32, 1, 0, 32, 1, 1024))`` against channels-last's
  ``Reorder(1024, (30, 30, 16, 3, 3), (32, 1, 0, 32, 1))``: not the same hardware, and at
  one lane the live windows are 2115 elements against 201.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from finn.dataflow.buffers import live_window, slots
from finn.dataflow.gemm import k, m, n
from finn.dataflow.maps import PositionMap, chain, im2col, reshape, transpose
from finn.dataflow.reindex import Reader, Reindex, Route, composed, route
from finn.dataflow.schedule import Index, Schedule
from finn.dataflow.steps import Step, Steps
from finn.dataflow.traversal import (
    BeatSequence,
    Grouped,
    Reorder,
    Traversal,
    regrouped,
)

b0, c, h, w = (Index(name) for name in ("b0", "c", "h", "w"))

# (H, W, C, N, SIMD): CNV's five 3 x 3 convolutions.
LAYERS = {
    "conv0": (32, 32, 3, 64, 3),
    "conv1": (30, 30, 64, 64, 32),
    "conv2": (14, 14, 64, 128, 32),
    "conv3": (12, 12, 128, 128, 32),
    "conv4": (5, 5, 128, 256, 32),
}
MATMUL_PE = 4
INTO_KINDS = {
    "simd": (Step.REORDER, Step.MARKERS),
    "one": (Step.WIDTH, Step.REORDER, Step.MARKERS),
}

# One case per layer and side: into the Conv at PE = SIMD and at PE 1, out of the Conv.
CASES = [
    (name, side, pe)
    for name, (_, _, _, _, simd) in LAYERS.items()
    for side, pe in (("into-pe-simd", simd), ("into-pe-1", 1), ("out", MATMUL_PE))
]
CASE_IDS = [f"{name}-{side}" for name, side, _ in CASES]


def thresholding(
    shape: tuple[int, ...],
    access: tuple[Index, ...],
    channels: int,
    height: int,
    width: int,
    pe: int,
) -> Traversal:
    """The Thresholding schedule ``(b0, h, w, c)``, ``c`` in lanes at ``pe``, over a tensor
    of either layout."""
    schedule = Schedule({b0: 1, c: channels, h: height, w: width}, {c: pe}, (b0, h, w, c))
    return schedule.present(shape, access, lanes=(c,))


def addresses(reorder: Reorder) -> npt.NDArray[np.int64]:
    """The read addresses of a reorder's ``input_gen``: the digits of ``dims`` times
    ``coefs``, outer first."""
    found = np.zeros(1, dtype=np.int64)
    for extent, coef in zip(reorder.dims, reorder.coefs):
        found = (found[:, None] + np.arange(extent, dtype=np.int64)[None, :] * coef).ravel()
    return found


def reorders(found: Steps) -> list[Reorder]:
    return [hop.reorder for hop in found.hops if hop.reorder is not None]


def same_hardware(kept: Steps, channels_last: Steps) -> bool:
    """The same step kinds; each reorder the same frame and the same address sequence."""
    if kept.kinds != channels_last.kinds:
        return False
    for one, other in zip(reorders(kept), reorders(channels_last)):
        if one.frame_beats != other.frame_beats or not np.array_equal(
            addresses(one), addresses(other)
        ):
            return False
    return True


def derived(reindex: Reindex) -> Steps:
    """The steps of a reindex whose one reader needs no insertion or drop: its route has
    nothing before, no stage, and its steps after."""
    found = route(reindex)
    assert isinstance(found, Route)
    assert not found.before
    assert found.stage is None
    return found.after


def window_map(height: int, width: int, channels: int) -> PositionMap:
    """The ``(1, H, W, C)`` image read as the ``(OH * OW, 9 * C)`` matrix of its 3 x 3
    windows: ``im2col``, then the Reshape that flattens its rows and columns."""
    window = im2col((1, height, width, channels), (3, 3))
    return chain(window, reshape(window.shape, (window.shape[1] * window.shape[2], 9 * channels)))


def matmul_port(
    height: int, width: int, channels: int, n_out: int, simd: int
) -> tuple[BeatSequence, tuple[int, ...]]:
    """The MatMul's x port ``(m, k)`` at SIMD, with its closing marker, and the matrix's shape."""
    rows, depth = (height - 2) * (width - 2), 9 * channels
    schedule = Schedule({m: rows, n: n_out, k: depth}, {n: MATMUL_PE, k: simd}, (m, n, k))
    port = schedule.present((rows, depth), (m, k), lanes=(k,))
    return BeatSequence(port, markers=(schedule.closing((k,)),)), (rows, depth)


def into_conv(name: str, pe: int) -> tuple[Reindex, Reindex]:
    """The kept reindex (the threshold over NCHW, its reader through the transpose, the
    window and the reshape) and the channels-last one (the threshold over NHWC)."""
    height, width, channels, n_out, simd = LAYERS[name]
    stream, shape = matmul_port(height, width, channels, n_out, simd)
    window = window_map(height, width, channels)
    assert window.shape == shape
    kept_map = chain(transpose((1, channels, height, width), (0, 2, 3, 1)), window)
    kept_port = thresholding(
        (1, channels, height, width), (b0, c, h, w), channels, height, width, pe
    )
    last_port = thresholding(
        (1, height, width, channels), (b0, h, w, c), channels, height, width, pe
    )
    kept = Reindex(BeatSequence(kept_port), (Reader(kept_map, stream),))
    last = Reindex(BeatSequence(last_port), (Reader(window, stream),))
    return kept, last


def out_of_conv(name: str) -> tuple[Reindex, Reindex]:
    """The kept reindex (the MatMul's y port, the reader the threshold over NCHW, through
    the reshape and the transpose) and the channels-last one (the reshape alone)."""
    height, width, _, n_out, simd = LAYERS[name]
    channels = LAYERS[name][2]
    oh, ow = height - 2, width - 2
    schedule = Schedule({m: oh * ow, n: n_out, k: 9 * channels}, {n: MATMUL_PE, k: simd}, (m, n, k))
    y = schedule.present((oh * ow, n_out), (m, n), lanes=(n,), reduces=(k,))
    to_y = reshape((oh * ow, n_out), (1, oh, ow, n_out))
    kept_map = chain(to_y, transpose((1, oh, ow, n_out), (0, 3, 1, 2)))
    kept_port = thresholding((1, n_out, oh, ow), (b0, c, h, w), n_out, oh, ow, MATMUL_PE)
    last_port = thresholding((1, oh, ow, n_out), (b0, h, w, c), n_out, oh, ow, MATMUL_PE)
    kept = Reindex(BeatSequence(y), (Reader(kept_map, BeatSequence(kept_port)),))
    last = Reindex(BeatSequence(y), (Reader(to_y, BeatSequence(last_port)),))
    return kept, last


def reindexes(name: str, side: str, pe: int) -> tuple[Reindex, Reindex]:
    return out_of_conv(name) if side == "out" else into_conv(name, pe)


def _both(case: tuple[str, str, int]) -> tuple[Reindex, Reindex]:
    return reindexes(*case)


# --- the five layers, both sides of the Conv ---------------------------------------------


@pytest.mark.parametrize(("name", "side", "pe"), CASES, ids=CASE_IDS)
def test_a_kept_transpose_is_the_same_hardware_as_the_channels_last_graph(
    name: str, side: str, pe: int
) -> None:
    kept, last = reindexes(name, side, pe)
    kept_steps, last_iterations = derived(kept), derived(last)
    expected: tuple[Step, ...]
    if side == "out":
        expected = ()
    elif pe == 1:
        expected = INTO_KINDS["one"]
    else:
        expected = INTO_KINDS["simd"]
    assert kept_steps.kinds == last_iterations.kinds == expected
    assert same_hardware(kept_steps, last_iterations)
    if side == "out":
        assert kept_steps.describe() == last_iterations.describe() == "direct"
        assert not kept_steps and not last_iterations


def test_the_kept_transpose_is_the_same_hardware_in_15_of_15_cases() -> None:
    same = [same_hardware(derived(kept), derived(last)) for kept, last in map(_both, CASES)]
    assert len(same) == 15
    assert sum(same) == 15


# --- the control: a threshold that walks the NCHW tensor in its own order -----------------


def control_reindexes() -> tuple[Reindex, Reindex, Traversal]:
    """conv0 into the Conv: the threshold walks ``(b0, c, h, w)`` one lane, plane by plane
    (kept), against FINN's ``(b0, h, w, c)`` at one lane (channels-last)."""
    height, width, channels, n_out, simd = LAYERS["conv0"]
    stream, _ = matmul_port(height, width, channels, n_out, simd)
    window = window_map(height, width, channels)
    plane = Schedule({b0: 1, c: channels, h: height, w: width}, {}, (b0, c, h, w)).present(
        (1, channels, height, width), (b0, c, h, w)
    )
    kept_map = chain(transpose((1, channels, height, width), (0, 2, 3, 1)), window)
    last_port = thresholding(
        (1, height, width, channels), (b0, h, w, c), channels, height, width, 1
    )
    kept = Reindex(BeatSequence(plane), (Reader(kept_map, stream),))
    last = Reindex(BeatSequence(last_port), (Reader(window, stream),))
    return kept, last, plane


def live_elements(reindex: Reindex, composed_order: Traversal | Grouped) -> int:
    """The live window at one lane of the producer's stream against the composed one."""
    return live_window(slots(regrouped(reindex.producer.form, 1), regrouped(composed_order, 1)))


def test_a_threshold_walking_the_tensor_in_its_own_order_is_not_the_same_hardware() -> None:
    kept, last, _ = control_reindexes()
    kept_steps, last_iterations = derived(kept), derived(last)
    assert not same_hardware(kept_steps, last_iterations)
    assert reorders(kept_steps) == [Reorder(3072, (30, 30, 16, 3, 3, 3), (32, 1, 0, 32, 1, 1024))]
    assert reorders(last_iterations) == [Reorder(1024, (30, 30, 16, 3, 3), (32, 1, 0, 32, 1))]
    assert live_elements(kept, composed(kept).form) == 2115
    assert live_elements(last, composed(last).form) == 201
