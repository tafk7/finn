# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Position maps on the geometries that matter: what composing them gives, against what
the line's ``finn.dataflow`` states, and what the steps and buffers between a producer
and a consumer come to.

Each test states its geometry and the numbers it asserts:

- CNV's five 3x3 windows, each at PE 1, 4 and 16: a plain MatMul ``(m, n, k)`` read
  through ``im2col`` against the line's windowed MatMul (``gemm.Window``, schedule
  ``(oh, ow, n, kh, kw, c)``), then against a MatMul kernel placed in the line's test
  support;
- the live window of each at PE = N;
- SIMD against the channels on an 8 x 8 image;
- padding, factored into an insertion of constants and a window over the padded image;
- DepthToSpace, the inverse transposes and Concat.

Every live window is in beats of the producer, which presents its tensor row-major, its
innermost axis in lanes (SIMD lanes for a window's image, the channels for the others).
"""

from __future__ import annotations

import pytest

import functools
import numpy as np
import numpy.typing as npt
from dataflow_next.harness import translate
from math import prod
from support.kernel_specs import matmul as matmul_spec
from support.kernel_specs.base import placed
from support.kernel_specs.thresholding import tensor
from typing import Any

import finn.dataflow.gemm as line_gemm
import finn.dataflow.schedule as line_schedule
import finn.dataflow.steps as line_steps
import finn.dataflow.traversal as line_traversal
from finn.dataflow_next.buffers import live_window, slots
from finn.dataflow_next.gemm import k, m, n
from finn.dataflow_next.maps import (
    PositionMap,
    Unmappable,
    chain,
    concat,
    depth_to_space,
    factor_fills,
    identity,
    im2col,
    padded_im2col,
    reads,
    reshape,
    through,
    transpose,
)
from finn.dataflow_next.schedule import Schedule
from finn.dataflow_next.steps import steps
from finn.dataflow_next.traversal import (
    BeatSequence,
    Marker,
    Reorder,
    Traversal,
    offsets,
    regrouped,
    vector_major,
)

Array = npt.NDArray[np.int64]

# (layer, H, W, C, N, SIMD): CNV's five 3x3 convolutions.
LAYERS = {
    "conv0": (32, 32, 3, 64, 3),
    "conv1": (30, 30, 64, 64, 32),
    "conv2": (14, 14, 64, 128, 32),
    "conv3": (12, 12, 128, 128, 32),
    "conv4": (5, 5, 128, 256, 32),
}
PES = (1, 4, 16)
CASES = [(name, pe) for name in LAYERS for pe in PES]
CASE_IDS = [f"{name}-pe{pe}" for name, pe in CASES]


def reorders(found: Any) -> list[Any]:
    """Each reorder among the hops of ``found`` (the line's steps or the revision's)."""
    return [hop.reorder for hop in found.hops if hop.reorder is not None]


def loops(form: Any) -> tuple[Any, Any]:
    return form.beat_loops, form.lane_loops


# --- CNV's windows ---------------------------------------------------------------------


@functools.cache
def window_map(name: str) -> PositionMap:
    """The ``(1, H, W, C)`` image of a layer read as the ``(M, K)`` matrix of its 3x3 windows:
    ``im2col``, then the Reshape that flattens its rows and columns."""
    h, w, c = LAYERS[name][:3]
    window = im2col((1, h, w, c), (3, 3))
    return chain(window, reshape(window.shape, (window.shape[1] * window.shape[2], 9 * c)))


@functools.cache
def composed_window(name: str, pe: int) -> tuple[Traversal, Marker, Traversal]:
    """The plain MatMul ``(m, n, k)`` at ``pe`` and SIMD, read through the window map: its
    traversal of the ``(1, H, W, C)`` image, its marker and its own port's traversal of
    the ``(M, K)`` matrix."""
    n_out, simd = LAYERS[name][3:]
    flat = window_map(name)
    rows, depth = flat.shape
    schedule = Schedule({m: rows, n: n_out, k: depth}, {n: pe, k: simd}, (m, n, k))
    port = schedule.present((rows, depth), (m, k), lanes=(k,))
    return through(flat, port), schedule.closing((k,)), port


@functools.cache
def line_window(name: str, pe: int) -> tuple[Any, Any]:
    """Today's form: the same MatMul read through a ``gemm.Window`` of the line, on the
    schedule ``(oh, ow, n, kh, kw, c)``: its traversal of the ``(H * W, C)`` image and its
    marker."""
    h, w, c, n_out, simd = LAYERS[name]
    window = line_gemm.Window(image=(h, w), size=(3, 3))
    schedule = line_schedule.Schedule(
        {**window.extents, line_gemm.n: n_out, line_gemm.c: c},
        {line_gemm.n: pe, line_gemm.c: simd},
        (
            line_gemm.oh,
            line_gemm.ow,
            line_gemm.n,
            line_gemm.kh,
            line_gemm.kw,
            line_gemm.c,
        ),
    )
    form = schedule.present((h * w, c), window.signature.x, lanes=(line_gemm.c,))
    return form, schedule.closing((line_gemm.kh, line_gemm.kw, line_gemm.c))


@functools.cache
def enumerated(name: str) -> tuple[Array, Array]:
    return reads(window_map(name))


@pytest.mark.parametrize(("name", "pe"), CASES, ids=CASE_IDS)
def test_the_composed_window_is_the_enumeration(name: str, pe: int) -> None:
    # What the composed traversal presents is what the MatMul's port asks for, read
    # through the map: the same source offsets, position by position.
    composed, _, port = composed_window(name, pe)
    source, offset = enumerated(name)
    assert (source == 0).all()
    assert np.array_equal(offsets(composed), offset[offsets(port)])


@pytest.mark.parametrize(("name", "pe"), CASES, ids=CASE_IDS)
def test_a_plain_matmul_through_im2col_is_the_lines_windowed_matmul(name: str, pe: int) -> None:
    # The same offsets, the same loops and the same marker: the line's form, translated
    # to the revision's values, is the composition (the line's image is (H * W, C), the
    # composition's (1, H, W, C): the same flat offsets).
    composed, marker, _ = composed_window(name, pe)
    form, closing = line_window(name, pe)
    today = translate.forward(form)
    assert isinstance(today, Traversal)
    assert prod(today.shape) == prod(composed.shape)
    assert loops(today) == loops(composed)
    assert np.array_equal(offsets(today), offsets(composed))
    assert translate.forward(closing) == marker


INPUT_GEN = {
    # (FM_SIZE, DIMS at PE 1 (its third entry is N / PE), COEFS)
    "conv0": (1024, (30, 30, 64, 3, 3), (32, 1, 0, 32, 1)),
    "conv1": (1800, (28, 28, 64, 3, 6), (60, 2, 0, 60, 1)),
    "conv2": (392, (12, 12, 128, 3, 6), (28, 2, 0, 28, 1)),
    "conv3": (576, (10, 10, 128, 3, 12), (48, 4, 0, 48, 1)),
    "conv4": (100, (3, 3, 256, 3, 12), (20, 4, 0, 20, 1)),
}


@pytest.mark.parametrize(("name", "pe"), CASES, ids=CASE_IDS)
def test_the_steps_and_the_input_generator_of_a_row_major_producer(name: str, pe: int) -> None:
    # From a producer presenting the image row-major, SIMD lanes a beat, the steps are one
    # reorder and the markers; its frame, dims and coefficients are the line's, which
    # are FINN's input generator's FM_SIZE, DIMS and COEFS.
    h, w, c, n_out, simd = LAYERS[name]
    composed, marker, _ = composed_window(name, pe)
    form, closing = line_window(name, pe)
    found = steps(
        BeatSequence(vector_major((1, h, w, c), simd)), BeatSequence(composed, markers=(marker,))
    )
    today = line_steps.steps(
        line_traversal.BeatSequence(line_traversal.vector_major((h * w, c), simd)),
        line_traversal.BeatSequence(form, markers=(closing,)),
    )
    assert found.describe() == today.describe() == "reorder -> markers"
    [reorder], [old] = reorders(found), reorders(today)
    assert translate.forward(old) == reorder
    frame, dims, coefs = INPUT_GEN[name]
    assert reorder == Reorder(frame, (*dims[:2], n_out // pe, *dims[3:]), coefs)
    assert (old.frame_beats, old.dims, old.coefs) == (
        reorder.frame_beats,
        reorder.dims,
        reorder.coefs,
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [("conv0", 66), ("conv1", 124), ("conv2", 60), ("conv3", 104), ("conv4", 48)],
)
def test_a_windows_live_window_at_pe_equal_to_n(name: str, expected: int) -> None:
    # PE = N: the activations are not replayed. The window holds this many producer beats.
    h, w, c, n_out, simd = LAYERS[name]
    composed = composed_window(name, n_out)[0]
    assert live_window(slots(vector_major((1, h, w, c), simd), composed)) == expected


# --- SIMD against the channels ---------------------------------------------------------


def eight_by_eight_port(channels: int, simd: int) -> tuple[PositionMap, Schedule, Traversal]:
    """The 8 x 8 image's window map flattened to ``(36, 9 * channels)``, the schedule of a
    MatMul of N = 4 with K folded by ``simd`` and N not folded, and the plain port that
    schedule presents of the flattened reader: ``(m, k)`` beats, ``k`` in lanes."""
    window = im2col((1, 8, 8, channels), (3, 3))
    flat = chain(window, reshape(window.shape, (36, 9 * channels)))
    schedule = Schedule({m: 36, n: 4, k: 9 * channels}, {k: simd}, (m, n, k))
    return flat, schedule, schedule.present(flat.shape, (m, k), lanes=(k,))


def eight_by_eight(channels: int, simd: int) -> Traversal:
    """An 8 x 8 image of ``channels`` read through ``im2col`` by a MatMul of N = 4, K folded
    by ``simd`` and N not folded: the lanes the composition finds."""
    flat, _, port = eight_by_eight_port(channels, simd)
    return through(flat, port)


@pytest.mark.parametrize(
    ("channels", "simd", "lanes"),
    [(6, 3, 3), (3, 9, 9), (6, 18, 18), (4, 12, 12)],
    ids=["C6-SIMD3", "C3-SIMD9", "C6-SIMD18", "C4-SIMD12"],
)
def test_simd_dividing_the_channels_or_a_whole_number_of_pixels_composes(
    channels: int, simd: int, lanes: int
) -> None:
    # SIMD divides C (the line's own rule), or is a whole number of a window row's pixels
    # (3 pixels of 3 channels, 3 of 6, 3 of 4): the lanes are one run of that many elements.
    assert [(loop.extent, loop.stride) for loop in eight_by_eight(channels, simd).lane_loops] == [
        (lanes, 1)
    ]


@pytest.mark.parametrize(
    ("channels", "simd", "lanes", "beats"),
    [(6, 9, (9, 1), (2, 9)), (4, 6, (6, 1), (2, 6))],
    ids=["C6-SIMD9", "C4-SIMD6"],
)
def test_simd_of_a_part_of_a_window_row_composes_as_one_run(
    channels: int, simd: int, lanes: tuple[int, int], beats: tuple[int, int]
) -> None:
    # A window row is KW * C contiguous elements (18 at C 6, 12 at C 4): the canonical map
    # merges its kw and c modes. SIMD dividing the run but not C (a pixel and a half in
    # both cases) reads it in whole beats: lanes one run of SIMD, and a beat loop of the
    # row's beats innermost. A map that kept kw and c apart would refuse these.
    found = eight_by_eight(channels, simd)
    assert [(loop.extent, loop.stride) for loop in found.lane_loops] == [lanes]
    assert (found.beat_loops[-1].extent, found.beat_loops[-1].stride) == beats


# What the chain of steps delivers: the producer is row-major with ``lanes`` lanes; the
# consumer is the composed window (``eight_by_eight``) with its closing markers.
STEPS_OF_A_PART_OF_A_ROW = [
    (6, 9, 3, Reorder(128, (6, 6, 4, 3, 6), (16, 2, 0, 16, 1))),
    (6, 9, 6, Reorder(64, (6, 6, 4, 3, 3), (8, 1, 0, 8, 1))),
    (4, 6, 4, Reorder(64, (6, 6, 4, 3, 3), (8, 1, 0, 8, 1))),
]


@pytest.mark.parametrize(
    ("channels", "simd", "lanes", "reorder"),
    STEPS_OF_A_PART_OF_A_ROW,
    ids=["C6-SIMD9-3lanes", "C6-SIMD9-6lanes", "C4-SIMD6-4lanes"],
)
def test_the_steps_for_simd_of_a_part_of_a_window_row_deliver_the_enumeration(
    channels: int, simd: int, lanes: int, reorder: Reorder
) -> None:
    flat, schedule, port = eight_by_eight_port(channels, simd)
    composed = through(flat, port)
    producer = vector_major((1, 8, 8, channels), lanes)
    found = steps(BeatSequence(producer), BeatSequence(composed, markers=(schedule.closing((k,)),)))
    assert found.describe() == "reorder -> width_conversion -> markers"
    assert [hop.reorder for hop in found.hops if hop.reorder] == [reorder]

    # What is compared. The sink's elements in the sink's order, by three routes:
    #   enumeration: the flattened map's source offset (``reads``) at each element the plain
    #     port presents (``offsets`` of the port, offsets of the reader's flat positions);
    #   composed: ``offsets`` of the composed traversal of the source;
    #   delivered: what the chain's reorder presents, beat by beat, at the producer's lanes.
    #     The reorder's output is the sink's traversal regrouped to those lanes (the first
    #     hop's sink); its beats are whole beats of the producer (``slots``: the producer
    #     beat each one reads), which the reorder's own loop nest emits in the same order.
    enumeration = reads(flat)[1][offsets(port)]
    assert np.array_equal(offsets(composed), enumeration)
    reordered = found.hops[0].sink.form
    assert reordered == regrouped(composed, lanes)
    assert np.array_equal(offsets(reordered), enumeration)
    beat_of = slots(producer, reordered)
    emitted = np.indices(reorder.dims).reshape(len(reorder.dims), -1).T @ np.array(reorder.coefs)
    frames = len(beat_of) // len(emitted)
    frame = np.arange(frames).repeat(len(emitted)) * reorder.frame_beats
    assert np.array_equal(beat_of, frame + np.tile(emitted, frames))
    delivered = offsets(producer).reshape(producer.beats, lanes)[beat_of].ravel()
    assert np.array_equal(delivered, enumeration)
    # The width conversion after it regroups the same elements to the schedule's lanes.
    assert found.hops[1].sink.form == composed
    assert np.array_equal(offsets(found.hops[1].sink.form), delivered)


@pytest.mark.parametrize(("channels", "simd"), [(6, 9), (4, 6)], ids=["C6-SIMD9", "C4-SIMD6"])
def test_a_producer_of_simd_lanes_is_no_producer_when_simd_does_not_divide_the_channels(
    channels: int, simd: int
) -> None:
    # A row-major producer's lanes divide its innermost axis, the channels: the producer
    # itself is refused where it is built, before any steps are asked for.
    with pytest.raises(ValueError, match="lanes must divide the innermost extent"):
        vector_major((1, 8, 8, channels), simd)


@pytest.mark.parametrize(
    ("channels", "simd"),
    [(6, 27), (4, 18), (4, 9)],
    ids=["C6-SIMD27", "C4-SIMD18", "C4-SIMD9"],
)
def test_simd_carrying_across_a_window_row_is_refused(channels: int, simd: int) -> None:
    # SIMD of a row and a half (C 6, SIMD 27: lanes), a row and a half (C 4, SIMD 18:
    # lanes) or three quarters of a row (C 4, SIMD 9: its beats): the next element is in
    # the next window row, not the next in memory.
    with pytest.raises(Unmappable) as info:
        eight_by_eight(channels, simd)
    assert [finding.code for finding in info.value.findings] == ["loop-straddles-mode"]


# --- against a placed kernel -----------------------------------------------------------


def placed_stream(name: str, pe: int) -> Any:
    """The activations' stream of a MatMul kernel read through a ``Window``, placed as the
    line's test support places it (the packed core, ``pe`` and SIMD committed)."""
    h, w, c, n_out, simd = LAYERS[name]
    window = line_gemm.Window((h, w), (3, 3))
    x, weights = tensor((h * w, c), "INT4"), tensor((9 * c, n_out), "INT4")
    case = matmul_spec.matmul()
    case["inputs"] = {"x_channel": x, "w_channel": weights}
    case["facts"] = {**case["facts"], "weights": weights}
    case["outputs"] = {"y_channel": matmul_spec._results(x, weights, window=window)}
    kernel: Any = placed(case, {"pe": pe, "simd": simd, **case["choices"]}, window=window)
    return kernel.x_stream.sequence


@pytest.mark.parametrize(("name", "pe"), CASES, ids=CASE_IDS)
def test_the_composed_window_is_a_placed_kernels_activation_stream(name: str, pe: int) -> None:
    # The kernel's ``x_stream``, translated, has the composition's loops, elements and marker.
    composed, marker, _ = composed_window(name, pe)
    stream = translate.forward(placed_stream(name, pe))
    assert isinstance(stream, BeatSequence)
    assert loops(stream.form) == loops(composed)
    assert np.array_equal(offsets(stream.form), offsets(composed))
    assert stream.markers == (marker,)


# --- padding ---------------------------------------------------------------------------


def stream_with_fills(insertion_shape: tuple[int, ...], filled: npt.NDArray[np.bool_]) -> Array:
    """The padded tensor in row-major order as an insertion presents it: each source element,
    in its order, as its number, with each constant (-1) where it falls."""
    stream = np.full(filled.size, -1, dtype=np.int64)
    stream[~filled] = np.arange(int((~filled).sum()))
    assert stream.size == prod(insertion_shape)
    return stream


@pytest.mark.parametrize(
    ("image", "pad", "stride", "simd", "pe", "constants", "padded_window", "plain_window"),
    [
        ((8, 8, 4), 1, 1, 4, 2, 144, 23, 19),
        ((8, 8, 4), 1, 1, 2, 4, 144, 46, 38),
        ((8, 8, 4), 2, 1, 1, 1, 320, 108, 76),
        ((9, 7, 3), 1, 2, 3, 1, 108, 21, 17),
        ((32, 32, 3), 1, 1, 3, 4, 396, 71, 67),
    ],
    ids=["8x8x4-pad1-simd4-pe2", "8x8x4-pad1-simd2-pe4", "8x8x4-pad2", "9x7x3-stride2", "32x32x3"],
)
def test_a_padded_window_is_an_insertion_then_a_window(
    image: tuple[int, int, int],
    pad: int,
    stride: int,
    simd: int,
    pe: int,
    constants: int,
    padded_window: int,
    plain_window: int,
) -> None:
    # A 3 x 3 window of N = 8 at ``stride``, ``pad`` zeros around the image (a Pad, then the
    # window), a row-major producer in SIMD lanes, PE as given. The constants are inserted
    # in the padded image's row-major order, the producer's elements in theirs: the
    # insertion moves nothing; the window over the padded image is one reorder.
    h, w, c = image
    padding, window = padded_im2col((1, h, w, c), (3, 3), (stride, stride), pads=(pad,) * 4)
    insertion, over_padded = factor_fills(padding, window)  # refuses out of order
    assert insertion.count == constants
    padded_shape = (1, h + 2 * pad, w + 2 * pad, c)
    assert insertion.shape == padded_shape

    # The factorization is exact: the window's read, by enumeration, of the source through
    # the pad (-1 where it reads a constant) is its read of the insertion's stream.
    rows = window.shape[1] * window.shape[2]
    flat = chain(window, reshape(window.shape, (rows, 9 * c)))
    schedule = Schedule({m: rows, n: 8, k: 9 * c}, {n: pe, k: simd}, (m, n, k))
    port = schedule.present(flat.shape, (m, k), lanes=(k,))
    over = through(flat, port)  # the window's traversal of the padded image
    label, offset = reads(padding)
    composite = np.where(label < 0, -1, offset)[offsets(over)]
    assert np.array_equal(
        composite, stream_with_fills(padded_shape, insertion.inserted())[offsets(over)]
    )
    assert over_padded == window

    # The window's steps from the padded image's row-major stream, and the live windows
    # (in producer beats) of the window over the padded image and over the image alone.
    producer = vector_major(padded_shape, simd)
    found = steps(BeatSequence(producer), BeatSequence(over, markers=(schedule.closing((k,)),)))
    assert found.describe() == "reorder -> markers"
    assert live_window(slots(producer, over)) == padded_window

    bare = im2col((1, h, w, c), (3, 3), (stride, stride))
    bare_rows = bare.shape[1] * bare.shape[2]
    bare_flat = chain(bare, reshape(bare.shape, (bare_rows, 9 * c)))
    bare_schedule = Schedule({m: bare_rows, n: 8, k: 9 * c}, {n: pe, k: simd}, (m, n, k))
    bare_over = through(bare_flat, bare_schedule.present(bare_flat.shape, (m, k), lanes=(k,)))
    assert live_window(slots(vector_major((1, h, w, c), simd), bare_over)) == plain_window


# --- DepthToSpace ----------------------------------------------------------------------


def applied(pmap: PositionMap, values: Array) -> Array:
    """What ``pmap`` reads of one source of ``values``, by its enumeration."""
    source, offset = reads(pmap)
    assert (source == 0).all()
    return values.ravel()[offset].reshape(pmap.shape)


@pytest.mark.parametrize(
    ("image", "block", "modes", "frame", "dims", "coefs"),
    [
        ((4, 6, 8), 2, ((4, 48), (2, 4), (6, 8), (4, 1)), 24, (2, 6, 2), (2, 4, 1)),
        ((3, 3, 18), 3, ((3, 54), (3, 6), (3, 18), (6, 1)), 27, (3, 3, 3), (3, 9, 1)),
    ],
    ids=["4x6x8-block2", "3x3x18-block3"],
)
def test_depth_to_space_is_its_three_steps_as_one_map(
    image: tuple[int, int, int],
    block: int,
    modes: tuple[tuple[int, int], ...],
    frame: int,
    dims: tuple[int, ...],
    coefs: tuple[int, ...],
) -> None:
    # DepthToSpace as qonnx writes it (DCR, channels last): Reshape (H, W, b, b, C'),
    # Transpose to (H, b, W, b, C'), Reshape (H * b, W * b, C'). One map equals the three
    # applied in turn, in canonical modes; at C' lanes at both ends, to a row-major consumer
    # it is one reorder.
    h, w, c = image
    depth = c // (block * block)
    split = reshape(image, (h, w, block, block, depth))
    swap = transpose(split.shape, (0, 2, 1, 3, 4))
    merge = reshape(swap.shape, (h * block, w * block, depth))
    one = chain(split, chain(swap, merge))
    assert one == depth_to_space(image, block, channels_last=True)
    assert tuple((mode.extent, mode.stride) for mode in one.pieces[0].modes) == modes
    x = np.arange(prod(image), dtype=np.int64).reshape(image) + 1000
    assert np.array_equal(applied(one, x), applied(merge, applied(swap, applied(split, x))))
    producer = vector_major(image, depth)
    consumer = vector_major((h * block, w * block, depth), depth)
    found = steps(BeatSequence(producer), BeatSequence(through(one, consumer)))
    assert found.describe() == "reorder"
    assert reorders(found) == [Reorder(frame, dims, coefs)]


@pytest.mark.parametrize(("image", "block"), [((4, 6, 8), 2), ((3, 3, 18), 3), ((2, 5, 12), 2)])
def test_depth_to_space_in_crd_mode_is_onnxs_three_steps(
    image: tuple[int, int, int], block: int
) -> None:
    # ONNX's CRD DepthToSpace on NCHW is Reshape (N, C', b, b, H, W), Transpose
    # (0, 1, 4, 2, 5, 3), Reshape (N, C', H * b, W * b); the NHWC map reads the same image
    # transposed to and from NHWC. (qonnx has no routine for it: the spec's, in NumPy.)
    h, w, c = image
    depth = c // (block * block)
    nhwc = np.arange(prod(image), dtype=np.int64).reshape(1, *image) + 1000
    nchw = nhwc.transpose(0, 3, 1, 2)
    expected = (
        nchw.reshape(1, depth, block, block, h, w)
        .transpose(0, 1, 4, 2, 5, 3)
        .reshape(1, depth, h * block, w * block)
    )
    pmap = depth_to_space((1, *image), block, "CRD", channels_last=True)
    assert np.array_equal(applied(pmap, nhwc), expected.transpose(0, 2, 3, 1))
    # And it is not DCR's, whose channel groups are ordered the other way.
    assert pmap != depth_to_space((1, *image), block, "DCR", channels_last=True)


# --- inverse transposes ----------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 8, 4, 4), (1, 3, 5, 7)], ids=["1x8x4x4", "1x3x5x7"])
def test_inverse_transposes_compose_to_the_identity_and_connect_directly(
    shape: tuple[int, int, int, int],
) -> None:
    # NCHW to NHWC, then back: the identity, so the channel connects with no step, the
    # width W of the image in lanes at both ends.
    batch, c, h, w = shape
    to_nhwc = transpose(shape, (0, 2, 3, 1))
    to_nchw = transpose((batch, h, w, c), (0, 3, 1, 2))
    both = chain(to_nhwc, to_nchw)
    assert both == identity(shape)
    form = vector_major(shape, w)
    found = steps(BeatSequence(form), BeatSequence(through(both, form)))
    assert found.describe() == "direct"
    assert through(both, form).lane_loops == form.lane_loops


# --- Concat ----------------------------------------------------------------------------


def concat_facts(h: int, w: int, c1: int, c2: int, pe: int) -> dict[str, Any]:
    """What a row-major consumer at ``pe`` lanes reads of Concat of two ``(h, w, c)`` tensors
    along the channels, each read in its own row-major order, from ``reads``."""
    pmap = concat(((h, w, c1), (h, w, c2)), 2)
    source, offset = reads(pmap)
    consumer = offsets(vector_major((h, w, c1 + c2), pe)).reshape(-1, pe)
    from_source, at = source[consumer], offset[consumer]
    one_source = bool((from_source == from_source[:, :1]).all())
    facts: dict[str, Any] = {"each_beat_one_source": one_source}
    if one_source:
        beat_source = from_source[:, 0]
        facts["first_source_in_row_major"] = np.array_equal(
            at[from_source == 0], np.arange(h * w * c1)
        )
        facts["second_source_in_row_major"] = np.array_equal(
            at[from_source == 1], np.arange(h * w * c2)
        )
        facts["first_pixel"] = "".join("AB"[each] for each in beat_source[: (c1 + c2) // pe])
        # The rule: pixel by pixel, C1 / PE beats of the first source, then C2 / PE of the second.
        rule = np.tile(np.repeat([0, 1], [c1 // pe, c2 // pe]), h * w)
        facts["follows_the_rule"] = np.array_equal(beat_source, rule)
    return facts


@pytest.mark.parametrize(
    ("geometry", "first_pixel"),
    [((4, 4, 8, 4, 4), "AAB"), ((4, 4, 8, 4, 2), "AAAABB")],
    ids=["C8-4-PE4", "C8-4-PE2"],
)
def test_concat_along_the_channels_takes_whole_beats_from_each_source(
    geometry: tuple[int, int, int, int, int], first_pixel: str
) -> None:
    # (H, W, C1, C2, PE) with PE dividing both parts: each consumer beat reads one source,
    # each source in its own row-major order, and the beats follow the rule.
    assert concat_facts(*geometry) == {
        "each_beat_one_source": True,
        "first_source_in_row_major": True,
        "second_source_in_row_major": True,
        "first_pixel": first_pixel,
        "follows_the_rule": True,
    }


def test_concat_where_pe_divides_neither_part_has_beats_of_both_sources() -> None:
    # (4, 4, 6, 2, 4): the first pixel's second beat reads the first source's last two
    # channels and the second source's two.
    assert concat_facts(4, 4, 6, 2, 4) == {"each_beat_one_source": False}
