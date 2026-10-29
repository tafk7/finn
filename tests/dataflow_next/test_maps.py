# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Position maps: the type, its constructors against NumPy, and composition against the
enumeration.

Every constructor is checked by applying its map to source tensors of distinct values
and comparing with NumPy's own operation (or an explicit loop). ``through`` and
``chain`` are checked position by position against the maps applied in turn.
"""

from __future__ import annotations

import pytest

import numpy as np
import numpy.typing as npt
import random
from collections.abc import Callable
from itertools import permutations, product
from math import prod
from qonnx.custom_op.general.im2col import im2col_indices_nchw

import finn.dataflow_next.maps as maps_module
from finn.dataflow_next.maps import (
    DATA_DEPENDENT_READ,
    MAX_PIECES,
    Drop,
    Insertion,
    Piece,
    PositionMap,
    Unmappable,
    broadcast,
    chain,
    concat,
    depth_to_space,
    factor_drop,
    factor_fills,
    identity,
    im2col,
    pad,
    padded_im2col,
    reads,
    reshape,
    sliced,
    squeeze,
    through,
    transpose,
)
from finn.dataflow_next.schedule import Index, Schedule
from finn.dataflow_next.traversal import Loop, Traversal, tile, vector_major

Array = npt.NDArray[np.int64]


def sources_of(pmap: PositionMap) -> list[Array]:
    """A tensor of distinct values for each source: source ``i`` holds ``1000 * (i + 1)`` on."""
    return [
        np.arange(prod(shape), dtype=np.int64).reshape(shape) + 1000 * (index + 1)
        for index, shape in enumerate(pmap.sources)
    ]


def apply(pmap: PositionMap, tensors: list[Array]) -> Array:
    """The tensor ``pmap`` reads from ``tensors``, by its enumeration."""
    source, offset = reads(pmap)
    out = np.full(prod(pmap.shape), -(10**9), dtype=np.int64)
    if pmap.fill is not None:
        out[source < 0] = pmap.fill
    for index, tensor in enumerate(tensors):
        at = source == index
        out[at] = tensor.ravel()[offset[at]]
    assert (source >= 0).all() or pmap.fill is not None
    return out.reshape(pmap.shape)


def refused(code: str, build: Callable[[], object]) -> Unmappable:
    with pytest.raises(Unmappable) as info:
        build()
    assert isinstance(info.value, ValueError)
    assert [finding.code for finding in info.value.findings] == [code]
    return info.value


def whole(shape: tuple[int, ...], source: tuple[int, ...], modes: tuple[Loop, ...]) -> PositionMap:
    return PositionMap(shape, (source,), (Piece((0,) * len(shape), shape, 0, 0, modes),))


# --- the type --------------------------------------------------------------------------


def test_a_piece_keeps_its_modes_canonical() -> None:
    # A (2, 3) box read row-major from a 6-element source: two modes that merge into one.
    split = Piece((0, 0), (2, 3), 0, 0, (Loop(2, 3), Loop(3, 1), Loop(1, 7)))
    assert split.modes == (Loop(6, 1),)
    assert split == Piece((0, 0), (2, 3), 0, 0, (Loop(6, 1),))
    assert split.size == 6 and split.reach == 5
    assert split.body().tolist() == [0, 1, 2, 3, 4, 5]


def test_a_map_orders_its_pieces_and_compares_by_value() -> None:
    first = Piece((0, 0), (2, 2), 0, 0, (Loop(4, 1),))
    second = Piece((0, 2), (2, 3), 1, 0, (Loop(6, 1),))
    one = PositionMap((2, 5), ((2, 2), (2, 3)), (second, first))
    other = PositionMap((2, 5), ((2, 2), (2, 3)), (first, second))
    assert one == other and hash(one) == hash(other)
    assert one.pieces == (first, second)
    assert one == concat(((2, 2), (2, 3)), axis=1)
    assert identity((2, 3)).strided and not one.strided


def test_every_spelling_of_one_map_is_equal() -> None:
    # A (2, 4) reader of an 8-element source, row-major: one piece with merged modes.
    plain = reshape((8,), (2, 4))
    spellings = [
        # unit modes, and modes that merge (4 at stride 1 inside 2 at stride 4)
        whole((2, 4), (8,), (Loop(1, 5), Loop(2, 4), Loop(4, 1), Loop(1, 0))),
        # a fill stated although the pieces cover every position
        PositionMap((2, 4), ((8,),), plain.pieces, 3),
        # two column boxes of one source, in either order, together one strided box
        PositionMap(
            (2, 4),
            ((8,),),
            (
                Piece((0, 2), (2, 2), 0, 2, (Loop(2, 4), Loop(2, 1))),
                Piece((0, 0), (2, 2), 0, 0, (Loop(2, 4), Loop(2, 1))),
            ),
        ),
        # four row halves
        PositionMap(
            (2, 4),
            ((8,),),
            tuple(
                Piece((row, half * 2), (1, 2), 0, row * 4 + half * 2, (Loop(2, 1),))
                for row in (1, 0)
                for half in (1, 0)
            ),
        ),
    ]
    for spelling in spellings:
        assert spelling == plain and hash(spelling) == hash(plain)
        assert spelling.pieces == plain.pieces and spelling.fill is None
    # A Pad's border stated as many pieces is the one-piece Pad.
    padded = pad((2, 2), (1, 1, 1, 1), constant=4)
    rows = PositionMap(
        (4, 4),
        ((2, 2),),
        tuple(Piece((1 + row, 1), (1, 2), 0, 2 * row, (Loop(2, 1),)) for row in (0, 1)),
        4,
    )
    assert rows == padded and hash(rows) == hash(padded)
    assert padded.pieces == (Piece((1, 1), (2, 2), 0, 0, (Loop(4, 1),)),)


def test_cutting_a_map_into_boxes_changes_nothing() -> None:
    """Random maps, restated as pieces over random boxes (each box's reads factored from
    the enumeration), equal themselves."""
    rng = random.Random(5)
    maps = [
        transpose((3, 4, 2), (2, 0, 1)),
        pad((4, 5), (1, 2, 0, 1), constant=1),
        concat(((2, 3), (2, 2), (2, 1)), axis=1),
        padded_im2col((1, 4, 4, 2), (2, 2), pads=(1, 0, 1, 1))[0],
        chain(*padded_im2col((1, 5, 4, 1), (3, 2), pads=(1, 1, 1, 1))),
        sliced((6, 5), (1, 0), (6, 5), None, (2, 2)),
    ]
    restated = 0
    for pmap in maps:
        source, offset = (array.reshape(pmap.shape) for array in reads(pmap))
        for _ in range(10):
            bounds = [
                sorted({0, extent, *rng.sample(range(1, extent), min(2, extent - 1))})
                for extent in pmap.shape
            ]
            pieces = []
            for box in product(*([(a, b) for a, b in zip(cut, cut[1:])] for cut in bounds)):
                region = tuple(slice(a, b) for a, b in box)
                which = int(source[region].flat[0])
                if which < 0 or (source[region] != which).any():
                    continue
                strided = maps_module.strided_modes(offset[region].ravel())
                if strided is None:
                    break
                start = tuple(a for a, _ in box)
                extent = tuple(b - a for a, b in box)
                pieces.append(Piece(start, extent, which, *strided))
            else:
                if sum(piece.size for piece in pieces) != int((source >= 0).sum()):
                    continue  # a box mixed a source with the fill or another source
                again = PositionMap(pmap.shape, pmap.sources, tuple(pieces), pmap.fill)
                assert again == pmap and hash(again) == hash(pmap)
                restated += 1
    assert restated >= 20


def test_different_reads_are_different_maps() -> None:
    assert transpose((2, 2), (1, 0)) != identity((2, 2))
    assert pad((2,), (1, 0), 0) != pad((2,), (1, 0), 1)
    assert pad((2,), (1, 0), 0) != pad((2,), (0, 1), 0)


def test_reads_enumerates_every_position() -> None:
    # Concat of (2,) and (3,): a piece a source. A (2,) tensor padded by 1 and 2 with 7.
    joined = concat(((2,), (3,)), axis=0)
    assert reads(joined)[0].tolist() == [0, 0, 1, 1, 1]
    assert reads(joined)[1].tolist() == [0, 1, 0, 1, 2]
    padded = pad((2,), (1, 2), constant=7)
    assert padded.fill == 7
    assert reads(padded)[0].tolist() == [-1, 0, 0, -1, -1]
    assert apply(padded, sources_of(padded)).tolist() == [7, 1000, 1001, 7, 7]


@pytest.mark.parametrize(
    ("code", "build"),
    [
        ("box-invalid", lambda: Piece((0,), (2, 2), 0, 0, (Loop(4, 1),))),
        ("box-invalid", lambda: Piece((-1,), (2,), 0, 0, (Loop(2, 1),))),
        ("piece-invalid", lambda: Piece((0,), (2,), 0, -1, (Loop(2, 1),))),
        ("modes-mismatch", lambda: Piece((0,), (4,), 0, 0, (Loop(3, 1),))),
        ("shape-invalid", lambda: PositionMap((0,), ((1,),), ())),
        (
            "piece-outside-reader",
            lambda: whole((2,), (4,), (Loop(2, 1),)).__class__(
                (2,), ((4,),), (Piece((1,), (2,), 0, 0, (Loop(2, 1),)),)
            ),
        ),
        (
            "source-unknown",
            lambda: PositionMap((2,), ((2,),), (Piece((0,), (2,), 1, 0, (Loop(2, 1),)),)),
        ),
        ("read-outside-source", lambda: whole((3,), (4,), (Loop(3, 2),))),
        (
            "pieces-overlap",
            lambda: PositionMap(
                (4,),
                ((4,),),
                (Piece((0,), (3,), 0, 0, (Loop(3, 1),)), Piece((2,), (2,), 0, 0, (Loop(2, 1),))),
            ),
        ),
        (
            "fill-missing",
            lambda: PositionMap((3,), ((2,),), (Piece((0,), (2,), 0, 0, (Loop(2, 1),)),)),
        ),
        ("fill-not-integral", lambda: PositionMap((3,), ((2,),), (), 1.5)),  # type: ignore[arg-type]
    ],
)
def test_an_invalid_map_is_refused_with_its_code(code: str, build: Callable[[], object]) -> None:
    refused(code, build)


def test_the_data_dependent_read_code_is_named() -> None:
    assert DATA_DEPENDENT_READ == "data-dependent-read"
    assert maps_module.__doc__ is not None and "data-dependent-read" in maps_module.__doc__


# --- constructors against NumPy --------------------------------------------------------


@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 8, 4, 4), (3, 1, 2, 5)])
def test_transpose_is_numpys_for_every_permutation(shape: tuple[int, ...]) -> None:
    x = np.arange(prod(shape), dtype=np.int64).reshape(shape) + 1000
    for perm in permutations(range(len(shape))):
        assert np.array_equal(apply(transpose(shape, perm), [x]), np.transpose(x, perm))
    assert np.array_equal(apply(transpose(shape), [x]), np.transpose(x))
    refused("perm-invalid", lambda: transpose(shape, (0, 0, 1)))


@pytest.mark.parametrize(
    ("source", "shape"), [((2, 3, 4), (6, 4)), ((4, 6), (2, 2, 6)), ((5,), (5, 1))]
)
def test_reshape_is_numpys(source: tuple[int, ...], shape: tuple[int, ...]) -> None:
    x = np.arange(prod(source), dtype=np.int64).reshape(source) + 1000
    assert np.array_equal(apply(reshape(source, shape), [x]), x.reshape(shape))
    assert reshape(source, shape).strided
    refused("size-mismatch", lambda: reshape(source, (prod(source) + 1,)))


def test_squeeze_is_numpys() -> None:
    shape = (1, 3, 1, 4)
    x = np.arange(12, dtype=np.int64).reshape(shape) + 1000
    assert np.array_equal(apply(squeeze(shape), [x]), np.squeeze(x))
    assert np.array_equal(apply(squeeze(shape, (0,)), [x]), np.squeeze(x, 0))
    assert np.array_equal(apply(squeeze(shape, (-2,)), [x]), np.squeeze(x, -2))
    assert squeeze(shape).shape == (3, 4)
    refused("squeeze-not-unit", lambda: squeeze(shape, (1,)))
    refused("axis-invalid", lambda: squeeze(shape, (4,)))


@pytest.mark.parametrize(
    ("source", "shape", "reader"),
    [
        ((3,), (2, 4, 3), (2, 4, 3)),
        ((4, 1, 3), (4, 5, 3), (4, 5, 3)),
        ((1, 3), (2, 1), (2, 3)),  # Expand broadcasts both ways
        ((), (2, 3), (2, 3)),
    ],
)
def test_broadcast_is_numpys(
    source: tuple[int, ...], shape: tuple[int, ...], reader: tuple[int, ...]
) -> None:
    x = np.arange(prod(source), dtype=np.int64).reshape(source) + 1000
    pmap = broadcast(source, shape)
    assert pmap.shape == reader
    assert np.array_equal(apply(pmap, [x]), np.broadcast_to(x, reader))


def test_broadcast_refuses_incompatible_extents() -> None:
    refused("broadcast-incompatible", lambda: broadcast((3,), (2, 4)))


def explicit_im2col(
    x: Array, kernel: tuple[int, int], stride: tuple[int, int], dilation: tuple[int, int]
) -> Array:
    """qonnx's Im2Col, unpadded, by loops: out[b, oh, ow, (kh * KW + kw) * C + c]."""
    batch, h, w, c = x.shape
    (kh_, kw_), (sh, sw), (dh, dw) = kernel, stride, dilation
    oh_, ow_ = (h - dh * (kh_ - 1) - 1) // sh + 1, (w - dw * (kw_ - 1) - 1) // sw + 1
    out = np.zeros((batch, oh_, ow_, kh_ * kw_ * c), dtype=np.int64)
    for b, oh, ow, kh, kw, ch in product(*map(range, (batch, oh_, ow_, kh_, kw_, c))):
        out[b, oh, ow, (kh * kw_ + kw) * c + ch] = x[b, oh * sh + kh * dh, ow * sw + kw * dw, ch]
    return out


def qonnx_im2col(
    x: Array,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
    pads: tuple[int, int, int, int],
    pad_value: int,
) -> Array:
    """qonnx's Im2Col as its node executes it: NHWC to NCHW, its index routine, back."""
    batch, h, w, c = x.shape
    (kh, kw), (sh, sw), (dh, dw) = kernel, stride, dilation
    oh = (h + pads[0] + pads[2] - dh * (kh - 1) - 1) // sh + 1
    ow = (w + pads[1] + pads[3] - dw * (kw - 1) - 1) // sw + 1
    cols = im2col_indices_nchw(  # type: ignore[no-untyped-call]
        x.transpose(0, 3, 1, 2), h, w, kh, kw, list(pads), sh, sw, pad_value, dh, dw
    )
    out = cols.reshape(batch, c, kh, kw, oh, ow).transpose(0, 4, 5, 2, 3, 1)
    return np.asarray(out.reshape(batch, oh, ow, kh * kw * c), dtype=np.int64)


@pytest.mark.parametrize(
    ("image", "kernel", "stride", "dilation"),
    [
        ((1, 6, 6, 4), (3, 3), (1, 1), (1, 1)),
        ((1, 9, 7, 3), (3, 2), (2, 1), (1, 1)),
        ((2, 8, 8, 2), (2, 3), (1, 2), (2, 1)),
        ((1, 32, 32, 3), (3, 3), (1, 1), (1, 1)),
    ],
)
def test_im2col_is_qonnxs(
    image: tuple[int, int, int, int],
    kernel: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
) -> None:
    x = np.arange(prod(image), dtype=np.int64).reshape(image) + 1000
    found = apply(im2col(image, kernel, stride, dilation), [x])
    assert np.array_equal(found, explicit_im2col(x, kernel, stride, dilation))
    if image[0] == 1:  # qonnx's routine folds the batch into the patch
        assert np.array_equal(found, qonnx_im2col(x, kernel, stride, dilation, (0, 0, 0, 0), 0))


@pytest.mark.parametrize(
    ("image", "kernel", "stride", "pads", "pad_value"),
    [
        ((1, 8, 8, 4), (3, 3), (1, 1), (1, 1, 1, 1), 0),
        ((1, 9, 7, 3), (3, 3), (2, 2), (1, 1, 1, 1), -1),
        ((1, 6, 5, 2), (3, 2), (1, 2), (2, 0, 1, 1), 3),
    ],
)
def test_a_padded_im2col_is_pad_then_the_window(
    image: tuple[int, int, int, int],
    kernel: tuple[int, int],
    stride: tuple[int, int],
    pads: tuple[int, int, int, int],
    pad_value: int,
) -> None:
    x = np.arange(prod(image), dtype=np.int64).reshape(image) + 1000
    padding, window = padded_im2col(image, kernel, stride, (1, 1), pads, pad_value)
    in_turn = apply(window, [apply(padding, [x])])
    assert np.array_equal(in_turn, qonnx_im2col(x, kernel, stride, (1, 1), pads, pad_value))
    refused("pads-negative", lambda: padded_im2col(image, kernel, pads=(0, -1, 0, 0)))


def test_im2col_refuses_what_it_cannot_read() -> None:
    refused("rank-mismatch", lambda: im2col((6, 6, 4), (3, 3)))
    refused("window-larger-than-image", lambda: im2col((1, 2, 6, 4), (3, 3)))
    refused("window-invalid", lambda: im2col((1, 6, 6, 4), (3, 0)))


def depth_to_space_by_loops(x: Array, block: int, mode: str) -> Array:
    """ONNX DepthToSpace on NCHW, element by element: output (n, c', h * b + i, w * b + j)
    reads input channel (i * b + j) * C' + c' (DCR) or c' * b * b + i * b + j (CRD)."""
    batch, c, h, w = x.shape
    depth = c // (block * block)
    out = np.zeros((batch, depth, h * block, w * block), dtype=np.int64)
    for n, d, y, z, i, j in product(*map(range, (batch, depth, h, w, block, block))):
        channel = (
            (i * block + j) * depth + d if mode == "DCR" else d * block * block + i * block + j
        )
        out[n, d, y * block + i, z * block + j] = x[n, channel, y, z]
    return out


@pytest.mark.parametrize("mode", ["DCR", "CRD"])
@pytest.mark.parametrize(
    ("shape", "block"), [((1, 8, 4, 6), 2), ((2, 18, 3, 3), 3), ((1, 4, 1, 2), 2)]
)
def test_depth_to_space_is_onnxs(shape: tuple[int, int, int, int], block: int, mode: str) -> None:
    x = np.arange(prod(shape), dtype=np.int64).reshape(shape) + 1000
    expected = depth_to_space_by_loops(x, block, mode)
    pmap = depth_to_space(shape, block, mode)
    assert pmap.strided
    assert np.array_equal(apply(pmap, [x]), expected)
    # Channels last: the same, on the image transposed to NHWC and back.
    nhwc = (shape[0], shape[2], shape[3], shape[1])
    last = depth_to_space(nhwc, block, mode, channels_last=True)
    assert np.array_equal(apply(last, [x.transpose(0, 2, 3, 1)]), expected.transpose(0, 2, 3, 1))


def test_depth_to_space_refuses_what_it_cannot_split() -> None:
    refused("depth-indivisible", lambda: depth_to_space((1, 6, 2, 2), 2))
    refused("mode-unsupported", lambda: depth_to_space((1, 8, 2, 2), 2, "RDC"))
    refused("rank-mismatch", lambda: depth_to_space((8, 2, 2), 2))


def numpy_pad(x: Array, pads: tuple[int, ...], constant: int) -> Array:
    """ONNX Pad, constant mode: np.pad's growth, then the negative amounts cropped."""
    rank = x.ndim
    begins, ends = pads[:rank], pads[rank:]
    grown = np.pad(
        x, [(max(b, 0), max(e, 0)) for b, e in zip(begins, ends)], constant_values=constant
    )
    crop = tuple(
        slice(max(-b, 0), grown.shape[axis] - max(-e, 0))
        for axis, (b, e) in enumerate(zip(begins, ends))
    )
    return np.asarray(grown[crop], dtype=np.int64)


@pytest.mark.parametrize(
    ("shape", "pads", "constant"),
    [
        ((8, 8, 4), (1, 1, 0, 1, 1, 0), 0),
        ((3, 5), (2, 0, 1, 3), -4),
        ((4, 6), (-1, 2, 1, -2), 9),  # a crop on one side, growth on the other
        ((3, 3), (0, 0, 0, 0), 5),  # nothing inserted: no fill
        ((2, 4), (-2, 0, 1, 0), 1),  # all cropped: every position the fill
    ],
)
def test_pad_is_numpys(shape: tuple[int, ...], pads: tuple[int, ...], constant: int) -> None:
    x = np.arange(prod(shape), dtype=np.int64).reshape(shape) + 1000
    pmap = pad(shape, pads, constant)
    assert np.array_equal(apply(pmap, [x]), numpy_pad(x, pads, constant))
    assert (pmap.fill is None) == all(amount <= 0 for amount in pads)


def test_a_negative_pad_is_its_slice() -> None:
    # Cropping one row off the top and two columns off the right of a (4, 6) tensor.
    assert pad((4, 6), (-1, 0, 0, -2)) == sliced((4, 6), (1, 0), (4, 4))
    assert pad((4, 6), (-1, 0, 0, -2), constant=7) == sliced((4, 6), (1, 0), (4, 4))
    # A crop on one side and growth on the other: the slice, then the growth.
    grown = pad((4, 6), (-1, 2, 1, -2), constant=9)
    assert grown == chain(sliced((4, 6), (1, 0), (4, 4)), pad((3, 4), (0, 2, 1, 0), 9))
    assert grown.fill == 9


def test_pad_takes_an_integral_constant_only() -> None:
    assert pad((2,), (1, 0), 2.0).fill == 2
    assert pad((2,), (1, 0), -3).fill == -3
    refused("fill-not-integral", lambda: pad((2,), (1, 0), 0.5))
    refused("fill-not-integral", lambda: pad((2,), (1, 0), float("nan")))
    refused("pad-mode-unsupported", lambda: pad((2,), (1, 0), mode="reflect"))
    refused("pads-invalid", lambda: pad((2, 2), (1, 0)))


@pytest.mark.parametrize(
    ("shape", "starts", "ends", "axes", "steps", "expected"),
    [
        ((8, 6), (1, 0), (7, 6), None, (2, 1), (slice(1, 7, 2), slice(0, 6))),
        ((8, 6), (-3,), (1000,), (0,), None, (slice(5, 8), slice(None))),
        ((4, 5, 6), (1, 2), (-1, 6), (-1, 0), (3, 1), (slice(2, 4), slice(None), slice(1, 5, 3))),
    ],
)
def test_slice_is_numpys(
    shape: tuple[int, ...],
    starts: tuple[int, ...],
    ends: tuple[int, ...],
    axes: tuple[int, ...] | None,
    steps: tuple[int, ...] | None,
    expected: tuple[slice, ...],
) -> None:
    x = np.arange(prod(shape), dtype=np.int64).reshape(shape) + 1000
    pmap = sliced(shape, starts, ends, axes, steps)
    assert np.array_equal(apply(pmap, [x]), x[expected])
    assert len(pmap.pieces) == 1 and pmap.fill is None


def test_slice_refuses_steps_below_one_and_empty_results() -> None:
    refused("slice-step-unsupported", lambda: sliced((8,), (7,), (0,), None, (-1,)))
    refused("slice-empty", lambda: sliced((8,), (5,), (2,)))


@pytest.mark.parametrize(
    ("shapes", "axis"),
    [
        (((4, 4, 8), (4, 4, 4)), -1),
        (((2, 3), (1, 3), (4, 3)), 0),
        (((2, 1, 5), (2, 3, 5)), 1),
    ],
)
def test_concat_is_numpys_one_piece_a_source(
    shapes: tuple[tuple[int, ...], ...], axis: int
) -> None:
    pmap = concat(shapes, axis)
    tensors = sources_of(pmap)
    assert np.array_equal(apply(pmap, tensors), np.concatenate(tensors, axis=axis))
    assert [piece.source for piece in pmap.pieces] == list(range(len(shapes)))
    refused("concat-mismatch", lambda: concat(((2, 3), (2, 4)), 0))


# --- through ---------------------------------------------------------------------------


def check_through(pmap: PositionMap, form: Traversal) -> Traversal:
    """``through``, checked position by position: each beat and lane of the result is the
    source position the map reads at ``form``'s position there."""
    composed = through(pmap, form)
    assert composed.beats == form.beats and composed.lanes == form.lanes
    source, offset = reads(pmap)
    for beat, lane in product(range(form.beats), range(form.lanes)):
        flat = int(np.ravel_multi_index(form.position(beat, lane), form.shape))
        assert source[flat] == 0
        wanted = np.unravel_index(int(offset[flat]), composed.shape)
        assert composed.position(beat, lane) == tuple(int(i) for i in wanted)
    return composed


def test_through_the_identity_changes_nothing() -> None:
    form = tile(6, 8, 2, 4)
    assert through(identity((6, 8)), form) == form


def test_through_a_transpose_is_the_transposed_walk() -> None:
    # The (6, 4) transpose of a (4, 6) tensor, read row-major at 2 lanes: the source is
    # walked column by column, two rows a beat.
    composed = check_through(transpose((4, 6), (1, 0)), vector_major((6, 4), 2))
    assert composed == Traversal((4, 6), (Loop(6, 1), Loop(2, 12)), (Loop(2, 6),))


def test_through_im2col_is_the_windowed_schedule() -> None:
    # A plain MatMul (m, n, k) = (16, 2, 36) at SIMD 4 reading a 3x3 window of a (6, 6, 4)
    # image through Im2Col equals the schedule that walks the window itself.
    m, n, k = Index("m"), Index("n"), Index("k")
    oh, ow, kh, kw, c = (Index(name) for name in ("oh", "ow", "kh", "kw", "c"))
    window = im2col((1, 6, 6, 4), (3, 3))
    flat = chain(window, reshape(window.shape, (16, 36)))
    plain = Schedule({m: 16, n: 2, k: 36}, {k: 4}, (m, n, k)).present((16, 36), (m, k), lanes=(k,))
    windowed = Schedule(
        {oh: 4, ow: 4, n: 2, kh: 3, kw: 3, c: 4}, {c: 4}, (oh, ow, n, kh, kw, c)
    ).present((6, 6, 4), (oh + kh, ow + kw, c), lanes=(c,))
    composed = check_through(flat, plain)
    assert composed.shape == (1, 6, 6, 4)
    assert (composed.beat_loops, composed.lane_loops) == (
        windowed.beat_loops,
        windowed.lane_loops,
    )


def test_through_random_transposes_and_traversals() -> None:
    rng = random.Random(7)
    composed = 0
    for _ in range(60):
        shape = tuple(rng.choice((1, 2, 3, 4)) for _ in range(rng.choice((2, 3))))
        perm = list(range(len(shape)))
        rng.shuffle(perm)
        pmap = transpose(shape, perm)
        lanes = rng.choice([d for d in range(1, pmap.shape[-1] + 1) if pmap.shape[-1] % d == 0])
        form = vector_major(pmap.shape, lanes).replayed(rng.choice((1, 2)), inner_beats=1)
        try:
            check_through(pmap, form)
            composed += 1
        except Unmappable as error:
            assert error.findings[0].code in ("loop-straddles-mode", "carry-between-loops")
    assert composed >= 40


def test_through_refuses_a_carry_between_loops() -> None:
    # Reader (2, 2, 2), source (2, 2, 2) with its last two axes swapped: a beat loop and a
    # lane loop both step the reader's flat offset by 1 and present 0, 1, 1, 2. Each maps
    # to a source step of 2, but reader offset 2 is source offset 1: their sum carries.
    form = Traversal((2, 2, 2), (Loop(2, 1),), (Loop(2, 1),))
    error = refused("carry-between-loops", lambda: through(transpose((2, 2, 2), (0, 2, 1)), form))
    assert "no loop nest" in str(error)


def test_through_refuses_a_loop_straddling_a_mode() -> None:
    # Reader (3, 4), the transpose of (4, 3): stepping the reader by 3 crosses rows of 4.
    form = Traversal((3, 4), (Loop(4, 3),), ())
    refused("loop-straddles-mode", lambda: through(transpose((4, 3), (1, 0)), form))


def test_through_refuses_what_is_no_strided_map() -> None:
    refused("not-one-piece", lambda: through(pad((4,), (1, 1)), vector_major((6,), 1)))
    refused("not-one-piece", lambda: through(concat(((2,), (2,)), 0), vector_major((4,), 1)))
    refused("map-offset", lambda: through(sliced((6,), (2,), (6,)), vector_major((4,), 1)))
    refused("shape-mismatch", lambda: through(identity((6,)), vector_major((2, 3), 1)))
    refused("source-rank-zero", lambda: through(broadcast((), (4,)), vector_major((4,), 1)))


# --- chain -----------------------------------------------------------------------------


def check_chain(outer: PositionMap, inner: PositionMap, source: int = 0) -> PositionMap:
    """``chain``, checked against the two maps applied in turn to distinct tensors."""
    composed = chain(outer, inner, source)
    tensors = sources_of(composed)
    outer_tensors = tensors[source : source + len(outer.sources)]
    inner_tensors = [
        *tensors[:source],
        apply(outer, outer_tensors),
        *tensors[source + len(outer.sources) :],
    ]
    assert np.array_equal(apply(composed, tensors), apply(inner, inner_tensors))
    return composed


@pytest.mark.parametrize(
    ("image", "block", "modes"),
    [
        ((4, 6, 8), 2, ((4, 48), (2, 4), (6, 8), (4, 1))),
        ((3, 3, 18), 3, ((3, 54), (3, 6), (3, 18), (6, 1))),
    ],
)
def test_depth_to_space_is_its_three_steps_as_one_map(
    image: tuple[int, int, int], block: int, modes: tuple[tuple[int, int], ...]
) -> None:
    # DepthToSpace (DCR, channels last) as Reshape (H, W, b, b, C'), Transpose to
    # (H, b, W, b, C') and Reshape (H * b, W * b, C'): one strided map, stated by its
    # canonical modes (contiguous modes merged).
    h, w, c = image
    depth = c // (block * block)
    split = reshape(image, (h, w, block, block, depth))
    swap = transpose(split.shape, (0, 2, 1, 3, 4))
    merge = reshape(swap.shape, (h * block, w * block, depth))
    one = check_chain(split, check_chain(swap, merge))
    assert one == depth_to_space(image, block, channels_last=True)
    assert tuple((mode.extent, mode.stride) for mode in one.pieces[0].modes) == modes
    x = sources_of(one)[0]
    assert np.array_equal(apply(one, [x]), apply(merge, [apply(swap, [apply(split, [x])])]))


@pytest.mark.parametrize("shape", [(1, 8, 4, 4), (1, 3, 5, 7)])
def test_inverse_transposes_compose_to_the_identity(shape: tuple[int, int, int, int]) -> None:
    # NCHW to NHWC, then back.
    n, c, h, w = shape
    to_nhwc = transpose(shape, (0, 2, 3, 1))
    to_nchw = transpose((n, h, w, c), (0, 3, 1, 2))
    assert check_chain(to_nhwc, to_nchw) == identity(shape)
    form = vector_major(shape, w)
    assert through(chain(to_nhwc, to_nchw), form) == form


def test_chain_of_random_strided_maps_is_the_maps_in_turn() -> None:
    rng = random.Random(11)
    for _ in range(80):
        shape = tuple(rng.choice((1, 2, 3, 4)) for _ in range(rng.choice((2, 3))))
        perm = list(range(len(shape)))
        rng.shuffle(perm)
        first = transpose(shape, perm)
        choice = rng.choice(("transpose", "reshape", "slice", "broadcast"))
        if choice == "transpose":
            perm = list(range(len(shape)))
            rng.shuffle(perm)
            second = transpose(first.shape, perm)
        elif choice == "reshape":
            second = reshape(first.shape, (prod(first.shape),))
        elif choice == "slice":
            axis = rng.randrange(len(shape))
            second = sliced(
                first.shape, (rng.randrange(first.shape[axis]),), (1000,), (axis,), (2,)
            )
        else:
            second = broadcast(first.shape, (2, *first.shape))
        try:
            check_chain(first, second)
        except Unmappable as error:
            assert error.findings[0].code == "not-strided"


def test_chain_through_a_fill_cuts_the_reader_into_pieces() -> None:
    # A 3x3 window over an (8, 8, 4) image padded by one with the constant 5: the window's
    # reader is cut into boxes that each read the image or the fill.
    padding, window = padded_im2col((1, 8, 8, 4), (3, 3), pads=(1, 1, 1, 1), pad_value=5)
    composed = check_chain(padding, window)
    assert composed.fill == 5 and composed.sources == ((1, 8, 8, 4),)
    assert len(composed.pieces) > 1


def test_chain_reads_one_source_of_many() -> None:
    # Concat of A (2, 3) and the transpose of B (3, 2) on rows: B's transpose composed in.
    joined = concat(((2, 3), (2, 3)), 0)
    composed = check_chain(transpose((3, 2), (1, 0)), joined, source=1)
    assert composed.sources == ((2, 3), (3, 2))
    composed = check_chain(concat(((1, 3), (1, 3)), 0), joined, source=0)
    assert composed.sources == ((1, 3), (1, 3), (2, 3))


def test_chain_refuses_two_fills_and_mismatched_maps() -> None:
    once = pad((2,), (1, 1), constant=2)
    refused("fills-differ", lambda: chain(once, pad((4,), (1, 0), constant=1)))
    assert check_chain(once, pad((4,), (1, 0), constant=2)).fill == 2
    refused("shape-mismatch", lambda: chain(identity((2, 3)), identity((3, 2))))
    refused("source-unknown", lambda: chain(identity((2, 3)), identity((2, 3)), source=1))


def test_a_chain_in_no_strided_order_is_cut_into_pieces() -> None:
    # A (2, 3) tensor transposed and flattened reads offsets 0, 3, 1, 4, 2, 5; its middle
    # four, 3, 1, 4, 2, are in no strided order: one piece a position, exact, and no
    # traversal composes through it.
    flat = chain(transpose((2, 3), (1, 0)), reshape((3, 2), (6,)))
    middle = check_chain(flat, sliced((6,), (1,), (5,)))
    assert [piece.base for piece in middle.pieces] == [3, 1, 4, 2]
    refused("not-one-piece", lambda: through(middle, vector_major((4,), 1)))


def test_one_source_read_in_two_blocks_stays_two_pieces() -> None:
    # A (3, 4) tensor with its two column halves swapped: each half is strided, the whole
    # is not, so the cut falls where the step along a row breaks.
    halves = (
        Piece((0, 0), (3, 2), 0, 2, (Loop(3, 4), Loop(2, 1))),
        Piece((0, 2), (3, 2), 0, 0, (Loop(3, 4), Loop(2, 1))),
    )
    swapped = PositionMap((3, 4), ((3, 4),), halves)
    assert swapped.pieces == halves
    x = np.arange(12, dtype=np.int64).reshape(3, 4)
    assert np.array_equal(apply(swapped, [x]), np.concatenate((x[:, 2:], x[:, :2]), axis=1))


def test_cnv_conv1s_window_composes() -> None:
    # CNV's second 3x3 layer: a (30, 30, 64) image, read as a (784, 576) matrix.
    window = im2col((1, 30, 30, 64), (3, 3))
    flat = check_chain(window, reshape(window.shape, (784, 576)))
    assert flat.pieces[0].modes == (Loop(28, 1920), Loop(28, 64), Loop(3, 1920), Loop(192, 1))


# --- Pad's factorization ---------------------------------------------------------------


def stream_of(insertion: Insertion) -> Array:
    """What the insertion presents, in the padded tensor's row-major order: each source
    element, in its order, with each fill (-1) inserted where it falls."""
    filled = insertion.inserted()
    stream = np.full(filled.size, -1, dtype=np.int64)
    stream[~filled] = np.arange(int((~filled).sum()))
    return stream


@pytest.mark.parametrize(
    ("image", "pads", "stride", "inserted"),
    [
        ((1, 8, 8, 4), (1, 1, 1, 1), (1, 1), 144),
        ((1, 8, 8, 4), (2, 2, 2, 2), (1, 1), 320),
        ((1, 9, 7, 3), (1, 1, 1, 1), (2, 2), 108),
        ((1, 32, 32, 3), (1, 1, 1, 1), (1, 1), 396),
    ],
)
def test_a_padded_window_factors_into_an_insertion_and_a_window(
    image: tuple[int, int, int, int],
    pads: tuple[int, int, int, int],
    stride: tuple[int, int],
    inserted: int,
) -> None:
    padding, window = padded_im2col(image, (3, 3), stride, pads=pads)
    insertion, over_padded = factor_fills(padding, window)
    assert insertion.count == inserted and over_padded == window
    assert insertion.shape == window.sources[0]
    # The composite, by enumeration: each window position's source offset, or -1 (fill).
    source, offset = reads(padding)
    composite = np.where(source < 0, -1, offset)[reads(window)[1]]
    # The factorization: the window reading the insertion's stream.
    factored = stream_of(insertion)[reads(window)[1]]
    assert np.array_equal(composite, factored)


def test_an_insertion_out_of_order_is_refused() -> None:
    # Pad a (4, 4, 2) tensor's transpose: the covered positions read it column-major.
    swapped = chain(transpose((4, 4, 2), (1, 0, 2)), pad((4, 4, 2), (1, 1, 0, 1, 1, 0)))
    window = im2col((1, *swapped.shape), (3, 3))
    window = chain(reshape(swapped.shape, (1, *swapped.shape)), window)
    refused("insertion-out-of-order", lambda: factor_fills(swapped, window))
    refused("not-an-insertion", lambda: Insertion(concat(((2,), (2,)), 0)))
    padding = pad((4,), (1, 1))
    refused("shape-mismatch", lambda: factor_fills(padding, identity((4,))))
    refused("not-one-piece", lambda: factor_fills(padding, pad((6,), (1, 0))))


# --- Slice's factorization -------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "starts", "ends", "steps", "skipped"),
    [
        ((8, 6), (1, 0), (7, 6), (2, 3), 48 - 6),
        ((5, 6, 4), (0, 1, 1), (5, 6, 4), (2, 2, 2), 120 - 3 * 3 * 2),
    ],
)
def test_a_slice_factors_into_an_in_order_drop_and_a_map(
    shape: tuple[int, ...],
    starts: tuple[int, ...],
    ends: tuple[int, ...],
    steps: tuple[int, ...],
    skipped: int,
) -> None:
    kept = sliced(shape, starts, ends, None, steps)
    window = transpose(kept.shape)  # what remains, read transposed
    drop, over_kept = factor_drop(kept, window)
    assert drop.count == skipped and over_kept == window
    assert drop.shape == shape
    # The producer walks the source in increasing order, skipping the dropped positions:
    # what remains is the slice, in row-major order.
    x = np.arange(prod(shape), dtype=np.int64).reshape(shape)
    remains = x.ravel()[~drop.dropped()].reshape(kept.shape)
    region = tuple(slice(a, b, c) for a, b, c in zip(starts, ends, steps))
    assert np.array_equal(remains, x[region])
    assert np.array_equal(apply(window, [remains]), apply(chain(kept, window), [x]))


def test_a_drop_out_of_order_is_refused() -> None:
    # The slice of a transpose reads its source column by column.
    out_of_order = chain(transpose((4, 6), (1, 0)), sliced((6, 4), (0, 1), (6, 3)))
    refused("drop-out-of-order", lambda: factor_drop(out_of_order, identity(out_of_order.shape)))
    # A broadcast reads positions twice.
    refused("drop-out-of-order", lambda: Drop(broadcast((3,), (2, 3))))
    refused("not-a-drop", lambda: Drop(pad((3,), (1, 0))))
    kept = sliced((6,), (1,), (5,))
    refused("shape-mismatch", lambda: factor_drop(kept, identity((6,))))
    refused("map-offset", lambda: factor_drop(kept, sliced((4,), (1,), (4,))))


# --- the bound on a map's pieces --------------------------------------------------------


def interleaved(each: int) -> PositionMap:
    """A reader of ``2 * each`` positions that reads two sources of ``each`` element by
    element, alternating: one piece of one element a position."""
    both = concat([(each,), (each,)], 0)
    shuffle = chain(
        reshape((2 * each,), (2, each)),
        chain(transpose((2, each)), reshape((each, 2), (2 * each,))),
    )
    return chain(both, shuffle)


def test_a_map_at_the_bound_of_pieces_builds() -> None:
    found = interleaved(MAX_PIECES // 2)
    assert len(found.pieces) == MAX_PIECES
    source, offset = reads(found)
    assert source.tolist() == [0, 1] * (MAX_PIECES // 2)
    assert offset.tolist() == [at for at in range(MAX_PIECES // 2) for _ in range(2)]


def test_a_map_of_more_pieces_than_the_bound_is_refused() -> None:
    # One piece a position: 258 pieces, one over 256 and two over.
    error = refused("too-many-pieces", lambda: interleaved(MAX_PIECES // 2 + 1))
    assert str(MAX_PIECES) in str(error)
    assert error.findings[0].owner == "chain"
    # The composition stops at the bound, so it does not say how many pieces it would be.
    assert f"at least {MAX_PIECES + 1} pieces" in str(error)


def alternating(count: int) -> PositionMap:
    """``count`` positions built by hand, one piece each, alternating between two sources."""
    half = -(-count // 2)
    pieces = tuple(Piece((at,), (1,), at % 2, at // 2, (Loop(1, 1),)) for at in range(count))
    return PositionMap((count,), ((half,), (half,)), pieces)


def test_the_bound_counts_the_canonical_pieces_of_hand_built_ones() -> None:
    assert len(alternating(MAX_PIECES).pieces) == MAX_PIECES
    refused("too-many-pieces", lambda: alternating(MAX_PIECES + 1))
    # 300 one-position pieces of one run are one piece, cut anew: they are within the bound.
    run = tuple(Piece((at,), (1,), 0, at, (Loop(1, 1),)) for at in range(300))
    assert len(PositionMap((300,), ((300,),), run).pieces) == 1
