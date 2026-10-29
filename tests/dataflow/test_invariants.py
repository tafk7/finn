# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Invariants the package states but no constructor or other test holds.

Each is checked on values the package builds, drawn by a seeded generator (seed ``SEED``)
where the space is large, and on named cases where a refusal is the invariant:

* every traversal is ``regrouped(order, lanes)``: its loops, beats then lanes, cut again
  at its lane count give the traversal back;
* a form presents its pass ``passes(form)`` times a frame, whatever its replays and
  repetitions: ``passes * one_pass(form).beats == form.beats``;
* a port's beat times (``Schedule.beat_times``) increase strictly and lie below the
  schedule's ``iterations``;
* the value refusals of time: a source's words lie within its span, a kernel's ports
  move at iterations within its iterations, and its ports share one schedule;
* a strided position map and a traversal are one strided layout: a map composed through
  a reader walked once is the traversal of its source, and a traversal is the strided map
  from ``(beats * lanes,)`` that reads the offsets it presents;
* the constructors that refuse a value no test gave them: a tensor's rank and extents, an
  encoding's type, width and range, a loop's extent and stride, a traversal's rank, loop
  type and reach.
"""

from __future__ import annotations

import itertools
import random
from math import prod

import numpy as np
import pytest
from qonnx.core.ranges import TensorRange

from finn.dataflow import maps
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.schedule import Index, Pace, Schedule
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.timing import Kernel, Source, kernel
from finn.dataflow.traversal import (
    Grouped,
    Loop,
    Traversal,
    grouped,
    offsets,
    one_pass,
    passes,
    regrouped,
    vector_major,
)

SEED = 7
SAMPLE = 2_000

INT4 = ScalarEncoding(resolve_qonnx_datatype_name("INT4"))


# --- traversals -------------------------------------------------------------------------


def random_traversal(rng: random.Random) -> Traversal:
    """A traversal of a random tensor (rank one to three, extents one to four): its axes
    walked in a random order, a replay at a random depth, repeated, and cut into a random
    lane count where one closes whole loops."""
    rank = rng.choice((1, 2, 3))
    shape = tuple(rng.choice((1, 2, 3, 4)) for _ in range(rank))
    order = list(range(rank))
    rng.shuffle(order)
    form = Traversal.over(shape, [(axis, shape[axis], 1) for axis in order], ())
    if rng.random() < 0.5:
        inner = [prod(loop.extent for loop in form.beat_loops[at:]) for at in range(rank + 1)]
        form = form.replayed(rng.choice((2, 3)), inner_beats=rng.choice(inner))
    if rng.random() < 0.4:
        form = form.repeated(rng.choice((2, 3)))
    try:
        return regrouped(form, rng.choice((1, 2, 3, 4, 6)))
    except ValueError:
        return form


def traversals() -> list[Traversal]:
    rng = random.Random(SEED)
    return [random_traversal(rng) for _ in range(SAMPLE)]


def test_every_traversal_is_its_order_regrouped_at_its_lanes() -> None:
    """Beats then lanes make one loop nest (the traversal's order), and cutting it at the
    traversal's own lane count is the traversal; an aligned group is that traversal too."""
    sampled = traversals()
    assert any(form.lanes > 1 for form in sampled) and any(form.lanes == 1 for form in sampled)
    assert any(any(loop.stride == 0 for loop in form.beat_loops) for form in sampled)
    for form in sampled:
        order = Traversal(form.shape, (*form.beat_loops, *form.lane_loops), ())
        assert regrouped(order, form.lanes) == form, form
        assert grouped(order, form.lanes) == form, form
        assert regrouped(form, form.lanes) == form, form


def test_a_form_presents_its_pass_a_whole_number_of_times() -> None:
    """``passes * one_pass(form).beats == form.beats``, for a traversal with repetition and
    replay, and for the lane group of its order at lanes that cut it."""
    sampled = traversals()
    counts = set()
    for form in sampled:
        assert passes(form) * one_pass(form).beats == form.beats, form
        counts.add(passes(form))
        for lanes in (2, 3, 4):
            try:
                group = grouped(
                    Traversal(form.shape, (*form.beat_loops, *form.lane_loops), ()), lanes
                )
            except ValueError:
                continue
            assert passes(group) * one_pass(group).beats == group.beats, group
    assert {1, 2, 3} <= counts
    assert passes(vector_major((2, 3), 1).repeated(3)) == 3
    assert passes(vector_major((2, 3), 1).replayed(3, inner_beats=1)) == 1
    # A lane group of lanes that cut its rows: its pass is its order's, at the same lanes.
    cut = grouped(Traversal((4, 6), (Loop(6, 1), Loop(4, 6)), ()), 3)
    assert isinstance(cut, Grouped) and passes(cut) == 1 and one_pass(cut) == cut
    repeated = cut.repeated(2)
    assert passes(repeated) == 2 and one_pass(repeated) == cut


# --- beat times -------------------------------------------------------------------------


def random_schedule(rng: random.Random) -> Schedule:
    indices = [Index(name) for name in "abcd"[: rng.choice((1, 2, 3, 4))]]
    extents = {index: rng.choice((1, 2, 3, 4, 6)) for index in indices}
    factors = {
        index: rng.choice([d for d in (1, 2, 3) if extents[index] % d == 0]) for index in indices
    }
    order = list(indices)
    rng.shuffle(order)
    return Schedule(extents, factors, order)


def test_a_port_s_beat_times_increase_strictly_and_lie_below_the_schedule_s_iterations() -> None:
    rng = random.Random(SEED)
    cases = longest = 0
    for _ in range(SAMPLE):
        schedule = random_schedule(rng)
        order = list(schedule.order)
        # Disjoint reduced and held indices: each index is neither, reduced, or held.
        roles = [rng.choice(("walked", "walked", "reduced", "held")) for _ in order]
        reduces = tuple(index for index, role in zip(order, roles) if role == "reduced")
        holds = tuple(index for index, role in zip(order, roles) if role == "held")
        times = schedule.beat_times(reduces=reduces, holds=holds)
        assert times, (schedule, reduces, holds)
        assert all(a < b for a, b in itertools.pairwise(times)), (schedule, reduces, holds, times)
        assert 0 <= times[0] and times[-1] < schedule.iterations, (schedule, reduces, holds)
        # A port walking every index but those: one beat for each of their other indices' index
        # values.
        walked = prod(schedule.fold(i) for i, role in zip(order, roles) if role == "walked")
        assert len(times) == walked
        cases += 1
        longest = max(longest, len(times))
    assert cases == SAMPLE and longest > 20


def test_the_times_of_a_port_that_reads_every_beat_and_of_one_that_closes_a_reduction() -> None:
    m, n, k = Index("m"), Index("n"), Index("k")
    schedule = Schedule({m: 2, n: 3, k: 4}, order=(m, n, k))
    assert schedule.beat_times() == tuple(range(24))
    assert schedule.beat_times(reduces=(k,)) == (3, 7, 11, 15, 19, 23)
    assert schedule.beat_times(holds=(k,)) == (0, 4, 8, 12, 16, 20)


# --- the refusals of time ---------------------------------------------------------------


def test_a_source_s_words_lie_within_its_span() -> None:
    assert Source("s", (0, 1, 2), 3, 3).span == 3
    for span in (2, 1, 0):
        with pytest.raises(ValueError, match="lie within the source's span"):
            Source("s", (0, 1, 2), span, 4)
    # The words' first time does not count in the span: a source starting late has the same.
    assert Source("s", (5, 6, 7), 3, 3).span == 3
    with pytest.raises(ValueError, match="lie within the source's span"):
        Source("s", (5, 6, 7), 2, 3)


def test_a_kernel_s_ports_move_at_iterations_within_its_iterations() -> None:
    assert Kernel("k", 4, ((0, 3),), ((3,),)).iterations == 4
    for reads, writes in ((((0, 4),), ()), ((), ((4,),)), (((-1, 2),), ()), (((),), ())):
        with pytest.raises(ValueError, match="a port moves at least one word, at iterations 0..3"):
            Kernel("k", 4, reads, writes)
    with pytest.raises(ValueError, match="a kernel has a port"):
        Kernel("k", 4, (), ())


def test_a_kernel_s_ports_share_one_schedule() -> None:
    m, n, k = Index("m"), Index("n"), Index("k")
    one = Schedule({m: 2, n: 3, k: 4}, order=(m, n, k))
    other = Schedule({m: 2, n: 3, k: 4}, order=(m, k, n))
    built = kernel("k", (Pace(one),), (Pace(one, reduces=(k,)),))
    assert built.iterations == one.iterations
    assert all(port[-1] < built.iterations for port in (*built.reads, *built.writes))
    assert kernel("k", (Pace(one),), (Pace(Schedule({m: 2, n: 3, k: 4}, order=(m, n, k))),))
    with pytest.raises(ValueError, match="a kernel's ports share one schedule"):
        kernel("k", (Pace(one),), (Pace(other),))
    with pytest.raises(ValueError, match="a kernel's ports share one schedule"):
        kernel("k", (Pace(one), Pace(other)), ())


# --- strided maps and traversals --------------------------------------------------------

STRIDED_MAPS = {
    "identity (3, 4)": maps.identity((3, 4)),
    "reshape (6, 4) to (3, 8)": maps.reshape((6, 4), (3, 8)),
    "transpose (4, 6)": maps.transpose((4, 6)),
    "transpose (2, 3, 4) by (2, 0, 1)": maps.transpose((2, 3, 4), (2, 0, 1)),
    "transpose (3, 1, 4, 2) by (3, 1, 0, 2)": maps.transpose((3, 1, 4, 2), (3, 1, 0, 2)),
    "broadcast (1, 4) to (3, 4)": maps.broadcast((1, 4), (3, 4)),
    "squeeze (1, 4, 1)": maps.squeeze((1, 4, 1)),
    "window (1, 4, 4, 2) of 3": maps.im2col((1, 4, 4, 2), (3, 3)),
    "window (1, 5, 5, 1) of 2 by 2": maps.im2col((1, 5, 5, 1), (2, 2), (2, 2)),
    "window (2, 6, 6, 2) of 2, dilated 2": maps.im2col((2, 6, 6, 2), (2, 2), (1, 1), (2, 2)),
    "slice (6, 8) every other row": maps.sliced((6, 8), (0, 0), (6, 8), (0, 1), (2, 1)),
}


@pytest.mark.parametrize("name", STRIDED_MAPS)
def test_a_strided_map_is_the_traversal_of_its_source_read_one_element_a_beat(name: str) -> None:
    """The map composed through its reader walked once is the traversal of its source
    with the map's modes as its loops, and presents the offsets the map reads."""
    pmap = STRIDED_MAPS[name]
    assert pmap.strided
    piece = pmap.pieces[0]
    source = pmap.sources[piece.source]
    walked = maps.through(pmap, vector_major(pmap.shape, 1))
    assert walked == Traversal(source, piece.modes, ())
    assert np.array_equal(offsets(walked), maps.reads(pmap)[1])
    # At lanes that close whole loops of the reader the same elements come a beat, in the
    # same order: the lanes only cut the order.
    for lanes in (2, 3, 4):
        try:
            wide = maps.through(pmap, vector_major(pmap.shape, lanes))
        except (ValueError, maps.Unmappable):
            continue
        assert wide.lanes == lanes and np.array_equal(offsets(wide), offsets(walked))


def test_a_traversal_is_the_strided_map_from_its_elements_that_reads_its_offsets() -> None:
    """A traversal's loops, beats then lanes, are the modes of a strided map from
    ``(beats * lanes,)`` onto its tensor: the map reads the offsets the traversal
    presents, and a reader walked as the traversal's lanes composes back to it."""
    sampled = traversals()
    composed = 0
    for form in sampled:
        count = form.beats * form.lanes
        piece = maps.Piece((0,), (count,), 0, 0, (*form.beat_loops, *form.lane_loops))
        pmap = maps.PositionMap((count,), (form.shape,), (piece,))
        assert pmap.strided
        assert np.array_equal(maps.reads(pmap)[1], offsets(form)), form
        assert maps.through(pmap, vector_major((count,), 1)) == Traversal(
            form.shape, (*form.beat_loops, *form.lane_loops), ()
        )
        try:
            wide = maps.through(pmap, vector_major((count,), form.lanes))
        except maps.Unmappable:
            continue
        assert wide == form, form
        composed += 1
    assert composed > SAMPLE // 2


# --- the refusals no constructor test gives ---------------------------------------------


def test_the_values_a_constructor_refuses_that_no_test_gave_it() -> None:
    """A tensor's rank and extents and element, an encoding's type, width and range, a
    loop's extent and stride, a traversal's rank, loop type and reach."""
    with pytest.raises(ValueError, match="a tensor has at least one axis, each of positive"):
        Tensor((), INT4)
    for shape in ((2, 0), (0,), (2, -1), (2, True)):
        with pytest.raises(ValueError, match="a tensor has at least one axis, each of positive"):
            Tensor(shape, INT4)
    with pytest.raises(TypeError, match="a tensor's element is a ScalarEncoding"):
        Tensor((2, 3), "INT4")  # type: ignore[arg-type]
    assert Tensor((2, 3), INT4).size == 6

    int4 = resolve_qonnx_datatype_name("INT4")
    with pytest.warns(Warning, match="zero bits"):  # qonnx names INT0, and warns of it
        zero_bits = resolve_qonnx_datatype_name("INT0")
    with pytest.raises(ValueError, match="positive width"):
        ScalarEncoding(zero_bits)
    with pytest.raises(TypeError, match="a range is a TensorRange"):
        ScalarEncoding(int4, (-7, 7))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is not a range of INT4 values"):
        ScalarEncoding(int4, TensorRange(-100, 100))
    assert ScalarEncoding(int4, TensorRange(-7, 7)).stated

    for extent in (0, -1, True):
        with pytest.raises(ValueError, match="loop extent must be a positive integer"):
            Loop(extent, 1)
    for stride in (-1, 1.0):
        with pytest.raises(ValueError, match="a loop stride is a nonnegative number"):
            Loop(2, stride)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="operand of rank at least one"):
        Traversal((), (), ())
    with pytest.raises(ValueError, match="operand extent must be a positive integer"):
        Traversal((2, 0), (), ())
    with pytest.raises(TypeError, match="traversal loops are Loop values"):
        Traversal((4,), ((2, 1),), ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="leaves an operand of 4 elements"):
        Traversal((4,), (Loop(3, 2),), ())
    with pytest.raises(ValueError, match="addresses positions outside the operand"):
        Traversal((2, 2), (Loop(2, 2),), (Loop(2, 2),))
    assert Traversal((2, 2), (Loop(2, 2),), (Loop(2, 1),)).lanes == 2
