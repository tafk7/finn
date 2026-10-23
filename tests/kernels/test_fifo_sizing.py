# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FIFO sizing from both sides' beat patterns: input_gen's buffer as its RTL derives it,
the throttle, the least depth, the strategy proposing it through the seam, and the
seam's check of a stored transport against its stream.

The strategy's tests run on the Chain (``support.chain``), as the seam's do
(``test_explore``); the Chain needs no FIFO at any folding, so a FIFO it places and a
FIFO the seam refuses are made controllable by a double of the per-channel sizing.
"""

from __future__ import annotations

import random
from math import prod
from pathlib import Path
from typing import Any

import pytest
from support import chain
from support.helpers import FULL_DSP48E2, keyed, with_defaults
from support.refusals import refused
from support.xsim import requires_xsim

import finn.kernels.explore as explore_module
import finn.kernels.fifo_sizing as fifo_sizing
from finn import resources
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.rtlsim_bench import simulate as simulate_rtl
from finn.harness.toolchain import finnlib_root
from finn.kernels import input_generator
from finn.kernels.artifacts.rtl import evaluate
from finn.kernels.channels import Channel, Neighbours
from finn.kernels.explore import (
    Accepted,
    ExploreError,
    FifoCheck,
    Hold,
    Refused,
    Seam,
    SizeFifos,
    TargetCycles,
    dotted,
    explore,
)
from finn.kernels.fifo import least_depth_holding
from finn.kernels.fifo_sizing import (
    FIFO_LATENCY,
    Buffered,
    Flow,
    NotModelled,
    Pattern,
    Sized,
    converted,
    keeps,
    least_depth,
    sides,
    simulate,
)
from finn.kernels.input_generator import NestBuffer, nest_buffer, nest_geometry
from finn.kernels.network import Network
from finn.kernels.policies import Required
from finn.toolchain.launcher import host_toolchain

MEMBERS = chain.MEMBERS


# -- input_gen's buffer --------------------------------------------------------------------


def test_input_gen_s_constants_are_the_ones_its_rtl_evaluates() -> None:
    # DIMS {4, 49}, COEFS {0, 1}: 48 read ahead, plus a frame of write-ahead after the
    # burst free, 97; BUF_SIZE 2**clog2(97 + 1 + 2) = 128.
    geometry = nest_geometry(49, (4, 49), (0, 1))
    assert geometry.buffer_words == 128 and geometry.max_occupancy == 97
    assert geometry.frees == (True, False, False)
    assert geometry.read_steps == (1, -48, 1)
    assert geometry.free_steps == (49, 0, 0)
    assert geometry.capacity == 127


def test_input_gen_s_constants_are_evaluated_once_per_nest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def counted(*args: Any) -> Any:
        calls.append(args)
        return evaluate(*args)

    monkeypatch.setattr(input_generator, "evaluate", counted)
    input_generator._evaluated.cache_clear()
    nest = (7, (3, 7), (0, 1))
    first = nest_geometry(*nest)
    assert nest_geometry(*nest) == first and nest_geometry(7, [3, 7], [0, 1]) == first
    assert len(calls) == 1
    nest_geometry(7, (2, 7), (0, 1))
    assert len(calls) == 2


def test_a_cost_query_never_fetches_finnlib(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without a local FinnLib (no override, an empty cache), reading input_gen's
    constants refuses with the resource's error, naming it, and fetches nothing."""
    monkeypatch.delenv("FINN_RESOURCES_FINNLIB", raising=False)
    monkeypatch.setenv("FINN_RESOURCES_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FINN_RESOURCES_SYSTEM_CACHE", str(tmp_path / "system"))

    def fetched(*args: object) -> None:
        pytest.fail("a cost query fetched FinnLib")

    monkeypatch.setattr(resources._store, "fetch", fetched)
    with pytest.raises(resources.ResourceError, match="Resource finnlib is not in any cache"):
        nest_geometry(11, (2, 11), (0, 1))


def test_a_replay_nest_holds_its_frame_until_the_last_pass() -> None:
    buffer = nest_buffer(49, (4, 49), (0, 1))
    assert buffer.capacity == 127
    assert buffer.reads == tuple(range(49)) * 4
    assert buffer.freed == (0,) * 195 + (49,)


def test_an_identity_nest_frees_each_word_as_it_presents_it() -> None:
    buffer = nest_buffer(4, (4,), (1,))
    assert buffer.capacity == 3
    assert buffer.reads == (0, 1, 2, 3)
    assert buffer.freed == (1, 2, 3, 4)


def test_a_row_replay_frees_each_row_after_its_last_pass() -> None:
    # Three rows of four, each read twice: a row is freed once its second pass ends.
    buffer = nest_buffer(12, (3, 2, 4), (4, 0, 1))
    assert buffer.reads == tuple(
        4 * row + i for row in range(3) for _ in range(2) for i in range(4)
    )
    assert buffer.freed == tuple(
        4 * (beat // 8 + 1) if beat % 8 == 7 else 4 * (beat // 8) for beat in range(24)
    )


def test_a_nest_s_buffer_is_stepped_once_per_nest() -> None:
    # A list and a tuple name the same nest: one buffer, shared (it is immutable).
    input_generator._stepped.cache_clear()
    first = nest_buffer(12, (3, 2, 4), (4, 0, 1))
    again = nest_buffer(12, [3, 2, 4], [4, 0, 1])
    assert again == first and again is first
    assert input_generator._stepped.cache_info().misses == 1


# -- the throttle and the least depth --------------------------------------------------------


def test_a_width_conversion_sends_a_word_once_its_last_element_arrived() -> None:
    assert converted((3, 7), 16, 4) == (4, 5, 6, 7, 8, 9, 10, 11)
    assert converted(tuple(range(8)), 1, 4) == (4, 8)


# A bottleneck producer (no idle time) sending a frame's 8 words in a burst, and a consumer
# taking one every other cycle: the FIFO must hold what the burst runs ahead.
BURST = Pattern(tuple(range(8)), 16)
SLOW = Pattern(tuple(range(0, 16, 2)), 16)


def _late(supply: Pattern, accept: Any, period: int, depth: int, frames: int) -> list[int]:
    flow = simulate(supply, accept, period, depth, frames)
    assert flow is not None
    return list(flow.late)


def test_direct_throttles_a_burst_into_a_slower_consumer() -> None:
    late = _late(BURST, SLOW, 16, 0, 4)
    assert late[0] > 0 and late == sorted(late)


def test_the_least_depth_keeps_the_period_and_one_less_does_not() -> None:
    # Four words run ahead of the consumer, and one more is in flight through the FIFO.
    depth = least_depth(BURST, SLOW, 16)
    assert depth == 5
    assert _late(BURST, SLOW, 16, depth, 8) == [0] * 8
    assert max(_late(BURST, SLOW, 16, depth - 1, 8)) > 0


def test_idle_time_absorbs_the_throttle() -> None:
    # The same burst from a producer with half its period idle needs no FIFO.
    assert least_depth(Pattern(tuple(range(8)), 8), SLOW, 16) == 0


def test_a_replay_buffer_of_frames_absorbs_a_bottleneck_producer() -> None:
    # TFC's first MatMul behind its input_gen at 16 lanes (PE and SIMD): 49 words in,
    # 196 beats read; the buffer takes the next frame while it replays this one.
    buffered = Buffered(nest_buffer(49, (4, 49), (0, 1)), Pattern(tuple(range(196)), 196))
    assert least_depth(Pattern(tuple(range(4, 200, 4)), 196), buffered, 196) == 0


def test_a_fifo_s_depth_is_the_least_whose_storage_holds_the_words() -> None:
    # FinnLib's FIFO is a shift register of at least five words up to DEPTH 33.
    assert [least_depth_holding(words, 8, "auto", False) for words in (1, 4, 5, 6, 33)] == [
        2,
        2,
        2,
        6,
        33,
    ]
    # LUTRAM from 34 to 257 holds DEPTH words; block RAM beyond rounds up.
    assert least_depth_holding(40, 8, "auto", False) == 40
    assert least_depth_holding(300, 8, "auto", False) == 258


# -- a channel's sides, from the paces its ports state ----------------------------------------


def _folded() -> tuple[Seam, Network]:
    """The Chain as conversion and the cut leave it (``with_defaults``), folded to 12
    cycles a frame."""
    explorer = Seam(FULL_DSP48E2)
    explorer.held.update(chain.HELD)
    return explorer, TargetCycles(12).explore(explorer, with_defaults(chain.network()))


def chosen(explorer: Seam, network: Network) -> dict[str, object]:
    """``network``'s stored choices, each by ``member.key``."""
    return {dotted(key): value for key, value in explorer.chosen(network).items()}


def test_a_channel_s_sides_are_read_from_the_paces_its_ports_state() -> None:
    _, point = _folded()
    # hidden: first (PE 1, SIMD 4) closes a reduction of one step each beat, a lane a
    # word; the output adapter's vpc pairs them into activate's two lanes, a word the
    # cycle after its second lane. activate reads every beat.
    supply, accept = sides(point.member("hidden"), point.neighbours("hidden"))
    assert supply == Pattern((2, 4, 6, 8, 10, 12), 12)
    assert accept == Pattern(tuple(range(6)), 6)
    # levels: activate's six beats, four lanes a word after its vpc, into second's
    # input_gen, which second reads every beat.
    supply, accept = sides(point.member("levels"), point.neighbours("levels"))
    assert supply == Pattern((2, 4, 6), 6)
    assert isinstance(accept, Buffered)
    assert accept.reads == Pattern(tuple(range(12)), 12)


def test_a_boundary_and_a_memory_are_not_modelled() -> None:
    _, point = _folded()
    with pytest.raises(NotModelled, match="a boundary: not modelled"):
        sides(point.member("x"), point.neighbours("x"))
    with pytest.raises(NotModelled, match="a memory source: paced by its consumer"):
        sides(point.member("w1"), point.neighbours("w1"))


# -- the strategy ---------------------------------------------------------------------------


def test_size_fifos_proposes_every_transport_in_one_batch_and_says_why() -> None:
    explorer, point = _folded()
    strategy = SizeFifos()
    attempts = explorer.attempts
    sized = strategy.explore(explorer, point)
    assert explorer.attempts == attempts + 1
    transports = sized.channels
    assert {
        key: value for key, value in chosen(explorer, sized).items() if key.endswith(".transport")
    } == {f"{name}.transport": "direct" for name in transports}
    # It reopened every transport (each stored direct by the cut's policy)
    # and gave back nothing else.
    (_, run) = explorer.runs
    assert set(run.reopened) == {(name, "transport") for name in transports}
    assert run.kept == run.by_policy == {}
    report = strategy.report()
    assert report["period"] == 12 and report["fifo_bits"] == 0
    assert {name: row["why"] for name, row in strategy.channels().items()} == {
        "x": "a boundary: not modelled",
        "w1": "a memory source: paced by its consumer",
        "hidden": "direct absorbs it",
        "levels": "direct absorbs it",
        "w2": "a memory source: paced by its consumer",
        "y": "a boundary: not modelled",
    }


def test_it_sizes_at_the_stored_folding_and_refuses_a_point_whose_cycles_are_unknown() -> None:
    """No ``before folding`` path: it reads the stored folding (conversion's, on a point
    as conversion and the cut leave it); where a member's cycles wait on an open choice,
    it refuses, naming it."""
    explorer = Seam(FULL_DSP48E2)
    explorer.held.update(chain.HELD)
    point = chain.network()
    base = with_defaults(point)
    strategy = SizeFifos()
    sized = strategy.explore(explorer, base)
    assert strategy.report()["period"] == explorer.cost(base).bottleneck.cycles  # type: ignore[union-attr]
    assert {k: v for k, v in chosen(explorer, sized).items() if not k.endswith(".transport")} == {
        k: v for k, v in chosen(explorer, base).items() if not k.endswith(".transport")
    }
    with refused(
        "cycles-unknown",
        error=ExploreError,
        message="needs every member's cycles: .*first.compute",
    ):
        SizeFifos().explore(explorer, point)


def test_a_fifo_sized_before_is_sized_again_and_a_held_one_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every transport not held is reopened, its FIFO's depth and style with it, and
    sized again; a held transport is not reopened, nor its held depth (its style, not
    held, is the strategy's: reopened, and given back)."""
    explorer, point = _folded()
    placed = explorer.set(
        point, keyed({"levels.transport": "fifo", "levels.transport.fifo.depth": 64})
    )
    # The FIFO's style, brought into play, set by the fold rule (the cut's policy).
    assert isinstance(placed, Accepted)
    assert placed.settled == keyed({"levels.transport.fifo.buffer.ram_style": "auto"})
    sized = SizeFifos().explore(explorer, placed.network)
    assert chosen(explorer, sized)["levels.transport"] == "direct"
    assert ("levels", "transport.fifo.depth") in explorer.runs[-1].reopened
    explorer.held.update(keyed({"levels.transport": "fifo", "levels.transport.fifo.depth": 64}))
    kept = SizeFifos().explore(explorer, placed.network)
    assert chosen(explorer, kept)["levels.transport.fifo.depth"] == 64
    assert chosen(explorer, kept)["levels.transport.fifo.buffer.ram_style"] == "auto"
    assert [key for key in explorer.runs[-1].reopened if key[0] == "levels"] == [
        ("levels", "transport.fifo.buffer.ram_style")
    ]


def _hidden_needs_eight(
    channel: Channel, neighbours: Neighbours, period: int, **options: Any
) -> Sized:
    """A double of the per-channel sizing: ``hidden`` (first's output) needs a FIFO of
    eight words, every other channel none."""
    if channel.sides.source_owner == "first.compute.packed.y":
        return Sized(8, "least 8 words", 24, 8)
    return Sized(0, "direct absorbs it", 8, 0)


def test_a_fifo_is_proposed_with_its_depth_and_memory_in_the_same_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explorer, point = _folded()
    monkeypatch.setattr(explore_module, "size", _hidden_needs_eight)
    strategy = SizeFifos()
    found = strategy.explore(explorer, point)
    values = chosen(explorer, found)
    assert values["hidden.transport"] == "fifo"
    assert values["hidden.transport.fifo.depth"] == 8
    # Its style, never stored, is committed with its depth: the cut's policy's first
    # valid, FinnLib's ``auto``. Nothing is given back or set by policy after.
    assert values["hidden.transport.fifo.buffer.ram_style"] == "auto"
    assert explorer.runs[-1].kept == explorer.runs[-1].by_policy == {}
    assert strategy.report()["ram_style"] is None
    assert strategy.channels()["hidden"]["bits"] == 8 * 24
    assert strategy.report()["fifo_bits"] == 8 * 24
    # A fold after it keeps it (no transport is a folding strategy's), and nothing
    # refuses itself.
    done = explore(explorer, found, [TargetCycles(12)])
    assert explorer.refusals(done) == {}
    assert chosen(explorer, done)["hidden.transport.fifo.depth"] == 8


def test_a_style_named_is_committed_with_the_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    explorer, point = _folded()
    monkeypatch.setattr(explore_module, "size", _hidden_needs_eight)
    for style in ("auto", "distributed"):
        values = chosen(explorer, SizeFifos(ram_style=style).explore(explorer, point))
        assert values["hidden.transport.fifo.depth"] == 8
        assert values["hidden.transport.fifo.buffer.ram_style"] == style


def test_a_depth_is_sized_for_the_style_its_fifo_is_built_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sizing without a style sizes each FIFO for the style it is built in: its held
    style, else its earlier one, else ``auto`` (the cut's policy's first valid). Above 257
    words ``auto`` is block RAM,
    which holds more than its DEPTH, and ``distributed`` holds DEPTH words: a FIFO stored
    ``distributed`` before is sized at the words it needs, which the check keeps."""

    # Every modelled channel (hidden, levels) needs 300 words; the check keeps a FIFO
    # that holds them, and reads no other channel.
    def holds(channel: Channel, neighbours: Neighbours, period: int, words: int) -> bool:
        sides(channel, neighbours)
        return words >= 300

    monkeypatch.setattr(fifo_sizing, "least_depth", lambda supply, accept, period: 300)
    monkeypatch.setattr(explore_module, "holds", holds)
    style = ("hidden", "transport.fifo.buffer.ram_style")
    for held in (False, True):
        explorer, point = _folded()
        placed = explorer.set(
            point,
            keyed(
                {
                    "hidden.transport": "fifo",
                    "hidden.transport.fifo.depth": 400,
                    "hidden.transport.fifo.buffer.ram_style": "distributed",
                }
            ),
        )
        assert isinstance(placed, Accepted)
        if held:
            explorer.held[style] = "distributed"
        sized = SizeFifos().explore(explorer, placed.network)
        values = chosen(explorer, sized)
        # hidden, stored distributed: DEPTH 300, 300 words.
        assert values["hidden.transport.fifo.buffer.ram_style"] == "distributed"
        assert values["hidden.transport.fifo.depth"] == 300
        # levels, no style before: auto, block RAM from DEPTH 258 (514 words).
        assert values["levels.transport.fifo.buffer.ram_style"] == "auto"
        assert values["levels.transport.fifo.depth"] == 258
        found = explorer.check_fifos(sized)
        assert found.checked == ("hidden", "levels")
        assert found.stale == {} and found.findings == ()


def test_a_style_held_as_none_is_sized_as_auto_and_left_to_its_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A held style is the hold's, never committed by the sizing: one held as ``None`` is
    sized as ``auto``, never as a style named "None", and the stored style is left as it
    is. A ``Hold`` stating ``None`` is refused where it is set: no value of the style."""
    sized: dict[str, str] = {}

    def sizing(channel: Channel, neighbours: Neighbours, period: int, **options: Any) -> Sized:
        sized[channel.sides.source_owner or ""] = options["ram_style"]
        return _hidden_needs_eight(channel, neighbours, period)

    monkeypatch.setattr(explore_module, "size", sizing)
    explorer, point = _folded()
    placed = explorer.set(
        point,
        keyed(
            {
                "hidden.transport": "fifo",
                "hidden.transport.fifo.depth": 8,
                "hidden.transport.fifo.buffer.ram_style": "distributed",
            }
        ),
    )
    assert isinstance(placed, Accepted)
    explorer.held[("hidden", "transport.fifo.buffer.ram_style")] = None
    found = SizeFifos().explore(explorer, placed.network)
    assert sized["first.compute.packed.y"] == "auto"
    assert set(sized.values()) == {"auto"}
    assert chosen(explorer, found)["hidden.transport.fifo.buffer.ram_style"] == "distributed"
    with refused(
        "pin-refused",
        error=ExploreError,
        owner="held",
        message="hidden.transport.fifo.buffer.ram_style: .*request-refused",
    ):
        Hold({"hidden": {"transport.fifo.buffer.ram_style": None}}).explore(
            Seam(FULL_DSP48E2), placed.network
        )


MARKERS = Finding(
    FindingKind.REJECTION, "fifo-markers", "hidden.transport.fifo", "what arrives carries markers"
)


def test_a_refused_fifo_falls_back_to_direct_and_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    _, point = _folded()

    class Refusing(Seam):
        def set(self, network: Any, changes: Any) -> Any:
            if changes.get(("hidden", "transport")) == "fifo":
                self.attempts += 1
                return Refused({("hidden", ""): (MARKERS,)})
            return super().set(network, changes)

    monkeypatch.setattr(explore_module, "size", _hidden_needs_eight)
    refusing = Refusing(FULL_DSP48E2)
    strategy = SizeFifos()
    found = strategy.explore(refusing, point)
    assert chosen(refusing, found)["hidden.transport"] == "direct"
    row = strategy.channels()["hidden"]
    assert row["transport"] == "direct" and row["bits"] == 0
    assert row["why"] == (
        "least 8 words; refused: hidden: hidden.transport.fifo: fifo-markers: what arrives "
        "carries markers"
    )


# -- a stored transport checked against its stream ---------------------------------------

#: A burst of 16 words from a kernel of ``span`` cycles a frame, and a consumer taking one
#: word every ``gap`` cycles: the sizing tests' patterns, at a size where the period alone
#: changes the need.
FRAME_WORDS = 16


def _burst(span: int) -> Pattern:
    return Pattern(tuple(range(FRAME_WORDS)), span)


def _every(gap: int, span: int) -> Pattern:
    return Pattern(tuple(range(0, FRAME_WORDS * gap, gap)), max(span, FRAME_WORDS * gap))


def test_the_check_is_sizing_s_own_test_at_the_words_a_transport_holds() -> None:
    """``keeps`` at the least depth holds, one word less does not. The period alone moves
    the need with the sides unchanged: none at 48 cycles a frame, 5 words at 40, 9 at 32,
    so a depth sized at 40 fails at 32; a slower consumer needs more than the 9, a faster
    one none."""
    supply, accept = _burst(32), _every(2, 32)
    sized = least_depth(supply, accept, 32)
    assert sized == 9
    assert keeps(supply, accept, 32, sized) and not keeps(supply, accept, 32, sized - 1)
    assert [least_depth(supply, accept, period) for period in (48, 40, 32)] == [0, 5, 9]
    assert keeps(supply, accept, 40, 5) and not keeps(supply, accept, 32, 5)
    assert keeps(supply, accept, 48, 0) and not keeps(supply, accept, 40, 0)
    assert least_depth(_burst(48), _every(3, 48), 48) == 11
    assert not keeps(_burst(48), _every(3, 48), 48, sized)
    assert keeps(_burst(32), _every(1, 32), 32, sized)


def _with_fifo(depth: int = 8) -> tuple[Seam, Network]:
    """The folded Chain with a FIFO of ``depth`` on ``levels`` (a shift register of
    ``max(5, depth)`` words)."""
    explorer, point = _folded()
    placed = explorer.set(
        point,
        keyed(
            {
                "levels.transport": "fifo",
                "levels.transport.fifo.depth": depth,
                "levels.transport.fifo.buffer.ram_style": "auto",
            }
        ),
    )
    assert isinstance(placed, Accepted)
    return explorer, placed.network


def _levels_short(channel: Channel, neighbours: Neighbours, period: int, words: int) -> bool:
    """A double of the check (the Chain needs no FIFO at any folding): ``levels``
    (activate's output) needs more words than its transport holds; every other channel's
    holds."""
    return not str(channel.sides.source_owner).startswith("activate.")


DEPTH = ("levels", "transport.fifo.depth")


def test_a_fifo_that_holds_its_stream_is_kept_and_the_check_says_what_it_read() -> None:
    explorer, network = _with_fifo()
    checked, found = explorer.checked(network)
    assert checked is network
    assert found == FifoCheck(
        12,
        checked=("hidden", "levels"),
        not_modelled={
            "x": "a boundary: not modelled",
            "w1": "a memory source: paced by its consumer",
            "w2": "a memory source: paced by its consumer",
            "y": "a boundary: not modelled",
        },
    )


def test_a_fifo_too_small_for_its_stream_is_dropped_named_and_left_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explorer, network = _with_fifo()
    monkeypatch.setattr(explore_module, "holds", _levels_short)
    checked, found = explorer.checked(network)
    (finding,) = found.stale[DEPTH]
    assert (finding.kind, finding.code, finding.owner) == (
        FindingKind.REJECTION,
        "fifo-stale",
        "levels.transport.fifo.depth",
    )
    assert finding.message == (
        "depth 8 (8 words) no longer keeps its producer within its idle time at the "
        "bottleneck's 12 cycles a frame: size it again (SizeFifos)"
    )
    assert dict(finding.details) == {"tensor": "levels", "words": 8, "period": 12}
    assert found.findings == ()
    # Left open, a required choice; the FIFO and the rest kept, the bottleneck readable,
    # so a later strategy folds and FIFO sizing sizes it again.
    values = chosen(explorer, checked)
    assert "levels.transport.fifo.depth" not in values
    assert {k: v for k, v in chosen(explorer, network).items() if k != dotted(DEPTH)} == values
    offered = explorer.choices(checked)
    assert [choice.key for choice in offered if choice.policy == Required()] == [DEPTH]
    assert explorer.cost(checked).bottleneck == explorer.cost(network).bottleneck
    sized = SizeFifos().explore(explorer, checked)
    assert chosen(explorer, sized)["levels.transport"] == "direct"


def test_a_held_fifo_too_small_is_kept_and_named(monkeypatch: pytest.MonkeyPatch) -> None:
    explorer, network = _with_fifo()
    explorer.held.update(keyed({"levels.transport.fifo.depth": 8}))
    monkeypatch.setattr(explore_module, "holds", _levels_short)
    checked, found = explorer.checked(network)
    assert checked is network and found.stale == {}
    (finding,) = found.findings
    assert (finding.kind, finding.code, finding.owner) == (
        FindingKind.LIMITATION,
        "fifo-held-stale",
        "levels.transport.fifo.depth",
    )
    assert finding.message.startswith("held at 8 (8 words), which no longer keeps")


def test_a_direct_that_throttles_its_producer_is_kept_and_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explorer, network = _folded()
    monkeypatch.setattr(explore_module, "holds", _levels_short)
    checked, found = explorer.checked(network)
    assert checked is network and found.stale == {}
    (finding,) = found.findings
    assert (finding.kind, finding.code, finding.owner) == (
        FindingKind.LIMITATION,
        "direct-throttles",
        "levels.transport",
    )
    assert finding.message == (
        "direct does not keep its producer within its idle time at the bottleneck's 12 "
        "cycles a frame: its producer is throttled (SizeFifos places a FIFO)"
    )


def test_an_open_depth_and_unknown_cycles_are_not_checked() -> None:
    explorer = Seam(FULL_DSP48E2)
    assert explorer.check_fifos(chain.network()) == FifoCheck(None)
    _, network = _with_fifo()
    opened, _ = explorer.reopen(network, [DEPTH])
    found = explorer.check_fifos(opened)
    assert found.period == 12 and found.checked == ("hidden",)


def test_only_the_analytical_method_exists() -> None:
    with refused("sizing-invalid", error=ExploreError, message="analytical"):
        SizeFifos(method="simulated")
    with refused("sizing-invalid", error=ExploreError, message=r"\bmargin\b"):
        SizeFifos(margin=-1)


# -- in XSim: the model against the RTL ------------------------------------------------------


def _cycles(log: str, tag: str) -> list[int]:
    return [int(line.split()[-1]) for line in log.splitlines() if line.startswith(tag + " ")]


#: input_gen's nests: a frame replayed four times (TFC's first MatMul), rows replayed
#: twice (a MatMul's activations across two output folds), a transpose, the identity.
NESTS = (
    (49, (4, 49), (0, 1)),
    (12, (3, 2, 4), (4, 0, 1)),
    (9, (3, 3), (1, 3)),
    (4, (4,), (1,)),
)


@requires_xsim
@pytest.mark.parametrize(("frame", "dims", "coefs"), NESTS)
def test_input_gen_accepts_its_input_at_the_cycles_its_model_predicts(
    tmp_path: Path, frame: int, dims: tuple[int, ...], coefs: tuple[int, ...]
) -> None:
    """The runtime free rule ``nest_buffer`` keeps in Python, cycle-exact against the RTL:
    an always-valid source into ``input_gen``, its output read every cycle, over six
    frames. The constants are the RTL's own (``nest_geometry``); what this pins is the
    pointers' stepping and the accept, write and read timing ``simulate`` assumes."""
    frames, beats = 6, prod(dims)

    def vector(values: tuple[int, ...]) -> str:
        return "'{" + ", ".join(map(str, values)) + "}"

    log = simulate_rtl(
        [finnlib_root() / "rtl/shape/input_gen.sv"],
        f"""
module check;
    logic clk = 0, rst = 1;
    logic [7:0] idat = 0;
    wire [7:0] odat;
    wire irdy, ovld;
    wire [{len(dims) - 1}:0] olst;
    int cycle = 0, accepted = 0, presented = 0;
    wire ivld = !rst && accepted < {frames * frame};
    always #5 clk = !clk;
    input_gen #(
        .DATA_WIDTH(8), .FM_SIZE({frame}), .D({len(dims)}),
        .DIMS({vector(dims)}), .COEFS({vector(coefs)})
    ) dut (.clk, .rst, .idat, .ivld, .irdy, .odat, .ovld, .olst, .ordy(1'b1));
    always @(posedge clk) if (!rst) begin
        cycle <= cycle + 1;
        if (ivld && irdy) begin
            $display("ACCEPT %0d", cycle);
            accepted <= accepted + 1;
            idat <= idat + 1;
        end
        if (ovld) presented <= presented + 1;
    end
    initial begin
        repeat (4) @(negedge clk);
        rst = 0;
        wait (presented == {frames * beats});
        $display("INPUT_GEN_PASS");
        $finish;
    end
    initial begin #1000000; $fatal(1, "input_gen watchdog"); end
endmodule
""",
        tmp_path,
        toolchain=host_toolchain(),
    )
    measured = _cycles(log, "ACCEPT")
    supply = Pattern(tuple(range(frame)), frame)  # always valid: never ahead of the source
    accept = Buffered(nest_buffer(frame, dims, coefs), Pattern(tuple(range(beats)), beats))
    flow = simulate(supply, accept, frame, 0, frames)
    assert flow is not None
    model = list(flow.taken)
    assert [at - measured[0] for at in measured] == [at - model[0] for at in model]


#: Stage 1's unit case, at a size where a FIFO's depth is its capacity (above the shift
#: register's five words): a producer with no idle time sending its frame's 16 words in a
#: burst at the start of a 32-cycle period, a consumer taking one every other cycle.
FRAME, PERIOD, GAP = 16, 32, 2


@requires_xsim
def test_the_proposed_depth_holds_the_period_in_xsim_and_one_less_misses_it(
    tmp_path: Path,
) -> None:
    """FinnLib's FIFO between a producer and a consumer that present stage 1's patterns
    (testbench processes: no kernel on the trunk presents a burst into a slower
    consumer). The producer resumes its schedule when stalled; the consumer takes a word
    once its spacing has passed. At the proposed depth the producer is never late; one
    word less, it is later each frame, as the model says, cycle for cycle: every word
    leaves the producer when the model says it does."""
    burst = Pattern(tuple(range(FRAME)), PERIOD)
    slow = Pattern(tuple(range(0, PERIOD, GAP)), PERIOD)
    least = least_depth(burst, slow, PERIOD)
    assert least == 9
    depth = least_depth_holding(least, 8, "auto", False)
    assert depth == 9
    frames = 8
    for case in (depth, depth - 1):
        directory = tmp_path / f"depth{case}"
        directory.mkdir()
        log = simulate_rtl(
            [finnlib_root() / "rtl/infra/fifo.sv"],
            f"""
module check;
    logic clk = 0, rst = 1;
    wire [7:0] odat;
    wire irdy, ovld;
    int cycle = 0, sent = 0, ready = 0, taken = 0, last = 0;
    wire ivld = !rst && sent < {frames * FRAME} && cycle >= ready;
    wire ordy = !rst && (taken == 0 || cycle >= last + {GAP});
    always #5 clk = !clk;
    fifo #(.DEPTH({case}), .DATA_WIDTH(8), .RAM_STYLE("auto")) dut (
        .clk, .rst, .idat(sent[7:0]), .ivld, .irdy, .odat, .ovld, .ordy);
    always @(posedge clk) if (!rst) begin
        automatic int next = sent + 1;
        automatic int due = next / {FRAME} * {PERIOD} + next % {FRAME};
        automatic int gap = next % {FRAME} == 0 ? {PERIOD - FRAME + 1} : 1;
        cycle <= cycle + 1;
        if (ivld && irdy) begin
            $display("LEFT %0d", cycle);
            sent <= next;
            ready <= due > cycle + gap ? due : cycle + gap;
        end
        if (ovld && ordy) begin
            $display("TAKEN %0d", cycle);
            taken <= taken + 1;
            last <= cycle;
        end
    end
    initial begin
        repeat (4) @(negedge clk);
        rst = 0;
        wait (taken == {frames * FRAME});
        $display("FIFO_SIZING_PASS");
        $finish;
    end
    initial begin #1000000; $fatal(1, "FIFO sizing watchdog"); end
endmodule
""",
            directory,
            toolchain=host_toolchain(),
        )
        left = _cycles(log, "LEFT")
        late = [left[(k + 1) * FRAME - 1] - (k * PERIOD + FRAME - 1) for k in range(frames)]
        flow = simulate(burst, slow, PERIOD, case, frames)
        assert flow is not None
        assert left == list(flow.left)
        assert late == list(flow.late)
        if case == depth:
            assert late == [0] * frames
        else:
            assert max(late) > 0


# -- the presentation by its events, against the model beat by beat --------------------------


def simulate_by_beat(
    supply: Pattern, accept: Any, period: int, depth: int, frames: int
) -> Flow | None:
    """The reference ``simulate`` is checked against: the model as reviewed and checked in
    XSim (``test_input_gen_accepts_its_input_at_the_cycles_its_model_predicts``), an
    input_gen's presentation stepped beat by beat. A test oracle, nothing in ``src`` reads
    it."""
    words, gaps = len(supply.times), supply.gaps()
    accept_gaps = accept.gaps() if isinstance(accept, Pattern) else accept.reads.gaps()
    left: list[int] = []
    taken: list[int] = []
    late: list[int] = []
    buffered = accept if isinstance(accept, Buffered) else None
    shown: list[int] = []
    freed_at: list[int] = []
    freed_count = 0

    def present_until(count: int) -> bool:
        nonlocal freed_count
        assert buffered is not None
        buffer, frame = buffered.buffer, len(buffered.buffer.reads)
        frame_words, read_gaps = buffer.freed[-1], accept_gaps
        while freed_count < count:
            beat = len(shown)
            k, i = divmod(beat, frame)
            word = k * frame_words + buffer.reads[i]
            if word >= len(taken):
                return False
            at = taken[word] + 2
            if shown:
                at = max(at, shown[-1] + read_gaps[i])
            shown.append(at)
            now = k * frame_words + buffer.freed[i]
            while freed_count < now:
                freed_at.append(at)
                freed_count += 1
        return True

    for n in range(frames * words):
        k, j = divmod(n, words)
        ready = k * period + supply.times[j]
        if left:
            ready = max(ready, left[-1] + gaps[j])
        if depth and n >= depth:
            ready = max(ready, taken[n - depth] + 1)
        at = ready + (FIFO_LATENCY if depth else 0)
        if taken:
            at = max(at, taken[-1] + 1)
        if buffered is None:
            assert isinstance(accept, Pattern)
            reads = accept_gaps[n % len(accept.times)]
            if taken:
                at = max(at, taken[-1] + reads)
        else:
            room = n - buffered.buffer.capacity + 1
            if room > 0:
                if not present_until(room):
                    return None
                at = max(at, freed_at[room - 1] + 1)
        taken.append(at)
        left.append(at if not depth else ready)
        if j == words - 1:
            late.append(left[-1] - (k * period + supply.times[-1]))
    return Flow(tuple(left), tuple(taken), tuple(late))


def _random_pattern(rng: random.Random, beats: int, slack: int) -> Pattern:
    times = [rng.randint(0, 3)]
    for _ in range(beats - 1):
        times.append(times[-1] + rng.choice((0, 1, 1, 1, 2, 3, 5)))
    return Pattern(tuple(times), times[-1] - times[0] + 1 + rng.randint(0, slack))


def _random_buffer(rng: random.Random, frame_words: int) -> NestBuffer:
    """A nest-like buffer: its frame's words read in passes, some re-read, each freed at or
    after its last read; now and then freed counts out of order, or beyond the frame's own
    words before its last beat (``simulate`` reads any)."""
    reads = [
        word
        for sweep in range(rng.randint(1, 5))
        for word in range(frame_words)
        if sweep == 0 or rng.random() < 0.7
    ]
    last = {word: beat for beat, word in enumerate(reads)}
    order = sorted(range(frame_words), key=last.__getitem__)
    freed, count = [], 0
    for beat in range(len(reads)):
        while count < frame_words and last[order[count]] <= beat and rng.random() < 0.8:
            count += 1
        freed.append(count)
    if rng.random() < 0.1:
        freed = [rng.randint(0, frame_words + 2) for _ in reads]
    freed[-1] = frame_words
    return NestBuffer(rng.randint(1, 3 * frame_words), tuple(reads), tuple(freed))


def test_the_presentation_by_its_events_is_the_model_beat_by_beat_on_random_sides() -> None:
    """The same ``Flow`` (or deadlock), word for word, on random patterns and nest-like
    buffers, a stream frame of one to three of the buffer's frames (the Chain's
    ``levels``: three words through an input_gen frame of one)."""
    rng = random.Random(20261014)
    outcomes = []
    for _ in range(1500):
        frame_words = rng.randint(1, 24)
        frames_a_stream = rng.choice((1, 1, 2, 3))
        supply = _random_pattern(rng, frame_words * frames_a_stream, 30)
        accept: Pattern | Buffered
        if rng.random() < 0.6:
            buffer = _random_buffer(rng, frame_words)
            accept = Buffered(buffer, _random_pattern(rng, len(buffer.reads) * frames_a_stream, 10))
            span = max(supply.beats, accept.reads.beats)
        else:
            accept = _random_pattern(rng, frame_words, 30)
            span = max(supply.beats, accept.beats)
        period = span + rng.randint(0, span)
        depth = rng.choice((0, 0, 1, 2, 3, 5, 8, rng.randint(0, 3 * frame_words)))
        frames = rng.choice((2, 4, 16))
        flow = simulate(supply, accept, period, depth, frames)
        assert flow == simulate_by_beat(supply, accept, period, depth, frames)
        outcomes.append((isinstance(accept, Buffered), flow is None))
    # Buffered sides both flowing and deadlocked were compared.
    assert {(True, False), (True, True), (False, False)} <= set(outcomes)


def _sizing_sides() -> list[tuple[Pattern, Pattern | Buffered, tuple[int, ...]]]:
    """The sizing tests' patterns and ``NESTS``, each nest read every beat and every third,
    with periods around their spans."""
    found: list[tuple[Pattern, Pattern | Buffered, tuple[int, ...]]] = [
        (BURST, SLOW, (16, 20, 24, 32)),
        (Pattern(tuple(range(8)), 8), SLOW, (16,)),
        (_burst(32), _every(2, 32), (32, 40, 48, 64)),
        (_burst(48), _every(3, 48), (48, 64)),
        (_burst(32), _every(1, 32), (32,)),
    ]
    for frame, dims, coefs in NESTS:
        beats, buffer = prod(dims), nest_buffer(frame, dims, coefs)
        supply = Pattern(tuple(range(frame)), frame)
        reader = Pattern(tuple(range(beats)), beats)
        slow = Pattern(tuple(range(0, 3 * beats, 3)), 3 * beats)
        found.append((supply, Buffered(buffer, reader), (frame, beats, 2 * beats)))
        found.append((supply, Buffered(buffer, slow), (3 * beats, 4 * beats)))
    found.append(
        (
            Pattern(tuple(range(4, 200, 4)), 196),
            Buffered(nest_buffer(49, (4, 49), (0, 1)), Pattern(tuple(range(196)), 196)),
            (196, 250),
        )
    )
    return found


def test_the_presentation_by_its_events_is_the_model_beat_by_beat_on_the_sizing_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same ``Flow`` at every depth to 13 words, and the same least depth, on the
    sizing tests' bursts and input_gen's nests."""
    for supply, accept, periods in _sizing_sides():
        for period in periods:
            for depth in range(14):
                for frames in (4, 16):
                    expected = simulate_by_beat(supply, accept, period, depth, frames)
                    assert simulate(supply, accept, period, depth, frames) == expected
            if supply.beats <= period:
                least = least_depth(supply, accept, period)
                with monkeypatch.context() as patched:
                    patched.setattr(fifo_sizing, "simulate", simulate_by_beat)
                    assert least_depth(supply, accept, period) == least
