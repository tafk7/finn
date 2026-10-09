# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FIFO sizing from both ends' beat patterns: input_gen's buffer as its RTL derives it,
the throttle, the least depth, and the strategy proposing it through the seam.

The strategy's tests run on the Chain (``kernels.chain``), as the seam's do
(``test_explore``); the Chain needs no FIFO at any folding, so a FIFO it places and a
FIFO the seam refuses are made controllable by a double of the per-channel sizing.
"""

from __future__ import annotations

from math import prod
from pathlib import Path
from typing import Any

import pytest

import finn.kernels.explore as explore_module
from finn import resources
from finn.core.executors.xsim.rtl import simulate as simulate_rtl
from finn.core.space import design_space
from finn.harness.toolchain import finnlib_root
from finn.kernels import input_generator
from finn.kernels.artifacts.rtl import evaluate
from finn.kernels.channels import Channel
from finn.kernels.explore import (
    Completed,
    ExploreError,
    Ranked,
    Refused,
    Seam,
    SizeFifos,
    TargetCycles,
    explore,
)
from finn.kernels.fifo import least_depth_holding
from finn.kernels.fifo_sizing import (
    NotModelled,
    Pattern,
    Replay,
    Sized,
    converted,
    ends,
    least_depth,
    simulate,
)
from finn.kernels.input_generator import nest_buffer, nest_geometry
from kernels import chain
from kernels.helpers import FULL_DSP48E2, Lanes
from kernels.xsim import requires_xsim

MEMBERS = ("x", "w1", "hidden", "levels", "w2", "y", "first", "activate", "second")


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
    replay = Replay(nest_buffer(49, (4, 49), (0, 1)), Pattern(tuple(range(196)), 196))
    assert least_depth(Pattern(tuple(range(4, 200, 4)), 196), replay, 196) == 0


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


# -- a channel's ends, from the paces its ports state -----------------------------------------


def _folded() -> tuple[Seam, Any]:
    explorer = Seam(MEMBERS, platform=FULL_DSP48E2)
    return explorer, TargetCycles(12).explore(explorer, design_space(chain.Chain()))


def test_a_channel_s_ends_are_read_from_the_paces_its_ports_state() -> None:
    _, point = _folded()
    # hidden: first (PE 1, SIMD 4) closes a reduction of one step each beat, a lane a
    # word; the output adapter's vpc pairs them into activate's two lanes, a word the
    # cycle after its second lane. activate reads every beat.
    supply, accept = ends(point.hidden)
    assert supply == Pattern((2, 4, 6, 8, 10, 12), 12)
    assert accept == Pattern(tuple(range(6)), 6)
    # levels: activate's six beats, four lanes a word after its vpc, into second's
    # input_gen, which second reads every beat.
    supply, accept = ends(point.levels)
    assert supply == Pattern((2, 4, 6), 6)
    assert isinstance(accept, Replay)
    assert accept.reads == Pattern(tuple(range(12)), 12)


def test_a_boundary_and_a_memory_are_not_modelled() -> None:
    _, point = _folded()
    with pytest.raises(NotModelled, match="a boundary: not modelled"):
        ends(point.x)
    with pytest.raises(NotModelled, match="a memory source: paced by its consumer"):
        ends(point.w1)


# -- the strategy ---------------------------------------------------------------------------


def test_size_fifos_proposes_every_open_transport_in_one_batch_and_says_why() -> None:
    explorer, point = _folded()
    strategy = SizeFifos()
    attempts = explorer.attempts
    sized = strategy.explore(explorer, point)
    assert explorer.attempts == attempts + 1
    transports = {
        name: getattr(sized, name) for name in MEMBERS if isinstance(getattr(sized, name), Channel)
    }
    assert {
        key: value for key, value in explorer.chosen(sized).items() if key.endswith(".transport")
    } == {f"{name}.transport": "direct" for name in transports}
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


def test_before_folding_it_sizes_at_the_completed_folding() -> None:
    """Q-B: with no folding chosen, it reads the period and the ends from the copy the
    seam completes (the baseline folding), commits the transports only, and the seam
    records what it read."""
    explorer, point = Seam(MEMBERS, platform=FULL_DSP48E2), design_space(chain.Chain())
    strategy = SizeFifos()
    sized = strategy.explore(explorer, point)
    (read,) = explorer.reads
    assert read["first.compute.packed.pe"] == 1
    completed = explorer.cost(explorer.complete(point).point).bottleneck
    assert completed is not None and strategy.report()["period"] == completed.cycles
    assert all(key.endswith(".transport") for key in explorer.chosen(sized))
    assert explorer.chosen(sized)

    # A policy that completes nothing leaves the period unknown, and it says so.
    class Nothing:
        name = label = "nothing"

        def complete(
            self, seam: Seam, point: Any, *, sizing: bool = False, members: Any = None
        ) -> Completed[Any]:
            return Completed(point, {}, {}, None, ())

    with pytest.raises(ExploreError, match="needs every member's cycles"):
        SizeFifos().explore(Seam(MEMBERS, platform=FULL_DSP48E2, completion=Nothing()), point)


def _hidden_needs_eight(channel: Channel, period: int, **options: Any) -> Sized:
    """A double of the per-channel sizing: ``hidden`` (first's output) needs a FIFO of
    eight words, every other channel none."""
    if channel.endpoints.source_owner == "first.compute.packed.y":
        return Sized(8, "least 8 words", 24, 8)
    return Sized(0, "direct absorbs it", 8, 0)


def test_a_fifo_is_proposed_with_its_depth_and_memory_in_the_same_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explorer, point = _folded()
    monkeypatch.setattr(explore_module, "size", _hidden_needs_eight)
    strategy = SizeFifos()
    found = strategy.explore(explorer, point)
    chosen = explorer.chosen(found)
    assert chosen["hidden.transport"] == "fifo"
    assert chosen["hidden.transport.fifo.buffer.depth"] == 8
    assert chosen["hidden.transport.fifo.buffer.ram_style"] == "auto"
    assert strategy.channels()["hidden"]["bits"] == 8 * 24
    assert strategy.report()["fifo_bits"] == 8 * 24
    # The rest of the chain explores to the end around it.
    done = explore(explorer, found, [Ranked(Lanes())])
    assert explorer.refusals(done) == {}


def test_a_refused_fifo_falls_back_to_direct_and_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    _, point = _folded()

    class Refusing(Seam):
        def attempt(self, point: Any, batch: Any) -> Any:
            if batch.get("hidden.transport") == "fifo":
                self.attempts += 1
                return Refused({"hidden": "fifo-markers: what arrives carries markers"})
            return super().attempt(point, batch)

    monkeypatch.setattr(explore_module, "size", _hidden_needs_eight)
    refusing = Refusing(MEMBERS, platform=FULL_DSP48E2)
    strategy = SizeFifos()
    found = strategy.explore(refusing, point)
    assert refusing.chosen(found)["hidden.transport"] == "direct"
    row = strategy.channels()["hidden"]
    assert row["transport"] == "direct" and row["bits"] == 0
    assert (
        row["why"] == "least 8 words; refused: hidden: fifo-markers: what arrives carries markers"
    )


def test_only_the_analytical_method_exists() -> None:
    with pytest.raises(ExploreError, match="analytical"):
        SizeFifos(method="simulated")
    with pytest.raises(ExploreError, match="margin"):
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
    )
    measured = _cycles(log, "ACCEPT")
    supply = Pattern(tuple(range(frame)), frame)  # always valid: never ahead of the source
    accept = Replay(nest_buffer(frame, dims, coefs), Pattern(tuple(range(beats)), beats))
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
