# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The DSE seam over a root: choices, attempts, cost, and the strategies that search it.

On the Chain (``kernels.chain``: MatMul ``first``, Thresholding ``activate`` at a
fixed PE, MatMul ``second``; three rows of four), nothing chosen, its platform at
5 ns. No ONNX: the seam reads a root and its members; owners are the KernelOps'
(``tests/kernel_ops/test_explore.py``).
"""

from __future__ import annotations

from typing import Any

import pytest

from finn.core.space import design_space
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.explore import (
    Accepted,
    Bottleneck,
    ExploreError,
    Placeholder,
    Refused,
    Seam,
    TargetCycles,
    TargetThroughput,
    explore,
)
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernels import chain
from kernels.helpers import FULL_DSP48E2

MEMBERS = ("x", "w1", "hidden", "levels", "w2", "y", "first", "activate", "second")


def seam() -> tuple[Seam, Any]:
    return Seam(MEMBERS, platform=FULL_DSP48E2), design_space(chain.Chain())


def test_a_choice_names_its_space_class_and_whether_its_cases_are_ordered() -> None:
    explorer, point = seam()
    offered = {choice.key: choice for choice in explorer.choices(point)}
    pe = offered["first.compute.packed.pe"]
    assert pe.space_type is PackedDotpKernel and pe.ordered and pe.cases == (1, 2, 4)
    assert offered["activate.ultra_stages"].space_type is ThresholdingAxiKernel
    # A list of values states no order; a forced Decision is never a choice.
    assert not offered["levels.transport"].ordered
    assert not offered["first.compute.packed.reducer"].ordered
    assert "first.compute" not in offered and "w1.source" not in offered


def test_an_attempt_is_refused_for_a_forced_decision_or_a_case_that_is_not_viable() -> None:
    explorer, point = seam()
    forced = explorer.attempt(point, {"first.compute": "int8_dsp58"})
    assert isinstance(forced, Refused) and "forced to 'packed'" in forced.why["first.compute"]
    three = explorer.attempt(point, {"first.compute.packed.pe": 3})
    assert isinstance(three, Refused) and "3 is not viable" in three.why["first.compute.packed.pe"]
    assert isinstance(explorer.attempt(point, {"nowhere.pe": 1}), Refused)


def test_an_attempt_never_changes_a_committed_choice() -> None:
    explorer, point = seam()
    direct = explorer.attempt(point, {"y.transport": "direct"})
    assert isinstance(direct, Accepted)
    again = explorer.attempt(direct.point, {"y.transport": "fifo"})
    assert isinstance(again, Refused) and again.why == {"y.transport": "committed to 'direct'"}


def test_an_attempt_that_leaves_a_decision_without_a_viable_case_is_refused() -> None:
    """Viable is not feasible: a batch can dead-end another Decision, and the seam says
    which and why."""
    explorer, point = seam()
    batch = {"first.compute.packed.simd": 1, "first.compute.packed.compute_pumping": True}
    outcome = explorer.attempt(point, batch)
    assert isinstance(outcome, Refused)
    assert "dotp-pumping" in "".join(outcome.why.values())
    assert isinstance(explorer.attempt(point, {**batch, "first.compute.packed.simd": 2}), Accepted)


def test_a_batch_commits_a_selector_and_a_choice_nested_under_it_together() -> None:
    explorer, point = seam()
    batch = {"hidden.transport": "fifo", "hidden.transport.fifo.buffer.depth": 16}
    outcome = explorer.attempt(point, batch)
    assert isinstance(outcome, Accepted)
    assert explorer.chosen(outcome.point) == batch
    # Alone, the nested choice does not apply: the engine refuses it.
    alone = explorer.attempt(point, {"hidden.transport.fifo.buffer.depth": 16})
    assert isinstance(alone, Refused) and list(alone.why) == ["hidden.transport.fifo.buffer.depth"]


def test_a_fifo_depth_is_known_by_membership_and_proposed_through_an_attempt() -> None:
    explorer, point = seam()
    fifo = explorer.attempt(point, {"levels.transport": "fifo"})
    assert isinstance(fifo, Accepted)
    offered = {choice.key: choice for choice in explorer.choices(fifo.point)}
    depth = offered["levels.transport.fifo.buffer.depth"]
    assert depth.cases is None and depth.ordered
    small = explorer.attempt(fifo.point, {depth.key: 1})
    assert isinstance(small, Refused) and "domain-membership" in small.why[depth.key]
    sized = explorer.attempt(fifo.point, {depth.key: 16})
    assert isinstance(sized, Accepted)
    # The FIFO sits before the input adapter that makes second's markers, so the
    # channel carries it: explored to the end, nothing refuses itself.
    explored = Placeholder(lanes=2).explore(explorer, sized.point)
    assert explorer.refusals(explored) == {}
    assert [stage.label for stage in explored.levels.stages][0] == "transport.fifo.buffer"
    word = explored.levels.transport.word_bits
    assert explorer.cost(explored).buffering["levels"] >= 16 * word


def test_cost_names_what_a_member_waits_on_then_its_cycles_and_the_bottleneck_ties() -> None:
    explorer, point = seam()
    cost = explorer.cost(point)
    assert cost.waiting["first"] == ("first.compute.packed.pe",)
    assert cost.cycles["activate"] == 6  # its PE is the Chain's
    assert cost.bottleneck is None
    explored = Placeholder(lanes=2).explore(explorer, point)
    cost = explorer.cost(explored)
    assert cost.cycles["first"] == explored.first.compute.schedule.beat_count == 3 * 2 * 2
    # Every member at the most cycles is named: the activation replays feed the MatMuls
    # at their pace.
    assert cost.bottleneck == Bottleneck(("x", "levels", "first", "second"), 12)


def test_target_cycles_folds_the_least_parallelism_that_meets_the_budget() -> None:
    explorer, point = seam()
    strategy = TargetCycles(12)
    folded = strategy.explore(explorer, point)
    chosen = explorer.chosen(folded)
    # 3 rows x 4 x 4 at 12 cycles: PE x SIMD = 4, the earlier axis least.
    assert chosen == {
        "first.compute.packed.pe": 1,
        "first.compute.packed.simd": 4,
        "second.compute.packed.pe": 1,
        "second.compute.packed.simd": 4,
    }
    assert strategy.report()["bottleneck"] == {
        "members": ["x", "hidden", "levels", "first", "second"],
        "cycles": 12,
    }
    assert strategy.relaxed_to is None


def test_a_budget_no_member_meets_is_relaxed_to_the_bottleneck_reached() -> None:
    explorer, point = seam()
    strategy = TargetCycles(1)
    folded = explore(explorer, point, [strategy, Placeholder()])
    # The activation's fixed PE takes 6 cycles: the MatMuls fold to meet that, no faster.
    assert strategy.relaxed_to == 6
    chosen = explorer.chosen(folded)
    assert chosen["first.compute.packed.pe"] == 2 and chosen["first.compute.packed.simd"] == 4
    assert explorer.cost(folded).bottleneck == Bottleneck(
        ("x", "levels", "first", "activate", "second"), 6
    )
    unrelaxed = TargetCycles(1, relax=False).explore(explorer, point)
    assert explorer.chosen(unrelaxed)["first.compute.packed.pe"] == 4


def test_a_target_throughput_is_a_cycles_budget_at_the_platform_s_clock() -> None:
    explorer, point = seam()
    # 5 ns: 1e9 / (5 x 16,666,666) frames a second leaves 12 cycles a frame.
    strategy = TargetThroughput(16_666_666)
    folded = strategy.explore(explorer, point)
    assert strategy.cycles == 12
    assert explorer.chosen(folded) == explorer.chosen(TargetCycles(12).explore(explorer, point))
    assert strategy.report()["fps"] == 16_666_666
    with pytest.raises(ExploreError, match="needs the seam's platform"):
        TargetThroughput(1000).explore(Seam(MEMBERS), point)
