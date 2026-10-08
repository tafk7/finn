# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The DSE seam over a root: choices, attempts, cost, completion, and the strategies that
search it.

On the Chain (``kernels.chain``: MatMul ``first``, Thresholding ``activate`` at a
fixed PE, MatMul ``second``; three rows of four), nothing chosen, its platform at
5 ns. No ONNX: the seam reads a root and its members; owners are the KernelOps'
(``tests/kernel_ops/test_explore.py``).
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import pytest

import finn.kernels.explore as explore_module
from finn.core.space import design_space
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.explore import (
    Accepted,
    Baseline,
    Bottleneck,
    ExploreError,
    MaxThroughput,
    Placeholder,
    Ranked,
    Refused,
    ResourceBudgetWarning,
    Seam,
    TargetCycles,
    TargetThroughput,
    explore,
)
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.utilization import Resources
from kernels import chain
from kernels.helpers import FULL_DSP48E2, Lanes

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
    assert depth.cases is None and depth.ordered and depth.required
    small = explorer.attempt(fifo.point, {depth.key: 1})
    assert isinstance(small, Refused) and "domain-membership" in small.why[depth.key]
    sized = explorer.attempt(fifo.point, {depth.key: 16})
    assert isinstance(sized, Accepted)
    # The FIFO sits before the input adapter that makes second's markers, so the
    # channel carries it: explored to the end, nothing refuses itself.
    explored = Ranked(Lanes(2)).explore(explorer, sized.point)
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
    explored = Ranked(Lanes(2)).explore(explorer, point)
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
    folded = explore(explorer, point, [strategy])
    # The activation's fixed PE takes 6 cycles: the MatMuls fold to meet that, no faster.
    assert strategy.relaxed_to == 6
    chosen = explorer.chosen(folded)
    assert chosen["first.compute.packed.pe"] == 2 and chosen["first.compute.packed.simd"] == 4
    assert explorer.cost(explorer.complete(folded).point).bottleneck == Bottleneck(
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


def test_the_baseline_completes_every_choice_but_a_required_one_on_a_copy() -> None:
    explorer, point = seam()
    completed = Baseline().complete(explorer, point)
    # The point is unchanged; the copy holds the baseline: each kernel's first case.
    assert explorer.chosen(point) == {}
    assert completed.values == explorer.chosen(completed.point)
    assert completed.values["first.compute.packed.pe"] == 1
    assert completed.values["first.compute.packed.reducer"] == "tree"
    # Where it is costed, a transport takes its first case.
    assert {value for key, value in completed.values.items() if key.endswith("transport")} == {
        "direct"
    }
    assert set(completed.made_by.values()) == {"baseline"} and completed.sizing is None
    assert completed.open == () and explorer.cost(completed.point).bottleneck is not None
    # A FIFO chosen with no depth: the depth is required, so it is left open, named.
    fifo = explorer.attempt(point, {"levels.transport": "fifo"})
    assert isinstance(fifo, Accepted)
    left = Baseline().complete(explorer, fifo.point).open
    assert [(choice.key, choice.required) for choice in left] == [
        ("levels.transport.fifo.buffer.depth", True)
    ]


def test_the_debug_placeholder_refuses_a_required_choice_it_has_no_case_for() -> None:
    explorer, point = seam()
    completed = Placeholder().complete(explorer, point)
    assert set(completed.made_by.values()) == {"DEBUG: completed by placeholder"}
    fifo = explorer.attempt(point, {"levels.transport": "fifo"})
    assert isinstance(fifo, Accepted)
    with pytest.raises(ExploreError, match="no case to take .*levels.transport.fifo.buffer.depth"):
        Placeholder().complete(explorer, fifo.point)


def test_at_hardware_generation_the_baseline_sizes_fifos_at_the_completed_folding() -> None:
    explorer, point = seam()
    completed = Baseline().complete(explorer, point, sizing=True)
    assert completed.sizing is not None and completed.sizing["strategy"] == "size_fifos"
    transports = {key for key in completed.values if key.endswith(".transport")}
    assert transports and {completed.made_by[key] for key in transports} == {"size_fifos"}
    # The period sized at is the completed point's bottleneck.
    cost = explorer.cost(completed.point)
    assert cost.bottleneck is not None and completed.sizing["period"] == cost.bottleneck.cycles
    assert completed.open == ()
    # Every other value is the baseline's, as where it is costed.
    costed = Baseline().complete(explorer, point)
    assert {k: v for k, v in completed.values.items() if k not in transports} == {
        k: v for k, v in costed.values.items() if not k.endswith(".transport")
    }


def test_a_strategy_reads_a_completed_copy_and_the_seam_records_it() -> None:
    explorer, point = seam()
    assert explorer.reads == []
    completed = explorer.complete(point)
    assert explorer.reads == [completed.values]
    assert explorer.chosen(point) == {}


# -- the most throughput within resources -----------------------------------------------------

#: The Chain's part: its totals, of which a budget is a fraction. At each budget of cycles
#: the Chain folds to (``TargetCycles``), completed: 6 cycles, 478 LUT and 8 DSP; 12,
#: 375 and 8; 24, 330 and 4; 48 (the least parallelism), 325 and 2.
PART = replace(FULL_DSP48E2, resources=Resources(lut=1000, ff=2000, bram18=10, uram=0, dsp=10))


def within_part() -> tuple[Seam, Any]:
    return Seam(MEMBERS, platform=PART), design_space(chain.Chain())


def test_the_seam_states_the_root_s_own_resources_once_its_members_do() -> None:
    explorer, point = within_part()
    waiting = explorer.resources(point)
    assert isinstance(waiting, str) and waiting.startswith("waits on ")
    completed = explorer.complete(point).point
    assert explorer.resources(completed) == explorer.cost(completed).used


def test_max_throughput_keeps_the_least_budget_whose_point_fits() -> None:
    explorer, point = within_part()
    strategy = MaxThroughput({"dsp": 0.5})
    folded = strategy.explore(explorer, point)
    # Half the part's 10 DSPs: the 6- and 12-cycle points use 8, the 24-cycle one 4.
    assert explorer.chosen(folded) == {
        "first.compute.packed.pe": 1,
        "first.compute.packed.simd": 2,
        "second.compute.packed.pe": 1,
        "second.compute.packed.simd": 2,
    }
    # The folded point is returned with the rest open, for the explorers after it.
    assert explorer.choices(folded)
    report = strategy.report()
    assert report["budget"] == {"dsp": 5}
    assert report["bottleneck"] == {"members": ["x", "levels", "first", "second"], "cycles": 24}
    assert report["used"] == asdict(Resources(lut=330, ff=380, dsp=4))
    assert report["ratio"] == {"dsp": 0.8} and report["binding"] == "dsp" and report["fits"]
    tried = report["tried"]
    assert isinstance(tried, list)
    # The fastest (budget 1, relaxed to 6), no budget (48), then bisection from 7 to 48.
    assert [row["cycles"] for row in tried][:2] == [1, None]
    assert (tried[0]["relaxed_to"], tried[0]["bottleneck"], tried[0]["fits"]) == (6, 6, False)
    assert (tried[1]["bottleneck"], tried[1]["used"], tried[1]["fits"]) == (48, {"dsp": 2}, True)
    assert min(row["cycles"] for row in tried[2:] if row["fits"]) == 24
    assert report["fastest"] == 6 and report["monotone"] and report["departures"] == []


def test_max_throughput_folds_each_budget_as_target_cycles_alone_asking_each_point_once() -> None:
    """The search's folds share what they ask of the seam: each budget tried folds to the
    point ``TargetCycles`` alone folds it to, and its bottleneck and resources are those
    of that point completed, in fewer attempts than folding and completing each alone."""
    explorer, point = within_part()
    strategy = MaxThroughput({"dsp": 0.5})
    strategy.explore(explorer, point)
    alone = Seam(MEMBERS, platform=PART)
    for tried in strategy.tried:
        budget = explore_module._UNBOUNDED if tried.cycles is None else tried.cycles
        folder = TargetCycles(budget)
        folded = folder.explore(alone, point)
        assert explorer.chosen(tried.folded) == alone.chosen(folded)
        assert tried.relaxed_to == folder.relaxed_to
        completed = alone.complete(folded).point
        assert tried.reached == alone.cost(completed).bottleneck
        assert tried.used == alone.resources(completed)
    assert explorer.attempts < alone.attempts


def test_max_throughput_names_the_resource_that_binds_among_those_budgeted() -> None:
    explorer, point = within_part()
    # The fastest point fits both budgets: one fold, LUT the nearer its budget.
    both = MaxThroughput({"lut": 0.5, "dsp": 1.0})
    both.explore(explorer, point)
    report = both.report()
    assert report["ratio"] == {"lut": 0.956, "dsp": 0.8} and report["binding"] == "lut"
    assert [tried.cycles for tried in both.tried] == [1]


def test_max_throughput_warns_and_keeps_the_least_parallelism_where_nothing_fits() -> None:
    explorer, point = within_part()
    strategy = MaxThroughput({"dsp": 0.1, "lut": 1.0})
    with pytest.warns(ResourceBudgetWarning, match="most of dsp: dsp 2 of 1"):
        folded = strategy.explore(explorer, point)
    assert explorer.chosen(folded)["first.compute.packed.simd"] == 1
    report = strategy.report()
    assert report["fits"] is False and report["binding"] == "dsp"
    assert [tried.cycles for tried in strategy.tried] == [1, None]


def test_max_throughput_states_where_a_budget_departs_from_its_assumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bisection assumes a larger budget folds to more cycles and fewer resources. A budget
    whose fold reaches more cycles than itself (an end's converter, SZ6) departs from
    that: here every budget from 13 to 23 folds as no budget does, to 48 cycles. The
    search is not corrected: it keeps the least budget that fits, 13, at 48 cycles where
    budget 27 reached 24, and states each departure."""

    class Departing(TargetCycles):
        def _fold_all(self, seam: Seam, point: Any, budget: int) -> Any:
            return super()._fold_all(seam, point, 48 if 13 <= budget < 24 else budget)

    monkeypatch.setattr(explore_module, "TargetCycles", Departing)
    explorer, point = within_part()
    strategy = MaxThroughput({"dsp": 0.5})
    strategy.explore(explorer, point)
    report = strategy.report()
    tried = report["tried"]
    assert isinstance(tried, list)
    assert [row["cycles"] for row in tried] == [1, None, 27, 17, 12, 15, 14, 13]
    assert report["bottleneck"] == {"members": ["x", "levels", "first", "second"], "cycles": 48}
    assert not report["monotone"]
    departures = report["departures"]
    assert isinstance(departures, list)
    assert "budget 13 relaxed to 48, reaching 48" in departures
    assert "budget 27 reached 24, fewer than budget 17's 48" in departures


def test_max_throughput_needs_a_budget_of_named_resources_and_the_part_s_totals() -> None:
    with pytest.raises(ExploreError, match="needs a budget"):
        MaxThroughput({})
    with pytest.raises(ExploreError, match=r"no resource is named \['luts'\]"):
        MaxThroughput({"luts": 0.5})
    with pytest.raises(ExploreError, match="is no budget"):
        MaxThroughput({"lut": 0})
    with pytest.raises(ExploreError, match="is no fraction"):
        MaxThroughput({"lut": "half"})  # type: ignore[dict-item]
    explorer, point = seam()
    with pytest.raises(ExploreError, match="platform's resources"):
        MaxThroughput({"lut": 0.5}).explore(explorer, point)
