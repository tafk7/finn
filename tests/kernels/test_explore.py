# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The DSE seam over a root: choices, attempts, cost, the operations that keep a point
complete (baseline, reopen, settle, set), and the strategies that search it.

On the Chain (``kernels.chain``: MatMul ``first``, Thresholding ``activate`` at a
fixed PE, MatMul ``second``; three rows of four), nothing chosen, its platform at
5 ns. No ONNX: the seam reads a root and its members; owners are the KernelOps'
(``tests/kernel_ops/test_explore.py``).
"""

from __future__ import annotations

import re
from dataclasses import asdict, replace
from typing import Any

import pytest

import finn.kernels.explore as explore_module
from finn.core.space import (
    Finding,
    FindingKind,
    design_space,
    finding_record,
    finding_text,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.channels import Channel, end_of
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.explore import (
    Accepted,
    Bottleneck,
    DebugPlaceholder,
    ExploreError,
    MaxThroughput,
    Ranked,
    Rebaseline,
    Refused,
    Seam,
    Stated,
    TargetCycles,
    TargetThroughput,
    Unsettled,
    explore,
)
from finn.kernels.matmul import MatMulKernel
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.utilization import total
from finn.kernels.values.semantics import INTEGER_TENSOR, IntegerTensor
from finn.platform.capabilities import RESOURCE_NAMES, Resources
from kernels import chain
from kernels.helpers import FULL_DSP48E2, FULL_DSP58, Lanes, Root, view_of

MEMBERS = ("x", "w1", "hidden", "levels", "w2", "y", "first", "activate", "second")


def seam() -> tuple[Seam, Any]:
    return Seam(MEMBERS, platform=FULL_DSP48E2), design_space(chain.Chain())


def folding(explorer: Seam, point: Any) -> dict[str, object]:
    """``point``'s stored folding: each PE and SIMD."""
    return {k: v for k, v in explorer.chosen(point).items() if k.endswith((".pe", ".simd"))}


#: The Chain's foldings at a budget of 12 cycles a frame: PE x SIMD = 4, the earlier axis
#: least.
FOLDED_12 = {
    "first.compute.packed.pe": 1,
    "first.compute.packed.simd": 4,
    "second.compute.packed.pe": 1,
    "second.compute.packed.simd": 4,
}


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


def said(why: Any) -> str:
    """Every finding of a refusal's ``why``, as text."""
    return "; ".join(finding_text(finding) for found in why.values() for finding in found)


def test_an_attempt_is_refused_for_a_forced_decision_or_a_case_that_is_not_viable() -> None:
    """The seam's own refusals are findings too, each owned by the key it refuses."""
    explorer, point = seam()
    forced = explorer.attempt(point, {"first.compute": "int8_dsp58"})
    assert isinstance(forced, Refused)
    (found,) = forced.why["first.compute"]
    assert (found.kind, found.code, found.owner) == (
        FindingKind.REJECTION,
        "forced",
        "first.compute",
    )
    assert "forced to 'packed'" in found.message
    three = explorer.attempt(point, {"first.compute.packed.pe": 3})
    assert isinstance(three, Refused)
    (found,) = three.why["first.compute.packed.pe"]
    assert found.code == "case-not-viable" and "3 is not viable" in found.message
    nowhere = explorer.attempt(point, {"nowhere.pe": 1})
    assert isinstance(nowhere, Refused)
    assert [f.code for f in nowhere.why["nowhere.pe"]] == ["not-a-decision"]


def test_an_attempt_replaces_a_stored_value_but_never_a_held_one() -> None:
    explorer, point = seam()
    direct = explorer.attempt(point, {"y.transport": "direct"})
    assert isinstance(direct, Accepted)
    fifo = explorer.attempt(direct.point, {"y.transport": "fifo"})
    assert isinstance(fifo, Accepted) and explorer.chosen(fifo.point) == {"y.transport": "fifo"}
    # A key held for the run (a pin) takes no other value.
    explorer.held["y.transport"] = "direct"
    held = explorer.attempt(direct.point, {"y.transport": "fifo"})
    assert isinstance(held, Refused)
    assert held.why == {
        "y.transport": (
            Finding(
                FindingKind.REJECTION,
                "held",
                "y.transport",
                "held to 'direct' by a pin",
            ),
        )
    }
    assert isinstance(explorer.attempt(direct.point, {"y.transport": "direct"}), Accepted)


def test_an_attempt_replacing_what_forces_a_choice_may_set_it_in_the_same_batch() -> None:
    """At SIMD 1 pumping is forced off; a batch replacing SIMD by 2 may set it on, and a
    batch that leaves it forced may not."""
    explorer, point = seam()
    base = explorer.baseline(point)
    pumped = {"first.compute.packed.simd": 2, "first.compute.packed.compute_pumping": True}
    outcome = explorer.attempt(base, pumped)
    assert isinstance(outcome, Accepted)
    assert explorer.chosen(outcome.point)["first.compute.packed.compute_pumping"] is True
    alone = explorer.attempt(base, {"first.compute.packed.compute_pumping": True})
    assert isinstance(alone, Refused) and "forced to False" in said(alone.why)


def test_an_attempt_that_leaves_a_decision_without_a_viable_case_is_refused() -> None:
    """Viable is not feasible: a batch can dead-end another Decision, and the seam says
    which and why."""
    explorer, point = seam()
    batch = {"first.compute.packed.simd": 1, "first.compute.packed.compute_pumping": True}
    outcome = explorer.attempt(point, batch)
    assert isinstance(outcome, Refused)
    assert "dotp-pumping" in said(outcome.why)
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
    assert isinstance(small, Refused)
    assert "domain-membership" in [finding.code for finding in small.why[depth.key]]
    sized = explorer.attempt(fifo.point, {depth.key: 16})
    assert isinstance(sized, Accepted)
    # The FIFO sits before the input adapter that makes second's markers, so the
    # channel carries it: folded and settled around it (no transport is a folding
    # strategy's), nothing refuses itself.
    explored = TargetCycles(12).explore(explorer, sized.point)
    assert explorer.refusals(explored) == {}
    assert explorer.chosen(explored)[depth.key] == 16
    labels = [stage.label for stage in explored.levels.stages]
    assert labels.index("transport.fifo.buffer") < labels.index("adapter.input_gen.input_gen")
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
    # 3 rows x 4 x 4 at 12 cycles: PE x SIMD = 4, the earlier axis least.
    assert folding(explorer, folded) == FOLDED_12
    # The point is settled: every other choice at its baseline, none open.
    assert explorer.choices(folded) == ()
    # At SIMD 4, each MatMul's pumping is a choice: its baseline, off.
    assert explorer.chosen(folded) == {
        **explorer.chosen(explorer.baseline(point)),
        **FOLDED_12,
        "first.compute.packed.compute_pumping": False,
        "second.compute.packed.compute_pumping": False,
    }
    report = strategy.report()
    assert report["bottleneck"] == {
        "members": ["x", "hidden", "levels", "first", "second"],
        "cycles": 12,
    }
    assert (report["bottleneck_of"], report["met"], report["unfolded"]) == ("folded", True, {})
    # The Chain's cores are forced (a DSP48E2 platform): no case is tried.
    assert report["cases"] == {}
    assert strategy.relaxed_to is None


def test_a_budget_no_member_meets_is_relaxed_to_the_bottleneck_reached() -> None:
    explorer, point = seam()
    strategy = TargetCycles(1)
    folded = explore(explorer, point, [strategy])
    # The activation's fixed PE takes 6 cycles: the MatMuls fold to meet that, no faster.
    assert strategy.relaxed_to == 6
    chosen = explorer.chosen(folded)
    assert chosen["first.compute.packed.pe"] == 2 and chosen["first.compute.packed.simd"] == 4
    assert explorer.cost(folded).bottleneck == Bottleneck(
        ("x", "levels", "first", "activate", "second"), 6
    )
    unrelaxed = TargetCycles(1, relax=False)
    assert explorer.chosen(unrelaxed.explore(explorer, point))["first.compute.packed.pe"] == 4
    # A miss is stated: the bottleneck reached and that it does not meet the budget.
    report = unrelaxed.report()
    assert report["bottleneck"]["cycles"] == 6  # type: ignore[index]
    assert "activate" in report["bottleneck"]["members"]  # type: ignore[index]
    assert (report["relaxed_to"], report["met"]) == (None, False)
    assert strategy.report()["met"] is False


def test_target_cycles_states_the_bottleneck_of_the_settled_point_where_it_leaves_one_open() -> (
    None
):
    """A member whose cycles wait on a choice that is not its own is left to settling;
    the report still states the bottleneck, of the point settled, and whether it meets
    the budget."""

    class Partial(TargetCycles):
        def _fold(self, seam: Seam, point: Any, name: str, budget: int) -> Any:
            if name == "second":
                return explore_module._Folded(None, point, "left by the test")
            return super()._fold(seam, point, name, budget)

    explorer, point = seam()
    strategy = Partial(12)
    folded = strategy.explore(explorer, point)
    # Settled: second at its baseline, the least parallelism.
    assert explorer.chosen(folded)["second.compute.packed.pe"] == 1
    assert explorer.chosen(folded)["second.compute.packed.simd"] == 1
    report = strategy.report()
    # second, and the channels whose cycles wait on its folding.
    unfolded = report["unfolded"]
    assert isinstance(unfolded, dict) and unfolded["second"] == "left by the test"
    assert all("second." in why for name, why in unfolded.items() if name != "second")
    # Settled at its baseline, second takes 48 cycles: the budget is missed, and said.
    assert report["bottleneck"]["cycles"] == 48  # type: ignore[index]
    assert "second" in report["bottleneck"]["members"]  # type: ignore[index]
    assert (report["bottleneck_of"], report["met"]) == ("settled", False)


def dsp58_matmul() -> tuple[Seam, Any]:
    """The Chain's first MatMul alone, on a DSP58 platform: its core an open choice."""

    x_tensor = Tensor((chain.ROWS, chain.INPUTS), ScalarEncoding(chain.A))

    class Single(Root):
        y_tensor = view_of("first", "result_tensor")
        w_tensor = view_of("first", "w_carried")
        w_contents: IntegerTensor = view_of("first", "w_stored", INTEGER_TENSOR)
        x_end = end_of("first", "x_end")
        w_end = end_of("first", "w_end")
        y_end = end_of("first", "y_end")
        x = Channel(platform=FULL_DSP58, tensor=x_tensor, port="s_axis_0", consumer=x_end)
        w = Channel(tensor=w_tensor, contents=w_contents, consumer=w_end, platform=FULL_DSP58)
        y = Channel(tensor=y_tensor, port="m_axis_0", producer=y_end, platform=FULL_DSP58)
        first = MatMulKernel(
            m=chain.ROWS,
            n=chain.HIDDEN,
            k=chain.INPUTS,
            activation_dtype=chain.A,
            weights_dtype=chain.W,
            platform=FULL_DSP58,
            x_tensor=x_tensor,
            y_tensor=y_tensor,
            w_contents=chain.W1,
        )

    return Seam(("x", "w", "y", "first"), platform=FULL_DSP58), design_space(Single())


def test_target_cycles_folds_each_case_of_an_unordered_choice_its_cycles_wait_on() -> None:
    """On a DSP58 platform a MatMul's core is an open, unordered choice (``packed`` or
    ``int8_dsp58``) its cycles wait on. Each case is folded, and the case kept meets the
    budget with the least parallelism, then the fewest resources; no order is imposed
    on the cases, and the report names the case taken and why."""
    explorer, point = dsp58_matmul()
    offered = {choice.key: choice for choice in explorer.choices(point)}
    assert not offered["first.compute"].ordered
    assert set(offered["first.compute"].cases or ()) == {"packed", "int8_dsp58"}
    strategy = TargetCycles(6)
    folded = strategy.explore(explorer, point)
    report = strategy.report()
    assert (report["met"], report["unfolded"]) == (True, {})
    (row,) = report["cases"].values()  # type: ignore[attr-defined]
    taken = row["taken"]
    assert explorer.chosen(folded)["first.compute"] == taken
    assert explorer.chosen(folded)[f"first.compute.{taken}.simd"]
    folds = row["folded"]
    assert set(folds) == {"packed", "int8_dsp58"} and row["refused"] == {}
    # Both cores fold to the most cycles within 6; tied, the lighter by dominance is kept.
    assert {case: each["cycles"] for case, each in folds.items()} == {
        "packed": 6,
        "int8_dsp58": 6,
    }
    assert "tied on cycles" in row["why"] and "dominance" in row["why"]
    used = {case: Resources(**each["resources"]) for case, each in folds.items()}
    other = next(case for case in used if case != taken)
    assert used[taken] != used[other]
    assert all(
        getattr(used[taken], name) <= getattr(used[other], name) for name in RESOURCE_NAMES
    ) or ("listing order" in row["why"])
    # What it compared is the member settled in each case; the point it returns is settled.
    assert explorer.choices(folded) == ()


def test_target_cycles_keeps_the_case_with_the_least_parallelism_that_meets_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cycles come first: where one case meets the budget at more cycles than another, it
    is kept whatever it uses; where no case meets it, the fewest cycles."""
    explorer, point = dsp58_matmul()
    cost = Seam.cost

    def slower_int8(self: Seam, at: Any, members: Any = None) -> Any:
        found = cost(self, at, members)
        if "first" in found.cycles and self.chosen(at).get("first.compute") == "int8_dsp58":
            found.cycles = {**found.cycles, "first": found.cycles["first"] + 1}
        return found

    monkeypatch.setattr(Seam, "cost", slower_int8)
    strategy = TargetCycles(12)
    strategy.explore(explorer, point)
    (row,) = strategy.report()["cases"].values()  # type: ignore[attr-defined]
    folds = {case: each["cycles"] for case, each in row["folded"].items()}
    assert row["taken"] == max(folds, key=folds.__getitem__) and "least parallelism" in row["why"]
    assert all("resources" not in each for each in row["folded"].values())
    missed = TargetCycles(1, relax=False)
    missed.explore(explorer, point)
    (row,) = missed.report()["cases"].values()  # type: ignore[attr-defined]
    assert row["taken"] == "packed" and row["why"].startswith("no case meets the budget of 1")
    assert missed.report()["met"] is False


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


# -- keeping a point complete -------------------------------------------------------------


def test_the_baseline_takes_each_open_choice_s_first_viable_case_and_leaves_required_ones() -> None:
    """``Seam.baseline`` stores each kernel's first case, every transport ``direct``; a
    required choice stays open."""
    explorer, point = seam()
    base = explorer.baseline(point)
    values = explorer.chosen(base)
    assert values["first.compute.packed.pe"] == 1
    assert values["first.compute.packed.reducer"] == "tree"
    assert {value for key, value in values.items() if key.endswith("transport")} == {"direct"}
    assert explorer.choices(base) == () and explorer.cost(base).bottleneck is not None
    fifo = explorer.attempt(base, {"levels.transport": "fifo"})
    assert isinstance(fifo, Accepted)
    assert [(c.key, c.required) for c in explorer.choices(explorer.baseline(fifo.point))] == [
        ("levels.transport.fifo.buffer.depth", True)
    ]


def test_reopen_widens_the_clear_to_the_stored_choices_the_engine_names() -> None:
    """Clearing a MatMul's folding leaves its input channel's adapter memory unresolved:
    the engine refuses the clear naming it, and the reopen clears it too. Nothing
    outside the reach is touched."""
    explorer, point = seam()
    base = explorer.baseline(point)
    opened, reach = explorer.reopen(base, ["first.compute.packed.pe"])
    assert reach == ("first.compute.packed.pe", "x.adapter.input_gen.input_gen.ram_style")
    assert set(explorer.chosen(base)) - set(explorer.chosen(opened)) == set(reach)
    # A key not stored is nothing to reopen.
    assert explorer.reopen(point, ["first.compute.packed.pe"]) == (point, ())
    # A selector's nested choices are the engine's to name too.
    sized = explorer.attempt(
        base, {"hidden.transport": "fifo", "hidden.transport.fifo.buffer.depth": 16}
    )
    assert isinstance(sized, Accepted)
    _, reach = explorer.reopen(sized.point, ["hidden.transport"])
    assert "hidden.transport.fifo.buffer.depth" in reach


def test_settle_keeps_each_prior_value_still_accepted_and_rebaselines_the_rest() -> None:
    explorer, point = seam()
    prior = {
        **explorer.chosen(explorer.baseline(point)),
        "first.compute.packed.reducer": "compressor",
        "second.compute.packed.pe": 3,  # no divisor of four: not viable
    }
    settled = explorer.settle(point, prior)
    chosen = explorer.chosen(settled)
    assert chosen["first.compute.packed.reducer"] == "compressor"
    assert chosen["second.compute.packed.pe"] == 1  # its baseline
    assert explorer.choices(settled) == ()
    # Only the members named, where named.
    first = explorer.settle(point, prior, ("first",))
    assert {explorer.member_of(key) for key in explorer.chosen(first)} == {"first"}


def test_set_clears_a_stored_choice_the_change_makes_invalid_then_settles() -> None:
    """Pumping on at SIMD 2, then SIMD replaced by 1: the engine refuses the attempt,
    naming the pumping choice; ``set`` clears it with the change, and settling finds it
    forced off, nothing to store."""
    explorer, point = seam()
    base = explorer.baseline(point)
    pumped = explorer.set(
        base, {"first.compute.packed.simd": 2, "first.compute.packed.compute_pumping": True}
    )
    assert isinstance(pumped, Accepted)
    refused = explorer.attempt(pumped.point, {"first.compute.packed.simd": 1})
    assert isinstance(refused, Refused)
    assert "pumping requires SIMD >= 2" in said(
        {key: why for key, why in refused.why.items() if key.endswith("compute_pumping")}
    )
    replaced = explorer.set(pumped.point, {"first.compute.packed.simd": 1})
    assert isinstance(replaced, Accepted)
    chosen = explorer.chosen(replaced.point)
    assert chosen["first.compute.packed.simd"] == 1
    assert "first.compute.packed.compute_pumping" not in chosen
    assert explorer.choices(replaced.point) == ()
    # A change of the change's own key that is not taken is refused.
    assert isinstance(explorer.set(base, {"first.compute.packed.pe": 3}), Refused)


def test_a_held_choice_a_change_reaches_must_settle_back_or_the_change_is_refused() -> None:
    explorer, point = seam()
    pumped = explorer.set(
        explorer.baseline(point),
        {"first.compute.packed.simd": 2, "first.compute.packed.compute_pumping": True},
    )
    assert isinstance(pumped, Accepted)
    explorer.held["first.compute.packed.compute_pumping"] = True
    refused = explorer.set(pumped.point, {"first.compute.packed.simd": 1})
    assert isinstance(refused, Refused)
    ((key, (found,)),) = refused.why.items()
    assert (key, found.code, found.owner) == ("first.compute.packed.compute_pumping", "held", key)
    assert found.message == "held to True by a pin, now False after the change"
    # Reached and still viable, it comes back: SIMD 4 keeps pumping viable.
    kept = explorer.set(pumped.point, {"first.compute.packed.simd": 4})
    assert isinstance(kept, Accepted)
    assert explorer.chosen(kept.point)["first.compute.packed.compute_pumping"] is True
    with pytest.raises(Unsettled, match="compute_pumping: held to True"):
        explorer.settle(explorer.reopen(pumped.point, ["first.compute.packed.simd"])[0], {})


def test_a_strategy_reopens_its_scope_and_keeps_what_it_does_not_decide() -> None:
    """A strategy changes only what it decides: a reducer an earlier strategy set stays,
    though the folding strategy reopens every choice of a member; the run records what
    it reopened (the engine's reach included) and what settling kept and re-baselined."""
    explorer, point = seam()
    base = explorer.baseline(point)
    compressor = explorer.set(base, {"second.compute.packed.reducer": "compressor"})
    assert isinstance(compressor, Accepted)
    folded = TargetCycles(12).explore(explorer, compressor.point)
    chosen = explorer.chosen(folded)
    assert chosen["second.compute.packed.reducer"] == "compressor"
    assert folding(explorer, folded) == FOLDED_12
    (run,) = explorer.runs
    assert run.strategy == "target_cycles"
    assert "second.compute.packed.reducer" in run.reopened
    assert "levels.adapter.input_gen.input_gen.ram_style" in run.reopened
    assert not any(key.endswith(".transport") for key in run.reopened)
    assert run.kept["second.compute.packed.reducer"] == "compressor"
    assert set(run.kept) | set(run.baselined) <= set(run.reopened) | set(chosen)
    # Again on the folded point: the same folding, nothing new.
    again = TargetCycles(12).explore(explorer, folded)
    assert explorer.chosen(again) == chosen


def test_a_pin_holds_its_keys_and_a_later_strategy_folds_around_it() -> None:
    """Pinned first and held (SR6): the target throughput folds the other MatMul and
    keeps the pinned one; its relax step reaches the pin's bottleneck."""
    explorer, point = seam()
    base = explorer.baseline(point)
    explorer.held.update({"first.compute.packed.pe": 1, "first.compute.packed.simd": 1})
    strategy = TargetCycles(1)
    folded = strategy.explore(explorer, base)
    chosen = explorer.chosen(folded)
    assert (chosen["first.compute.packed.pe"], chosen["first.compute.packed.simd"]) == (1, 1)
    # first at 48 cycles binds: second folds to meet it, no faster.
    assert strategy.relaxed_to == 48
    assert (chosen["second.compute.packed.pe"], chosen["second.compute.packed.simd"]) == (1, 1)
    assert "first.compute.packed.pe" not in explorer.runs[-1].reopened


def test_the_baseline_strategy_resets_every_choice_not_held() -> None:
    explorer, point = seam()
    folded = TargetCycles(12).explore(explorer, explorer.baseline(point))
    explorer.held["first.compute.packed.simd"] = 4
    reset = Rebaseline().explore(explorer, folded)
    assert folding(explorer, reset) == {
        **{key: 1 for key in FOLDED_12},
        "first.compute.packed.simd": 4,
    }
    assert explorer.chosen(reset) == {
        **explorer.chosen(explorer.baseline(point)),
        "first.compute.packed.simd": 4,
    }
    assert Rebaseline().report() == {"strategy": "baseline"}


def test_the_debug_placeholder_takes_a_required_choice_s_first_case_or_refuses_it() -> None:
    """With nothing required open it sets nothing; a FIFO with no depth is a required
    choice known by membership only, which it refuses, named."""
    explorer, point = seam()
    base = explorer.baseline(point)
    placeholder = DebugPlaceholder()
    assert explorer.chosen(placeholder.explore(explorer, base)) == explorer.chosen(base)
    assert placeholder.report() == {"strategy": "placeholder", "set": {}}
    assert placeholder.label == "DEBUG: set by placeholder"
    fifo = explorer.attempt(base, {"levels.transport": "fifo"})
    assert isinstance(fifo, Accepted)
    with pytest.raises(ExploreError, match="known by membership only: .*levels.transport.fifo"):
        DebugPlaceholder().explore(explorer, fifo.point)


# -- the most throughput within resources -----------------------------------------------------

#: The Chain's part: its totals, of which a budget is a fraction. At each budget of cycles
#: the Chain folds to (``TargetCycles``), settled: 6 cycles, 482 LUT and 8 DSP; 12,
#: 379 and 8; 24, 331 and 4; 48 (the least parallelism), 328 and 2.
PART = replace(FULL_DSP48E2, resources=Resources(lut=1000, ff=2000, bram18=10, uram=0, dsp=10))


def within_part() -> tuple[Seam, Any]:
    return Seam(MEMBERS, platform=PART), design_space(chain.Chain())


def test_the_seam_states_the_root_s_own_resources_once_its_members_do() -> None:
    explorer, point = within_part()
    waiting = explorer.resources(point)
    # Not yet: why, its findings (the open choices it waits on).
    assert isinstance(waiting, tuple) and waiting
    assert {finding.code for finding in waiting} == {"decision-unassigned"}
    base = explorer.baseline(point)
    used = explorer.cost(base).used
    assert used is not None and explorer.resources(base) == Stated(used, {})


def test_max_throughput_keeps_the_least_budget_whose_point_fits() -> None:
    explorer, point = within_part()
    strategy = MaxThroughput({"dsp": 0.5})
    folded = strategy.explore(explorer, point)
    # Half the part's 10 DSPs: the 6- and 12-cycle points use 8, the 24-cycle one 4.
    assert folding(explorer, folded) == {
        "first.compute.packed.pe": 1,
        "first.compute.packed.simd": 2,
        "second.compute.packed.pe": 1,
        "second.compute.packed.simd": 2,
    }
    # The point is returned settled, as it was costed.
    assert explorer.choices(folded) == ()
    report = strategy.report()
    assert report["budget"] == {"dsp": 5} and report["kept_budget"] is not None
    assert report["bottleneck"] == {"members": ["x", "levels", "first", "second"], "cycles": 24}
    assert (report["lower_bound"], report["unstated"]) == (False, {})
    assert report["used"] == asdict(Resources(lut=331, ff=380, dsp=4))
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
    of that point settled, in fewer attempts than folding and settling each alone."""
    explorer, point = within_part()
    strategy = MaxThroughput({"dsp": 0.5})
    strategy.explore(explorer, point)
    searched = explorer.attempts
    alone = Seam(MEMBERS, platform=PART)
    for tried in strategy.tried:
        budget = explore_module._UNBOUNDED if tried.cycles is None else tried.cycles
        folder = TargetCycles(budget)
        folded = folder.explore(alone, point)
        assert explorer.chosen(explorer.settle(tried.folded, {})) == alone.chosen(folded)
        assert tried.relaxed_to == folder.relaxed_to
        assert tried.reached == alone.cost(folded).bottleneck
        assert tried.stated == alone.resources(folded)
    assert searched < alone.attempts


def test_max_throughput_names_the_resource_that_binds_among_those_budgeted() -> None:
    explorer, point = within_part()
    # The fastest point fits both budgets: one fold, LUT the nearer its budget.
    both = MaxThroughput({"lut": 0.5, "dsp": 1.0})
    both.explore(explorer, point)
    report = both.report()
    assert report["ratio"] == {"lut": 0.964, "dsp": 0.8} and report["binding"] == "lut"
    assert [tried.cycles for tried in both.tried] == [1]


def test_max_throughput_states_and_keeps_the_least_parallelism_where_nothing_fits() -> None:
    """Never a refusal and never a Python warning (RC5): a ``resource-budget``
    limitation, in its ``findings`` and its report, naming the binding resource."""
    explorer, point = within_part()
    strategy = MaxThroughput({"dsp": 0.1, "lut": 1.0})
    folded = strategy.explore(explorer, point)
    assert explorer.chosen(folded)["first.compute.packed.simd"] == 1
    (found,) = strategy.findings
    assert (found.kind, found.code, found.owner) == (
        FindingKind.LIMITATION,
        "resource-budget",
        "max_throughput",
    )
    assert "most of dsp: dsp 2 of 1" in found.message
    assert dict(found.details)["binding"] == "dsp"
    report = strategy.report()
    assert report["findings"] == [finding_record(found)]
    assert report["fits"] is False and report["binding"] == "dsp"
    assert [tried.cycles for tried in strategy.tried] == [1, None]


def test_max_throughput_states_where_a_budget_departs_from_its_assumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bisection assumes a larger budget folds to more cycles and fewer resources. A budget
    whose fold reaches more cycles than itself (an end's converter, SZ6) departs from
    that: here every budget from 13 to 23 folds as no budget does, to 48 cycles. The
    search is not corrected, and states each departure; but of every point it tried it
    keeps the one that fits with the fewest cycles: budget 27's, at 24 cycles, not the
    least budget that fits, 13, at 48."""

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
    assert [row["bottleneck"] for row in tried if row["cycles"] == 13] == [48]
    assert report["kept_budget"] == 27
    assert report["bottleneck"] == {"members": ["x", "levels", "first", "second"], "cycles": 24}
    assert report["used"] == asdict(Resources(lut=331, ff=380, dsp=4)) and report["fits"]
    assert not report["monotone"]
    departures = report["departures"]
    assert isinstance(departures, list)
    assert "budget 13 relaxed to 48, reaching 48" in departures
    assert "budget 27 reached 24, fewer than budget 17's 48" in departures


def test_max_throughput_keeps_the_tie_on_cycles_that_uses_the_least(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two fitting points at the same cycles: the one using less of the budget is kept,
    not the last the bisection reached. Here budget 24 also folds ``second`` to 6
    cycles: 24 cycles still, at 6 DSPs where budget 27's point uses 4."""

    class Heavier(TargetCycles):
        def _fold(self, seam: Seam, point: Any, name: str, budget: int) -> Any:
            return super()._fold(
                seam, point, name, 6 if (name, budget) == ("second", 24) else budget
            )

    monkeypatch.setattr(explore_module, "TargetCycles", Heavier)
    explorer, point = within_part()
    strategy = MaxThroughput({"dsp": 0.6})
    strategy.explore(explorer, point)
    rows = {row.cycles: (row.reached.cycles, row.used.dsp, row.fits) for row in strategy.tried}
    assert rows[27] == (24, 4, True) and rows[24] == (24, 6, True)
    # The bisection's last fitting point.
    assert [row.cycles for row in strategy.tried if row.fits][-1] == 24
    report = strategy.report()
    assert report["kept_budget"] == 27 and report["used"]["dsp"] == 4  # type: ignore[index]


def test_max_throughput_budgets_what_is_stated_where_a_member_states_no_resources() -> None:
    """A member that states no resources (here a compressor reducer, whose resources are
    not characterised) is allowed (RC5): the budget is checked against the members that
    state theirs, a lower bound, and it states so (``resources-unstated``), naming the
    member and why, its findings the causes."""
    explorer, point = within_part()
    pinned = explorer.attempt(point, {"first.compute.packed.reducer": "compressor"})
    assert isinstance(pinned, Accepted)
    strategy = MaxThroughput({"dsp": 0.5})
    folded = strategy.explore(explorer, pinned.point)
    (found,) = strategy.findings
    assert found.code == "resources-unstated" and found.kind is FindingKind.LIMITATION
    assert re.search("lower bound.*first .*compressor", found.message)
    assert found.causes == strategy.unstated()["first"]
    assert explorer.chosen(folded)["first.compute.packed.reducer"] == "compressor"
    report = strategy.report()
    assert report["lower_bound"] is True and report["fits"] is True
    unstated = report["unstated"]
    assert isinstance(unstated, dict) and list(unstated) == ["first"]
    assert "not characterised" in " ".join(record["message"] for record in unstated["first"])
    # What is stated, without first's: at 6 cycles, second's 4 DSPs of the budget's 5.
    stated = explorer.resources(folded)
    assert isinstance(stated, Stated) and stated.lower_bound
    assert stated.used == total(
        used for name, used in explorer.cost(folded).resources.items() if name != "first"
    )
    assert report["used"] == asdict(stated.used) and report["bottleneck"]["cycles"] == 6  # type: ignore[index]


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
