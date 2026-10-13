# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The DSE seam over a network: choices, changes, cost, the operations that keep a
network complete (baseline, reopen, settle), and the strategies that search it.

On the Chain (``kernels.chain``: MatMul ``first``, Thresholding ``activate`` at a
fixed PE, MatMul ``second``; three rows of four), nothing chosen, its platform at
5 ns. No ONNX: the seam reads a network and its members, each key a member and the key
in its space; owners are the KernelOps' (``tests/kernel_ops/test_explore.py``). A test
names a key ``member.key`` (``helpers.keyed``).
"""

from __future__ import annotations

import re
from dataclasses import asdict, replace
from typing import Any

import pytest

import finn.kernels.explore as explore_module
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
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
from finn.kernels.network import Fact, Interface, Network, dotted, wire
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.utilization import total
from finn.platform.capabilities import RESOURCE_NAMES, Resources
from kernels import chain
from kernels.helpers import FULL_DSP48E2, FULL_DSP58, Lanes, keyed


def held(explorer: Seam) -> Seam:
    """``explorer`` holding the Chain's fixed PE."""
    explorer.held.update(chain.HELD)
    return explorer


def seam() -> tuple[Seam, Network]:
    return held(Seam(FULL_DSP48E2)), chain.network()


def chosen(explorer: Seam, network: Network) -> dict[str, object]:
    """``network``'s stored choices, each by ``member.key``."""
    return {dotted(key): value for key, value in explorer.chosen(network).items()}


def folding(explorer: Seam, network: Network) -> dict[str, object]:
    """``network``'s stored folding: each MatMul's PE and SIMD (the thresholding's PE is
    the Chain's, held)."""
    return {
        k: v
        for k, v in chosen(explorer, network).items()
        if k.endswith((".pe", ".simd")) and not k.startswith("activate.")
    }


def offered(explorer: Seam, network: Network) -> dict[str, Any]:
    """``network``'s open choices, each by ``member.key``."""
    return {dotted(choice.key): choice for choice in explorer.choices(network)}


#: The Chain's foldings at a budget of 12 cycles a frame: PE x SIMD = 4, the earlier axis
#: least.
FOLDED_12 = {
    "first.compute.packed.pe": 1,
    "first.compute.packed.simd": 4,
    "second.compute.packed.pe": 1,
    "second.compute.packed.simd": 4,
}


def test_a_choice_names_its_space_class_and_whether_its_cases_are_ordered() -> None:
    explorer, network = seam()
    found = offered(explorer, explorer.baseline(network))
    first = offered(explorer, network)
    pe = first["first.compute.packed.pe"]
    assert pe.key == ("first", "compute.packed.pe")
    assert pe.space_type is PackedDotpKernel and pe.ordered and pe.cases == (1, 2, 4)
    assert first["activate.ultra_stages"].space_type is ThresholdingAxiKernel
    assert not first["first.compute.packed.reducer"].ordered
    # A channel's choices are offered once its ends are known (its kernels folded); a
    # list of values states no order; a forced Decision is never a choice.
    assert "levels.transport" not in first
    level = explorer.reopen(explorer.baseline(network), [("levels", "transport")])[0]
    assert not offered(explorer, level)["levels.transport"].ordered
    assert "first.compute" not in found and "w1.source" not in found


def said(why: Any) -> str:
    """Every finding of a refusal's ``why``, as text."""
    return "; ".join(str(finding) for found in why.values() for finding in found)


def test_a_change_is_refused_for_a_forced_decision_or_a_case_that_is_not_viable() -> None:
    """The seam's own refusals are findings too, each owned by the key it refuses."""
    explorer, network = seam()
    forced = explorer.set(network, keyed({"first.compute": "int8_dsp58"}))
    assert isinstance(forced, Refused)
    (found,) = forced.why[("first", "compute")]
    assert (found.kind, found.code, found.owner) == (
        FindingKind.REJECTION,
        "forced",
        "first.compute",
    )
    assert "forced to 'packed'" in found.message
    three = explorer.set(network, keyed({"first.compute.packed.pe": 3}))
    assert isinstance(three, Refused)
    (found,) = three.why[("first", "compute.packed.pe")]
    assert found.code == "case-not-viable" and "3 is not viable" in found.message
    nowhere = explorer.set(network, keyed({"nowhere.pe": 1}))
    assert isinstance(nowhere, Refused)
    assert [f.code for f in nowhere.why[("nowhere", "pe")]] == ["not-a-decision"]


def test_a_change_replaces_a_stored_value_but_never_a_held_one() -> None:
    explorer, network = seam()
    base = explorer.baseline(network)
    fifo = explorer.set(base, keyed({"y.transport": "fifo"}))
    assert isinstance(fifo, Accepted) and chosen(explorer, fifo.network)["y.transport"] == "fifo"
    # A key held for the run (a pin) takes no other value.
    explorer.held[("y", "transport")] = "direct"
    held = explorer.set(base, keyed({"y.transport": "fifo"}))
    assert isinstance(held, Refused)
    assert held.why == {
        ("y", "transport"): (
            Finding(
                FindingKind.REJECTION,
                "held",
                "y.transport",
                "held to 'direct' by a pin",
            ),
        )
    }
    assert isinstance(explorer.set(base, keyed({"y.transport": "direct"})), Accepted)


def test_a_change_replacing_what_forces_a_choice_may_set_it_in_the_same_batch() -> None:
    """At SIMD 1 pumping is forced off; a batch replacing SIMD by 2 may set it on, and a
    batch that leaves it forced may not."""
    explorer, network = seam()
    base = explorer.baseline(network)
    pumped = keyed({"first.compute.packed.simd": 2, "first.compute.packed.compute_pumping": True})
    outcome = explorer.set(base, pumped)
    assert isinstance(outcome, Accepted)
    assert chosen(explorer, outcome.network)["first.compute.packed.compute_pumping"] is True
    alone = explorer.set(base, keyed({"first.compute.packed.compute_pumping": True}))
    assert isinstance(alone, Refused) and "forced to False" in said(alone.why)


def test_a_change_that_leaves_a_decision_without_a_viable_case_is_refused() -> None:
    """Viable is not feasible: a batch can dead-end another Decision, and the seam says
    which and why."""
    explorer, network = seam()
    batch = keyed({"first.compute.packed.simd": 1, "first.compute.packed.compute_pumping": True})
    outcome = explorer.set(network, batch)
    assert isinstance(outcome, Refused)
    assert "dotp-pumping" in said(outcome.why)
    simd = keyed({"first.compute.packed.simd": 2})
    assert isinstance(explorer.set(network, {**batch, **simd}), Accepted)


def test_a_batch_commits_a_selector_and_a_choice_nested_under_it_together() -> None:
    explorer, network = seam()
    base = explorer.baseline(network)
    batch = keyed({"hidden.transport": "fifo", "hidden.transport.fifo.buffer.depth": 16})
    outcome = explorer.set(base, batch)
    assert isinstance(outcome, Accepted) and not outcome.dropped
    assert {
        k: v for k, v in chosen(explorer, outcome.network).items() if k.startswith("hidden.")
    } == {
        "hidden.transport": "fifo",
        "hidden.transport.fifo.buffer.depth": 16,
    }
    # Alone, the nested choice does not apply: the engine refuses it.
    alone = explorer.set(base, keyed({"hidden.transport.fifo.buffer.depth": 16}))
    assert isinstance(alone, Refused)
    assert list(alone.why) == [("hidden", "transport.fifo.buffer.depth")]


def test_a_change_drops_what_it_makes_stale_names_it_and_settles_it() -> None:
    """A FIFO on ``hidden`` with its depth, then ``hidden`` direct: the depth no longer
    applies, so the change drops it, named with why, and nothing is left open. A held
    choice is never dropped: the change is refused, naming it."""
    explorer, network = seam()
    base = explorer.baseline(network)
    fifo = explorer.set(
        base, keyed({"hidden.transport": "fifo", "hidden.transport.fifo.buffer.depth": 16})
    )
    assert isinstance(fifo, Accepted)
    direct = explorer.set(fifo.network, keyed({"hidden.transport": "direct"}))
    assert isinstance(direct, Accepted)
    # Its style was never stored (``set`` settles only what it drops): the depth alone.
    assert set(direct.dropped) == {("hidden", "transport.fifo.buffer.depth")}
    assert "hidden.transport.fifo.buffer.depth" not in chosen(explorer, direct.network)
    assert explorer.choices(direct.network) == ()
    explorer.held[("hidden", "transport.fifo.buffer.depth")] = 16
    refused = explorer.set(fifo.network, keyed({"hidden.transport": "direct"}))
    assert isinstance(refused, Refused)
    (found,) = refused.why[("hidden", "transport.fifo.buffer.depth")]
    assert found.code == "held" and "dropped by the change" in found.message


def test_a_fifo_depth_is_known_by_membership_and_proposed_through_a_change() -> None:
    explorer, network = seam()
    base = explorer.baseline(network)
    fifo = explorer.set(base, keyed({"levels.transport": "fifo"}))
    assert isinstance(fifo, Accepted)
    depth = offered(explorer, fifo.network)["levels.transport.fifo.buffer.depth"]
    assert depth.cases is None and depth.ordered and depth.required
    small = explorer.set(fifo.network, {depth.key: 1})
    assert isinstance(small, Refused)
    assert "domain-membership" in [finding.code for finding in small.why[depth.key]]
    sized = explorer.set(fifo.network, {depth.key: 16})
    assert isinstance(sized, Accepted)
    # The FIFO sits before the input adapter that makes second's markers, so the
    # channel carries it: folded and settled around it (no transport is a folding
    # strategy's), nothing refuses itself.
    explored = TargetCycles(12).explore(explorer, sized.network)
    assert explorer.refusals(explored) == {}
    assert explorer.chosen(explored)[depth.key] == 16
    levels = explored.member("levels")
    labels = [stage.label for stage in levels.stages]
    assert labels.index("transport.fifo.buffer") < labels.index("adapter.input_gen.input_gen")
    word = levels.transport.buffer.word_bits
    assert explorer.cost(explored).buffering["levels"] >= 16 * word


def test_cost_names_what_a_member_waits_on_then_its_cycles_and_the_bottleneck_ties() -> None:
    explorer, network = seam()
    cost = explorer.cost(network)
    assert cost.waiting["first"] == (("first", "compute.packed.pe"),)
    # A channel whose end waits on its kernel's folding waits on it too.
    assert cost.waiting["x"] == (("first", "compute.packed.pe"),)
    assert cost.cycles["activate"] == 6  # its PE is the Chain's
    assert cost.bottleneck is None
    explored = Ranked(Lanes(2)).explore(explorer, network)
    cost = explorer.cost(explored)
    first = explored.member("first")
    assert cost.cycles["first"] == first.compute.schedule.beat_count == 3 * 2 * 2
    # Every member at the most cycles is named, kernels first: the activation replays
    # feed the MatMuls at their pace.
    assert cost.bottleneck == Bottleneck(("first", "second", "x", "levels"), 12)


def test_target_cycles_folds_the_least_parallelism_that_meets_the_budget() -> None:
    explorer, network = seam()
    strategy = TargetCycles(12)
    folded = strategy.explore(explorer, network)
    # 3 rows x 4 x 4 at 12 cycles: PE x SIMD = 4, the earlier axis least.
    assert folding(explorer, folded) == FOLDED_12
    # The network is settled: every other choice at its baseline, none open.
    assert explorer.choices(folded) == ()
    # At SIMD 4, each MatMul's pumping is a choice: its baseline, off.
    assert chosen(explorer, folded) == {
        **chosen(explorer, explorer.baseline(network)),
        **FOLDED_12,
        "first.compute.packed.compute_pumping": False,
        "second.compute.packed.compute_pumping": False,
    }
    report = strategy.report()
    assert report["bottleneck"] == {
        "members": ["first", "second", "x", "hidden", "levels"],
        "cycles": 12,
    }
    assert (report["bottleneck_of"], report["met"], report["unfolded"]) == ("folded", True, {})
    # The Chain's cores are forced (a DSP48E2 platform): no case is tried.
    assert report["cases"] == {}
    assert strategy.relaxed_to is None


def test_a_budget_no_member_meets_is_relaxed_to_the_bottleneck_reached() -> None:
    explorer, network = seam()
    strategy = TargetCycles(1)
    folded = explore(explorer, network, [strategy])
    # The activation's fixed PE takes 6 cycles: the MatMuls fold to meet that, no faster.
    assert strategy.relaxed_to == 6
    values = chosen(explorer, folded)
    assert values["first.compute.packed.pe"] == 2 and values["first.compute.packed.simd"] == 4
    assert explorer.cost(folded).bottleneck == Bottleneck(
        ("first", "activate", "second", "x", "levels"), 6
    )
    unrelaxed = TargetCycles(1, relax=False)
    assert chosen(explorer, unrelaxed.explore(explorer, network))["first.compute.packed.pe"] == 4
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
    the report still states the bottleneck, of the network settled, and whether it meets
    the budget."""

    class Partial(TargetCycles):
        def _fold(self, seam: Seam, network: Any, name: str, budget: int) -> Any:
            if name == "second":
                return explore_module._Folded(None, network, "left by the test")
            return super()._fold(seam, network, name, budget)

    explorer, network = seam()
    strategy = Partial(12)
    folded = strategy.explore(explorer, network)
    # Settled: second at its baseline, the least parallelism.
    assert chosen(explorer, folded)["second.compute.packed.pe"] == 1
    assert chosen(explorer, folded)["second.compute.packed.simd"] == 1
    report = strategy.report()
    # second, and the channels whose cycles wait on its folding.
    unfolded = report["unfolded"]
    assert isinstance(unfolded, dict) and unfolded["second"] == "left by the test"
    assert all("second." in why for name, why in unfolded.items() if name != "second")
    # Settled at its baseline, second takes 48 cycles: the budget is missed, and said.
    assert report["bottleneck"]["cycles"] == 48  # type: ignore[index]
    assert "second" in report["bottleneck"]["members"]  # type: ignore[index]
    assert (report["bottleneck_of"], report["met"]) == ("settled", False)


def dsp58_matmul() -> tuple[Seam, Network]:
    """The Chain's first MatMul alone, on a DSP58 platform: its core an open choice."""

    x_tensor = Tensor((chain.ROWS, chain.INPUTS), ScalarEncoding(chain.A))

    def first() -> MatMulKernel:
        return MatMulKernel(
            weights_dtype=chain.W,
            platform=FULL_DSP58,
            x_tensor=x_tensor,
            w_contents=chain.W1,
        )

    y_tensor = design_space(first()).y_tensor
    network = Network(
        {
            "x": wire(
                platform=FULL_DSP58,
                tensor=x_tensor,
                port="s_axis_0",
                consumer=Interface("first", "x_end"),
            ),
            "w": wire(
                tensor=Fact("first", "w_carried"),
                contents=Fact("first", "w_stored"),
                consumer=Interface("first", "w_end"),
                platform=FULL_DSP58,
            ),
            "y": wire(
                tensor=y_tensor,
                port="m_axis_0",
                producer=Interface("first", "y_end"),
                platform=FULL_DSP58,
            ),
            "first": first(),
        }
    )
    return Seam(FULL_DSP58), network


def test_target_cycles_folds_each_case_of_an_unordered_choice_its_cycles_wait_on() -> None:
    """On a DSP58 platform a MatMul's core is an open, unordered choice (``packed`` or
    ``int8_dsp58``) its cycles wait on. Each case is folded, and the case kept meets the
    budget with the least parallelism, then the fewest resources; no order is imposed
    on the cases, and the report names the case taken and why."""
    explorer, network = dsp58_matmul()
    compute = offered(explorer, network)["first.compute"]
    assert not compute.ordered and set(compute.cases or ()) == {"packed", "int8_dsp58"}
    strategy = TargetCycles(6)
    folded = strategy.explore(explorer, network)
    report = strategy.report()
    assert (report["met"], report["unfolded"]) == (True, {})
    (row,) = report["cases"].values()  # type: ignore[attr-defined]
    assert list(report["cases"]) == ["first.compute"]  # type: ignore[call-overload]
    taken = row["taken"]
    assert chosen(explorer, folded)["first.compute"] == taken
    assert chosen(explorer, folded)[f"first.compute.{taken}.simd"]
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
    # What it compared is the member settled in each case; the network it returns is
    # settled.
    assert explorer.choices(folded) == ()


def test_target_cycles_keeps_the_case_with_the_least_parallelism_that_meets_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cycles come first: where one case meets the budget at more cycles than another, it
    is kept whatever it uses; where no case meets it, the fewest cycles."""
    explorer, network = dsp58_matmul()
    cost = Seam.cost

    def slower_int8(self: Seam, at: Any, members: Any = None) -> Any:
        found = cost(self, at, members)
        if "first" in found.cycles and self.chosen(at).get(("first", "compute")) == "int8_dsp58":
            found.cycles = {**found.cycles, "first": found.cycles["first"] + 1}
        return found

    monkeypatch.setattr(Seam, "cost", slower_int8)
    strategy = TargetCycles(12)
    strategy.explore(explorer, network)
    (row,) = strategy.report()["cases"].values()  # type: ignore[attr-defined]
    folds = {case: each["cycles"] for case, each in row["folded"].items()}
    assert row["taken"] == max(folds, key=folds.__getitem__) and "least parallelism" in row["why"]
    assert all("resources" not in each for each in row["folded"].values())
    missed = TargetCycles(1, relax=False)
    missed.explore(explorer, network)
    (row,) = missed.report()["cases"].values()  # type: ignore[attr-defined]
    assert row["taken"] == "packed" and row["why"].startswith("no case meets the budget of 1")
    assert missed.report()["met"] is False


def test_a_target_throughput_is_a_cycles_budget_at_the_platform_s_clock() -> None:
    explorer, network = seam()
    # 5 ns: 1e9 / (5 x 16,666,666) frames a second leaves 12 cycles a frame.
    strategy = TargetThroughput(16_666_666)
    folded = strategy.explore(explorer, network)
    assert strategy.cycles == 12
    assert explorer.chosen(folded) == explorer.chosen(TargetCycles(12).explore(explorer, network))
    assert strategy.report()["fps"] == 16_666_666
    with pytest.raises(ExploreError, match="needs the seam's platform"):
        TargetThroughput(1000).explore(Seam(), network)


# -- keeping a network complete -------------------------------------------------------------


def test_the_baseline_takes_each_open_choice_s_first_viable_case_and_leaves_required_ones() -> None:
    """``Seam.baseline`` stores each kernel's first case, every transport ``direct``; a
    required choice stays open."""
    explorer, network = seam()
    base = explorer.baseline(network)
    values = chosen(explorer, base)
    assert values["first.compute.packed.pe"] == 1
    assert values["first.compute.packed.reducer"] == "tree"
    assert {value for key, value in values.items() if key.endswith("transport")} == {"direct"}
    assert explorer.choices(base) == () and explorer.cost(base).bottleneck is not None
    fifo = explorer.set(base, keyed({"levels.transport": "fifo"}))
    assert isinstance(fifo, Accepted)
    assert [(c.key, c.required) for c in explorer.choices(explorer.baseline(fifo.network))] == [
        (("levels", "transport.fifo.buffer.depth"), True)
    ]


def test_reopen_clears_what_the_engine_names_and_a_channel_waits_on_its_kernel() -> None:
    """Clearing a MatMul's folding leaves its channels' ends open: they wait, their
    stored choices with them, and come back when the folding does. Within a member, what
    the engine names is cleared too. Nothing outside the reach is touched."""
    explorer, network = seam()
    base = explorer.baseline(network)
    opened, reach = explorer.reopen(base, [("first", "compute.packed.pe")])
    assert reach == (("first", "compute.packed.pe"),)
    assert set(explorer.chosen(base)) - set(explorer.chosen(opened)) == set(reach)
    held = opened.unbound("x")
    assert held is not None and held.stored == {
        key[1]: value for key, value in explorer.chosen(base).items() if key[0] == "x"
    }
    # The folding back, the channel is bound again, its choices replayed.
    again = explorer.set(opened, keyed({"first.compute.packed.pe": 1}))
    assert isinstance(again, Accepted) and not again.dropped
    assert explorer.chosen(again.network) == explorer.chosen(base)
    # A key not stored is nothing to reopen.
    assert explorer.reopen(network, [("first", "compute.packed.pe")]) == (network, ())
    # A selector's nested choices are the engine's to name too.
    sized = explorer.set(
        base, keyed({"hidden.transport": "fifo", "hidden.transport.fifo.buffer.depth": 16})
    )
    assert isinstance(sized, Accepted)
    _, reach = explorer.reopen(sized.network, [("hidden", "transport")])
    assert ("hidden", "transport.fifo.buffer.depth") in reach


def test_settle_keeps_each_prior_value_still_accepted_and_rebaselines_the_rest() -> None:
    explorer, network = seam()
    prior = {
        **explorer.chosen(explorer.baseline(network)),
        **keyed(
            {
                "first.compute.packed.reducer": "compressor",
                "second.compute.packed.pe": 3,  # no divisor of four: not viable
            }
        ),
    }
    settled = explorer.settle(network, prior)
    values = chosen(explorer, settled)
    assert values["first.compute.packed.reducer"] == "compressor"
    assert values["second.compute.packed.pe"] == 1  # its baseline
    assert explorer.choices(settled) == ()
    # Only the members named, where named (and the Chain's fixed PE, committed).
    first = explorer.settle(network, prior, ("first",))
    assert {key[0] for key in explorer.chosen(first)} == {"first", "activate"}


def test_set_drops_a_stored_choice_the_change_makes_invalid_then_settles() -> None:
    """Pumping on at SIMD 2, then SIMD replaced by 1: the engine refuses the change
    beside the pumping choice; ``set`` drops it, named, and settling finds it forced off,
    nothing to store."""
    explorer, network = seam()
    base = explorer.baseline(network)
    pumped = explorer.set(
        base, keyed({"first.compute.packed.simd": 2, "first.compute.packed.compute_pumping": True})
    )
    assert isinstance(pumped, Accepted)
    replaced = explorer.set(pumped.network, keyed({"first.compute.packed.simd": 1}))
    assert isinstance(replaced, Accepted)
    (pumping,) = [key for key in replaced.dropped if key[1].endswith("compute_pumping")]
    assert "pumping requires SIMD >= 2" in said({pumping: replaced.dropped[pumping]})
    values = chosen(explorer, replaced.network)
    assert values["first.compute.packed.simd"] == 1
    assert "first.compute.packed.compute_pumping" not in values
    assert explorer.choices(replaced.network) == ()
    # A change of the change's own key that is not taken is refused.
    assert isinstance(explorer.set(base, keyed({"first.compute.packed.pe": 3})), Refused)


def test_a_held_choice_a_change_would_drop_refuses_the_change() -> None:
    explorer, network = seam()
    pumped = explorer.set(
        explorer.baseline(network),
        keyed({"first.compute.packed.simd": 2, "first.compute.packed.compute_pumping": True}),
    )
    assert isinstance(pumped, Accepted)
    pumping = ("first", "compute.packed.compute_pumping")
    explorer.held[pumping] = True
    refused = explorer.set(pumped.network, keyed({"first.compute.packed.simd": 1}))
    assert isinstance(refused, Refused)
    ((key, (found,)),) = refused.why.items()
    assert (key, found.code, found.owner) == (
        pumping,
        "held",
        "first.compute.packed.compute_pumping",
    )
    assert found.message.startswith("held to True by a pin, dropped by the change: ")
    assert "pumping requires SIMD >= 2" in found.message
    # Not reached, it stays: SIMD 4 keeps pumping viable.
    kept = explorer.set(pumped.network, keyed({"first.compute.packed.simd": 4}))
    assert isinstance(kept, Accepted)
    assert explorer.chosen(kept.network)[pumping] is True
    opened = explorer.reopen(pumped.network, [("first", "compute.packed.simd")])[0]
    with pytest.raises(
        Unsettled, match="compute_pumping: pin-unsettled: held to True"
    ) as unsettled:
        explorer.settle(opened, {})
    (finding,) = unsettled.value.findings
    assert finding.owner == "first.compute.packed.compute_pumping"


def test_a_strategy_reopens_its_scope_and_keeps_what_it_does_not_decide() -> None:
    """A strategy changes only what it decides: a reducer an earlier strategy set stays,
    though the folding strategy reopens every choice of a member; the run records what
    it reopened and what settling kept and re-baselined."""
    explorer, network = seam()
    base = explorer.baseline(network)
    compressor = explorer.set(base, keyed({"second.compute.packed.reducer": "compressor"}))
    assert isinstance(compressor, Accepted)
    folded = TargetCycles(12).explore(explorer, compressor.network)
    values = chosen(explorer, folded)
    assert values["second.compute.packed.reducer"] == "compressor"
    assert folding(explorer, folded) == FOLDED_12
    (run,) = explorer.runs
    assert run.strategy == "target_cycles"
    assert ("second", "compute.packed.reducer") in run.reopened
    assert ("levels", "adapter.input_gen.input_gen.ram_style") in run.reopened
    assert not any(key[1].endswith("transport") for key in run.reopened)
    assert run.kept[("second", "compute.packed.reducer")] == "compressor"
    assert set(run.kept) | set(run.baselined) <= set(run.reopened) | set(explorer.chosen(folded))
    # Again on the folded network: the same folding, nothing new.
    again = TargetCycles(12).explore(explorer, folded)
    assert explorer.chosen(again) == explorer.chosen(folded)


def test_a_pin_holds_its_keys_and_a_later_strategy_folds_around_it() -> None:
    """Pinned first and held (SR6): the target throughput folds the other MatMul and
    keeps the pinned one; its relax step reaches the pin's bottleneck."""
    explorer, network = seam()
    base = explorer.baseline(network)
    explorer.held.update(keyed({"first.compute.packed.pe": 1, "first.compute.packed.simd": 1}))
    strategy = TargetCycles(1)
    folded = strategy.explore(explorer, base)
    values = chosen(explorer, folded)
    assert (values["first.compute.packed.pe"], values["first.compute.packed.simd"]) == (1, 1)
    # first at 48 cycles binds: second folds to meet it, no faster.
    assert strategy.relaxed_to == 48
    assert (values["second.compute.packed.pe"], values["second.compute.packed.simd"]) == (1, 1)
    assert ("first", "compute.packed.pe") not in explorer.runs[-1].reopened


def test_the_baseline_strategy_resets_every_choice_not_held() -> None:
    explorer, network = seam()
    folded = TargetCycles(12).explore(explorer, explorer.baseline(network))
    explorer.held[("first", "compute.packed.simd")] = 4
    reset = Rebaseline().explore(explorer, folded)
    assert folding(explorer, reset) == {
        **{key: 1 for key in FOLDED_12},
        "first.compute.packed.simd": 4,
    }
    assert chosen(explorer, reset) == {
        **chosen(explorer, explorer.baseline(network)),
        "first.compute.packed.simd": 4,
    }
    assert Rebaseline().report() == {"strategy": "baseline"}


def test_the_debug_placeholder_takes_a_required_choice_s_first_case_or_refuses_it() -> None:
    """With nothing required open it sets nothing; a FIFO with no depth is a required
    choice known by membership only, which it refuses, named."""
    explorer, network = seam()
    base = explorer.baseline(network)
    placeholder = DebugPlaceholder()
    assert explorer.chosen(placeholder.explore(explorer, base)) == explorer.chosen(base)
    assert placeholder.report() == {"strategy": "placeholder", "set": {}}
    assert placeholder.label == "DEBUG: set by placeholder"
    fifo = explorer.set(base, keyed({"levels.transport": "fifo"}))
    assert isinstance(fifo, Accepted)
    with pytest.raises(ExploreError, match="known by membership only: .*levels.transport.fifo"):
        DebugPlaceholder().explore(explorer, fifo.network)


# -- the most throughput within resources -----------------------------------------------------

#: The Chain's part: its totals, of which a budget is a fraction. At each budget of cycles
#: the Chain folds to (``TargetCycles``), settled: 6 cycles, 482 LUT and 8 DSP; 12,
#: 379 and 8; 24, 331 and 4; 48 (the least parallelism), 328 and 2.
PART = replace(FULL_DSP48E2, resources=Resources(lut=1000, ff=2000, bram18=10, uram=0, dsp=10))


def within_part() -> tuple[Seam, Network]:
    return held(Seam(PART)), chain.network()


def test_the_seam_states_the_network_s_resources_once_its_members_do() -> None:
    explorer, network = within_part()
    waiting = explorer.resources(network)
    # Not yet: why, its findings (the open choices it waits on).
    assert isinstance(waiting, tuple) and waiting
    assert {finding.code for finding in waiting} == {"decision-unassigned"}
    base = explorer.baseline(network)
    used = explorer.cost(base).used
    assert used is not None and explorer.resources(base) == Stated(used, {})


def test_max_throughput_keeps_the_least_budget_whose_point_fits() -> None:
    explorer, network = within_part()
    strategy = MaxThroughput({"dsp": 0.5})
    folded = strategy.explore(explorer, network)
    # Half the part's 10 DSPs: the 6- and 12-cycle points use 8, the 24-cycle one 4.
    assert folding(explorer, folded) == {
        "first.compute.packed.pe": 1,
        "first.compute.packed.simd": 2,
        "second.compute.packed.pe": 1,
        "second.compute.packed.simd": 2,
    }
    # The network is returned settled, as it was costed.
    assert explorer.choices(folded) == ()
    report = strategy.report()
    assert report["budget"] == {"dsp": 5} and report["kept_budget"] is not None
    assert report["bottleneck"] == {"members": ["first", "second", "x", "levels"], "cycles": 24}
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
    network ``TargetCycles`` alone folds it to, and its bottleneck and resources are those
    of that network settled, in fewer changes than folding and settling each alone."""
    explorer, network = within_part()
    strategy = MaxThroughput({"dsp": 0.5})
    strategy.explore(explorer, network)
    searched = explorer.attempts
    alone = held(Seam(PART))
    for tried in strategy.tried:
        budget = explore_module._UNBOUNDED if tried.cycles is None else tried.cycles
        folder = TargetCycles(budget)
        folded = folder.explore(alone, network)
        assert explorer.chosen(explorer.settle(tried.folded, {})) == alone.chosen(folded)
        assert tried.relaxed_to == folder.relaxed_to
        assert tried.reached == alone.cost(folded).bottleneck
        assert tried.stated == alone.resources(folded)
    assert searched < alone.attempts


def test_max_throughput_names_the_resource_that_binds_among_those_budgeted() -> None:
    explorer, network = within_part()
    # The fastest point fits both budgets: one fold, LUT the nearer its budget.
    both = MaxThroughput({"lut": 0.5, "dsp": 1.0})
    both.explore(explorer, network)
    report = both.report()
    assert report["ratio"] == {"lut": 0.964, "dsp": 0.8} and report["binding"] == "lut"
    assert [tried.cycles for tried in both.tried] == [1]


def test_max_throughput_states_and_keeps_the_least_parallelism_where_nothing_fits() -> None:
    """Never a refusal and never a Python warning (RC5): a ``resource-budget``
    limitation, in its ``findings`` and its report, naming the binding resource."""
    explorer, network = within_part()
    strategy = MaxThroughput({"dsp": 0.1, "lut": 1.0})
    folded = strategy.explore(explorer, network)
    assert chosen(explorer, folded)["first.compute.packed.simd"] == 1
    (found,) = strategy.findings
    assert (found.kind, found.code, found.owner) == (
        FindingKind.LIMITATION,
        "resource-budget",
        "max_throughput",
    )
    assert "most of dsp: dsp 2 of 1" in found.message
    assert dict(found.details)["binding"] == "dsp"
    report = strategy.report()
    assert report["findings"] == [found.record()]
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
        def _fold_all(self, seam: Seam, network: Any, budget: int) -> Any:
            return super()._fold_all(seam, network, 48 if 13 <= budget < 24 else budget)

    monkeypatch.setattr(explore_module, "TargetCycles", Departing)
    explorer, network = within_part()
    strategy = MaxThroughput({"dsp": 0.5})
    strategy.explore(explorer, network)
    report = strategy.report()
    tried = report["tried"]
    assert isinstance(tried, list)
    assert [row["cycles"] for row in tried] == [1, None, 27, 17, 12, 15, 14, 13]
    assert [row["bottleneck"] for row in tried if row["cycles"] == 13] == [48]
    assert report["kept_budget"] == 27
    assert report["bottleneck"] == {"members": ["first", "second", "x", "levels"], "cycles": 24}
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
        def _fold(self, seam: Seam, network: Any, name: str, budget: int) -> Any:
            return super()._fold(
                seam, network, name, 6 if (name, budget) == ("second", 24) else budget
            )

    monkeypatch.setattr(explore_module, "TargetCycles", Heavier)
    explorer, network = within_part()
    strategy = MaxThroughput({"dsp": 0.6})
    strategy.explore(explorer, network)
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
    explorer, network = within_part()
    pinned = explorer.set(network, keyed({"first.compute.packed.reducer": "compressor"}))
    assert isinstance(pinned, Accepted)
    strategy = MaxThroughput({"dsp": 0.5})
    folded = strategy.explore(explorer, pinned.network)
    (found,) = strategy.findings
    assert found.code == "resources-unstated" and found.kind is FindingKind.LIMITATION
    assert re.search("lower bound.*first .*compressor", found.message)
    assert found.causes == strategy.unstated()["first"]
    assert chosen(explorer, folded)["first.compute.packed.reducer"] == "compressor"
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
    explorer, network = seam()
    with pytest.raises(ExploreError, match="platform's resources"):
        MaxThroughput({"lut": 0.5}).explore(explorer, network)
