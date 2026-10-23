# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The DSE seam over a network: choices, changes, cost, a strategy's run (reopen, search,
restore), and the strategies that search it.

On the Chain (``support.chain``: MatMul ``first``, Thresholding ``activate`` at a
fixed PE, MatMul ``second``; three rows of four), its platform at 5 ns: as conversion and
the cut leave it (``support.helpers.with_defaults``: each kernel's choices by its
policies, each channel's by the cut's), or with nothing chosen
(``chain.network()``) where a test reads what is open. No ONNX: the seam reads a network
and its members, each key a member and the key in its space; owners are the KernelOps'
(``tests/kernel_ops/test_explore.py``). A test names a key ``member.key``
(``helpers.keyed``).
"""

from __future__ import annotations

import re
from dataclasses import asdict, replace
from typing import Any

import pytest
from qonnx.core.datatype import DataType
from support import chain
from support.helpers import (
    FULL_DSP48E2,
    FULL_DSP58,
    Lanes,
    choose,
    keyed,
    matmul_point,
    with_defaults,
)
from support.refusals import refused

import finn.kernels.explore as explore_module
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import design_space, inspection
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.explore import (
    Accepted,
    Bottleneck,
    DebugPlaceholder,
    Defaults,
    ExploreError,
    Hold,
    MaxThroughput,
    Ranked,
    Refused,
    Seam,
    SizeFifos,
    Stated,
    TargetCycles,
    TargetThroughput,
    dotted,
    explore,
)
from finn.kernels.matmul import MatMulKernel
from finn.kernels.network import Fact, Interface, Network, NetworkError, wire
from finn.kernels.policies import Required
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.utilization import total
from finn.platform.capabilities import RESOURCE_NAMES, Resources


def held(explorer: Seam) -> Seam:
    """``explorer`` holding the Chain's fixed PE."""
    explorer.held.update(chain.HELD)
    return explorer


def seam() -> tuple[Seam, Network]:
    """A seam holding the Chain's fixed PE, and the Chain as conversion and the cut leave
    it."""
    return held(Seam(FULL_DSP48E2)), with_defaults(chain.network())


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


def state(network: Network, name: str) -> inspection.ChoiceState:
    """The state of choice ``name`` (``member.key``) in ``network``."""
    ((owner, key),) = keyed({name: None})
    return dict(network.states((owner,)))[(owner, key)]


#: The Chain's foldings at a budget of 12 cycles a frame: PE x SIMD = 4, the earlier axis
#: least.
FOLDED_12 = {
    "first.pe": 1,
    "first.simd": 4,
    "second.pe": 1,
    "second.simd": 4,
}


def test_a_choice_names_its_space_class_and_whether_its_cases_are_ordered() -> None:
    """Every open choice in play is offered, one possible value or several, each refused
    value with the rule it would break; one whose presence waits on another open choice
    is not in play yet, and a channel's choices are offered once its contracts are known."""
    explorer, base = seam()
    fresh = offered(explorer, chain.network())
    # Each core is open: on a DSP48E2 platform one value is possible, and offered.
    compute = fresh["first.compute"]
    assert compute.key == ("first", "compute") and not compute.ordered
    assert compute.cases == ("packed",)
    ((rule, (finding,)),) = compute.refused["int8_dsp58"].why.items()
    assert (rule, finding.code) == (("first", "compute.int8_dsp58.core_supported"), "dotp-target")
    assert fresh["activate.ultra_stages"].space_type is ThresholdingAxiKernel
    # The folding is MatMul's, offered while its core is open; the channels wait on their
    # kernels' contracts, and a core's own choices on its presence.
    pe = fresh["first.pe"]
    assert pe.key == ("first", "pe")
    assert pe.space_type is MatMulKernel and pe.ordered and pe.cases == (1, 2, 4)
    assert "levels.transport" not in fresh and "first.compute.packed.reducer" not in fresh
    cored = choose(chain.network(), {"first.compute": "packed"})
    first = offered(explorer, cored)
    assert first["first.pe"].cases == (1, 2, 4)
    reducer = first["first.compute.packed.reducer"]
    assert reducer.space_type is PackedDotpKernel and not reducer.ordered
    # As conversion and the cut leave it, nothing is open; a transport reopened is
    # offered, a list of values stating no order.
    assert offered(explorer, base) == {}
    level = explorer.reopen(base, [("levels", "transport")])[0]
    assert not offered(explorer, level)["levels.transport"].ordered


def said(why: Any) -> str:
    """Every finding of a refusal's ``why``, as text."""
    return "; ".join(str(finding) for found in why.values() for finding in found)


def test_a_change_is_refused_by_the_rule_it_breaks_or_a_value_outside_its_domain() -> None:
    """The engine's refusals: a rule by its key, a value by the choice's; the seam's own
    are findings too, each owned by the key it refuses."""
    explorer, network = seam()
    core = explorer.set(network, keyed({"first.compute": "int8_dsp58"}))
    assert isinstance(core, Refused)
    rule = ("first", "compute.int8_dsp58.core_supported")
    (found,) = core.why[rule]
    assert (found.kind, found.code) == (FindingKind.REJECTION, "dotp-target")
    assert not core.standing.get(rule)
    three = explorer.set(network, keyed({"first.pe": 3}))
    assert isinstance(three, Refused)
    (found,) = three.why[("first", "pe")]
    assert found.code == "domain-membership"
    nowhere = explorer.set(network, keyed({"nowhere.pe": 1}))
    assert isinstance(nowhere, Refused)
    assert [f.code for f in nowhere.why[("nowhere", "pe")]] == ["not-a-decision"]


def test_a_change_replaces_a_stored_value_but_never_a_held_one() -> None:
    explorer, base = seam()
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
                "held to 'direct'",
            ),
        )
    }
    assert isinstance(explorer.set(base, keyed({"y.transport": "direct"})), Accepted)


def test_a_rule_a_value_breaks_names_the_stored_choices_standing_in_its_way() -> None:
    """At SIMD 1 pumping on breaks the rule that pumping needs lanes: SIMD, stored,
    stands in its way, and a batch replacing it may set pumping on. A batch that names
    both and breaks the rule is refused by it, nothing standing."""
    explorer, base = seam()
    alone = explorer.set(base, keyed({"first.compute.packed.compute_pumping": True}))
    assert isinstance(alone, Refused)
    rule = ("first", "compute.packed.pumping_has_lanes")
    assert [f.code for f in alone.why[rule]] == ["dotp-pumping"]
    assert alone.standing == {rule: (("first", "simd"),)}
    pumped = keyed({"first.simd": 2, "first.compute.packed.compute_pumping": True})
    outcome = explorer.set(base, pumped)
    assert isinstance(outcome, Accepted)
    assert chosen(explorer, outcome.network)["first.compute.packed.compute_pumping"] is True
    batch = keyed({"first.simd": 1, "first.compute.packed.compute_pumping": True})
    both = explorer.set(base, batch)
    assert isinstance(both, Refused)
    assert "dotp-pumping" in said(both.why) and not both.standing.get(rule)


def test_a_batch_commits_a_selector_and_a_choice_nested_under_it_together() -> None:
    explorer, base = seam()
    batch = keyed({"hidden.transport": "fifo", "hidden.transport.fifo.depth": 16})
    outcome = explorer.set(base, batch)
    # The FIFO's style, brought into play by the change, is set by the cut's policy (the
    # fold rule: first valid, ``auto``), the strategy's, named.
    assert isinstance(outcome, Accepted) and not outcome.cleared
    assert outcome.settled == keyed({"hidden.transport.fifo.buffer.ram_style": "auto"})
    assert {
        k: v for k, v in chosen(explorer, outcome.network).items() if k.startswith("hidden.")
    } == {
        "hidden.transport": "fifo",
        "hidden.transport.fifo.depth": 16,
        "hidden.transport.fifo.buffer.ram_style": "auto",
    }
    # Alone, the nested choice is stored out of play: its case is not selected.
    alone = explorer.set(base, keyed({"hidden.transport.fifo.depth": 16}))
    assert isinstance(alone, Accepted)
    assert state(alone.network, "hidden.transport.fifo.depth") == inspection.Dormant(16)


def test_a_change_that_puts_a_choice_out_of_play_keeps_it_dormant_and_a_held_one_refuses() -> None:
    """A FIFO on ``hidden`` with its depth, then ``hidden`` direct: the depth is out of
    play, kept dormant, and nothing is open. A held choice the change would put out of
    play refuses the change, naming it: a hold keeps its case in play."""
    explorer, base = seam()
    fifo = explorer.set(
        base, keyed({"hidden.transport": "fifo", "hidden.transport.fifo.depth": 16})
    )
    assert isinstance(fifo, Accepted)
    direct = explorer.set(fifo.network, keyed({"hidden.transport": "direct"}))
    assert isinstance(direct, Accepted) and not direct.cleared and not direct.settled
    depth = "hidden.transport.fifo.depth"
    assert chosen(explorer, direct.network)[depth] == 16
    assert state(direct.network, depth) == inspection.Dormant(16)
    assert explorer.choices(direct.network) == ()
    # The FIFO again: the depth back in play, as it was.
    again = explorer.set(direct.network, keyed({"hidden.transport": "fifo"}))
    assert isinstance(again, Accepted) and state(again.network, depth) == inspection.Set(16)
    explorer.held[("hidden", "transport.fifo.depth")] = 16
    refused_ = explorer.set(fifo.network, keyed({"hidden.transport": "direct"}))
    assert isinstance(refused_, Refused)
    (found,) = refused_.why[("hidden", "transport.fifo.depth")]
    assert found.code == "held" and "put out of play (dormant)" in found.message


def test_a_fifo_depth_is_known_by_membership_and_proposed_through_a_change() -> None:
    explorer, base = seam()
    fifo = explorer.set(base, keyed({"levels.transport": "fifo"}))
    assert isinstance(fifo, Accepted)
    depth = offered(explorer, fifo.network)["levels.transport.fifo.depth"]
    assert depth.cases is None and depth.ordered and depth.policy == Required()
    small = explorer.set(fifo.network, {depth.key: 1})
    assert isinstance(small, Refused)
    assert "domain-membership" in [finding.code for finding in small.why[depth.key]]
    # Its style is a choice of its own, brought into play with it: the fold rule sets it by
    # the cut's policy (first valid, ``auto``) and leaves the depth, required, open.
    style = ("levels", "transport.fifo.buffer.ram_style")
    assert fifo.settled == {style: "auto"} and list(fifo.unsettled) == [depth.key]
    assert dotted(style) not in offered(explorer, fifo.network)
    sized = explorer.set(fifo.network, {depth.key: 16})
    assert isinstance(sized, Accepted)
    # The FIFO sits before the input adapter that makes second's markers, so the
    # channel carries it: folded around it (no transport is a folding strategy's),
    # nothing refuses itself.
    explored = TargetCycles(12).explore(explorer, sized.network)
    assert explorer.refusals(explored) == {}
    assert explorer.chosen(explored)[depth.key] == 16
    levels = explored.member("levels")
    labels = [stage.label for stage in levels.stages]
    assert labels.index("transport.fifo.buffer") < labels.index("adapter.input_gen.input_gen")
    word = levels.transport.buffer.input_data_width
    assert explorer.cost(explored).buffering["levels"] >= 16 * word


def test_cost_names_what_a_member_waits_on_then_its_cycles_and_the_bottleneck_ties() -> None:
    explorer = held(Seam(FULL_DSP48E2))
    network = chain.network()
    cost = explorer.cost(network)
    # Nothing chosen: a MatMul's cycles wait on its core, open.
    assert cost.waiting["first"] == (("first", "compute"),)
    # A channel whose contract waits on its kernel's core waits on it too.
    assert cost.waiting["x"] == (("first", "compute"),)
    assert cost.cycles["activate"] == 6  # its PE is the Chain's
    assert cost.bottleneck is None
    cored = choose(network, {"first.compute": "packed", "second.compute": "packed"})
    assert explorer.cost(cored).waiting["first"] == (("first", "pe"),)
    explored = Ranked(Lanes(2)).explore(explorer, network)
    cost = explorer.cost(explored)
    first = explored.member("first")
    assert cost.cycles["first"] == first.compute.schedule.beat_count == 3 * 2 * 2
    # Every member at the most cycles is named, kernels first: the activation replays
    # feed the MatMuls at their pace.
    assert cost.bottleneck == Bottleneck(("first", "second", "x", "levels"), 12)


#: What the fold tries of each unordered choice a member's cycles wait on, on the Chain
#: at a DSP48E2 platform: each MatMul's core, reopened, with the one value possible,
#: which it keeps. A channel's choices are not a fold's.
CASES_12 = {"first.compute": "packed", "second.compute": "packed"}


def test_target_cycles_folds_the_least_parallelism_that_meets_the_budget() -> None:
    explorer, network = seam()
    strategy = TargetCycles(12)
    folded = strategy.explore(explorer, network)
    # 3 rows x 4 x 4 at 12 cycles: PE x SIMD = 4, the earlier axis least.
    assert folding(explorer, folded) == FOLDED_12
    # Every other reopened choice back at its prior value, none open; each channel,
    # rebuilt at its new contracts, keeps what they accept.
    assert explorer.choices(folded) == ()
    assert chosen(explorer, folded) == {**chosen(explorer, network), **FOLDED_12}
    (run,) = explorer.runs
    assert (run.by_policy, run.unset, run.cleared, run.settled) == ({}, {}, {}, {})
    report = strategy.report()
    assert report["bottleneck"] == {
        "members": ["first", "second", "x", "hidden", "levels"],
        "cycles": 12,
    }
    assert (report["bottleneck_of"], report["met"], report["unfolded"]) == ("folded", True, {})
    # Each unordered choice the cycles wait on is folded in its one possible value.
    cases = report["cases"]
    assert isinstance(cases, dict)
    assert {key: row["taken"] for key, row in cases.items()} == CASES_12
    assert strategy.relaxed_to is None


def test_a_budget_no_member_meets_is_relaxed_to_the_bottleneck_reached() -> None:
    explorer, network = seam()
    strategy = TargetCycles(1)
    folded = explore(explorer, network, [strategy])
    # The activation's fixed PE takes 6 cycles: the MatMuls fold to meet that, no faster.
    assert strategy.relaxed_to == 6
    values = chosen(explorer, folded)
    assert values["first.pe"] == 2 and values["first.simd"] == 4
    assert explorer.cost(folded).bottleneck == Bottleneck(
        ("first", "activate", "second", "x", "levels"), 6
    )
    unrelaxed = TargetCycles(1, relax=False)
    assert chosen(explorer, unrelaxed.explore(explorer, network))["first.pe"] == 4
    # A miss is stated: the bottleneck reached and that it does not meet the budget.
    report = unrelaxed.report()
    assert report["bottleneck"]["cycles"] == 6  # type: ignore[index]
    assert "activate" in report["bottleneck"]["members"]  # type: ignore[index]
    assert (report["relaxed_to"], report["met"]) == (None, False)
    assert strategy.report()["met"] is False


def test_target_cycles_states_the_bottleneck_of_the_restored_point_where_it_leaves_one_open() -> (
    None
):
    """A member whose cycles wait on a choice it does not fold is left open by the fold,
    then given back its prior values; the report still states the bottleneck, of the
    network restored, and whether it meets the budget."""

    class Partial(TargetCycles):
        def _fold(self, seam: Seam, network: Any, name: str, budget: int) -> Any:
            if name == "second":
                return explore_module._Folded(None, network, "left by the test")
            return super()._fold(seam, network, name, budget)

    explorer, network = seam()
    strategy = Partial(12)
    folded = strategy.explore(explorer, network)
    # Restored: second at its prior folding, the least parallelism.
    assert chosen(explorer, folded)["second.pe"] == 1
    assert chosen(explorer, folded)["second.simd"] == 1
    report = strategy.report()
    # second, and the channels whose cycles wait on its core.
    unfolded = report["unfolded"]
    assert isinstance(unfolded, dict) and unfolded["second"] == "left by the test"
    assert all("second." in why for name, why in unfolded.items() if name != "second")
    # Restored at its prior folding, second takes 48 cycles: the budget is missed, and said.
    assert report["bottleneck"]["cycles"] == 48  # type: ignore[index]
    assert "second" in report["bottleneck"]["members"]  # type: ignore[index]
    assert (report["bottleneck_of"], report["met"]) == ("restored", False)


def dsp58_matmul() -> tuple[Seam, Network]:
    """The Chain's first MatMul alone, on a DSP58 platform, as conversion and the cut leave
    it: its core ``packed``, the first valid, and ``int8_dsp58`` possible too."""

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
                consumer=Interface("first", "x_contract"),
            ),
            "w": wire(
                tensor=Fact("first", "carried_weights"),
                contents=Fact("first", "stored_weights"),
                consumer=Interface("first", "w_contract"),
                platform=FULL_DSP58,
            ),
            "y": wire(
                tensor=y_tensor,
                port="m_axis_0",
                producer=Interface("first", "y_contract"),
                platform=FULL_DSP58,
            ),
            "first": first(),
        }
    )
    return Seam(FULL_DSP58), with_defaults(network)


def test_target_cycles_folds_each_case_of_an_unordered_choice_its_cycles_wait_on() -> None:
    """On a DSP58 platform a MatMul's core is an unordered choice (``packed`` or
    ``int8_dsp58``) its cycles wait on, once reopened. Each possible value is folded, and
    the one kept meets the budget with the least parallelism, then the fewest resources;
    no order is imposed on the values, and the report names the value taken and why."""
    explorer, network = dsp58_matmul()
    assert chosen(explorer, network)["first.compute"] == "packed"
    compute = offered(explorer, explorer.reopen(network, [("first", "compute")])[0])
    assert not compute["first.compute"].ordered
    assert set(compute["first.compute"].cases or ()) == {"packed", "int8_dsp58"}
    strategy = TargetCycles(6)
    folded = strategy.explore(explorer, network)
    report = strategy.report()
    assert (report["met"], report["unfolded"]) == (True, {})
    (row,) = report["cases"].values()  # type: ignore[attr-defined]
    assert list(report["cases"]) == ["first.compute"]  # type: ignore[call-overload]
    taken = row["taken"]
    assert chosen(explorer, folded)["first.compute"] == taken
    assert chosen(explorer, folded)["first.simd"]
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
    # Each case's resources are compared as it is kept: the member's reopened choices
    # given back, and the case's own others, never stored, set by their policy. The INT8
    # core uses no LUTs: it is kept, and its pumping, never stored, is set by the fold's
    # policy, reported; nothing is left open.
    assert taken == "int8_dsp58"
    pumping = ("first", "compute.int8_dsp58.compute_pumping")
    (run,) = explorer.runs
    assert run.by_policy == {pumping: False} and explorer.chosen(folded)[pumping] is False
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
    with refused("platform-unstated", error=ExploreError, message="needs the seam's platform"):
        TargetThroughput(1000).explore(Seam(), network)


# -- a strategy's run: reopen, search, restore --------------------------------------------------


def test_the_defaults_set_each_open_choice_by_its_policy_and_leave_required_ones() -> None:
    """``Defaults`` on a network with nothing chosen sets each kernel's choices by its
    policies (each first valid here) and each channel's by the cut's, every transport
    ``direct``; a required choice stays open."""
    explorer = held(Seam(FULL_DSP48E2))
    base = Defaults().explore(explorer, chain.network())
    values = chosen(explorer, base)
    assert values["first.compute"] == "packed"
    assert values["first.pe"] == 1
    assert values["first.compute.packed.reducer"] == "tree"
    assert values["first.compute.packed.compute_pumping"] is False
    assert {value for key, value in values.items() if key.endswith("transport")} == {"direct"}
    assert explorer.choices(base) == () and explorer.cost(base).bottleneck is not None
    assert values == chosen(explorer, with_defaults(chain.network()))
    fifo = explorer.set(base, keyed({"levels.transport": "fifo"}))
    assert isinstance(fifo, Accepted)
    assert [(c.key, c.policy) for c in explorer.choices(with_defaults(fifo.network))] == [
        (("levels", "transport.fifo.depth"), Required())
    ]


def test_reopen_clears_its_keys_in_one_change_and_a_channel_waits_on_its_kernel() -> None:
    """Clearing a MatMul's folding leaves its channels' contracts open: they wait, their
    stored choices with them, and come back when the folding does. A clear never breaks
    a rule: a choice under a selector cleared waits on it, stored, and comes back in play
    with it. Nothing else is touched."""
    explorer, base = seam()
    opened, reach = explorer.reopen(base, [("first", "pe")])
    assert reach == (("first", "pe"),)
    assert set(explorer.chosen(base)) - set(explorer.chosen(opened)) == set(reach)
    held_ = opened.unbound("x")
    assert held_ is not None and held_.stored == {
        key[1]: value for key, value in explorer.chosen(base).items() if key[0] == "x"
    }
    # The folding back, the channel is bound again, its choices set again.
    again = explorer.set(opened, keyed({"first.pe": 1}))
    assert isinstance(again, Accepted) and not again.cleared and not again.settled
    assert explorer.chosen(again.network) == explorer.chosen(base)
    # A key not stored is nothing to reopen.
    network = chain.network()
    assert explorer.reopen(network, [("first", "pe")]) == (network, ())
    # A selector's nested choice waits on its presence, stored.
    sized = explorer.set(
        base, keyed({"hidden.transport": "fifo", "hidden.transport.fifo.depth": 16})
    )
    assert isinstance(sized, Accepted)
    depth = "hidden.transport.fifo.depth"
    waiting, reach = explorer.reopen(sized.network, [("hidden", "transport")])
    assert reach == (("hidden", "transport"),)
    assert state(waiting, depth) == inspection.Waiting(16, ("transport",))
    # The FIFO again, it is in play; another transport puts it out of play, kept.
    fifo = explorer.set(waiting, keyed({"hidden.transport": "fifo"}))
    assert isinstance(fifo, Accepted) and state(fifo.network, depth) == inspection.Set(16)
    direct = explorer.set(waiting, keyed({"hidden.transport": "direct"}))
    assert isinstance(direct, Accepted) and state(direct.network, depth) == inspection.Dormant(16)


def test_restore_gives_back_each_prior_value_still_accepted_and_sets_the_rest_by_policy() -> None:
    """A run's last step: the reopened choices the search left open take their prior
    values back, one batch; where the batch is refused, each alone, ranked, and a prior
    value refused there is set by the choice's policy instead, named (``by_policy``)."""
    explorer, base = seam()
    keys = [("first", "compute.packed.reducer"), ("second", "pe")]
    opened, reach = explorer.reopen(base, keys)
    assert reach == tuple(keys)
    prior = keyed(
        {
            "first.compute.packed.reducer": "compressor",
            "second.pe": 3,  # no divisor of four: outside its domain
        }
    )
    restored = explorer.restore(opened, prior, "the test")
    values = chosen(explorer, restored.network)
    assert values["first.compute.packed.reducer"] == "compressor"
    assert values["second.pe"] == 1  # first valid
    assert restored.kept == {("first", "compute.packed.reducer"): "compressor"}
    assert restored.by_policy == {("second", "pe"): 1}
    assert restored.unset == {} and explorer.choices(restored.network) == ()
    # Only what is open is given back: a stored value is never replaced.
    assert explorer.restore(base, prior).network is base


def test_set_refuses_a_change_that_breaks_a_rule_naming_what_stands_in_its_way() -> None:
    """Pumping on at SIMD 2, then SIMD replaced by 1: the rule that pumping needs lanes
    would be newly false, so ``set`` refuses the change by the rule's key, pumping
    standing in its way; nothing is committed."""
    explorer, base = seam()
    pumped = explorer.set(
        base, keyed({"first.simd": 2, "first.compute.packed.compute_pumping": True})
    )
    assert isinstance(pumped, Accepted)
    replaced = explorer.set(pumped.network, keyed({"first.simd": 1}))
    assert isinstance(replaced, Refused)
    rule = ("first", "compute.packed.pumping_has_lanes")
    ((key, (found,)),) = replaced.why.items()
    assert (key, found.code) == (rule, "dotp-pumping")
    assert replaced.standing == {rule: (("first", "compute.packed.compute_pumping"),)}
    assert "pumping requires SIMD >= 2" in said(replaced.why)
    # Pumping replaced in the same batch, the rule holds: accepted.
    both = explorer.set(
        pumped.network,
        keyed({"first.simd": 1, "first.compute.packed.compute_pumping": False}),
    )
    assert isinstance(both, Accepted)
    # A change of the change's own key outside its domain is refused.
    assert isinstance(explorer.set(base, keyed({"first.pe": 3})), Refused)


def test_a_held_choice_a_change_would_put_out_of_play_refuses_the_change() -> None:
    """A held choice a change would put out of play refuses the change, naming it; one a
    change leaves breaking a rule refuses it by the rule, as any other; a strategy that
    cannot restore a held choice never reopens it."""
    explorer, base = seam()
    # The activations' adapter memory: a folding of four lanes a side needs no adapter,
    # so the change puts its stored style out of play, kept; held, the change is refused.
    style = ("x", "adapter_ram_style")
    folding = keyed({"first.simd": 4, "first.pe": 4})
    unheld = explorer.set(base, folding)
    assert isinstance(unheld, Accepted) and not unheld.cleared
    assert state(unheld.network, dotted(style)) == inspection.Dormant("distributed")
    explorer.held[style] = "distributed"
    dropped = explorer.set(base, folding)
    assert isinstance(dropped, Refused)
    ((key, (found,)),) = dropped.why.items()
    assert (key, found.code, found.owner) == (
        style,
        "held",
        "x.adapter_ram_style",
    )
    assert found.message == "held to 'distributed', put out of play (dormant) by the change"
    del explorer.held[style]
    pumped = explorer.set(
        base, keyed({"first.simd": 2, "first.compute.packed.compute_pumping": True})
    )
    assert isinstance(pumped, Accepted)
    pumping = ("first", "compute.packed.compute_pumping")
    explorer.held[pumping] = True
    # One lane: the change is refused by the rule pumping needs lanes, not by the hold.
    ruled = explorer.set(pumped.network, keyed({"first.simd": 1}))
    assert isinstance(ruled, Refused)
    assert "pumping requires SIMD >= 2" in said(ruled.why)
    assert not any(f.code == "held" for why in ruled.why.values() for f in why)
    # Not reached, it stays: SIMD 4 keeps pumping possible.
    kept = explorer.set(pumped.network, keyed({"first.simd": 4}))
    assert isinstance(kept, Accepted)
    assert explorer.chosen(kept.network)[pumping] is True
    # A fold reopens SIMD, never the held pumping: SIMD's first value under it is 2.
    folded = TargetCycles(48).explore(explorer, pumped.network)
    assert explorer.chosen(folded)[pumping] is True
    assert explorer.chosen(folded)[("first", "simd")] == 2
    assert pumping not in explorer.runs[-1].reopened


def test_a_strategy_reopens_its_scope_and_keeps_what_it_does_not_decide() -> None:
    """A strategy changes only what it decides: a reducer an earlier strategy set stays,
    though the folding strategy reopens every choice of a member; the run records what
    it reopened and what it gave back."""
    explorer, base = seam()
    compressor = explorer.set(base, keyed({"second.compute.packed.reducer": "compressor"}))
    assert isinstance(compressor, Accepted)
    folded = TargetCycles(12).explore(explorer, compressor.network)
    values = chosen(explorer, folded)
    assert values["second.compute.packed.reducer"] == "compressor"
    assert folding(explorer, folded) == FOLDED_12
    (run,) = explorer.runs
    assert run.strategy == "target_cycles"
    assert ("second", "compute.packed.reducer") in run.reopened
    # A channel's choices are not a fold's: they stay as stored, and one standing in a
    # fold's way is the fold rule's to clear and set again (``Seam.set``).
    assert not any(key[0] in folded.channels for key in run.reopened)
    assert run.kept[("second", "compute.packed.reducer")] == "compressor"
    assert set(run.kept) | set(run.by_policy) <= set(run.reopened)
    # Again on the folded network: the same folding, nothing new.
    again = TargetCycles(12).explore(explorer, folded)
    assert explorer.chosen(again) == explorer.chosen(folded)


def test_a_pin_holds_its_keys_and_a_later_strategy_folds_around_it() -> None:
    """A ``Hold`` first (SR6): the target throughput folds the other MatMul and
    keeps the held one; its relax step reaches the hold's bottleneck."""
    explorer, base = seam()
    explorer.held.update(keyed({"first.pe": 1, "first.simd": 1}))
    strategy = TargetCycles(1)
    folded = strategy.explore(explorer, base)
    values = chosen(explorer, folded)
    assert (values["first.pe"], values["first.simd"]) == (1, 1)
    # first at 48 cycles binds: second folds to meet it, no faster.
    assert strategy.relaxed_to == 48
    assert (values["second.pe"], values["second.simd"]) == (1, 1)
    assert ("first", "pe") not in explorer.runs[-1].reopened


def test_a_hold_a_rule_refuses_names_the_rule_and_what_stands_in_its_way() -> None:
    """A ``Hold`` is one batch: a value a rule refuses is refused (``pin-refused``), the
    rule and the stored choices standing in its way named."""
    explorer, base = seam()
    with refused(
        "pin-refused",
        error=ExploreError,
        owner="held",
        message=r"pumping_has_lanes: .*dotp-pumping.*\(standing: first\.simd\)",
    ):
        Hold({"first": {"compute.packed.compute_pumping": True}}).explore(explorer, base)


def test_the_defaults_strategy_resets_every_choice_not_held() -> None:
    explorer, base = seam()
    folded = TargetCycles(12).explore(explorer, base)
    explorer.held[("first", "simd")] = 4
    reset = Defaults().explore(explorer, folded)
    assert folding(explorer, reset) == {
        **{key: 1 for key in FOLDED_12},
        "first.simd": 4,
    }
    assert chosen(explorer, reset) == {**chosen(explorer, base), "first.simd": 4}
    assert Defaults().report() == {"strategy": "defaults", "unset": {}}


def test_the_debug_placeholder_takes_a_required_choice_s_first_case_or_refuses_it() -> None:
    """With nothing required open it sets nothing; a FIFO with no depth is a required
    choice known by membership only, which it refuses, named."""
    explorer, base = seam()
    placeholder = DebugPlaceholder()
    assert explorer.chosen(placeholder.explore(explorer, base)) == explorer.chosen(base)
    assert placeholder.report() == {"strategy": "placeholder", "set": {}}
    assert placeholder.label == "DEBUG: set by placeholder"
    fifo = explorer.set(base, keyed({"levels.transport": "fifo"}))
    assert isinstance(fifo, Accepted)
    with refused(
        "placeholder-no-case",
        error=ExploreError,
        message="known by membership only: .*levels.transport.fifo",
    ):
        DebugPlaceholder().explore(explorer, fifo.network)


# -- the most throughput within resources -----------------------------------------------------

#: The Chain's part: its totals, of which a budget is a fraction. At each budget of cycles
#: the Chain folds to (``TargetCycles``), restored: 6 cycles, 482 LUT and 8 DSP; 12,
#: 379 and 8; 24, 331 and 4; 48 (the least parallelism), 328 and 2.
PART = replace(FULL_DSP48E2, resources=Resources(lut=1000, ff=2000, bram18=10, uram=0, dsp=10))


def within_part() -> tuple[Seam, Network]:
    return held(Seam(PART)), with_defaults(chain.network())


def test_the_seam_states_the_network_s_resources_once_its_members_do() -> None:
    explorer, base = within_part()
    waiting = explorer.resources(chain.network())
    # Not yet: why, its findings (the open choices it waits on).
    assert isinstance(waiting, tuple) and waiting
    assert {finding.code for finding in waiting} == {"decision-unassigned"}
    used = explorer.cost(base).used
    assert used is not None and explorer.resources(base) == Stated(used, {})


def test_max_throughput_keeps_the_least_budget_whose_point_fits() -> None:
    explorer, network = within_part()
    strategy = MaxThroughput({"dsp": 0.5})
    folded = strategy.explore(explorer, network)
    # Half the part's 10 DSPs: the 6- and 12-cycle points use 8, the 24-cycle one 4.
    assert folding(explorer, folded) == {
        "first.pe": 1,
        "first.simd": 2,
        "second.pe": 1,
        "second.simd": 2,
    }
    # The network is returned restored, as it was costed.
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
    of that network restored, in fewer changes than folding and restoring each alone."""
    explorer, network = within_part()
    strategy = MaxThroughput({"dsp": 0.5})
    strategy.explore(explorer, network)
    searched = explorer.attempts
    prior = {key: network.chosen()[key] for key in explorer.runs[-1].reopened}
    alone = held(Seam(PART))
    for tried in strategy.tried:
        budget = explore_module._UNBOUNDED if tried.cycles is None else tried.cycles
        folder = TargetCycles(budget)
        folded = folder.explore(alone, network)
        restored = explorer.restore(tried.folded, prior).network
        assert explorer.chosen(restored) == alone.chosen(folded)
        assert tried.relaxed_to == folder.relaxed_to
        assert tried.reached == alone.cost(folded).bottleneck
        assert tried.stated == alone.resources(folded)
    assert searched < alone.attempts


def test_a_core_no_value_of_which_is_possible_refuses_the_cost_strategies_naming_why() -> None:
    """A MatMul whose weights no core takes (unsigned weights on a DSP48E2 part): no value
    of its core is possible. Stored ``packed``, its bind is refused by the rule, by name;
    left open, its cycles wait on it, and so do its channels'. Sizing the FIFOs raises
    ``cycles-unknown``, naming what each member waits on; the throughput search's fold
    raises the core's own refusal (``choice-no-valid-value``, owned by it, each value's
    rules its causes: the weight port's for ``packed``). Never a bare code."""
    unsigned = matmul_point(
        m=3,
        n=4,
        k=4,
        activation_dtype=DataType["INT3"],
        weights_dtype=DataType["UINT3"],
        platform=PART,
    )
    with pytest.raises(NetworkError, match="compute.packed.w_element_taken: dtype-family"):
        choose(unsigned, {"matmul.compute": "packed"})
    with refused(
        "cycles-unknown",
        error=ExploreError,
        message="needs every member's cycles: matmul: matmul.compute",
    ):
        SizeFifos().explore(Seam(PART), unsigned)
    with refused(
        "choice-no-valid-value",
        error=ExploreError,
        owner="matmul.compute",
        message="'packed': matmul.compute.packed.w_element_taken: .*dtype-family",
    ) as caught:
        MaxThroughput({"dsp": 0.5}).explore(Seam(PART), unsigned)
    assert ("compute.packed.w_element_taken", "dtype-family") in {
        (cause.owner, cause.code) for cause in caught.finding.causes
    }


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
    assert chosen(explorer, folded)["first.simd"] == 1
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
    whose fold reaches more cycles than itself (a mover's converter, SZ6) departs from
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
    with refused("budget-invalid", error=ExploreError, message="needs a budget"):
        MaxThroughput({})
    with refused("budget-invalid", error=ExploreError, message=r"\['luts'\]"):
        MaxThroughput({"luts": 0.5})
    with refused("budget-invalid", error=ExploreError, message=r"within lut: a fraction of 0"):
        MaxThroughput({"lut": 0})
    with refused("budget-invalid", error=ExploreError, message=r"'half'"):
        MaxThroughput({"lut": "half"})  # type: ignore[dict-item]
    explorer, network = seam()
    with refused("platform-unstated", error=ExploreError, owner="max_throughput"):
        MaxThroughput({"lut": 0.5}).explore(explorer, network)
