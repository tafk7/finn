# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The most throughput within resources (``MaxThroughput``) through the exploration, and
the warning over the part: on the ``ip`` shell (the part, its total its partition's) and
in the Zynq shell (``pynq``: its two ends and its static region join the partition's).

The budget is a fraction of the part's totals (xczu3eg: 70 560 LUT, 141 120 FF, 432
RAMB18, 360 DSP), against the shell's total, the report's ``resources.used``. On
each shell, the search's logic is on the Chain as KernelOps (``support.models``),
where a search takes seconds, and one search on TFC_W2A2 for Ultra96 at 5 ns is the
end-to-end check (bisection over 17 or 18 folds by ``TargetCycles``, minutes each).
The search on the Chain without the KernelOps is ``tests/kernels/test_explore.py``'s.

Folded by ``TargetCycles`` and settled, the Chain on ip reaches 3 cycles a frame at
669 LUT (its fastest), 12 at 359, 24 at 321 and 48 (its least parallelism) at 298. In
the Zynq shell its ends hold it at 16 cycles at the fastest, and its shell's total is
about 9 500 LUT more: the ends' and the static region's.

TFC's least parallelism (no budget) reaches 50 176 cycles a frame at 1 131 LUT on ip,
10 875 in the Zynq shell; at 3 136 cycles it uses 1 196 and 10 940. Each end-to-end
search budgets between the two: of the budgets measured, the cheapest search that
still bisects both ways to a point that fits, its folds all at budgets of thousands
of cycles. R3's budgets (``{lut: 0.5}``, ``0.07`` and ``0.2``) are its record's probe's.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from qonnx.core.modelwrapper import ModelWrapper
from support.models import TARGET, kernel_model
from support.tfc import ULTRA96

from finn.core.graph.annotations.platform import write_target
from finn.core.reporting.findings import Finding, FindingKind
from finn.custom_op.kernels.shell import verify_partition
from finn.kernels.explore import (
    Explorer,
    MaxThroughput,
    Pinned,
    Rebaseline,
    SizeFifos,
    TargetThroughput,
)
from finn.kernels.utilization import total
from finn.platform.capabilities import Resources
from finn.platform.resolve import resolve_target
from finn.platform.target import Target
from finn.transformation.kernels import explore_kernel_choices, kernel_choices_config

ULTRA96_IP = resolve_target(part=ULTRA96.part, period_ns=ULTRA96.platform.period_ns)
SHELLS = {"ip": ULTRA96_IP, "pynq": ULTRA96}


def within(lut: float) -> list[Explorer]:
    """The search within ``lut`` of the part's LUTs, then FIFO sizing."""
    return [MaxThroughput({"lut": lut}), SizeFifos()]


def explore_each(
    model: ModelWrapper, strategies: list[Explorer]
) -> tuple[dict[str, Any], tuple[Finding, ...]]:
    """``model`` explored by each of ``strategies`` in turn, an exploration each, as a
    build's strategy steps run them: the last exploration's report, with each one's
    strategy entry in order (``strategies``), and every finding, in order."""
    results = [explore_kernel_choices(model, each) for each in strategies]
    report = {**results[-1].report, "strategies": [each.report["strategy"] for each in results]}
    return report, tuple(finding for each in results for finding in each.findings)


def explored(model: ModelWrapper, target: Target, strategies: list[Explorer]) -> dict[str, Any]:
    """A copy of ``model`` explored for ``target`` by each of ``strategies`` in turn
    (``Rebaseline()`` first: from every choice's baseline); the report (``explore_each``)."""
    model = ModelWrapper(model.model.__deepcopy__())
    write_target(model, target)
    return explore_each(model, strategies)[0]


# -- on ip ---------------------------------------------------------------------------------


def test_the_search_keeps_the_least_budget_whose_point_fits_and_reports_it() -> None:
    """Half a percent of the part's LUTs (352) on ip: the fastest fold does not fit, the
    least parallelism does, and bisection between them keeps the least budget that
    fits. The report states the point against the budget, read from settled costs."""
    report = explored(kernel_model(), TARGET, [Rebaseline(), *within(0.005)])
    _, searched, sized = report["strategies"]
    assert searched["within"] == {"lut": 0.005} and searched["budget"] == {"lut": 352}
    tried = searched["tried"]
    brackets = [(row["cycles"], row["relaxed_to"], row["bottleneck"], row["fits"]) for row in tried]
    assert brackets[:2] == [(1, 3, 3, False), (None, None, 48, True)]
    assert searched["fastest"] == 3
    # Budgets 15, 21 and 23 fold to 12 cycles at 359 LUT, over; 24 and 26 to 24 at 321.
    assert min(row["cycles"] for row in tried[2:] if row["fits"]) == 24
    assert searched["bottleneck"] == report["bottleneck"] and report["bottleneck"]["cycles"] == 24
    used = report["resources"]["used"]
    assert searched["used"] == used and used["lut"] == 321 and searched["fits"]
    # On ip, the shell's total is its partition's.
    assert used == report["resources"]["shell"]["partition"]
    assert searched["binding"] == "lut" and searched["ratio"] == {"lut": round(321 / 352, 4)}
    assert searched["monotone"] and searched["departures"] == []
    # It reopened the folding and settled the rest back: it changed the folding alone.
    assert searched["reopened"] > searched["kept"] > 0
    assert {key.rpartition(".")[2] for key in searched["changed"]} <= {
        "pe",
        "simd",
        "compute_pumping",
    }
    assert sized["strategy"] == "size_fifos" and sized["fifo_bits"] == 0


def searched_tfc(
    tfc: ModelWrapper, shell: str, fraction: float, budget: int, cycles: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """max_throughput {lut: fraction}, then size_fifos, on TFC: bisection between the fastest
    fold, which does not fit, and the least parallelism, which does, through budgets
    that fit and budgets that do not, keeps the least budget that fits, ``cycles``,
    and the report states its point against the budget. The search's entry and the
    report."""
    report = explored(tfc, SHELLS[shell], within(fraction))
    searched, sized = report["strategies"]
    assert searched["strategy"] == "max_throughput" and sized["strategy"] == "size_fifos"
    assert searched["within"] == {"lut": fraction} and searched["budget"] == {"lut": budget}
    used = report["resources"]["used"]
    # The search costs the shell's total; sizing places no FIFO here (FIFOs would join
    # the total after the search, NOTE §4.2), so the report's total is the search's.
    assert sized["fifo_bits"] == 0
    assert searched["used"] == used and searched["fits"] and used["lut"] <= budget
    assert searched["binding"] == "lut" and searched["ratio"] == {
        "lut": round(used["lut"] / budget, 4)
    }
    assert searched["bottleneck"] == report["bottleneck"]
    assert report["bottleneck"]["cycles"] == cycles
    tried = searched["tried"]
    assert tried[0]["cycles"] == 1 and searched["fastest"] == tried[0]["bottleneck"]
    assert not tried[0]["fits"]
    assert (tried[1]["cycles"], tried[1]["bottleneck"], tried[1]["fits"]) == (None, 50176, True)
    bisected = tried[2:]
    assert {row["fits"] for row in bisected} == {True, False}
    assert min(row["cycles"] for row in bisected if row["fits"]) == cycles
    assert all(row["cycles"] < cycles for row in bisected if not row["fits"])
    assert len(tried) == 18
    return searched, report


@pytest.mark.slow
def test_tfc_on_ip_searches_to_the_least_budget_that_fits(tfc: ModelWrapper) -> None:
    """1.70 % of the part's LUTs (1 199), between the least parallelism's 1 169 and the
    1 234 of 3 136 cycles: 3 584 cycles at 1 197, the shell's total its partition's."""
    searched, report = searched_tfc(tfc, "ip", 0.0170, 1199, 3584)
    resources = report["resources"]
    assert resources["used"]["lut"] == 1197 and resources["used"] == resources["shell"]["partition"]
    # TargetCycles folds by cycles alone: a looser budget can pick a costlier shape.
    assert "no budget uses more than budget 25089: lut 1169 > 1162" in searched["departures"]


# -- in the Zynq shell ----------------------------------------------------------------------


def test_in_the_zynq_shell_the_search_costs_the_shell_s_total() -> None:
    """The whole part on pynq: the fastest fold, relaxed to the ends' 16 cycles, fits at
    once, so it is the one fold. What it costs is the shell's total, its ends and its
    static region beside its partition."""
    report = explored(kernel_model(), ULTRA96, [Rebaseline(), *within(1.0)])
    searched = report["strategies"][1]
    assert [(row["cycles"], row["relaxed_to"], row["fits"]) for row in searched["tried"]] == [
        (1, 16, True)
    ]
    assert report["bottleneck"] == {"members": ["x", "y"], "cycles": 16}
    shell = report["resources"]["shell"]
    parts = [shell["partition"], *shell["ends"].values(), *shell["static_region"].values()]
    summed = total(Resources(**part) for part in parts)
    assert (
        searched["used"]
        == report["resources"]["used"]
        == {name: getattr(summed, name) for name in shell["partition"]}
    )
    assert searched["used"]["lut"] > shell["partition"]["lut"] + 9000
    assert searched["binding"] == "lut" and searched["fits"]


def test_where_nothing_fits_the_search_states_it_and_keeps_the_least_parallelism() -> None:
    """A tenth of the part's LUTs on pynq is less than the shell's static region alone:
    neither the fastest fold nor the least parallelism fits. The search states it (a
    ``resource-budget`` limitation, in its report and the result), naming the binding
    resource, and keeps the least parallelism; the point is explored and
    saved all the same (RC5). Here the least parallelism uses more LUTs than the fastest
    fold (a wider end), a departure the report states. FIFOs join after the search:
    sizing places one on the output, so the report's total is above the search's."""
    model = kernel_model()
    write_target(model, ULTRA96)
    report, findings = explore_each(model, [Rebaseline(), *within(0.1)])
    _, searched, sized = report["strategies"]
    (budget,) = searched["findings"]
    assert (budget["kind"], budget["code"]) == ("limitation", "resource-budget")
    assert "most of lut: lut 9882 of 7056" in budget["message"]
    assert budget["details"] == {"binding": "lut", "over": ("lut",)}
    # Before sizing, the output's direct throttles its producer: no exploration checks it,
    # and the build's verification names it.
    assert [finding.code for finding in findings] == ["resource-budget"]
    unsized = kernel_model()
    write_target(unsized, ULTRA96)
    explore_kernel_choices(unsized, Rebaseline())
    _, check = verify_partition(unsized, "the chain")
    assert [(f.kind, f.code) for f in check.findings] == [
        (FindingKind.LIMITATION, "direct-throttles")
    ]
    assert [row["cycles"] for row in searched["tried"]] == [1, None]
    assert searched["fits"] is False and searched["binding"] == "lut"
    assert report["bottleneck"]["cycles"] == 48 and searched["bottleneck"]["cycles"] == 48
    assert not searched["monotone"]
    assert searched["departures"] == ["no budget uses more than budget 1: lut 9882 > 9838"]
    # The least parallelism is every MatMul's baseline: nothing changed, every value saved.
    assert kernel_choices_config(model) and searched["committed"] == 0
    assert sized["fifo_bits"] > 0
    assert report["resources"]["used"]["lut"] > searched["used"]["lut"] == 9882


@pytest.mark.slow
def test_tfc_in_the_zynq_shell_searches_to_the_least_budget_that_fits(tfc: ModelWrapper) -> None:
    """15.57 % of the part's LUTs (10 986), between the least parallelism's shell total,
    10 913, and the 11 077 of 2 048 cycles: 3 136 cycles at 10 978, the ends and the
    static region beside the partition."""
    searched, report = searched_tfc(tfc, "pynq", 0.1557, 10986, 3136)
    resources = report["resources"]
    assert resources["used"]["lut"] == 10978 > resources["shell"]["partition"]["lut"] + 9000
    # The input end bounds the shell: the fastest fold, relaxed to 53, reaches 62 at
    # the end (SZ6), a departure stated and not corrected.
    tried = searched["tried"]
    assert (tried[0]["relaxed_to"], tried[0]["bottleneck"]) == (53, 62)
    assert "budget 1 relaxed to 53, reaching 62" in searched["departures"]


# -- over the part --------------------------------------------------------------------------


def test_a_point_over_the_part_is_a_finding_naming_the_binding_resource() -> None:
    """Whichever strategy chose it: the Chain at its fastest (target_throughput 2e8, a
    cycle a frame at 5 ns, relaxed to 3) uses 669 LUT, more than a part of 500 has; a
    limitation names the binding resource, and the point is explored and saved all the
    same (RC5)."""
    small = Resources(lut=500, ff=141120, bram18=432, uram=0, dsp=360)
    part = replace(TARGET, platform=replace(TARGET.platform, resources=small))
    report = explored(kernel_model(), part, [Rebaseline(), TargetThroughput(2e8)])
    resources = report["resources"]
    assert report["bottleneck"]["cycles"] == 3
    assert resources["binding"] == "lut" and list(resources["over"]) == ["lut"]
    assert resources["over"]["lut"] == {"used": 669, "platform": 500}
    (over,) = resources["findings"]
    assert (over["kind"], over["code"]) == ("limitation", "resources-over-part")
    assert over["message"].startswith("the point uses more than the platform's part has")
    assert "most of lut: lut 669 of 500" in over["message"]


def test_a_member_that_states_no_resources_is_budgeted_as_a_lower_bound(tmp_path: Path) -> None:
    """A compressor reducer pinned on ``first`` states no resources (RC5: rank and state
    it): the search budgets what is stated and says so (a ``resources-unstated``
    limitation, its member's findings the causes), naming it, and the report's total is
    a lower bound, naming it too, with its findings."""
    pinned = tmp_path / "compressor.json"
    pinned.write_text('{"first": {"compute.packed.reducer": "compressor"}}')
    model = ModelWrapper(kernel_model().model.__deepcopy__())
    write_target(model, TARGET)
    report, findings = explore_each(model, [Rebaseline(), Pinned(pinned), *within(0.005)])
    searched = report["strategies"][2]
    assert searched["lower_bound"] is True and list(searched["unstated"]) == ["first"]
    assert all(record["code"] for record in searched["unstated"]["first"])
    (unstated,) = [f for f in findings if f.code == "resources-unstated"]
    assert unstated.kind is FindingKind.LIMITATION and unstated.owner == "max_throughput"
    assert re.search(r"\bfirst .*compressor", unstated.message)
    assert dict(unstated.details)["members"] == ("first",) and unstated.causes
    assert searched["findings"][0]["code"] == "resources-unstated"
    resources = report["resources"]
    assert resources["lower_bound"] is True and resources["shell"] is None
    assert list(resources["unstated"]) == ["first"]
    assert resources["counted"].startswith("a lower bound")
    stated = total(
        Resources(**row["resources"])
        for row in report["members"].values()
        if row["resources"] is not None
    )
    assert resources["used"] == {name: getattr(stated, name) for name in resources["used"]}
    assert report["members"]["first"]["resources"] is None
