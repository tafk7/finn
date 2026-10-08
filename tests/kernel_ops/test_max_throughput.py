# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The most throughput within resources (``MaxThroughput``) through the exploration, and
the warning over the part: on the ``ip`` shell (the part, its total its partition's) and
in the Zynq shell (``pynq``: its two ends and its static region join the partition's).

The budget is a fraction of the part's totals (xczu3eg: 70 560 LUT, 141 120 FF, 432
RAMB18, 360 DSP), against the shell's total, the report's ``resources.used``. On
each shell, the search's logic is on the Chain as KernelOps (``kernel_ops.models``),
where a search takes seconds, and one search on TFC_W2A2 for Ultra96 at 5 ns is the
end-to-end check (bisection over 17 or 18 folds by ``TargetCycles``, minutes each).
The search on the Chain without the KernelOps is ``tests/kernels/test_explore.py``'s.

Folded by ``TargetCycles`` and completed, the Chain on ip reaches 3 cycles a frame at
669 LUT (its fastest), 12 at 355, 24 at 320 and 48 (its least parallelism) at 295. In
the Zynq shell its ends hold it at 16 cycles at the fastest, and its shell's total is
about 9 500 LUT more: the ends' and the static region's.

TFC's least parallelism (no budget) reaches 50 176 cycles a frame at 1 034 LUT on ip,
10 778 in the Zynq shell; at 3 136 cycles it uses 1 143 and 10 887. Each end-to-end
search budgets between the two: of the budgets measured, the cheapest search that
still bisects both ways to a point that fits, its folds all at budgets of thousands
of cycles. R3's budgets (``{lut: 0.5}``, ``0.07`` and ``0.2``) are its record's probe's.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.kernels.base import write_target
from finn.kernels.explore import ResourceBudgetWarning
from finn.kernels.target import Target
from finn.kernels.utilization import Resources, total
from finn.platform import resolve_target
from finn.transformation.kernels import (
    explore_kernel_choices,
    kernel_choices_config,
    strategy,
)
from kernel_ops.models import TARGET, kernel_model
from kernel_ops.tfc import ULTRA96

ULTRA96_IP = resolve_target(part=ULTRA96.part, period_ns=ULTRA96.platform.period_ns)
SHELLS = {"ip": ULTRA96_IP, "pynq": ULTRA96}
SIZE_FIFOS = {"strategy": "size_fifos"}


def within(lut: float) -> list[dict[str, Any]]:
    return [{"strategy": "max_throughput", "within": {"lut": lut}}, SIZE_FIFOS]


def explored(
    model: ModelWrapper, target: Target, specs: list[dict[str, Any]], *, fresh: bool = False
) -> dict[str, Any]:
    """A copy of ``model`` explored for ``target`` by the strategies ``specs`` write, as a
    build lists them (``fresh``: from no choice); the report."""
    model = ModelWrapper(model.model.__deepcopy__())
    write_target(model, target)
    strategies = [strategy(spec) for spec in specs]
    return dict(explore_kernel_choices(model, strategies, fresh=fresh).report)


# -- on ip ---------------------------------------------------------------------------------


def test_the_search_keeps_the_least_budget_whose_point_fits_and_reports_it() -> None:
    """Half a percent of the part's LUTs (352) on ip: the fastest fold does not fit, the
    least parallelism does, and bisection between them keeps the least budget that
    fits. The report states the point against the budget, read from completed costs."""
    report = explored(kernel_model(), TARGET, within(0.005), fresh=True)
    searched, sized = report["strategies"]
    assert searched["within"] == {"lut": 0.005} and searched["budget"] == {"lut": 352}
    tried = searched["tried"]
    brackets = [(row["cycles"], row["relaxed_to"], row["bottleneck"], row["fits"]) for row in tried]
    assert brackets[:2] == [(1, 3, 3, False), (None, None, 48, True)]
    assert searched["fastest"] == 3
    # Budgets 15, 21 and 23 fold to 12 cycles at 355 LUT, over; 24 and 26 to 24 at 320.
    assert min(row["cycles"] for row in tried[2:] if row["fits"]) == 24
    assert searched["bottleneck"] == report["bottleneck"] and report["bottleneck"]["cycles"] == 24
    used = report["resources"]["used"]
    assert searched["used"] == used and used["lut"] == 320 and searched["fits"]
    # On ip, the shell's total is its partition's.
    assert used == report["resources"]["shell"]["partition"]
    assert searched["binding"] == "lut" and searched["ratio"] == {"lut": round(320 / 352, 4)}
    assert searched["monotone"] and searched["departures"] == []
    assert searched["read_completed"].startswith("read completed choices: ")
    assert sized["strategy"] == "size_fifos" and sized["fifo_bits"] == 0


@pytest.fixture(scope="module")
def tfc(tfc_kernel_ops: Path) -> ModelWrapper:
    """TFC_W2A2 as KernelOps for Ultra96 at 5 ns."""
    return ModelWrapper(str(tfc_kernel_ops))


def searched_tfc(
    tfc: ModelWrapper, shell: str, fraction: float, budget: int, cycles: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """[max_throughput {lut: fraction}, size_fifos] on TFC: bisection between the fastest
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
    # What it committed it chose from completed costs, and the report says so.
    assert searched["read_completed"].startswith("read completed choices: ")
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
    """1.6 % of the part's LUTs (1 128), between the least parallelism's 1 034 and the
    1 143 of 3 136 cycles: 3 584 cycles at 1 116, the shell's total its partition's."""
    searched, report = searched_tfc(tfc, "ip", 0.016, 1128, 3584)
    resources = report["resources"]
    assert resources["used"]["lut"] == 1116 and resources["used"] == resources["shell"]["partition"]
    # TargetCycles folds by cycles alone: a looser budget can pick a costlier shape.
    assert "no budget uses more than budget 25089: lut 1034 > 1027" in searched["departures"]


# -- in the Zynq shell ----------------------------------------------------------------------


def test_in_the_zynq_shell_the_search_costs_the_shell_s_total() -> None:
    """The whole part on pynq: the fastest fold, relaxed to the ends' 16 cycles, fits at
    once, so it is the one fold. What it costs is the shell's total, its ends and its
    static region beside its partition."""
    report = explored(kernel_model(), ULTRA96, within(1.0), fresh=True)
    searched = report["strategies"][0]
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


def test_where_nothing_fits_the_search_warns_and_keeps_the_least_parallelism() -> None:
    """A tenth of the part's LUTs on pynq is less than the shell's static region alone:
    neither the fastest fold nor the least parallelism fits. The search warns, naming
    the binding resource, and keeps the least parallelism; the point is explored and
    saved all the same (RC5). Here the least parallelism uses more LUTs than the fastest
    fold (a wider end), a departure the report states. FIFOs join after the search:
    sizing places one on the output, so the report's total is above the search's."""
    model = kernel_model()
    write_target(model, ULTRA96)
    strategies = [strategy(spec) for spec in within(0.1)]
    with pytest.warns(ResourceBudgetWarning, match="most of lut: lut 9879 of 7056"):
        report = explore_kernel_choices(model, strategies, fresh=True).report
    searched, sized = report["strategies"]
    assert [row["cycles"] for row in searched["tried"]] == [1, None]
    assert searched["fits"] is False and searched["binding"] == "lut"
    assert report["bottleneck"]["cycles"] == 48 and searched["bottleneck"]["cycles"] == 48
    assert not searched["monotone"]
    assert searched["departures"] == ["no budget uses more than budget 1: lut 9879 > 9834"]
    assert kernel_choices_config(model) and searched["committed"] > 0
    assert sized["fifo_bits"] > 0
    assert report["resources"]["used"]["lut"] > searched["used"]["lut"] == 9879


@pytest.mark.slow
def test_tfc_in_the_zynq_shell_searches_to_the_least_budget_that_fits(tfc: ModelWrapper) -> None:
    """15.5 % of the part's LUTs (10 936), between the least parallelism's shell total,
    10 778, and the 11 011 of 2 048 cycles: 3 136 cycles at 10 887, the ends and the
    static region beside the partition."""
    searched, report = searched_tfc(tfc, "pynq", 0.155, 10936, 3136)
    resources = report["resources"]
    assert resources["used"]["lut"] == 10887 > resources["shell"]["partition"]["lut"] + 9000
    # The input end bounds the shell: the fastest fold, relaxed to 53, reaches 62 at
    # the end (SZ6), a departure stated and not corrected.
    tried = searched["tried"]
    assert (tried[0]["relaxed_to"], tried[0]["bottleneck"]) == (53, 62)
    assert "budget 1 relaxed to 53, reaching 62" in searched["departures"]


# -- over the part --------------------------------------------------------------------------


def test_a_point_over_the_part_warns_naming_the_binding_resource() -> None:
    """Whichever strategy chose it: the Chain at its fastest (target_throughput 2e8, a
    cycle a frame at 5 ns, relaxed to 3) uses 669 LUT, more than a part of 500 has; a
    warning names the binding resource, and the point is explored and saved all the
    same (RC5)."""
    small = Resources(lut=500, ff=141120, bram18=432, uram=0, dsp=360)
    part = replace(TARGET, platform=replace(TARGET.platform, resources=small))
    with pytest.warns(ResourceBudgetWarning, match="most of lut: lut 669 of 500"):
        report = explored(
            kernel_model(), part, [{"strategy": "target_throughput", "fps": 2e8}], fresh=True
        )
    resources = report["resources"]
    assert report["bottleneck"]["cycles"] == 3
    assert resources["binding"] == "lut" and list(resources["over"]) == ["lut"]
    assert resources["over"]["lut"] == {"used": 669, "platform": 500}
    assert resources["warning"].startswith("the point uses more than the platform's part has")
