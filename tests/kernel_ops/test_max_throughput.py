# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The most throughput within resources (``MaxThroughput``), and the warning over the
part, on TFC_W2A2 for Ultra96 at 5 ns: on the ``ip`` shell (the part, its total its
partition's) and in the Zynq shell (``pynq``: its two ends and its static region
join the partition's).

The budget is a fraction of the part's totals (xczu3eg: 70 560 LUT, 141 120 FF, 432
RAMB18, 360 DSP), against the shell's total, the report's ``resources.used``. Each
search takes minutes on ``ip`` (bisection over about 18 folds by ``TargetCycles``), so
each test explores once.
"""

from __future__ import annotations

from typing import Any

import pytest
from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.kernels.base import write_target
from finn.kernels.explore import ResourceBudgetWarning
from finn.kernels.target import Target
from finn.platform import resolve_target
from finn.transformation.kernels import (
    InferKernelTensors,
    ToKernelOps,
    explore_kernel_choices,
    strategy,
)
from kernel_ops.tfc import ULTRA96, streamlined

ULTRA96_IP = resolve_target(part=ULTRA96.part, period_ns=ULTRA96.platform.period_ns)
SHELLS = {"ip": ULTRA96_IP, "pynq": ULTRA96}
SIZE_FIFOS = {"strategy": "size_fifos"}

#: TFC's bottleneck at 1e6 fps ([target_throughput 1e6, size_fifos]) on ``ip``.
AT_1E6_CYCLES = 196


@pytest.fixture(scope="module")
def tfc(tmp_path_factory: pytest.TempPathFactory) -> ModelWrapper:
    """TFC_W2A2 as KernelOps for Ultra96 at 5 ns (half a minute)."""
    source = streamlined(tmp_path_factory.mktemp("tfc"))
    return source.transform(ToKernelOps(ULTRA96)).transform(InferKernelTensors())


def explored(tfc: ModelWrapper, target: Target, specs: list[dict[str, Any]]) -> dict[str, Any]:
    """TFC explored for ``target`` by the strategies ``specs`` write, as a build lists
    them; the report."""
    model = ModelWrapper(tfc.model.__deepcopy__())
    write_target(model, target)
    return dict(explore_kernel_choices(model, [strategy(spec) for spec in specs]).report)


def within(lut: float) -> list[dict[str, Any]]:
    return [{"strategy": "max_throughput", "within": {"lut": lut}}, SIZE_FIFOS]


@pytest.mark.slow
@pytest.mark.parametrize("shell", ["ip", "pynq"])
def test_tfc_within_half_the_part_s_luts_reports_its_point_against_the_budget(
    tfc: ModelWrapper, shell: str
) -> None:
    """[max_throughput {lut: 0.5}, size_fifos]: the point reached, its bottleneck, its
    resources against the budget (the shell's total) and the binding resource."""
    report = explored(tfc, SHELLS[shell], within(0.5))
    searched, sized = report["strategies"]
    assert searched["strategy"] == "max_throughput" and sized["strategy"] == "size_fifos"
    assert searched["within"] == {"lut": 0.5} and searched["budget"] == {"lut": 35280}
    used = report["resources"]["used"]
    partition = report["resources"]["shell"]["partition"]
    # The search costs the shell's total; sizing places no FIFO here (FIFOs would join
    # the total after the search, NOTE §4.2), so the report's total is the search's.
    assert sized["fifo_bits"] == 0
    assert searched["used"] == used and searched["fits"] and used["lut"] <= 35280
    # What it committed it chose from completed costs, and the report says so.
    assert searched["read_completed"].startswith("read completed choices: ")
    assert searched["binding"] == "lut" and searched["ratio"] == {
        "lut": round(used["lut"] / 35280, 4)
    }
    assert searched["bottleneck"] == report["bottleneck"]
    tried = searched["tried"]
    assert tried[0]["cycles"] == 1 and searched["fastest"] == tried[0]["bottleneck"]
    # Faster than 1e6 fps, at three to four times its partition.
    assert partition == {"lut": 19703, "ff": 27560, "bram18": 57, "uram": 0, "dsp": 272}
    if shell == "ip":
        # The fastest fold (1 cycle, 1.5 M LUT) does not fit; no budget (50 176
        # cycles) does; bisection between them keeps budget 49, which reaches 49.
        assert used == partition
        assert report["bottleneck"]["cycles"] == 49
        assert [row["cycles"] for row in tried[:2]] == [1, None]
        assert (tried[0]["bottleneck"], tried[0]["fits"]) == (1, False)
        assert (tried[1]["bottleneck"], tried[1]["fits"]) == (50176, True)
        assert min(row["cycles"] for row in tried[2:] if row["fits"]) == 49
        assert len(tried) == 18
    else:
        # The input end bounds the shell: the fastest fold, relaxed to 53, reaches 62
        # at the end (SZ6), and fits with the shell's 9 534 LUT beside the partition.
        assert report["bottleneck"] == {"members": ["Reshape_0_out0"], "cycles": 62}
        assert used["lut"] == 29237 and used["lut"] > partition["lut"]
        assert [(row["cycles"], row["relaxed_to"]) for row in tried] == [(1, 53)]
        assert searched["departures"] == ["budget 1 relaxed to 53, reaching 62"]


@pytest.mark.slow
@pytest.mark.parametrize(
    ("shell", "fraction", "budget", "at_1e6"),
    [
        # Below the 1e6-fps point's partition, 5 202 LUT, on the part alone.
        ("ip", 0.07, 4939, {"cycles": 197, "bottleneck": 196, "used": {"lut": 5202}}),
        # Below the 1e6-fps point's shell total in the Zynq shell, 14 946 LUT; the
        # input end holds that point at 200 cycles.
        ("pynq", 0.2, 14112, {"cycles": 209, "bottleneck": 200, "used": {"lut": 14946}}),
    ],
)
def test_tfc_within_fewer_luts_than_its_1e6_fps_point_folds_slower_and_fits(
    tfc: ModelWrapper, shell: str, fraction: float, budget: int, at_1e6: dict[str, Any]
) -> None:
    """A LUT budget below what the 1e6-fps point uses: the bottleneck rises above its
    196 cycles a frame, and the total fits."""
    report = explored(tfc, SHELLS[shell], within(fraction))
    searched = report["strategies"][0]
    assert searched["budget"] == {"lut": budget} and budget < at_1e6["used"]["lut"]
    used = report["resources"]["used"]
    assert searched["fits"] and used["lut"] <= budget and searched["binding"] == "lut"
    # Both reach 256 cycles a frame, the partition at 4 040 LUT.
    assert report["bottleneck"]["cycles"] == 256 > AT_1E6_CYCLES
    assert report["resources"]["shell"]["partition"]["lut"] == 4040
    # Bisection tried the budget that folds as the 1e6-fps point does, which does not fit.
    row = next(row for row in searched["tried"] if row["cycles"] == at_1e6["cycles"])
    assert (row["bottleneck"], row["used"], row["fits"]) == (
        at_1e6["bottleneck"],
        at_1e6["used"],
        False,
    )


@pytest.mark.slow
def test_tfc_over_the_part_warns_naming_the_binding_resource(tfc: ModelWrapper) -> None:
    """Whichever strategy chose it: TFC at a cycle a frame (target_throughput 2e8 at
    5 ns) uses far more than the part has, a warning naming the binding resource, and
    the point is explored and saved all the same (RC5)."""
    with pytest.warns(ResourceBudgetWarning, match="most of lut: lut "):
        report = explored(tfc, ULTRA96_IP, [{"strategy": "target_throughput", "fps": 2e8}])
    resources = report["resources"]
    assert report["bottleneck"]["cycles"] == 1
    assert resources["binding"] == "lut" and "lut" in resources["over"]
    assert resources["over"]["lut"]["platform"] == 70560
    assert resources["warning"].startswith("the point uses more than the platform's part has")
