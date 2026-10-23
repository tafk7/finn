# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Thresholding's threshold memories: Decisions over what thresholding.sv can express.

Stage s of M = clog2(N + 1) keeps a memory of depth ``base * 2**s``; the RTL
puts a stage in UltraRAM from its URAM trigger, else in block RAM from its BRAM
trigger, else distributed when the BRAM trigger is set, else leaves it to
Vivado. ``styles`` restates that rule, so each case below is read back as the
RTL would assign it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2

from finn.core.space import ConfigurationResult, Inapplicable, design_space, inspection
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.library import library_module
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.platform.capabilities import Platform

INT8 = DataType["INT8"]
# One row of the tables' four channels.
INPUT = Tensor((1, 4), ScalarEncoding(INT8))


def table(count: int, channels: int = 4) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return (tuple(tuple(range(count)) for _ in range(channels)),)


def configured(count: int = 3, pe: int = 1, **memory: object) -> ThresholdingAxiKernel:
    base = design_space(
        ThresholdingAxiKernel(
            input_tensor=INPUT,
            threshold_dtype=INT8,
            thresholds=IntegerTensorValue.of(table(count)),
            bias=0,
            platform=FULL_DSP48E2,
        )
    )
    report = base.try_with_choices(use_axilite=False, deep_pipeline=False, pe=pe, **memory)
    assert report.accepted, report
    return report.instance


def triggers(point: ThresholdingAxiKernel) -> tuple[int, int]:
    parameters = dict(library_module(point).parameters)
    return parameters["DEPTH_TRIGGER_BRAM"], parameters["DEPTH_TRIGGER_URAM"]


def styles(point: ThresholdingAxiKernel) -> tuple[str, ...]:
    """Each stage's resource, as thresholding.sv assigns it from the triggers."""
    bram, uram = triggers(point)
    found = []
    for stage in range(point.stages):
        depth = point.stage_depth(stage)
        if uram and depth >= uram:
            found.append("ultra")
        elif bram and depth >= bram:
            found.append("block")
        else:
            found.append("distributed" if bram else "auto")
    return tuple(found)


@pytest.mark.parametrize(("count", "stages"), ((1, 1), (2, 2), (3, 2), (4, 3), (7, 3), (8, 4)))
def test_the_stage_counts_range_over_the_rtl_stages(count: int, stages: int) -> None:
    point = configured(count, ram_style="auto", ultra_stages=0)
    assert point.stages == stages
    base = design_space(
        ThresholdingAxiKernel(
            input_tensor=INPUT,
            threshold_dtype=INT8,
            thresholds=IntegerTensorValue.of(table(count)),
            bias=0,
            platform=FULL_DSP48E2,
        )
    )
    for ultra in range(stages + 1):
        assert base.try_with_choices(ultra_stages=ultra).accepted
    assert not base.try_with_choices(ultra_stages=stages + 1).accepted
    assert not base.try_with_choices(ultra_stages=-1).accepted


@pytest.mark.parametrize(
    ("memory", "expected", "assigned"),
    (
        ({"ram_style": "auto", "ultra_stages": 0}, (0, 0), ("auto", "auto")),
        (
            {"ram_style": "distributed", "block_stages": 0, "ultra_stages": 0},
            (16, 0),
            ("distributed", "distributed"),
        ),
        (
            {"ram_style": "distributed", "block_stages": 1, "ultra_stages": 0},
            (8, 0),
            ("distributed", "block"),
        ),
        ({"ram_style": "auto", "ultra_stages": 1}, (0, 8), ("auto", "ultra")),
        (
            {"ram_style": "distributed", "block_stages": 1, "ultra_stages": 1},
            (4, 8),
            ("block", "ultra"),
        ),
        (
            {"ram_style": "distributed", "block_stages": 0, "ultra_stages": 1},
            (8, 8),
            ("distributed", "ultra"),
        ),
    ),
)
def test_each_case_maps_to_the_triggers_the_rtl_reads(
    memory: dict[str, object], expected: tuple[int, int], assigned: tuple[str, ...]
) -> None:
    # N = 3: two stages; four channels, PE 1: memories of depth 4 and 8.
    point = configured(**memory)
    assert [point.stage_depth(stage) for stage in range(point.stages)] == [4, 8]
    assert triggers(point) == expected
    assert styles(point) == assigned


def broken(report: ConfigurationResult[ThresholdingAxiKernel]) -> list[tuple[str, set[str]]]:
    """Each rule a refused change breaks, with its codes."""
    return [
        (rule.rule, {finding.code for finding in rule.result.findings}) for rule in report.broken
    ]


def test_more_stages_than_the_rtl_has_are_refused() -> None:
    """Three stages placed where the table has two: the change is refused by the rule
    on the triggers the RTL reads."""
    base = configured(ram_style="distributed", block_stages=0, ultra_stages=0)
    report = base.try_with_choices(block_stages=2, ultra_stages=1)
    assert not report.accepted
    assert broken(report) == [("depth_triggers", {"threshold-memory"})]


def test_block_stages_apply_only_to_distributed_memories() -> None:
    """Under ``auto`` the block stages are out of play: set, the value is kept dormant,
    and the triggers do not read it."""
    base = configured(ram_style="auto", ultra_stages=0)
    report = base.try_with_choices(block_stages=1)
    assert report.accepted
    state = dict(inspection.states(report.instance))["block_stages"]
    assert state == inspection.Dormant(1)
    assert triggers(report.instance) == triggers(base) == (0, 0)


def test_every_stage_in_ultraram_has_one_spelling() -> None:
    """With every stage in UltraRAM none is left: ``ram_style`` (and with it
    ``block_stages``) does not apply, so the assignment is ``ultra_stages`` alone; a
    style set beside it is kept dormant and changes nothing."""
    point = configured(ultra_stages=2)
    assert triggers(point) == (0, 4) and styles(point) == ("ultra", "ultra")
    assert isinstance(point.query(ThresholdingAxiKernel.ram_style), Inapplicable)
    for style in ("auto", "distributed"):
        report = point.try_with_choices(ram_style=style)
        assert report.accepted
        assert dict(inspection.states(report.instance))["ram_style"] == inspection.Dormant(style)
        assert triggers(report.instance) == (0, 4)
    # One stage left: it takes a ram_style.
    left = configured(ram_style="distributed", block_stages=0, ultra_stages=1)
    assert styles(left) == ("distributed", "ultra")


def test_the_choices_do_not_move_with_pe() -> None:
    """Counted in stages, a choice stays valid when PE changes; the depths it maps to move."""
    memory = {"ram_style": "distributed", "block_stages": 1, "ultra_stages": 0}
    for pe, bram in ((1, 8), (2, 4), (4, 2)):
        point = configured(pe=pe, **memory)
        assert triggers(point) == (bram, 0)
        assert styles(point) == ("distributed", "block")


def test_the_platform_narrows_the_memories_and_a_control_bus_the_axilite() -> None:
    """An UltraRAM stage needs UltraRAM that takes initial contents, runtime-writable
    thresholds a control bus (which a bare kernel has not): on a platform without them,
    each value is refused by the rule that names it, nothing standing, and only what
    remains is possible."""

    def point(platform: Platform) -> ThresholdingAxiKernel:
        return design_space(
            ThresholdingAxiKernel(
                input_tensor=INPUT,
                threshold_dtype=INT8,
                thresholds=IntegerTensorValue.of(table(3)),
                bias=0,
                platform=platform,
            )
        )

    def refused(viable: inspection.Viable) -> dict[object, list[tuple[str, tuple[str, ...]]]]:
        return {
            item.value: [(rule.rule, rule.standing) for rule in item.broken]
            for item in viable.values or ()
            if isinstance(item, inspection.Refused)
        }

    full = point(FULL_DSP48E2)
    ultra, axilite = inspection.viable(
        full, [ThresholdingAxiKernel.ultra_stages, ThresholdingAxiKernel.use_axilite]
    )
    assert ultra.cases == (0, 1, 2)
    assert axilite.cases == (False,) and refused(axilite) == {True: [("control_for_writes", ())]}
    bare = point(replace(FULL_DSP48E2, uram=False))
    (ultra,) = inspection.viable(bare, [ThresholdingAxiKernel.ultra_stages])
    assert ultra.cases == (0,)
    assert refused(ultra) == {1: [("uram_present", ())], 2: [("uram_present", ())]}
    zynq = point(replace(FULL_DSP48E2, uram_init=False))
    # A commit that breaks a rule is refused, naming it.
    report = zynq.try_with_choices(ultra_stages=1)
    assert not report.accepted
    assert broken(report) == [("uram_init_taken", {"uram-init"})]
