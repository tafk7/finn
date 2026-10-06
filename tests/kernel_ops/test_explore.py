# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exploring a model's KernelOps through the seam: owners, the strategies as a build
lists them, persistence, replay after a fact change, the report.

On the Chain as a model (MatMul ``first``, Thresholding ``activate``, MatMul
``second``; three rows of four) for Ultra96 without a shell, and an INT8 MatMul
retargeted from Ultra96 to a VCK190.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.space import inspection
from finn.custom_op.kernels.base import KernelOpError, kernel_op, read_target, write_target
from finn.custom_op.kernels.partition import partition_root
from finn.kernels.explore import ExploreError, Pinned, Placeholder, Seam, TargetThroughput
from finn.transformation.kernels import (
    ExploreKernelChoices,
    InferKernelTensors,
    explore_kernel_choices,
    kernel_choices_config,
    resolve_target,
    strategy,
)
from kernel_ops.models import matmul_model
from kernel_ops.test_choose import choices, kernel_model

VCK190 = resolve_target("xcvc1902-vsva2197-2MP-e-S", 5.0)


def seam_of(model: ModelWrapper) -> tuple[Seam, Any]:
    root = partition_root(model, model.graph.node)
    return Seam(root.members, root.owners, read_target(model).platform), root.point


def test_a_choice_names_the_node_and_attribute_that_persist_it() -> None:
    explorer, point = seam_of(kernel_model())
    offered = {choice.key: choice for choice in explorer.choices(point)}
    assert offered["first.compute.packed.pe"].owner == ("first", "compute.packed.pe")
    # An edge's choice is its consumer's.
    assert offered["levels.transport"].owner == ("second", "x.transport")
    assert explorer.key("second", "x.transport") == "levels.transport"
    assert explorer.key("second", "compute.packed.pe") == "second.compute.packed.pe"


def test_a_spec_names_its_strategy_and_its_parameters() -> None:
    made = strategy({"strategy": "target_throughput", "fps": 1000, "relax": False})
    assert isinstance(made, TargetThroughput) and made.fps == 1000 and not made.relax
    assert isinstance(strategy({"strategy": "placeholder"}), Placeholder)
    with pytest.raises(ValueError, match="names no kernel strategy"):
        strategy({"strategy": "size_fifos"})
    with pytest.raises(ValueError, match="unexpected keyword argument 'cycles'"):
        strategy({"strategy": "placeholder", "cycles": 3})


def test_exploring_saves_the_point_s_choices_and_reports_its_cost() -> None:
    model = kernel_model()
    explored = explore_kernel_choices(model, [Placeholder(lanes=2)])
    saved = choices(model)
    assert saved["first"]["compute.packed.pe"] == 2 and saved["second"]["x.transport"] == "direct"
    report = explored.report
    assert report["bottleneck"] == {"members": ["x", "levels", "first", "second"], "cycles": 12}
    # levels: the replay of a frame of two beats of two 2-bit levels.
    assert report["members"]["levels"] == {"cycles": 12, "buffering": 2 * 2 * 2}
    (placeholder,) = report["strategies"]
    assert placeholder["strategy"] == "placeholder" and placeholder["attempts"] > 0
    assert json.loads(json.dumps(report)) == report
    # The model replays to the explored point: nothing open, nothing stale.
    root = partition_root(model, model.graph.node)
    assert inspection.viable(root.point) == () and not root.dropped


def test_a_choice_left_open_is_refused_by_name_and_nothing_is_saved() -> None:
    model = kernel_model()
    with pytest.raises(KernelOpError, match="open choices no strategy chose: .*first.compute"):
        explore_kernel_choices(model, [])
    assert choices(model) == {"first": {}, "activate": {}, "second": {}}


def test_a_pinned_file_is_committed_and_the_rest_explored(tmp_path: Path) -> None:
    pinned = tmp_path / "pinned.json"
    pinned.write_text(
        json.dumps(
            {
                "first": {"compute.packed.pe": 1},
                "second": {"x.transport": "fifo", "x.transport.fifo.buffer.depth": 8},
            }
        )
    )
    model = kernel_model().transform(ExploreKernelChoices([Pinned(pinned), Placeholder(lanes=2)]))
    saved = choices(model)
    assert saved["first"]["compute.packed.pe"] == 1 and saved["first"]["compute.packed.simd"] == 2
    assert saved["second"]["x.transport.fifo.buffer.depth"] == 8
    # A whole kernel_choices.json pins another model to the same choices, in one batch:
    # the FIFO and its depth, nested under it, together.
    exported = tmp_path / "kernel_choices.json"
    exported.write_text(json.dumps(kernel_choices_config(model)))
    again = kernel_model().transform(ExploreKernelChoices([Pinned(exported)]))
    assert choices(again) == saved
    # What the root does not take is refused, named.
    pinned.write_text(json.dumps({"first": {"compute.packed.pe": 3}}))
    with pytest.raises(ExploreError, match="first.compute.packed.pe: 3 is not viable"):
        kernel_model().transform(ExploreKernelChoices([Pinned(pinned)]))
    pinned.write_text(json.dumps({"nobody": {"pe": 1}}))
    with pytest.raises(ExploreError, match="nobody.pe"):
        kernel_model().transform(ExploreKernelChoices([Pinned(pinned)]))


def test_saved_choices_are_pinned_and_fresh_explores_again() -> None:
    model = kernel_model()
    explore_kernel_choices(model, [Placeholder(lanes=2)])
    # A strategy fills only open choices: the saved folding stays.
    resumed = explore_kernel_choices(
        model, [TargetThroughput(1_000_000_000 // (5 * 3)), Placeholder()]
    )
    assert choices(model)["first"]["compute.packed.pe"] == 2
    assert resumed.report["strategies"][0]["attempts"] == 0
    # Fresh clears the nodes' choices first: 6 cycles a frame (the activation's) folds
    # each MatMul to PE 2, SIMD 4.
    explore_kernel_choices(
        model, [TargetThroughput(1_000_000_000 // (5 * 6)), Placeholder()], fresh=True
    )
    assert choices(model)["first"]["compute.packed.pe"] == 2
    assert choices(model)["first"]["compute.packed.simd"] == 4


def int8_matmul() -> ModelWrapper:
    model = matmul_model(annotate=(), infer=False)
    for tensor in ("x", "w"):
        model.set_tensor_datatype(tensor, DataType["INT8"])
    return model.transform(InferKernelTensors())


def test_a_retarget_drops_the_stale_folding_explores_it_again_and_clears_it() -> None:
    """An INT8 MatMul explored on Ultra96 (DSP48E2: the packed core forced, its folding
    saved nested under it), retargeted to a VCK190 (DSP58: the core open, so the nested
    keys inapplicable): replay drops them with why, exploring chooses the core and its
    folding again, and saving clears the stale attributes."""
    model = int8_matmul()
    explore_kernel_choices(model, [Placeholder()])
    saved = kernel_op(model, model.graph.node[0]).choices()
    assert "compute.packed.pe" in saved and "compute" not in saved
    write_target(model, VCK190)
    model = model.transform(InferKernelTensors())
    root = partition_root(model, model.graph.node)
    assert "first.compute.packed.pe" in root.dropped
    assert "first.compute" in {item.key for item in inspection.viable(root.point)}
    explored = explore_kernel_choices(model, [Placeholder()])
    assert "first.compute.packed.pe" in explored.report["dropped"]
    held = kernel_op(model, model.graph.node[0]).choices()
    assert "compute" in held
    again = partition_root(model, model.graph.node)
    assert not again.dropped and inspection.viable(again.point) == ()
