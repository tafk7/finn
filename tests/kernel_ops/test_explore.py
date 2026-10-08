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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from kernels.helpers import Lanes
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.space import inspection
from finn.custom_op.kernels.base import kernel_op, read_target, write_target
from finn.custom_op.kernels.shell import shell_root
from finn.kernels.explore import (
    Accepted,
    Baseline,
    ExploreError,
    Pinned,
    Placeholder,
    Ranked,
    Seam,
    SizeFifos,
    TargetThroughput,
)
from finn.kernels.utilization import Resources
from finn.platform import resolve_target
from finn.transformation.kernels import (
    ExploreKernelChoices,
    InferKernelTensors,
    completion,
    explore_kernel_choices,
    kernel_choices_config,
    strategy,
)
from finn.transformation.kernels.package import configured_root
from kernel_ops.models import matmul_model
from kernel_ops.test_choose import choices, kernel_model

VCK190 = resolve_target(part="xcvc1902-vsva2197-2MP-e-S", period_ns=5.0)


def seam_of(model: ModelWrapper) -> tuple[Seam, Any]:
    root = shell_root(model, model.graph.node)
    return Seam(root.members, root.owners, read_target(model).platform), root.point


def test_a_choice_names_the_node_and_attribute_that_persist_it() -> None:
    explorer, point = seam_of(kernel_model())
    offered = {choice.key: choice for choice in explorer.choices(point)}
    assert offered["partition.first.compute.packed.pe"].owner == ("first", "compute.packed.pe")
    # An edge's choice is its consumer's, a boundary channel's at the shell root's level.
    assert offered["partition.levels.transport"].owner == ("second", "x.transport")
    assert offered["y.transport"].owner == ("second", "y.transport")
    assert explorer.key("second", "x.transport") == "partition.levels.transport"
    assert explorer.key("second", "compute.packed.pe") == "partition.second.compute.packed.pe"
    assert explorer.key("second", "y.transport") == "y.transport"
    # A key's member is the longest member path that prefixes it, at a segment.
    assert explorer.member_of("partition.levels.transport") == "partition.levels"
    assert explorer.member_of("y.transport.fifo.buffer.depth") == "y"
    assert explorer.member_of("partition.levelsx.transport") is None
    assert explorer.member_of("partition") is None


def test_the_seam_reads_each_member_at_its_path() -> None:
    """Cost and refusals read members below the Partition by path, and an attempt checks
    the members its keys belong to."""
    explorer, point = seam_of(kernel_model())
    cost = explorer.cost(point)
    assert set(cost.waiting) | set(cost.cycles) == set(explorer.members)
    assert "partition.first" in cost.waiting
    hidden = explorer.cost(point, ("partition.hidden",))
    assert {*hidden.cycles, *hidden.waiting} == {"partition.hidden"}
    assert explorer.refusals(point) == {}
    outcome = explorer.attempt(point, {"partition.first.compute.packed.pe": 1})
    assert isinstance(outcome, Accepted)
    folded = explorer.cost(outcome.point, ("partition.first",))
    assert "partition.first" in folded.cycles or "partition.first" in folded.waiting


def test_a_spec_names_its_strategy_and_its_parameters() -> None:
    made = strategy({"strategy": "target_throughput", "fps": 1000, "relax": False})
    assert isinstance(made, TargetThroughput) and made.fps == 1000 and not made.relax
    assert isinstance(strategy({"strategy": "size_fifos", "margin": 2}), SizeFifos)
    with pytest.raises(ValueError, match="names no kernel strategy"):
        strategy({"strategy": "max_throughput"})
    with pytest.raises(ValueError, match="unexpected keyword argument 'cycles'"):
        strategy({"strategy": "size_fifos", "cycles": 3})
    # The placeholder is a completion policy, no strategy.
    with pytest.raises(ValueError, match="names no kernel strategy"):
        strategy({"strategy": "placeholder"})


def test_a_build_names_its_completion_policy() -> None:
    assert isinstance(completion("baseline"), Baseline)
    assert isinstance(completion("placeholder"), Placeholder)
    with pytest.raises(ValueError, match="names no kernel completion policy"):
        completion("minimum")


def test_exploring_saves_the_point_s_choices_and_reports_its_cost() -> None:
    model = kernel_model()
    explored = explore_kernel_choices(model, [Ranked(Lanes(2))])
    saved = choices(model)
    assert saved["first"]["compute.packed.pe"] == 2 and saved["second"]["x.transport"] == "direct"
    report = explored.report
    assert report["bottleneck"] == {
        "members": ["x", "partition.levels", "partition.first", "partition.second"],
        "cycles": 12,
    }
    # levels: the replay of a frame of two beats of two 2-bit levels, in input_gen's
    # buffer of BUF_SIZE 8 words for the nest {2, 2} {0, 1}.
    levels = report["members"]["partition.levels"]
    assert (levels["cycles"], levels["buffering"]) == (12, 8 * 2 * 2)
    # Every member states its resources; on the ip shell, which has no ends and no
    # static region, the total is their sum, the partition's, against the part's
    # totals.
    resources = report["resources"]
    assert resources["unstated"] == {}
    assert resources["used"] == {
        key: sum(member["resources"][key] for member in report["members"].values())
        for key in ("lut", "ff", "bram18", "uram", "dsp")
    }
    # Two packed dot products at PE 2, SIMD 2, both PE lanes packed in one pipe: a
    # slice a SIMD element each.
    assert resources["used"]["dsp"] == 4
    platform = read_target(model).platform.resources
    assert platform is not None and resources["platform"] == asdict(platform)
    assert resources["share"]["dsp"] == round(4 / platform.dsp, 4)
    assert resources["shell"] == {"partition": resources["used"], "ends": {}, "static_region": {}}
    assert "each end and its static region" in resources["counted"]
    (ranked,) = report["strategies"]
    assert ranked["strategy"] == "ranked" and ranked["attempts"] > 0
    # Nothing was left to complete.
    assert report["completed"] == {} and report["completion"] == {"policy": "baseline", "open": []}
    assert json.loads(json.dumps(report)) == report
    # The model replays to the explored point: nothing open, nothing stale.
    root = shell_root(model, model.graph.node)
    assert inspection.viable(root.point) == () and not root.dropped


def test_resources_are_a_total_only_once_every_member_states_them() -> None:
    """A member whose resources wait on an open choice (a transport, a memory style, a
    folding) says so, and the seam states no total from the others: a partial sum is
    never a total. Completed, every member states its own."""
    seam, point = seam_of(kernel_model())
    cost = seam.cost(point)
    assert cost.used is None and cost.resources == {}
    assert cost.unstated["x"] == "waits on x.transport"
    assert cost.unstated["partition.activate"] == "waits on partition.activate.ram_style"
    completed = seam.cost(seam.complete(point).point)
    assert completed.unstated == {}
    assert completed.used == sum(completed.resources.values(), Resources())


def test_what_no_strategy_chose_is_completed_on_a_copy_and_never_saved() -> None:
    """With no strategy, nothing is saved: the report lists every value the baseline
    completion takes, by owner, and the cost of the completed point; hardware
    generation completes the same values."""
    model = kernel_model()
    explored = explore_kernel_choices(model, [])
    assert choices(model) == {"first": {}, "activate": {}, "second": {}}
    report = explored.report
    assert report["strategies"] == [] and report["choices"] == {}
    completed = report["completed"]
    # The baseline: the least folding, the adder tree, auto memories.
    assert completed["first"]["compute.packed.pe"] == {"value": 1, "by": "baseline"}
    assert completed["first"]["compute.packed.simd"] == {"value": 1, "by": "baseline"}
    assert completed["first"]["compute.packed.reducer"] == {"value": "tree", "by": "baseline"}
    assert completed["first"]["w.source.memstream.ram_style"]["value"] == "auto"
    assert report["completion"]["open"] == [] and report["bottleneck"] is not None
    # The transports are sized on the completed copy (at hardware generation, sizing
    # is the default completion's): a choice of size_fifos, never saved either.
    assert completed["second"]["x.transport"]["by"] == "size_fifos"
    assert report["fifos"].startswith("sized at completion by baseline: ")
    assert explored.completed is not None
    values = {
        f"{node}.{attribute}": entry["value"]
        for node, held in completed.items()
        for attribute, entry in held.items()
    }
    point, _ = configured_root(model, "the chain")
    seam = seam_of(model)[0]
    assert {
        f"{node}.{attribute}": value
        for key, value in seam.chosen(point).items()
        for node, attribute in [seam.owner(key) or ("", key)]
    } == values
    # The point the exploration returns is the committed one: still open.
    assert explored.point is not explored.completed.point
    assert inspection.viable(explored.point) != ()


def test_a_strategy_that_reads_completed_values_is_flagged_and_its_choices_saved() -> None:
    """Q-B: a strategy that reads the completed copy (``Seam.complete``) commits as any
    other (DSE12); the report flags it, naming what it read."""

    class Reader:
        strategy = "reader"

        def explore(self, seam: Seam, point: Any) -> Any:
            seam.complete(point)
            return point

        def report(self) -> dict[str, object]:
            return {"strategy": self.strategy}

    model = kernel_model()
    report = explore_kernel_choices(model, [Reader(), SizeFifos()]).report
    reader, sizing = report["strategies"]
    assert reader["read_completed"].startswith("read completed choices: ")
    assert "first.compute.packed.pe" in reader["read_completed"]
    # Sizing before folding sizes at the completed folding, and its transports are saved.
    assert sizing["read_completed"].startswith("sized at completed folding (read completed ")
    assert choices(model)["second"]["x.transport"] in ("direct", "fifo")
    assert {"saved", "size_fifos"} >= {
        name for held in report["choices"].values() for name in held.values()
    }
    assert "first" not in report["choices"] or "compute.packed.pe" not in report["choices"]["first"]


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
    model = kernel_model().transform(ExploreKernelChoices([Pinned(pinned), Ranked(Lanes(2))]))
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
    explore_kernel_choices(model, [Ranked(Lanes(2))])
    # A strategy fills only open choices: the saved folding stays.
    resumed = explore_kernel_choices(model, [TargetThroughput(1_000_000_000 // (5 * 3))])
    assert choices(model)["first"]["compute.packed.pe"] == 2
    assert resumed.report["strategies"][0]["attempts"] == 0
    # Fresh clears the nodes' choices first: 6 cycles a frame (the activation's) folds
    # each MatMul to PE 2, SIMD 4.
    explore_kernel_choices(model, [TargetThroughput(1_000_000_000 // (5 * 6))], fresh=True)
    assert choices(model)["first"]["compute.packed.pe"] == 2
    assert choices(model)["first"]["compute.packed.simd"] == 4


def test_every_committed_choice_names_the_strategy_that_made_it() -> None:
    """The report attributes each choice the model holds after the exploration: to the
    strategy that committed it, or ``saved`` when the model held it before; each
    strategy counts its own, and says whether FIFOs were sized. What no strategy chose
    is completed, never attributed to one."""
    model = kernel_model()
    first = explore_kernel_choices(
        model, [TargetThroughput(1_000_000_000 // (5 * 6)), SizeFifos()]
    ).report
    made_by = {
        (node, attribute): strategy_name
        for node, held in first["choices"].items()
        for attribute, strategy_name in held.items()
    }
    # Every choice the model saved, and only those, is attributed.
    assert set(made_by) == {
        (node, attribute) for node, held in choices(model).items() for attribute in held
    }
    assert made_by[("first", "compute.packed.pe")] == "target_throughput"
    assert made_by[("second", "x.transport")] == "size_fifos"
    assert ("first", "compute.packed.reducer") not in made_by
    assert first["completed"]["first"]["compute.packed.reducer"]["by"] == "baseline"
    target, sizing = first["strategies"]
    assert target["committed"] == sum(v == "target_throughput" for v in made_by.values()) > 0
    assert sizing["committed"] == sum(v == "size_fifos" for v in made_by.values()) > 0
    assert "read_completed" not in sizing
    assert first["fifos"].startswith("sized by size_fifos: ")
    # Explored again, everything is the model's own: saved, nothing committed.
    again = explore_kernel_choices(model, [SizeFifos()]).report
    assert {v for held in again["choices"].values() for v in held.values()} == {"saved"}
    assert [each["committed"] for each in again["strategies"]] == [0]
    assert again["fifos"] == "not sized: size_fifos found no open transport (each was saved before)"


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
    explore_kernel_choices(model, [Ranked(Lanes())])
    saved = kernel_op(model, model.graph.node[0]).choices()
    assert "compute.packed.pe" in saved and "compute" not in saved
    write_target(model, VCK190)
    model = model.transform(InferKernelTensors())
    root = shell_root(model, model.graph.node)
    assert "partition.first.compute.packed.pe" in root.dropped
    assert "partition.first.compute" in {item.key for item in inspection.viable(root.point)}
    explored = explore_kernel_choices(model, [Ranked(Lanes())])
    assert "partition.first.compute.packed.pe" in explored.report["dropped"]
    held = kernel_op(model, model.graph.node[0]).choices()
    assert "compute" in held
    again = shell_root(model, model.graph.node)
    assert not again.dropped and inspection.viable(again.point) == ()
