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
import warnings
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from kernels.helpers import Lanes
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.space import inspection
from finn.custom_op.kernels.base import KernelOpError, kernel_op, read_target, write_target
from finn.custom_op.kernels.shell import configured_root, save_channels, shell_root
from finn.custom_op.partition.kernel_partitions import partition_body
from finn.kernels.explore import (
    Accepted,
    Baseline,
    ExploreError,
    MaxThroughput,
    Pinned,
    Placeholder,
    Ranked,
    ResourceBudgetWarning,
    Seam,
    SizeFifos,
    TargetThroughput,
)
from finn.kernels.utilization import Resources
from finn.platform import resolve_target
from finn.transformation.kernels import (
    ExploreKernelChoices,
    InferKernelTensors,
    ToKernelOps,
    completion,
    explore_kernel_choices,
    kernel_choices_config,
    strategy,
)
from finn.transformation.kernels.cut import CutKernelPartition
from kernel_ops.models import TARGET, matmul_model
from kernel_ops.test_choose import choices, kernel_model

VCK190 = resolve_target(part="xcvc1902-vsva2197-2MP-e-S", period_ns=5.0)


def seam_of(model: ModelWrapper) -> tuple[Seam, Any]:
    root = shell_root(model, model.graph.node)
    return Seam(root.members, root.owners, read_target(model).platform), root.point


def test_a_choice_names_the_node_or_tensor_that_persists_it() -> None:
    explorer, point = seam_of(kernel_model())
    offered = {choice.key: choice for choice in explorer.choices(point)}
    assert offered["first.compute.packed.pe"].owner == ("first", "compute.packed.pe")
    # A channel's choice is its tensor's: an edge's, a graph output's alike.
    assert offered["levels.transport"].owner == ("levels", "transport")
    assert offered["y.transport"].owner == ("y", "transport")
    assert explorer.key("levels", "transport") == "levels.transport"
    assert explorer.key("second", "compute.packed.pe") == "second.compute.packed.pe"
    assert explorer.key("y", "transport") == "y.transport"
    assert explorer.key("nobody", "transport") is None
    # A key's member is the longest member path that prefixes it, at a segment.
    assert explorer.member_of("levels.transport") == "levels"
    assert explorer.member_of("y.transport.fifo.buffer.depth") == "y"
    assert explorer.member_of("levelsx.transport") is None
    assert explorer.member_of("partition") is None


def test_the_seam_reads_each_member_at_its_path() -> None:
    """Cost and refusals read the root's members by path, and an attempt checks the
    members its keys belong to."""
    explorer, point = seam_of(kernel_model())
    cost = explorer.cost(point)
    assert set(cost.waiting) | set(cost.cycles) == set(explorer.members)
    assert "first" in cost.waiting
    hidden = explorer.cost(point, ("hidden",))
    assert {*hidden.cycles, *hidden.waiting} == {"hidden"}
    assert explorer.refusals(point) == {}
    outcome = explorer.attempt(point, {"first.compute.packed.pe": 1})
    assert isinstance(outcome, Accepted)
    folded = explorer.cost(outcome.point, ("first",))
    assert "first" in folded.cycles or "first" in folded.waiting


def test_a_spec_names_its_strategy_and_its_parameters() -> None:
    made = strategy({"strategy": "target_throughput", "fps": 1000, "relax": False})
    assert isinstance(made, TargetThroughput) and made.fps == 1000 and not made.relax
    assert isinstance(strategy({"strategy": "size_fifos", "margin": 2}), SizeFifos)
    made = strategy({"strategy": "max_throughput", "within": {"lut": 0.5, "dsp": 0.8}})
    assert isinstance(made, MaxThroughput) and made.within == {"lut": 0.5, "dsp": 0.8}
    # Its budget is required: no default fraction of the part.
    with pytest.raises(ValueError, match="missing 1 required positional argument: 'within'"):
        strategy({"strategy": "max_throughput"})
    with pytest.raises(ValueError, match=r"no resource is named \['luts'\]"):
        strategy({"strategy": "max_throughput", "within": {"luts": 0.5}})
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
    assert saved["first"]["compute.packed.pe"] == 2 and saved["levels"]["transport"] == "direct"
    report = explored.report
    assert report["bottleneck"] == {
        "members": ["x", "levels", "first", "second"],
        "cycles": 12,
    }
    # levels: the replay of a frame of two beats of two 2-bit levels, in input_gen's
    # buffer of BUF_SIZE 8 words for the nest {2, 2} {0, 1}.
    levels = report["members"]["levels"]
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


def test_a_point_over_the_part_s_resources_warns_naming_the_binding_resource() -> None:
    """Whichever strategy chose it, a point whose shell uses more than the part has is a
    warning naming the binding resource (its highest share), never a refusal (RC5):
    the choices are saved and the report states each resource over."""
    # Block RAM for the thresholds' stages, which Lanes(2) puts two of in block RAM.
    small = replace(
        TARGET.platform, resources=Resources(lut=100_000, ff=100_000, bram18=1_000, dsp=3)
    )
    model = kernel_model(replace(TARGET, platform=small))
    with pytest.warns(ResourceBudgetWarning, match="most of dsp: dsp 4 of 3"):
        explored = explore_kernel_choices(model, [Ranked(Lanes(2))])
    resources = explored.report["resources"]
    assert resources["binding"] == "dsp"
    assert resources["over"] == {"dsp": {"used": 4, "platform": 3}}
    assert resources["warning"].startswith("the point uses more than the platform's part has")
    assert choices(model)["first"]["compute.packed.pe"] == 2
    # Within the part: the binding resource is named, and nothing is over.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceBudgetWarning)
        within = explore_kernel_choices(kernel_model(), [Ranked(Lanes(2))]).report["resources"]
    assert (within["binding"], within["over"], within["warning"]) == ("dsp", {}, None)


def test_resources_are_a_total_only_once_every_member_states_them() -> None:
    """A member whose resources wait on an open choice (a transport, a memory style, a
    folding) says so, and the seam states no total from the others: a partial sum is
    never a total. Completed, every member states its own."""
    seam, point = seam_of(kernel_model())
    cost = seam.cost(point)
    assert cost.used is None and cost.resources == {}
    assert cost.unstated["x"] == "waits on x.transport"
    assert cost.unstated["activate"] == "waits on activate.ram_style"
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
    # The baseline: the least folding, the adder tree, each memory in the explicit style
    # its size orders first (the Chain's, all small: LUTRAM; never auto, last).
    assert completed["first"]["compute.packed.pe"] == {"value": 1, "by": "baseline"}
    assert completed["first"]["compute.packed.simd"] == {"value": 1, "by": "baseline"}
    assert completed["first"]["compute.packed.reducer"] == {"value": "tree", "by": "baseline"}
    assert completed["w1"]["source.memstream.ram_style"]["value"] == "distributed"
    assert completed["activate"]["ram_style"]["value"] == "distributed"
    assert completed["activate"]["block_stages"]["value"] == 0
    assert report["completion"]["open"] == [] and report["bottleneck"] is not None
    # The transports are sized on the completed copy (at hardware generation, sizing
    # is the default completion's): a choice of size_fifos, never saved either.
    assert completed["levels"]["transport"]["by"] == "size_fifos"
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
    assert choices(model)["levels"]["transport"] in ("direct", "fifo")
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
                "levels": {"transport": "fifo", "transport.fifo.buffer.depth": 8},
            }
        )
    )
    model = kernel_model().transform(ExploreKernelChoices([Pinned(pinned), Ranked(Lanes(2))]))
    saved = choices(model)
    assert saved["first"]["compute.packed.pe"] == 1 and saved["first"]["compute.packed.simd"] == 2
    assert saved["levels"]["transport.fifo.buffer.depth"] == 8
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


#: A memory pinned ``auto``: the first MatMul's weights (on their tensor) and the
#: thresholds' stages.
AUTO = {"w1": {"source.memstream.ram_style": "auto"}, "activate": {"ram_style": "auto"}}


def stated_auto(report: Mapping[str, Any]) -> None:
    """The report names each memory pinned ``auto`` unstated, Vivado's, and its total a
    lower bound; the completion takes every other memory explicit."""
    resources = report["resources"]
    assert resources["lower_bound"] and resources["counted"].startswith("a lower bound")
    reasons = resources["unstated"]
    assert len(reasons) == 2 and all(
        why.endswith("ram_style auto, placed by Vivado, not stated") for why in reasons.values()
    )
    assert "activate" in reasons
    assert report["completed"]["w2"]["source.memstream.ram_style"]["value"] == "distributed"


def test_a_memory_saved_or_pinned_auto_is_kept_and_stated_as_vivado_s(tmp_path: Path) -> None:
    """``auto`` stays a value of every memory, last (SZ18): a model saved with it replays
    it, nothing dropped, and a kernel_choices.json pinning it commits it (the thresholds'
    ``ram_style`` attribute too); either way the report names the memory unstated."""
    model = kernel_model()
    for node in model.graph.node:
        if node.name in AUTO:
            kernel_op(model, node).save(AUTO[node.name])
    save_channels(model, {"w1": AUTO["w1"]})
    model.save(tmp_path / "saved.onnx")
    saved = ModelWrapper(str(tmp_path / "saved.onnx"))
    report = explore_kernel_choices(saved, []).report
    assert report["dropped"] == {} and {node: choices(saved)[node] for node in AUTO} == AUTO
    stated_auto(report)
    pinned = tmp_path / "kernel_choices.json"
    pinned.write_text(json.dumps(AUTO))
    fresh = kernel_model()
    report = explore_kernel_choices(fresh, [Pinned(pinned)]).report
    assert {node: choices(fresh)[node] for node in AUTO} == AUTO
    assert report["strategies"][0]["committed"] == 2
    stated_auto(report)


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
    assert made_by[("levels", "transport")] == "size_fifos"
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
    assert "first.compute.packed.pe" in root.dropped
    assert "first.compute" in {item.key for item in inspection.viable(root.point)}
    explored = explore_kernel_choices(model, [Ranked(Lanes())])
    assert "first.compute.packed.pe" in explored.report["dropped"]
    held = kernel_op(model, model.graph.node[0]).choices()
    assert "compute" in held
    again = shell_root(model, model.graph.node)
    assert not again.dropped and inspection.viable(again.point) == ()


@pytest.mark.slow
def test_target_throughput_folds_tfc_through_each_matmul_s_core_on_a_dsp58_part(
    tfc_streamlined: Path, tmp_path: Path
) -> None:
    """On xcvc1902 (resolved by its family: DSP58) each MatMul's core is an open,
    unordered choice its cycles wait on: the target throughput folds each case and
    keeps one, so TFC at 1e6 frames a second meets its 200 cycles, and the report names
    the core taken for each MatMul and why."""
    model = ModelWrapper(str(tfc_streamlined)).transform(ToKernelOps(VCK190))
    parent = model.transform(InferKernelTensors()).transform(CutKernelPartition(tmp_path))
    _, body, _ = partition_body(parent)
    report = explore_kernel_choices(
        body, [strategy({"strategy": "target_throughput", "fps": 1e6})]
    ).report
    (target,) = report["strategies"]
    assert target["cycles"] == 200 and target["met"] is True and target["unfolded"] == {}
    assert target["bottleneck"]["cycles"] <= 200 and target["bottleneck_of"] == "folded"
    assert report["bottleneck"]["cycles"] <= 200
    cases = target["cases"]
    assert set(cases) == {f"MatMul_{index}.compute" for index in range(4)}
    for key, row in cases.items():
        node = key.split(".")[0]
        assert row["taken"] in ("packed", "int8_dsp58") and row["why"]
        assert set(row["folded"]) == {"packed", "int8_dsp58"}
        assert report["choices"][node]["compute"] == "target_throughput"
        assert report["choices"][node][f"compute.{row['taken']}.simd"] == "target_throughput"


def test_exploration_reads_a_partitions_body_and_refuses_any_other_node() -> None:
    """Which KernelOps go together is the cut's: a model holding a node that is not a
    KernelOp (the uncut model's host nodes) is refused, naming it."""
    model = kernel_model()
    model.graph.node[2].domain = ""  # a plain ONNX MatMul: not a KernelOp
    with pytest.raises(KernelOpError, match="second: not KernelOps; exploration reads"):
        explore_kernel_choices(model, [])
