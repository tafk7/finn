# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exploring a model's KernelOps through the seam, one strategy at a time: owners,
persistence of every value, pins that hold for every later exploration of the body,
replay after a fact change, the report as a log of the exploration.

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

from finn.core.graph.annotations.choices import FIFOS_STALE, HELD
from finn.core.graph.annotations.platform import read_target, write_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import partition_body
from finn.core.space import FindingKind, finding_record
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.shell import Partition, configured_root, network, save_channels
from finn.kernels.explore import (
    Accepted,
    DebugPlaceholder,
    ExploreError,
    Pinned,
    Ranked,
    Rebaseline,
    Seam,
    SizeFifos,
    TargetThroughput,
)
from finn.platform.capabilities import Resources
from finn.platform.resolve import resolve_target
from finn.transformation.kernels import (
    ExploreKernelChoices,
    InferKernelTensors,
    ToKernelOps,
    explore_kernel_choices,
    kernel_choices_config,
)
from finn.transformation.kernels.cut import CutKernelPartition
from kernel_ops.models import TARGET, matmul_model
from kernel_ops.test_builder_phase import TFC_BUILT
from kernel_ops.test_choose import choices, kernel_model

VCK190 = resolve_target(part="xcvc1902-vsva2197-2MP-e-S", period_ns=5.0)


def seam_of(model: ModelWrapper) -> tuple[Seam, Partition]:
    return Seam(read_target(model).platform), network(model, model.graph.node)


def test_a_choice_is_keyed_by_the_node_or_tensor_that_persists_it() -> None:
    explorer, network = seam_of(kernel_model())
    offered = {choice.key: choice for choice in explorer.choices(network)}
    # A kernel's choice is its node's (stored there at conversion); a channel's is its
    # tensor's: an edge's, a graph output's alike (open on a body no cut created).
    assert ("first", "compute.packed.pe") not in offered
    assert ("first", "compute.packed.pe") in explorer.chosen(network)
    assert ("levels", "transport") in offered and ("y", "transport") in offered
    # Kernels rank first, then the channels.
    assert network.members[:3] == ("first", "activate", "second")
    assert set(network.channels) == {"x", "w1", "hidden", "levels", "w2", "y"}


def test_the_seam_reads_each_member_by_name() -> None:
    """Cost and refusals read the network's members by name, and a change checks the
    members its keys belong to. A kernel's cycles are known from conversion on (its own
    choices at their baselines); a channel's resources wait on its open members' choices
    (its transport, its adapter's memory) on a body no cut created."""
    explorer, network = seam_of(kernel_model())
    cost = explorer.cost(network)
    assert set(cost.waiting) | set(cost.cycles) == set(network.members)
    assert "first" in cost.cycles
    # Each finding is owned in its member's space; the member names it.
    assert {(f.code, f.owner) for f in cost.unstated["levels"]} == {
        ("decision-unassigned", "transport"),
        ("decision-unassigned", "adapter.input_gen.input_gen.ram_style"),
    }
    hidden = explorer.cost(network, ("hidden",))
    assert {*hidden.cycles, *hidden.waiting} == {"hidden"}
    assert explorer.refusals(network) == {}
    outcome = explorer.set(network, {("first", "compute.packed.pe"): 1})
    assert isinstance(outcome, Accepted)
    folded = explorer.cost(outcome.network, ("first",))
    assert "first" in folded.cycles or "first" in folded.waiting


def test_exploring_saves_the_point_s_choices_and_reports_its_cost() -> None:
    model = kernel_model()
    explored = explore_kernel_choices(model, Ranked(Lanes(2)))
    saved = choices(model)
    assert saved["first"]["compute.packed.pe"] == 2 and saved["levels"]["transport"] == "direct"
    report = explored.report
    assert report["bottleneck"] == {
        "members": ["first", "second", "x", "levels"],
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
    ranked = report["strategy"]
    assert ranked["strategy"] == "ranked" and ranked["attempts"] > 0
    # Nothing is left open.
    assert report["open"] == [] and "completion" not in report and "completed" not in report
    assert json.loads(json.dumps(report)) == report
    # The model replays to the explored network: nothing open, nothing stale.
    bound = network(model, model.graph.node)
    assert bound.viable() == () and not bound.stale


def test_a_point_over_the_part_s_resources_is_a_finding_naming_the_binding_resource() -> None:
    """Whichever strategy chose it, a point whose shell uses more than the part has is a
    limitation naming the binding resource (its highest share), in the result and the
    report, never a refusal (RC5) and never a Python warning: the choices are saved and
    the report states each resource over."""
    # Block RAM for the thresholds' stages, which Lanes(2) puts two of in block RAM.
    small = replace(
        TARGET.platform, resources=Resources(lut=100_000, ff=100_000, bram18=1_000, dsp=3)
    )
    model = kernel_model(replace(TARGET, platform=small))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        explored = explore_kernel_choices(model, Ranked(Lanes(2)))
    (found,) = explored.findings
    assert (found.kind, found.code) == (FindingKind.LIMITATION, "resources-over-part")
    assert "most of dsp: dsp 4 of 3" in found.message
    assert dict(found.details)["binding"] == "dsp"
    resources = explored.report["resources"]
    assert resources["binding"] == "dsp"
    assert resources["over"] == {"dsp": {"used": 4, "platform": 3}}
    assert resources["findings"] == [finding_record(found)]
    assert found.message.startswith("the point uses more than the platform's part has")
    assert choices(model)["first"]["compute.packed.pe"] == 2
    # Within the part: the binding resource is named, and nothing is over.
    within = explore_kernel_choices(kernel_model(), Ranked(Lanes(2)))
    resources = within.report["resources"]
    assert (resources["binding"], resources["over"], resources["findings"]) == ("dsp", {}, [])
    assert within.findings == ()


def test_resources_are_a_total_only_once_every_member_states_them() -> None:
    """A member whose resources wait on an open choice (a transport, a memory style, a
    folding) says so, and the seam states no total from the others: a partial sum is
    never a total. Baselined, every member states its own. On a body no cut created, the
    kernels state theirs (their choices stored at conversion) and the channels wait."""
    seam, network = seam_of(kernel_model())
    cost = seam.cost(network)
    assert cost.used is None and set(cost.resources) == {"first", "activate", "second"}
    assert {(f.code, f.owner) for f in cost.unstated["x"]} == {
        ("decision-unassigned", "transport"),
        ("decision-unassigned", "adapter.input_gen.input_gen.ram_style"),
    }
    assert "activate" not in cost.unstated
    based = seam.cost(seam.baseline(network))
    assert based.unstated == {}
    assert based.used == sum(based.resources.values(), Resources())


def test_what_the_body_leaves_open_takes_its_baseline_and_every_value_is_saved() -> None:
    """With no strategy, every choice the body leaves open takes its baseline and is
    saved: the model holds every value (dense), the report names each as the
    baseline's, and hardware generation replays the same values, nothing open. On
    a body no cut created (``BaselineChoices`` not run), the KernelOps' own choices
    were stored at conversion and its channels' are open: the exploration baselines
    those."""
    model = kernel_model()
    explored = explore_kernel_choices(model)
    report = explored.report
    saved = choices(model)
    # The baseline: the least folding, the adder tree, each memory in the explicit style
    # its size orders first (the Chain's, all small: LUTRAM; never auto, last), every
    # transport direct (no size_fifos: nothing sized).
    assert saved["first"]["compute.packed.pe"] == saved["first"]["compute.packed.simd"] == 1
    assert saved["first"]["compute.packed.reducer"] == "tree"
    assert saved["w1"]["source.memstream.ram_style"] == "distributed"
    assert saved["activate"]["ram_style"] == "distributed"
    assert saved["activate"]["block_stages"] == 0
    assert saved["levels"]["transport"] == "direct"
    assert report["fifos"] == "not sized here (no size_fifos): each transport as it was"
    assert report["strategy"] is None and report["open"] == [] and report["held"] == {}
    by = {
        f"{node}.{attribute}": entry["by"]
        for node, held in report["choices"].items()
        for attribute, entry in held.items()
    }
    assert set(by.values()) == {"saved", "baseline"}
    assert sorted(key for key, who in by.items() if who == "baseline") == sorted(
        report["baselined"]
    )
    assert {key.split(".")[0] for key, who in by.items() if who == "saved"} == {
        "first",
        "activate",
        "second",
    }
    assert {key.split(".")[0] for key in report["baselined"]} == {
        "x",
        "w1",
        "hidden",
        "levels",
        "w2",
        "y",
    }
    values = {
        f"{node}.{attribute}": value
        for node, held in saved.items()
        for attribute, value in held.items()
    }
    assert {
        f"{node}.{attribute}": entry["value"]
        for node, held in report["choices"].items()
        for attribute, entry in held.items()
    } == values
    # Nothing is left open, where it is explored or generated.
    assert report["open"] == []
    point, _ = configured_root(model, "the chain")
    assert {
        f"{node}.{attribute}": value for (node, attribute), value in point.network.chosen().items()
    } == values
    assert explored.network.viable() == ()
    # Explored again, the body is the same and nothing is baselined.
    again = explore_kernel_choices(model).report
    assert again["baselined"] == [] and choices(model) == saved
    assert {e["by"] for held in again["choices"].values() for e in held.values()} == {"saved"}


def test_the_report_logs_what_each_strategy_changed_kept_and_re_baselined() -> None:
    """A log of the exploration (provenance is logged, never stored): each value its
    strategy changed, before and after, how many it reopened and settling kept, the ones
    settling re-baselined (here each MatMul's pumping, a choice once SIMD is 2), and who
    set each value: the body, the baseline or the strategy."""
    model = kernel_model()
    report = explore_kernel_choices(model, TargetThroughput(1_000_000_000 // (5 * 6))).report
    target = report["strategy"]
    assert target["changed"]["first.compute.packed.pe"] == [1, 2]
    assert target["changed"]["first.compute.packed.simd"] == [1, 4]
    assert target["changed"]["first.compute.packed.compute_pumping"] == [None, False]
    assert target["committed"] == len(target["changed"])
    assert target["re-baselined"] == [
        "first.compute.packed.compute_pumping",
        "second.compute.packed.compute_pumping",
    ]
    assert target["reopened"] > target["kept"] > 0 and "held" not in target
    entries = report["choices"]
    assert entries["first"]["compute.packed.pe"] == {"value": 2, "by": "target_throughput"}
    # The reducer, which the strategy does not decide, keeps the baseline conversion
    # stored; the transports, on a body no cut created, the exploration's baseline.
    assert entries["first"]["compute.packed.reducer"] == {"value": "tree", "by": "saved"}
    assert entries["levels"]["transport"] == {"value": "direct", "by": "baseline"}
    assert json.loads(json.dumps(report)) == report
    # Sizing next, a second exploration: it reopens the 6 transports and changes none (the
    # Chain needs no FIFO here); what the target set is now the body's.
    sized = explore_kernel_choices(model, SizeFifos()).report
    sizing = sized["strategy"]
    assert (sizing["reopened"], sizing["changed"], sizing["committed"]) == (6, {}, 0)
    assert sized["choices"]["first"]["compute.packed.pe"] == {"value": 2, "by": "saved"}
    assert sized["fifos"] == "sized by size_fifos: 6 channels, 0 FIFOs placed"


def test_a_pinned_file_is_committed_held_in_the_body_and_the_rest_explored(
    tmp_path: Path,
) -> None:
    """A pin is committed in one batch and recorded in the body (``HELD``), so a later
    exploration of the body holds it: the ranked policy, which reopens everything else,
    keeps it."""
    pinned = tmp_path / "pinned.json"
    pinned.write_text(
        json.dumps(
            {
                "first": {"compute.packed.pe": 1},
                "levels": {"transport": "fifo", "transport.fifo.buffer.depth": 8},
            }
        )
    )
    pinning = ExploreKernelChoices(Pinned(pinned))
    model = kernel_model().transform(pinning)
    held = {
        "first": {"compute.packed.pe": 1},
        "levels": {"transport": "fifo", "transport.fifo.buffer.depth": 8},
    }
    assert model.get(HELD) == held
    assert pinning.explored is not None
    assert pinning.explored.report["strategy"]["held"] == [
        "first.compute.packed.pe",
        "levels.transport",
        "levels.transport.fifo.buffer.depth",
    ]
    explorer = ExploreKernelChoices(Ranked(Lanes(2)))
    model = model.transform(explorer)
    saved = choices(model)
    assert saved["first"]["compute.packed.pe"] == 1 and saved["first"]["compute.packed.simd"] == 2
    assert saved["levels"]["transport.fifo.buffer.depth"] == 8
    assert explorer.explored is not None
    assert explorer.explored.report["held"] == held and model.get(HELD) == held
    # A whole kernel_choices.json pins another model to the same choices, in one batch:
    # the FIFO and its depth, nested under it, together.
    exported = tmp_path / "kernel_choices.json"
    exported.write_text(json.dumps(kernel_choices_config(model)))
    again = kernel_model().transform(ExploreKernelChoices(Pinned(exported)))
    assert choices(again) == saved
    # What the network does not take is refused, named.
    pinned.write_text(json.dumps({"first": {"compute.packed.pe": 3}}))
    with pytest.raises(ExploreError, match="first.compute.packed.pe: .*domain-membership"):
        kernel_model().transform(ExploreKernelChoices(Pinned(pinned)))
    pinned.write_text(json.dumps({"nobody": {"pe": 1}}))
    with pytest.raises(ExploreError, match="nobody.pe"):
        kernel_model().transform(ExploreKernelChoices(Pinned(pinned)))


#: A memory pinned ``auto``: the first MatMul's weights (on their tensor) and the
#: thresholds' stages.
AUTO = {"w1": {"source.memstream.ram_style": "auto"}, "activate": {"ram_style": "auto"}}


def auto(model: ModelWrapper) -> dict[str, dict[str, object]]:
    """The values ``model`` holds of ``AUTO``'s keys."""
    return {node: {key: choices(model)[node][key] for key in AUTO[node]} for node in AUTO}


def stated_auto(report: Mapping[str, Any]) -> None:
    """The report names each memory pinned ``auto`` unstated, Vivado's, and its total a
    lower bound; every other memory keeps its explicit baseline."""
    resources = report["resources"]
    assert resources["lower_bound"] and resources["counted"].startswith("a lower bound")
    reasons = resources["unstated"]
    assert len(reasons) == 2 and all(
        any(f["message"].endswith("ram_style auto, placed by Vivado, not stated") for f in why)
        for why in reasons.values()
    )
    assert "activate" in reasons
    assert report["choices"]["w2"]["source.memstream.ram_style"]["value"] == "distributed"


def test_a_memory_saved_or_pinned_auto_is_kept_and_stated_as_vivado_s(tmp_path: Path) -> None:
    """``auto`` stays a value of every memory, last (SZ18): a model saved with it replays
    it, nothing dropped, and a kernel_choices.json pinning it commits it (the thresholds'
    ``ram_style`` attribute too); either way the report names the memory unstated."""
    model = kernel_model()
    for node in model.graph.node:
        if node.name in AUTO:
            # By hand, a change names what it clears: the thresholds' block stages, stored
            # at their baseline at conversion, are no choice under auto.
            cleared: dict[str, object] = {"block_stages": None} if node.name == "activate" else {}
            kernel_op(model, node).save({**AUTO[node.name], **cleared})
    save_channels(model, {"w1": AUTO["w1"]})
    model.save(tmp_path / "saved.onnx")
    saved = ModelWrapper(str(tmp_path / "saved.onnx"))
    report = explore_kernel_choices(saved).report
    assert report["dropped"] == {} and auto(saved) == AUTO
    stated_auto(report)
    pinned = tmp_path / "kernel_choices.json"
    pinned.write_text(json.dumps(AUTO))
    fresh = kernel_model()
    report = explore_kernel_choices(fresh, Pinned(pinned)).report
    assert auto(fresh) == AUTO
    assert report["strategy"]["committed"] == 2
    stated_auto(report)


def test_a_strategy_changes_what_it_decides_and_baseline_starts_over() -> None:
    """Explored again, a saved value is replaced where a strategy decides it (no choice
    is closed once saved), and kept where none does; ``baseline`` resets every value not
    held."""
    model = kernel_model()
    explore_kernel_choices(model, Ranked(Lanes(2)))
    assert choices(model)["first"]["compute.packed.pe"] == 2
    # 6 cycles a frame (the activation's) folds each MatMul to PE 2, SIMD 4; the reducer,
    # which it does not decide, stays.
    compressor = save_choice(model, "second", "compute.packed.reducer", "compressor")
    resumed = explore_kernel_choices(compressor, TargetThroughput(1_000_000_000 // (5 * 6)))
    assert choices(compressor)["first"]["compute.packed.simd"] == 4
    assert choices(compressor)["second"]["compute.packed.reducer"] == "compressor"
    assert resumed.report["choices"]["second"]["compute.packed.reducer"]["by"] == "saved"
    # The baseline strategy: every value back at its baseline.
    explore_kernel_choices(compressor, Rebaseline())
    assert choices(compressor) == choices(explored_at_baseline())


def save_choice(model: ModelWrapper, node: str, key: str, value: object) -> ModelWrapper:
    """``model`` with ``key`` of ``node`` saved at ``value``, its other choices kept."""
    op = kernel_op(model, next(each for each in model.graph.node if each.name == node))
    op.save({**op.choices(), key: value})
    return model


def explored_at_baseline() -> ModelWrapper:
    """The Chain explored by nothing: every value at its baseline."""
    model = kernel_model()
    explore_kernel_choices(model)
    return model


def test_a_pin_holds_for_every_later_exploration_of_the_body(tmp_path: Path) -> None:
    """A pin recorded in the body holds for every later exploration of it, as one run of
    several strategies held it: a fold, FIFO sizing and going back to the baselines keep
    a pinned FIFO and a pinned folding, and pinning a held key to another value is
    refused, naming the earlier pin."""
    pinned = tmp_path / "pins.json"
    pins = {
        "first": {"compute.packed.simd": 1},
        "levels": {"transport": "fifo", "transport.fifo.buffer.depth": 8},
    }
    pinned.write_text(json.dumps(pins))
    model = kernel_model()
    explore_kernel_choices(model, Pinned(pinned))
    # 6 cycles a frame folds each MatMul to PE 2, SIMD 4 unpinned (the report test above).
    folded = explore_kernel_choices(model, TargetThroughput(1_000_000_000 // (5 * 6))).report
    assert folded["held"] == pins
    assert choices(model)["first"]["compute.packed.simd"] == 1
    assert choices(model)["second"]["compute.packed.simd"] == 4
    for strategy in (SizeFifos(), Rebaseline()):
        explore_kernel_choices(model, strategy)
        assert choices(model)["first"]["compute.packed.simd"] == 1
        assert choices(model)["levels"]["transport.fifo.buffer.depth"] == 8
    assert choices(model)["second"]["compute.packed.simd"] == 1  # back at its baseline
    assert model.get(HELD) == pins
    pinned.write_text(json.dumps({"first": {"compute.packed.simd": 2}}))
    with pytest.raises(ExploreError, match="held to 1 by an earlier pin"):
        explore_kernel_choices(model, Pinned(pinned))


def test_a_fifo_sized_before_a_folding_is_marked_stale_until_sized_again(
    tmp_path: Path,
) -> None:
    """A strategy that may change the folding, one no build step runs included, marks a
    body holding a FIFO as sized at another folding (``FIFOS_STALE``); FIFO sizing clears
    it, and the debug placeholder and the baselines alone leave it. A body without a
    FIFO, as the cut makes it, is not marked."""
    model = kernel_model()
    explore_kernel_choices(model, TargetThroughput(1_000_000_000 // (5 * 6)))
    assert model.get(FIFOS_STALE) is None
    pinned = tmp_path / "pins.json"
    pinned.write_text(
        json.dumps({"levels": {"transport": "fifo", "transport.fifo.buffer.depth": 8}})
    )
    explore_kernel_choices(model, Pinned(pinned))
    assert model.get(FIFOS_STALE) is True
    explore_kernel_choices(model, SizeFifos())
    assert model.get(FIFOS_STALE) is None
    explore_kernel_choices(model, Ranked(Lanes(2)))
    assert model.get(FIFOS_STALE) is True
    explore_kernel_choices(model, DebugPlaceholder())
    explore_kernel_choices(model)
    assert model.get(FIFOS_STALE) is True
    explore_kernel_choices(model, SizeFifos())
    assert model.get(FIFOS_STALE) is None


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
    explore_kernel_choices(model, Ranked(Lanes()))
    saved = kernel_op(model, model.graph.node[0]).choices()
    assert "compute.packed.pe" in saved and "compute" not in saved
    write_target(model, VCK190)
    model = model.transform(InferKernelTensors())
    bound = network(model, model.graph.node)
    assert "first.compute.packed.pe" in bound.stale
    assert ("first", "compute") in {key for key, _ in bound.viable()}
    explored = explore_kernel_choices(model, Ranked(Lanes()))
    assert "first.compute.packed.pe" in explored.report["dropped"]
    held = kernel_op(model, model.graph.node[0]).choices()
    assert "compute" in held
    again = network(model, model.graph.node)
    assert not again.stale and again.viable() == ()


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
    report = explore_kernel_choices(body, TargetThroughput(1e6)).report
    target = report["strategy"]
    assert target["cycles"] == 200 and target["met"] is True and target["unfolded"] == {}
    assert target["bottleneck"]["cycles"] <= 200 and target["bottleneck_of"] == "folded"
    assert report["bottleneck"]["cycles"] <= 200
    cases = target["cases"]
    assert set(cases) == {f"MatMul_{index}.compute" for index in range(4)}
    for key, row in cases.items():
        node = key.split(".")[0]
        assert row["taken"] in ("packed", "int8_dsp58") and row["why"]
        assert set(row["folded"]) == {"packed", "int8_dsp58"}
        # The core is set by the target throughput where it is not its baseline, packed,
        # which conversion stored.
        held = report["choices"][node]
        by = "saved" if row["taken"] == "packed" else "target_throughput"
        assert held["compute"] == {"value": row["taken"], "by": by}
        assert held[f"compute.{row['taken']}.simd"]["by"] == "target_throughput"


@pytest.mark.slow
def test_tfc_at_a_million_frames_a_second_is_saved_as_the_57_values_it_is_built_with(
    tfc: ModelWrapper,
) -> None:
    """TFC's body explored by target_throughput 1e6, then size_fifos, saves the 57 values
    it was built with when exploration left choices open and a completion policy
    completed them on a copy (``TFC_BUILT``): the target reopens the folding, the engine
    widens it to the four input buffers' memories, settling keeps the rest and gives
    each MatMul's pumping, a choice at SIMD 2 and more, its baseline; sizing places no
    FIFO."""
    model = ModelWrapper(tfc.model.__deepcopy__())
    target = explore_kernel_choices(model, TargetThroughput(1e6)).report["strategy"]
    report = explore_kernel_choices(model, SizeFifos()).report
    assert kernel_choices_config(model) == TFC_BUILT
    assert sum(map(len, TFC_BUILT.values())) == 57
    sizing = report["strategy"]
    assert target["re-baselined"] == [
        f"MatMul_{index}.compute.packed.compute_pumping" for index in range(4)
    ]
    assert (sizing["reopened"], sizing["changed"], sizing["fifo_bits"]) == (13, {}, 0)
    assert report["open"] == []
    # Emission replays the saved choices: the configured network is the explored one.
    point, _ = configured_root(model, "the partition")
    assert point.network.viable() == () and len(point.network.decisions()) > 57


def test_exploration_reads_a_partitions_body_and_refuses_any_other_node() -> None:
    """Which KernelOps go together is the cut's: a model holding a node that is not a
    KernelOp (the uncut model's host nodes) is refused, naming it."""
    model = kernel_model()
    model.graph.node[2].domain = ""  # a plain ONNX MatMul: not a KernelOp
    with pytest.raises(KernelOpError, match="second: not KernelOps; exploration reads"):
        explore_kernel_choices(model)
