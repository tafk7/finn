# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exploring a model's KernelOps through the seam, one strategy at a time: owners,
persistence of every value, pins that hold for every later exploration of the body,
replay after a fact change, the report as a log of the exploration.

On the Chain as a model (MatMul ``first``, Thresholding ``activate``, MatMul
``second``; three rows of four) for Ultra96 without a shell: as conversion leaves it, its
channels' choices not made (a body no cut created, ``kernel_model``), or as the cut leaves
it (``cut_model``: ``MakeCutChoices``, each channel's choices by the interim channel
rule); and an INT8 MatMul retargeted from Ultra96 to a VCK190.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from support.configure import configured, write_raw
from support.explore_models import choices, kernel_model
from support.helpers import Lanes
from support.models import TARGET, matmul_model
from support.refusals import refused
from support.tfc_built import TFC_BUILT, TFC_STORED

import finn.kernels.explore as explore_module
from finn.core.graph.annotations.choices import HELD, OBJECTIVE_FPS
from finn.core.graph.annotations.outputs import OUTPUT_VERIFIED
from finn.core.graph.annotations.platform import read_target, write_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import partition_body
from finn.core.reporting.findings import FindingKind
from finn.core.space import inspection
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import Partition, network
from finn.custom_op.kernels.shell import configured_network, verify_partition
from finn.kernels.explore import (
    Accepted,
    DebugPlaceholder,
    Defaults,
    ExploreError,
    Hold,
    Ranked,
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
from finn.transformation.kernels.cut import CutKernelPartition, MakeCutChoices

VCK190 = resolve_target(part="xcvc1902-vsva2197-2MP-e-S", period_ns=5.0)


def keys_of(held: Mapping[str, Any]) -> dict[str, list[str]]:
    """The held mark's form of ``{graph name: {key: value}}``: the keys, sorted, by owner
    (``HELD`` names keys; the values are the nodes' and tensors')."""
    return {owner: sorted(held[owner]) for owner in sorted(held)}


def seam_of(model: ModelWrapper) -> tuple[Seam, Partition]:
    return Seam(read_target(model).platform), network(model, model.graph.node)


def cut_model() -> ModelWrapper:
    """The Chain as the cut leaves it: ``kernel_model`` with each channel's choices made
    by the interim channel rule (``MakeCutChoices``, what the cut does to a body made by
    hand)."""
    return kernel_model().transform(MakeCutChoices())


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
    choices set by conversion's policies); a channel's resources wait on its open choices
    (its transport, its adapter) on a body no cut created."""
    explorer, network = seam_of(kernel_model())
    cost = explorer.cost(network)
    assert set(cost.waiting) | set(cost.cycles) == set(network.members)
    assert "first" in cost.cycles
    # Each finding is owned in its member's space; the member names it.
    assert {(f.code, f.owner) for f in cost.unstated["levels"]} == {
        ("decision-unassigned", "transport"),
        ("decision-unassigned", "adapter"),
    }
    hidden = explorer.cost(network, ("hidden",))
    assert {*hidden.cycles, *hidden.waiting} == {"hidden"}
    assert explorer.refusals(network) == {}
    outcome = explorer.set(network, {("first", "compute.packed.pe"): 1})
    assert isinstance(outcome, Accepted)
    folded = explorer.cost(outcome.network, ("first",))
    # pe=1 on the 48-cycle kernel: its cycles are known, nothing waits
    assert folded.cycles == {"first": 48}
    assert not folded.waiting


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
    # Every member states its resources; on the ip shell, which has no movers and no
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
    assert resources["shell"] == {"partition": resources["used"], "movers": {}, "static_region": {}}
    assert "each mover and its static region" in resources["counted"]
    ranked = report["strategy"]
    assert ranked["strategy"] == "ranked" and ranked["attempts"] > 0
    # Nothing is left open.
    assert report["open"] == [] and "completion" not in report and "completed" not in report
    assert json.loads(json.dumps(report)) == report
    # The model replays to the explored network: nothing open, nothing stale.
    bound = network(model, model.graph.node)
    assert bound.undecided() == () and not bound.stale


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
    assert resources["findings"] == [found.record()]
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
    never a total. As the cut leaves it, every member states its own. On a body no cut
    created, the kernels state theirs (their choices stored at conversion) and the
    channels wait."""
    seam, network = seam_of(kernel_model())
    cost = seam.cost(network)
    assert cost.used is None and set(cost.resources) == {"first", "activate", "second"}
    assert {(f.code, f.owner) for f in cost.unstated["x"]} == {
        ("decision-unassigned", "transport"),
        ("decision-unassigned", "adapter"),
    }
    assert "activate" not in cost.unstated
    based = seam.cost(seam_of(cut_model())[1])
    assert based.unstated == {}
    assert based.used == sum(based.resources.values(), Resources())


def test_exploring_with_no_strategy_sets_nothing_and_every_value_is_saved() -> None:
    """With no strategy, nothing is set: on a body no cut created, its KernelOps' choices
    were stored at conversion and its channels' are open, named in the report's ``open``
    (emission refuses them). As the cut leaves it, every value is stated, the report names
    each as the body's, and hardware generation replays the same values, nothing open."""
    bare = kernel_model()
    report = explore_kernel_choices(bare).report
    assert report["strategy"] is None and "baselined" not in report
    assert {key.split(".")[0] for key in report["open"]} == {
        "x",
        "w1",
        "hidden",
        "levels",
        "w2",
        "y",
    }
    assert {e["by"] for held in report["choices"].values() for e in held.values()} == {"saved"}
    model = cut_model()
    explored = explore_kernel_choices(model)
    report = explored.report
    saved = choices(model)
    # Conversion's and the cut's: the least folding, the adder tree, each memory in the
    # explicit style its size orders first (the Chain's, all small: LUTRAM; never auto,
    # last), every transport direct (no size_fifos: nothing sized).
    assert saved["first"]["compute.packed.pe"] == saved["first"]["compute.packed.simd"] == 1
    assert saved["first"]["compute.packed.reducer"] == "tree"
    assert saved["w1"]["source.memstream.ram_style"] == "distributed"
    assert saved["activate"]["ram_style"] == "distributed"
    assert saved["activate"]["block_stages"] == 0
    assert saved["levels"]["transport"] == "direct"
    assert report["fifos"] == "not sized here (no size_fifos): each transport as it was"
    assert report["strategy"] is None and report["open"] == [] and report["held"] == {}
    assert {e["by"] for held in report["choices"].values() for e in held.values()} == {"saved"}
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
    point, _ = configured_network(model, "the chain")
    assert {
        f"{node}.{attribute}": value for (node, attribute), value in point.network.chosen().items()
    } == values
    assert explored.network.undecided() == ()
    # Explored again, the body is the same.
    assert explore_kernel_choices(model).report["choices"] == report["choices"]
    assert choices(model) == saved


def test_the_report_logs_what_each_strategy_changed_kept_and_set() -> None:
    """A log of the exploration (provenance is logged, never stored): each value its
    strategy changed, before and after, how many it reopened and gave back, those it set by
    their policy (none here), and who set each value: the body, the strategy, or the
    interim channel rule (a channel's value its rebuild gave it: the levels' output adapter
    the fold brings into play)."""
    model = cut_model()
    report = explore_kernel_choices(model, TargetThroughput(1_000_000_000 // (5 * 6))).report
    target = report["strategy"]
    assert target["changed"]["first.compute.packed.pe"] == [1, 2]
    assert target["changed"]["first.compute.packed.simd"] == [1, 4]
    # Pumping, stored at conversion, is given back at SIMD 4: unchanged.
    assert "first.compute.packed.compute_pumping" not in target["changed"]
    assert target["changed"]["levels.output_adapter"] == [None, "vpc"]
    assert target["committed"] == len(target["changed"])
    assert (target["by_policy"], target["unset"], target["rederived"]) == ([], {}, {})
    assert target["interim"] == ["levels.output_adapter"]
    assert target["reopened"] > target["kept"] > 0 and "held" not in target
    entries = report["choices"]
    assert entries["first"]["compute.packed.pe"] == {"value": 2, "by": "target_throughput"}
    # The reducer, which the strategy does not decide, keeps what conversion stored; the
    # transports, what the cut stored.
    assert entries["first"]["compute.packed.reducer"] == {"value": "tree", "by": "saved"}
    assert entries["levels"]["transport"] == {"value": "direct", "by": "saved"}
    assert entries["levels"]["output_adapter"] == {"value": "vpc", "by": "interim channel rule"}
    assert json.loads(json.dumps(report)) == report
    # Sizing next, a second exploration: it reopens the 6 transports and changes none (the
    # Chain needs no FIFO here); what the target set is now the body's.
    sized = explore_kernel_choices(model, SizeFifos()).report
    sizing = sized["strategy"]
    assert (sizing["reopened"], sizing["changed"], sizing["committed"]) == (6, {}, 0)
    assert sized["choices"]["first"]["compute.packed.pe"] == {"value": 2, "by": "saved"}
    assert sized["fifos"] == "sized by size_fifos: 6 channels, 0 FIFOs placed"


def test_a_hold_is_committed_held_in_the_body_and_the_rest_explored() -> None:
    """A hold is committed in one batch and recorded in the body (``HELD``), so a later
    exploration of the body holds it: the ranked policy, which reopens everything else,
    keeps it. On a body no cut created, the ranked policy sets every open choice, the
    channels' too."""
    held: dict[str, dict[str, object]] = {
        "first": {"compute.packed.pe": 1},
        "levels": {"transport": "fifo", "transport.fifo.buffer.depth": 8},
    }
    pinning = ExploreKernelChoices(Hold(held))
    model = kernel_model().transform(pinning)
    assert model.get(HELD) == keys_of(held)
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
    assert explorer.explored.report["held"] == held and model.get(HELD) == keys_of(held)
    # A whole kernel_choices export holds another model to the same choices, in one batch:
    # the FIFO and its depth, nested under it, together.
    again = kernel_model().transform(ExploreKernelChoices(Hold(kernel_choices_config(model))))
    assert choices(again) == saved
    # What the network does not take is refused, named, the hold owning it.
    with refused(
        "pin-refused",
        error=ExploreError,
        owner="held",
        message=r"first\.compute\.packed\.pe: .*domain-membership",
    ):
        kernel_model().transform(ExploreKernelChoices(Hold({"first": {"compute.packed.pe": 3}})))
    with refused("member-unknown", error=ExploreError, owner="held", message=r"nobody\.pe"):
        kernel_model().transform(ExploreKernelChoices(Hold({"nobody": {"pe": 1}})))


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
    it, nothing dropped, and a hold of it commits it (the thresholds'
    ``ram_style`` attribute too); either way the report names the memory unstated."""
    model = kernel_model()
    configured(model, AUTO)
    model.save(tmp_path / "saved.onnx")
    saved = ModelWrapper(str(tmp_path / "saved.onnx"))
    report = explore_kernel_choices(saved).report
    assert report["dropped"] == {} and auto(saved) == AUTO
    stated_auto(report)
    fresh = cut_model()
    report = explore_kernel_choices(fresh, Hold(AUTO)).report
    assert auto(fresh) == AUTO
    assert report["strategy"]["committed"] == 2
    stated_auto(report)


def test_a_strategy_changes_what_it_decides_and_the_defaults_start_over() -> None:
    """Explored again, a saved value is replaced where a strategy decides it (no choice
    is closed once saved), and kept where none does; ``Defaults`` resets every value not
    held to conversion's policies and the interim channel rule's."""
    model = cut_model()
    explore_kernel_choices(model, Ranked(Lanes(2)))
    assert choices(model)["first"]["compute.packed.pe"] == 2
    # 6 cycles a frame (the activation's) folds each MatMul to PE 2, SIMD 4; the reducer,
    # which it does not decide, stays.
    compressor = save_choice(model, "second", "compute.packed.reducer", "compressor")
    resumed = explore_kernel_choices(compressor, TargetThroughput(1_000_000_000 // (5 * 6)))
    assert choices(compressor)["first"]["compute.packed.simd"] == 4
    assert choices(compressor)["second"]["compute.packed.reducer"] == "compressor"
    assert resumed.report["choices"]["second"]["compute.packed.reducer"]["by"] == "saved"
    # The defaults: every value back at conversion's and the cut's (the levels' output
    # adapter, brought into play by the fold, reopened and out of play again).
    explore_kernel_choices(compressor, Defaults())
    assert choices(compressor) == choices(cut_model())


def save_choice(model: ModelWrapper, node: str, key: str, value: object) -> ModelWrapper:
    """``model`` with ``key`` of ``node`` set at ``value``, its other choices kept."""
    return configured(model, {node: {key: value}})


def test_a_hold_holds_for_every_later_exploration_of_the_body() -> None:
    """A hold recorded in the body holds for every later exploration of it, as one run of
    several strategies held it: a fold, FIFO sizing and going back to the defaults keep
    a held FIFO and a held folding, and holding a held key at another value is
    refused, naming the value it is held to."""
    pins: dict[str, dict[str, object]] = {
        "first": {"compute.packed.simd": 1},
        "levels": {"transport": "fifo", "transport.fifo.buffer.depth": 8},
    }
    model = cut_model()
    explore_kernel_choices(model, Hold(pins))
    # 6 cycles a frame folds each MatMul to PE 2, SIMD 4 unpinned (the report test above).
    folded = explore_kernel_choices(model, TargetThroughput(1_000_000_000 // (5 * 6))).report
    assert folded["held"] == pins
    assert choices(model)["first"]["compute.packed.simd"] == 1
    assert choices(model)["second"]["compute.packed.simd"] == 4
    for strategy in (SizeFifos(), Defaults()):
        explore_kernel_choices(model, strategy)
        assert choices(model)["first"]["compute.packed.simd"] == 1
        assert choices(model)["levels"]["transport.fifo.buffer.depth"] == 8
    assert choices(model)["second"]["compute.packed.simd"] == 1  # back at its first valid
    assert model.get(HELD) == keys_of(pins)
    with refused(
        "pin-refused",
        error=ExploreError,
        owner="held",
        message=r"first\.compute\.packed\.simd: already held to 1\b",
    ):
        explore_kernel_choices(model, Hold({"first": {"compute.packed.simd": 2}}))


def test_a_second_hold_at_another_value_is_refused() -> None:
    """``Hold`` fixes values given as a dict, with no file: a key held at one value stays
    held, and a later ``Hold`` of it at another value is refused, naming the key and the
    value it is held to; at the same value it is accepted."""
    model = kernel_model()
    explore_kernel_choices(model, Hold({"first": {"compute.packed.simd": 1}}))
    assert model.get(HELD) == keys_of({"first": {"compute.packed.simd": 1}})
    explore_kernel_choices(model, Hold({"first": {"compute.packed.simd": 1}}))
    with refused(
        "pin-refused",
        error=ExploreError,
        owner="held",
        message=r"first\.compute\.packed\.simd: already held to 1\b",
    ):
        explore_kernel_choices(model, Hold({"first": {"compute.packed.simd": 2}}))
    assert choices(model)["first"]["compute.packed.simd"] == 1


def test_a_held_key_the_bound_network_does_not_state_is_refused_by_name() -> None:
    """The mark holds ``first``'s PE, but the node's attribute states a value its domain
    refuses (written past the writer, standing for a graph FINN did not write): the bind
    drops it, and the exploration is refused, naming the node and the key, the bind's
    findings its causes. The held value is not dropped and set another way."""
    model = cut_model()
    write_raw(model, "first", "compute.packed.simd", 2)
    write_raw(model, "first", "compute.packed.pe", 3)
    # SIMD is held too, and stated: only the refused key is the error's.
    model.set(HELD, {"first": ["compute.packed.pe", "compute.packed.simd"]})
    with refused(
        "held-not-taken",
        error=KernelOpError,
        owner="explore_kernel_choices",
        message=r"first\.compute\.packed\.pe: .*held",
        node="first",
        keys=("compute.packed.pe",),
    ) as caught:
        explore_kernel_choices(model)
    # The bind's own finding, its domain's: nothing else stands beside it.
    assert {(cause.code, cause.owner) for cause in caught.finding.causes} == {
        ("domain-membership", "compute.packed.pe"),
    }
    assert isinstance(caught.error, KernelOpError)
    assert caught.error.keys == ("first.compute.packed.pe",)
    assert model.get(HELD) == {"first": ["compute.packed.pe", "compute.packed.simd"]}


def test_a_held_key_whose_owner_is_no_member_of_the_bound_network_is_refused() -> None:
    """The mark names an owner the body does not have: refused by name, as ``Hold`` of
    one is (``member-unknown``), not left out of the seed."""
    model = kernel_model()
    model.set(HELD, {"nobody": ["pe"]})
    with refused(
        "held-owner-unknown",
        error=KernelOpError,
        owner="explore_kernel_choices",
        message=r"nobody\.pe: .*held",
        keys=("pe",),
    ):
        explore_kernel_choices(model)


def _levels_short(channel: Any, period: int, words: int) -> bool:
    """A double of the seam's check (the Chain needs no FIFO at any folding): ``levels``
    (activate's output) needs more words than its transport holds; every other channel's
    holds."""
    return not str(channel.sides.source_owner).startswith("activate.")


LEVELS_FIFO = {"transport": "fifo", "transport.fifo.buffer.depth": 8}
LEVELS_DEPTH = "levels.transport.fifo.buffer.depth"


class Checking:
    """A strategy that only checks the FIFOs and drops what the model calls too small
    (``Seam.checked``): the step a hybrid strategy composes before sizing again."""

    strategy = "checking"

    def explore(self, seam: Seam, network: Any) -> Any:
        checked, _ = seam.checked(network)
        return checked

    def report(self) -> dict[str, object]:
        return {"strategy": self.strategy}


def test_an_exploration_checks_its_fifos_only_where_its_strategy_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing marks a body, and an exploration checks nothing of itself: a fold that
    leaves a FIFO too small for its stream keeps it, its report saying its FIFOs were not
    checked, and the build's verification refuses it (``fifo-stale``), while binding alone
    builds it. A strategy that checks (``Seam.checked``) drops its depth, names it in the
    report's ``dropped`` and leaves it open, which emission refuses, its report saying the
    model judged them; sizing says so too. A held depth is the user's, kept and built,
    and the verification names it, a limitation."""
    model = cut_model()
    explore_kernel_choices(model, TargetThroughput(1_000_000_000 // (5 * 6)))
    configured(model, {"levels": LEVELS_FIFO})
    folded = explore_kernel_choices(model, TargetThroughput(1_000_000_000 // (5 * 12)))
    assert folded.report["strategy"]["changed"]
    assert choices(model)["levels"]["transport.fifo.buffer.depth"] == 8
    assert folded.report["dropped"] == {} and folded.findings == ()
    assert folded.report["fifos_judged"] == "not checked"
    assert "fifo_check" not in folded.report
    # The same FIFO too small for its stream: no exploration drops it of itself.
    monkeypatch.setattr(explore_module, "holds", _levels_short)
    kept = explore_kernel_choices(model).report
    assert kept["dropped"] == {} and kept["fifos_judged"] == "not checked"
    assert choices(model)["levels"]["transport.fifo.buffer.depth"] == 8
    configured_network(model, "the chain")
    with pytest.raises(KernelOpError) as stale:
        verify_partition(model, "the chain")
    assert [(f.code, f.owner) for f in stale.value.findings] == [("fifo-stale", LEVELS_DEPTH)]
    assert model.get(OUTPUT_VERIFIED) is None
    # A strategy that checks drops it.
    dropped = explore_kernel_choices(model, Checking()).report
    assert list(dropped["dropped"]) == [LEVELS_DEPTH]
    assert dropped["dropped"][LEVELS_DEPTH].startswith(
        f"{LEVELS_DEPTH}: fifo-stale: depth 8 (8 words) no longer keeps its producer"
    )
    assert dropped["fifos_judged"] == "model"
    assert dropped["fifo_check"]["period"] == dropped["bottleneck"]["cycles"]
    assert dropped["fifo_check"]["stale"] == [LEVELS_DEPTH]
    assert f"{LEVELS_DEPTH} (required)" in dropped["open"]
    levels = choices(model)["levels"]
    assert levels["transport"] == "fifo" and "transport.fifo.buffer.depth" not in levels
    with refused(
        "choices-open",
        error=KernelOpError,
        node="the chain",
        keys=(LEVELS_DEPTH,),
        message=r"\(required: run step_size_fifos",
    ):
        configured_network(model, "the chain")
    # Held, it is kept and built as held.
    pinned = explore_kernel_choices(model, Hold({"levels": {"transport.fifo.buffer.depth": 8}}))
    assert choices(model)["levels"]["transport.fifo.buffer.depth"] == 8
    assert pinned.findings == () and pinned.report["fifos_judged"] == "not checked"
    _, check = verify_partition(model, "the chain")
    assert [(f.kind, f.code, f.owner) for f in check.findings] == [
        (FindingKind.LIMITATION, "fifo-held-stale", LEVELS_DEPTH)
    ]
    assert check.stale == {} and "levels" in check.checked
    # Sizing judges by the model, and checks nothing.
    sized = explore_kernel_choices(cut_model(), SizeFifos()).report
    assert sized["fifos_judged"] == sized["strategy"]["fifos_judged"] == "model"
    assert "fifo_check" not in sized


def test_the_target_s_frames_a_second_go_when_an_exploration_changes_a_kernel_s_choice() -> None:
    """A target throughput records its frames a second (``OBJECTIVE_FPS``); an exploration
    that changes no kernel's choice (FIFO sizing, a hold of a channel's) keeps it, and one
    that changes any (whatever its strategy) removes it."""
    model = cut_model()
    explore_kernel_choices(model, TargetThroughput(1_000_000_000 // (5 * 6)))
    assert model.get(OBJECTIVE_FPS) == 1_000_000_000 // (5 * 6)
    explore_kernel_choices(model, SizeFifos())
    explore_kernel_choices(model, Hold({"levels": {"transport": "direct"}}))
    explore_kernel_choices(model, DebugPlaceholder())
    explore_kernel_choices(model)
    assert model.get(OBJECTIVE_FPS) == 1_000_000_000 // (5 * 6)
    changed = explore_kernel_choices(model, Ranked(Lanes(2))).report["strategy"]["changed"]
    assert any(key.startswith("first.") for key in changed)
    assert model.get(OBJECTIVE_FPS) is None


def int8_matmul() -> ModelWrapper:
    model = matmul_model(annotate=(), infer=False)
    for tensor in ("x", "w"):
        model.set_tensor_datatype(tensor, DataType["INT8"])
    return model.transform(InferKernelTensors())


def test_a_retarget_keeps_the_folding_and_another_core_puts_it_out_of_play() -> None:
    """An INT8 MatMul explored on Ultra96 (DSP48E2: the packed core the one possible, its
    folding saved nested under it), retargeted to a VCK190 (DSP58: the INT8 core possible
    too): the stored core and folding bind as they are, nothing stale, nothing open. The
    other core puts the packed folding out of play, kept (dormant), and opens its own; a
    hold of it leaves those open for a strategy, which sets them, reopening the dormant
    folding, no longer in play, gone."""
    model = int8_matmul()
    explore_kernel_choices(model, Ranked(Lanes()))
    saved = kernel_op(model, model.graph.node[0]).choices()
    assert saved["compute"] == "packed" and "compute.packed.pe" in saved
    write_target(model, VCK190)
    model = model.transform(InferKernelTensors())
    bound = network(model, model.graph.node)
    assert not bound.stale and bound.undecided() == ()
    assert bound.chosen(("first",)) == {("first", key): value for key, value in saved.items()}
    # The other core: the packed folding out of play, kept, the INT8 core's open.
    elsewhere = Seam(VCK190.platform).set(bound, {("first", "compute"): "int8_dsp58"})
    assert isinstance(elsewhere, Accepted) and not elsewhere.rederived
    states = dict(elsewhere.network.states(("first",)))
    assert states[("first", "compute.packed.pe")] == inspection.Dormant(saved["compute.packed.pe"])
    assert set(elsewhere.network.undecided()) == {
        ("first", "compute.int8_dsp58.pe"),
        ("first", "compute.int8_dsp58.simd"),
        ("first", "compute.int8_dsp58.compute_pumping"),
    }
    # Held, the core is set and its folding left open; explored, the folding is chosen,
    # the packed one's dormant attributes reopened and gone.
    report = explore_kernel_choices(model, Hold({"first": {"compute": "int8_dsp58"}})).report
    assert set(report["open"]) == {
        "first.compute.int8_dsp58.pe",
        "first.compute.int8_dsp58.simd",
        "first.compute.int8_dsp58.compute_pumping",
    }
    explore_kernel_choices(model, Ranked(Lanes()))
    held = kernel_op(model, model.graph.node[0]).choices()
    assert held["compute"] == "int8_dsp58" and "compute.int8_dsp58.pe" in held
    assert not any(key.startswith("compute.packed.") for key in held)
    again = network(model, model.graph.node)
    assert not again.stale and again.undecided() == ()


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
    completed them on a copy (``TFC_BUILT``), and the 27 conversion, the cut and the
    fold now store beside them, which forcing left unstored (``TFC_STORED``, both): the target
    reopens the kernels' choices, folds and gives the rest back; sizing places no FIFO."""
    model = ModelWrapper(tfc.model.__deepcopy__())
    target = explore_kernel_choices(model, TargetThroughput(1e6)).report["strategy"]
    report = explore_kernel_choices(model, SizeFifos()).report
    found = kernel_choices_config(model)
    assert found == TFC_STORED
    assert sum(map(len, TFC_BUILT.values())) == 57 and sum(map(len, found.values())) == 84
    sizing = report["strategy"]
    assert (target["by_policy"], target["unset"], target["rederived"]) == ([], {}, {})
    assert (sizing["reopened"], sizing["changed"], sizing["fifo_bits"]) == (13, {}, 0)
    assert report["open"] == []
    # Emission replays the saved choices: the configured network is the explored one.
    point, _ = configured_network(model, "the partition")
    assert point.network.undecided() == () and len(point.network.decisions()) > 79


def test_exploration_reads_a_partitions_body_and_refuses_any_other_node() -> None:
    """Which KernelOps go together is the cut's: a model holding a node that is not a
    KernelOp (the uncut model's host nodes) is refused, naming it."""
    model = kernel_model()
    model.graph.node[2].domain = ""  # a plain ONNX MatMul: not a KernelOp
    with refused(
        "body-not-kernel-ops",
        error=KernelOpError,
        owner="explore_kernel_choices",
        message="second: not KernelOps; exploration reads",
    ):
        explore_kernel_choices(model)
