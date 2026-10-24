# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exploring a model's KernelOps through the seam, one strategy at a time: owners,
persistence of every value, pins that hold for every later exploration of the body,
replay after a fact change, the report as a log of the exploration.

On the Chain as a model (MatMul ``first``, Thresholding ``activate``, MatMul
``second``; three rows of four) for Ultra96 without a shell: as conversion leaves it, its
channels' choices not made (a body no cut created, ``kernel_model``), or as the cut leaves
it (``cut_model``: ``MakeCutChoices``, each channel's choices by the channel's policies);
and an INT8 MatMul retargeted from Ultra96 to a VCK190.
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
from finn.builder.build_dataflow_strategies import HoldEntry
from finn.core.graph.annotations.outputs import OUTPUT_SEARCH, OUTPUT_VERIFIED
from finn.core.graph.annotations.platform import read_target, write_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import design_digest, partition_body, search_record
from finn.core.reporting.findings import FindingKind
from finn.core.search import Accepted, ExploreError, Hold, Ranked
from finn.core.space import inspection
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import network
from finn.custom_op.kernels.shell import configured_network, verify_partition
from finn.kernels.explore import Defaults, KernelExplorer, Seam, SizeFifos, TargetThroughput
from finn.kernels.partition import Partition
from finn.platform.capabilities import Resources
from finn.platform.resolve import resolve_target
from finn.transformation.kernels import (
    InferKernelTensors,
    ToKernelOps,
    explore_kernel_choices,
    kernel_choices_config,
)
from finn.transformation.kernels.cut import CutKernelPartition, MakeCutChoices

COMPUTE_PUMPING = "compute.packed.compute_pumping"
VCK190 = resolve_target(part="xcvc1902-vsva2197-2MP-e-S", period_ns=5.0)


def keys_of(held: Mapping[str, Any]) -> dict[str, list[str]]:
    """The held mark's form of ``{graph name: {key: value}}``: the keys, sorted, by owner
    (the search's record names keys; the values are the nodes' and tensors')."""
    return {owner: sorted(held[owner]) for owner in sorted(held)}


def seam_of(model: ModelWrapper) -> tuple[Seam, Partition]:
    return Seam(read_target(model).platform), network(model, model.graph.node)


def cut_model() -> ModelWrapper:
    """The Chain as the cut leaves it: ``kernel_model`` with each channel's choices made
    by the channel's policies (``MakeCutChoices``, what the cut does to a body made by
    hand)."""
    return kernel_model().transform(MakeCutChoices())


def test_a_choice_is_keyed_by_the_node_or_tensor_that_persists_it() -> None:
    explorer, network = seam_of(kernel_model())
    offered = {choice.key: choice for choice in explorer.choices(network)}
    # A kernel's choice is its node's (stored there at conversion); a channel's is its
    # tensor's: an edge's, a graph output's alike (open on a body no cut created).
    assert ("first", "pe") not in offered
    assert ("first", "pe") in explorer.chosen(network)
    assert ("levels", "transport") in offered and ("y", "transport") in offered
    # Kernels rank first, then the channels.
    assert network.members[:3] == ("first", "activate", "second")
    assert set(network.channels) == {"x", "w1", "hidden", "levels", "w2", "y"}


def test_the_seam_reads_each_member_by_name() -> None:
    """Cost and refusals read the network's members by name, and a change checks the
    members its keys belong to. A kernel's cycles are known from conversion on (its own
    choices set by conversion's policies); a channel's resources wait on its open choices
    (its transport, its adapter's memory) on a body no cut created."""
    explorer, network = seam_of(kernel_model())
    cost = explorer.cost(network)
    assert set(cost.waiting) | set(cost.cycles) == set(network.members)
    assert "first" in cost.cycles
    # Each finding is owned in its member's space; the member names it.
    assert {(f.code, f.owner) for f in cost.unstated["levels"]} == {
        ("decision-unassigned", "transport"),
        ("decision-unassigned", "adapter_ram_style"),
    }
    hidden = explorer.cost(network, ("hidden",))
    assert {*hidden.cycles, *hidden.waiting} == {"hidden"}
    assert explorer.refusals(network) == {}
    outcome = explorer.set(network, {("first", "pe"): 1})
    assert isinstance(outcome, Accepted)
    folded = explorer.cost(outcome.network, ("first",))
    # pe=1 on the 48-cycle kernel: its cycles are known, nothing waits
    assert folded.cycles == {"first": 48}
    assert not folded.waiting


def test_exploring_saves_the_point_s_choices_and_reports_its_cost() -> None:
    model = kernel_model()
    explored = explore_kernel_choices(model, [Ranked(Lanes(2))])
    saved = choices(model)
    assert saved["first"]["pe"] == 2 and saved["levels"]["transport"] == "direct"
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
    ranked = report["strategies"][0]
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
        explored = explore_kernel_choices(model, [Ranked(Lanes(2))])
    (found,) = explored.findings
    assert (found.kind, found.code) == (FindingKind.LIMITATION, "resources-over-part")
    assert "most of dsp: dsp 4 of 3" in found.message
    assert dict(found.details)["binding"] == "dsp"
    resources = explored.report["resources"]
    assert resources["binding"] == "dsp"
    assert resources["over"] == {"dsp": {"used": 4, "platform": 3}}
    assert resources["findings"] == [found.record()]
    assert found.message.startswith("the point uses more than the platform's part has")
    assert choices(model)["first"]["pe"] == 2
    # Within the part: the binding resource is named, and nothing is over.
    within = explore_kernel_choices(kernel_model(), [Ranked(Lanes(2))])
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
        ("decision-unassigned", "adapter_ram_style"),
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
    assert report["strategies"] == [] and "baselined" not in report
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
    # Conversion's and the channel's: the least folding, the adder tree, each memory in the
    # explicit style its size orders first (the Chain's, all small: LUTRAM; never auto,
    # last), every transport direct (no size_fifos: nothing sized).
    assert saved["first"]["pe"] == saved["first"]["simd"] == 1
    assert saved["first"]["compute.packed.reducer"] == "tree"
    assert saved["w1"]["source.ram_style"] == "distributed"
    assert saved["activate"]["ram_style"] == "distributed"
    assert saved["activate"]["block_stages"] == 0
    assert saved["levels"]["transport"] == "direct"
    assert report["strategies"] == [] and report["open"] == [] and report["held"] == {}
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
    their policy (none here), what the fold rule cleared and set for its changes (none
    here: no stored channel choice stands in a fold's way, and the levels' output adapter
    the fold brings into play is placed, no choice), and who set each value: the body or
    the strategy."""
    model = cut_model()
    report = explore_kernel_choices(model, [TargetThroughput(1_000_000_000 // (5 * 6))]).report
    target = report["strategies"][0]
    assert target["changed"]["first.pe"] == [1, 2]
    assert target["changed"]["first.simd"] == [1, 4]
    # Pumping, stored at conversion, is given back at SIMD 4: unchanged.
    assert "first.compute.packed.compute_pumping" not in target["changed"]
    assert target["committed"] == len(target["changed"])
    assert (target["by_policy"], target["unset"]) == ([], {})
    assert (target["cleared"], target["settled"]) == ({}, {})
    assert target["reopened"] > target["kept"] > 0 and "held" not in target
    entries = report["choices"]
    assert entries["first"]["pe"] == {"value": 2, "by": "target_throughput"}
    # The reducer, which the strategy does not decide, keeps what conversion stored; the
    # transports, what the cut stored.
    assert entries["first"]["compute.packed.reducer"] == {"value": "tree", "by": "saved"}
    assert entries["levels"]["transport"] == {"value": "direct", "by": "saved"}
    assert "output_adapter" not in entries["levels"]
    assert json.loads(json.dumps(report)) == report
    # Sizing next, a second exploration: it reopens the 6 transports and changes none (the
    # Chain needs no FIFO here); what the target set is now the body's.
    sized = explore_kernel_choices(model, [SizeFifos()]).report
    sizing = sized["strategies"][0]
    assert (sizing["reopened"], sizing["changed"], sizing["committed"]) == (6, {}, 0)
    assert sized["choices"]["first"]["pe"] == {"value": 2, "by": "saved"}
    assert sizing["fifos"] == "sized by size_fifos: 6 channels, 0 FIFOs placed"


def test_a_hold_is_committed_held_for_the_list_and_the_rest_explored() -> None:
    """A hold is committed in one batch and held for the entries after it, on one seam: the
    ranked policy, which reopens everything else, keeps it; the body records the keys held
    (``search_record``). On a body no cut created, the ranked policy sets every open
    choice, the channels' too."""
    held: dict[str, dict[str, object]] = {
        "first": {"pe": 1},
        "levels": {"transport": "fifo", "transport.fifo.depth": 8},
    }
    model = kernel_model()
    explored = explore_kernel_choices(model, [Hold(held), Ranked(Lanes(2))])
    assert search_record(model) == (
        "current",
        {"design": design_digest(model), "held": keys_of(held), "objective_fps": None},
    )
    assert explored.report["strategies"][0]["held"] == [
        "first.pe",
        "levels.transport",
        "levels.transport.fifo.depth",
    ]
    saved = choices(model)
    assert saved["first"]["pe"] == 1 and saved["first"]["simd"] == 2
    assert saved["levels"]["transport.fifo.depth"] == 8
    assert explored.report["held"] == held
    # A whole kernel_choices export holds another model to the same choices, in one batch:
    # the FIFO and its depth, nested under it, together.
    again = kernel_model()
    explore_kernel_choices(again, [Hold(kernel_choices_config(model))])
    assert choices(again) == saved
    # What the network does not take is refused, named, the hold owning it.
    with refused(
        "pin-refused",
        error=ExploreError,
        owner="hold",
        message=r"first\.pe: .*domain-membership",
    ):
        explore_kernel_choices(kernel_model(), [Hold({"first": {"pe": 3}})])
    with refused("member-unknown", error=ExploreError, owner="hold", message=r"nobody\.pe"):
        explore_kernel_choices(kernel_model(), [Hold({"nobody": {"pe": 1}})])


#: A memory pinned ``auto``: the first MatMul's weights (on their tensor) and the
#: thresholds' stages.
AUTO = {"w1": {"source.ram_style": "auto"}, "activate": {"ram_style": "auto"}}


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
    assert report["choices"]["w2"]["source.ram_style"]["value"] == "distributed"


def test_a_memory_saved_or_pinned_auto_is_kept_and_stated_as_vivado_s(tmp_path: Path) -> None:
    """``auto`` stays a value of every memory, last: a model saved with it replays
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
    report = explore_kernel_choices(fresh, [Hold(AUTO)]).report
    assert auto(fresh) == AUTO
    assert report["strategies"][0]["committed"] == 2
    stated_auto(report)


def test_a_strategy_changes_what_it_decides_and_the_defaults_start_over() -> None:
    """Explored again, a saved value is replaced where a strategy decides it (no choice
    is closed once saved), and kept where none does; ``Defaults`` resets every value not
    held to conversion's policies and the channel's policies'."""
    model = cut_model()
    explore_kernel_choices(model, [Ranked(Lanes(2))])
    assert choices(model)["first"]["pe"] == 2
    # 6 cycles a frame (the activation's) folds each MatMul to PE 2, SIMD 4; the reducer,
    # which it does not decide, stays.
    compressor = save_choice(model, "second", "compute.packed.reducer", "compressor")
    resumed = explore_kernel_choices(compressor, [TargetThroughput(1_000_000_000 // (5 * 6))])
    assert choices(compressor)["first"]["simd"] == 4
    assert choices(compressor)["second"]["compute.packed.reducer"] == "compressor"
    assert resumed.report["choices"]["second"]["compute.packed.reducer"]["by"] == "saved"
    # The defaults: every value back at conversion's and the channel's (the levels' output
    # adapter, brought into play by the fold, reopened and out of play again).
    explore_kernel_choices(compressor, [Defaults()])
    assert choices(compressor) == choices(cut_model())


def save_choice(model: ModelWrapper, node: str, key: str, value: object) -> ModelWrapper:
    """``model`` with ``key`` of ``node`` set at ``value``, its other choices kept."""
    return configured(model, {node: {key: value}})


def test_a_hold_holds_for_the_rest_of_the_list() -> None:
    """A hold holds for every entry after it in its list, on one seam: a fold, FIFO sizing and
    going back to the defaults keep a held FIFO and a held folding, and holding a held key
    at another value is refused, naming the value it is held to; the body then keeps what
    it had. A hold ends with its list: the next one holds only what it lists."""
    pins: dict[str, dict[str, object]] = {
        "first": {"simd": 1},
        "levels": {"transport": "fifo", "transport.fifo.depth": 8},
    }
    model = cut_model()
    # 6 cycles a frame folds each MatMul to PE 2, SIMD 4 unpinned (the report test above).
    folded = explore_kernel_choices(
        model, [Hold(pins), TargetThroughput(1_000_000_000 // (5 * 6))]
    ).report
    assert folded["held"] == pins
    assert choices(model)["first"]["simd"] == 1
    assert choices(model)["second"]["simd"] == 4
    explore_kernel_choices(model, [Hold(pins), SizeFifos(), Defaults()])
    assert choices(model)["first"]["simd"] == 1
    assert choices(model)["levels"]["transport.fifo.depth"] == 8
    assert choices(model)["second"]["simd"] == 1  # back at its first valid
    assert search_record(model)[1] == {
        "design": design_digest(model),
        "held": keys_of(pins),
        "objective_fps": None,
    }
    before = choices(model)
    with refused(
        "pin-refused",
        error=ExploreError,
        owner="hold",
        message=r"first\.simd: already held to 1\b",
    ):
        explore_kernel_choices(model, [Hold(pins), Defaults(), Hold({"first": {"simd": 2}})])
    assert choices(model) == before
    # Its list over, the hold is over: the next list's defaults reset what it did not list.
    explore_kernel_choices(model, [Defaults()])
    assert choices(model)["levels"]["transport"] == "direct"
    assert search_record(model)[1] == {
        "design": design_digest(model),
        "held": {},
        "objective_fps": None,
    }


def test_a_second_hold_at_another_value_is_refused() -> None:
    """``Hold`` fixes values given as a dict, with no file: a key held at one value stays
    held for the list, and a later ``Hold`` of it at another value is refused, naming the key
    and the value it is held to; at the same value it is accepted."""
    model = kernel_model()
    explore_kernel_choices(model, [Hold({"first": {"simd": 1}}), Hold({"first": {"simd": 1}})])
    assert search_record(model)[1] == {
        "design": design_digest(model),
        "held": {"first": ["simd"]},
        "objective_fps": None,
    }
    before = choices(model)
    with refused(
        "pin-refused",
        error=ExploreError,
        owner="hold",
        message=r"first\.simd: already held to 1\b",
    ):
        explore_kernel_choices(model, [Hold({"first": {"simd": 1}}), Hold({"first": {"simd": 2}})])
    assert choices(model) == before
    assert choices(model)["first"]["simd"] == 1


def crafted_record(model: ModelWrapper, held: dict[str, list[str]]) -> None:
    """``model`` with a search record that is its own (its design as it is now) and holds
    ``held``: what only a record written past the search can state, since a search holds
    only keys its network takes."""
    model.set(OUTPUT_SEARCH, {"design": design_digest(model), "held": held, "objective_fps": None})


def test_verification_refuses_a_held_key_not_stated() -> None:
    """The search's record holds a key of ``first`` that the bound network states no value
    for (a record that is the body's own, crafted: a search holds only keys its network
    takes): verification refuses it by name, ``held-not-taken``, never dropping it and setting
    it another way. The record's other keys, stated, are no part of the refusal."""
    model = cut_model()
    crafted_record(model, {"first": ["simd", "nonesuch"]})
    with refused(
        "held-not-taken",
        error=KernelOpError,
        owner="verify_partition",
        message=r"first\.nonesuch: held by the search's record",
        node="first",
        keys=("nonesuch",),
    ) as caught:
        verify_partition(model, "the chain")
    assert isinstance(caught.error, KernelOpError)
    assert caught.error.keys == ("first.nonesuch",)
    assert model.get(OUTPUT_VERIFIED) is None


def test_verification_refuses_a_held_key_whose_owner_is_no_member_of_the_network() -> None:
    """The search's record, the body's own, names an owner the body does not have: refused
    by name (``held-owner-unknown``), not left out."""
    model = cut_model()
    crafted_record(model, {"nobody": ["pe"]})
    with refused(
        "held-owner-unknown",
        error=KernelOpError,
        owner="verify_partition",
        message=r"nobody\.pe: held by the search's record",
        keys=("pe",),
    ):
        verify_partition(model, "the chain")


def _levels_short(channel: Any, neighbours: Any, period: int, words: int) -> bool:
    """A double of the seam's check (the Chain needs no FIFO at any folding): ``levels``
    (activate's output) needs more words than its transport holds; every other channel's
    holds."""
    return not str(channel.sides.source_owner).startswith("activate.")


LEVELS_FIFO = {"transport": "fifo", "transport.fifo.depth": 8}
LEVELS_DEPTH = "levels.transport.fifo.depth"


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
    """An exploration checks nothing of itself: a fold that
    leaves a FIFO too small for its stream keeps it, its report saying its FIFOs were not
    checked, and the build's verification refuses it (``fifo-stale``), while binding alone
    builds it. A strategy that checks (``Seam.checked``) drops its depth, names it in the
    report's ``dropped`` and leaves it open, which emission refuses, its report saying the
    model judged them; sizing says so too. A held depth is the user's, kept and built,
    and the verification names it, a limitation."""
    model = cut_model()
    explore_kernel_choices(model, [TargetThroughput(1_000_000_000 // (5 * 6))])
    configured(model, {"levels": LEVELS_FIFO})
    folded = explore_kernel_choices(model, [TargetThroughput(1_000_000_000 // (5 * 12))])
    assert folded.report["strategies"][0]["changed"]
    assert choices(model)["levels"]["transport.fifo.depth"] == 8
    assert folded.report["dropped"] == {} and folded.findings == ()
    assert folded.report["strategies"][0]["fifos_judged"] == "not checked"
    assert "fifo_check" not in folded.report["strategies"][0]
    # The same FIFO too small for its stream: no exploration drops it of itself.
    monkeypatch.setattr(explore_module, "holds", _levels_short)
    kept = explore_kernel_choices(model, [Hold({})]).report
    assert kept["dropped"] == {} and kept["strategies"][0]["fifos_judged"] == "not checked"
    assert choices(model)["levels"]["transport.fifo.depth"] == 8
    configured_network(model, "the chain")
    with pytest.raises(KernelOpError) as stale:
        verify_partition(model, "the chain")
    assert [(f.code, f.owner) for f in stale.value.findings] == [("fifo-stale", LEVELS_DEPTH)]
    assert model.get(OUTPUT_VERIFIED) is None
    # A strategy that checks drops it.
    dropped = explore_kernel_choices(model, [Checking()]).report
    assert list(dropped["dropped"]) == [LEVELS_DEPTH]
    assert dropped["dropped"][LEVELS_DEPTH].startswith(
        f"{LEVELS_DEPTH}: fifo-stale: depth 8 (8 words) no longer keeps its producer"
    )
    checking = dropped["strategies"][0]
    assert checking["fifos_judged"] == "model"
    assert checking["fifo_check"]["period"] == dropped["bottleneck"]["cycles"]
    assert checking["fifo_check"]["stale"] == [LEVELS_DEPTH]
    assert f"{LEVELS_DEPTH} (required)" in dropped["open"]
    levels = choices(model)["levels"]
    assert levels["transport"] == "fifo" and "transport.fifo.depth" not in levels
    with refused(
        "choices-open",
        error=KernelOpError,
        node="the chain",
        keys=(LEVELS_DEPTH,),
        message=r"\(required: add size_fifos",
    ):
        configured_network(model, "the chain")
    # Held, it is kept and built as held.
    pinned = explore_kernel_choices(model, [Hold({"levels": {"transport.fifo.depth": 8}})])
    assert choices(model)["levels"]["transport.fifo.depth"] == 8
    assert pinned.findings == () and pinned.report["strategies"][0]["fifos_judged"] == "not checked"
    _, check = verify_partition(model, "the chain")
    assert [(f.kind, f.code, f.owner) for f in check.findings] == [
        (FindingKind.LIMITATION, "fifo-held-stale", LEVELS_DEPTH)
    ]
    assert check.stale == {} and "levels" in check.checked
    # Sizing judges by the model, and checks nothing.
    sized = explore_kernel_choices(cut_model(), [SizeFifos()]).report
    assert sized["strategies"][0]["fifos_judged"] == "model"
    assert "fifo_check" not in sized["strategies"][0]


def test_each_entry_reports_its_own_runs_by_position() -> None:
    """Each entry of a list reports what its own runs did (the seam's runs, attempts and
    checks beyond where the entry began), not the list's: two target throughputs, each its
    own fps, cycles and changes, a second one's attempts as many as the first's, not the
    sum; then two holds, each its own keys, reopening and attempting nothing."""
    model = cut_model()
    report = explore_kernel_choices(
        model,
        [
            TargetThroughput(1_000_000_000 // (5 * 6)),
            TargetThroughput(1_000_000_000 // (5 * 12)),
            Hold({"first": {"pe": 1}}),
            Hold({"second": {"pe": 1}}),
        ],
    ).report
    first, second, one, other = report["strategies"]
    assert [entry["strategy"] for entry in report["strategies"]] == [
        "target_throughput",
        "target_throughput",
        "hold",
        "hold",
    ]
    assert (first["cycles"], second["cycles"]) == (6, 12)
    assert set(first["changed"]) == {
        "first.pe",
        "first.simd",
        "activate.pe",
        "second.pe",
        "second.simd",
    }
    assert set(second["changed"]) == {"first.pe", "activate.pe", "second.pe"}
    assert first["changed"]["first.pe"] == [1, 2] and second["changed"]["first.pe"] == [2, 1]
    assert (first["reopened"], second["reopened"]) == (16, 16)
    assert first["attempts"] == second["attempts"] > 0
    # A hold of what the folding already has changes, reopens and attempts nothing: its
    # report is its own, not the folding's before it.
    for entry, held in ((one, ["first.pe"]), (other, ["second.pe"])):
        assert entry["held"] == held
        assert (entry["changed"], entry["committed"], entry["reopened"], entry["attempts"]) == (
            {},
            0,
            0,
            0,
        )
    assert report["held"] == {"first": {"pe": 1}, "second": {"pe": 1}}


def test_set_by_labels_a_policy_value_by_its_own_entry() -> None:
    """Each value is labelled by the entry of the list that last changed it, the label the
    entry's own: an INT8 MatMul retargeted to a DSP58 part, its core held (``hold``), then
    ``Defaults`` sets the core's own choice by its policy (``defaults``); another list sets
    it by ``Ranked`` (``ranked``) and the ``Defaults`` after it, changing nothing, takes
    nothing over. What no entry changed stays ``saved``."""
    labels: dict[str, dict[str, object]] = {}
    core = Hold({"first": {"compute": "int8_dsp58"}})
    lists: dict[str, list[KernelExplorer]] = {
        "defaults": [core, Defaults()],
        "ranked": [core, Ranked(Lanes()), Defaults()],
    }
    for name, strategies in lists.items():
        model = int8_matmul()
        explore_kernel_choices(model, [Ranked(Lanes())])
        write_target(model, VCK190)
        model = model.transform(InferKernelTensors())
        report = explore_kernel_choices(model, strategies).report
        labels[name] = {key: entry["by"] for key, entry in report["choices"]["first"].items()}
    pumping = "compute.int8_dsp58.compute_pumping"
    assert labels["defaults"] == {
        "pe": "saved",
        "simd": "saved",
        "compute": "hold",
        pumping: "defaults",
    }
    assert labels["ranked"] == {
        "pe": "saved",
        "simd": "saved",
        "compute": "hold",
        pumping: "ranked",
    }


def test_the_fifo_rows_belong_to_the_entry_that_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A strategy that checks the FIFOs (``Seam.checked``) has the check in its entry
    (``fifos_judged``, ``fifo_check``), and the entries around it, which check nothing, say
    so; sizing judges by the model in its own entry and the one before it is not sized."""
    monkeypatch.setattr(explore_module, "holds", _levels_short)
    model = cut_model()
    configured(model, {"levels": LEVELS_FIFO})
    report = explore_kernel_choices(model, [Hold({}), Checking(), Hold({})]).report
    before, checking, after = report["strategies"]
    assert [entry["fifos_judged"] for entry in report["strategies"]] == [
        "not checked",
        "model",
        "not checked",
    ]
    assert "fifo_check" not in before and "fifo_check" not in after
    assert checking["fifo_check"]["stale"] == [LEVELS_DEPTH]
    assert list(report["dropped"]) == [LEVELS_DEPTH]
    monkeypatch.undo()
    sized = explore_kernel_choices(
        cut_model(), [TargetThroughput(1_000_000_000 // (5 * 6)), SizeFifos()]
    ).report
    target, sizing = sized["strategies"]
    assert target["fifos"] == "not sized here (no size_fifos): each transport as it was"
    assert target["fifos_judged"] == "not checked"
    assert sizing["fifos"] == "sized by size_fifos: 6 channels, 0 FIFOs placed"
    assert sizing["fifos_judged"] == "model"


def stale_first_pe() -> ModelWrapper:
    """The cut Chain whose first MatMul's PE was stored at a value outside its space (3,
    where 1, 2 or 4 divide its width): the bind drops it as stale."""
    model = cut_model()
    write_raw(model, "first", "pe", 3)
    return model


def test_a_hold_refused_on_a_dropped_key_refuses_the_step() -> None:
    """A hold of a key the bind dropped as stale, at a value the seam refuses, is refused as
    any hold is (``pin-refused``), the bind's ``domain-membership`` finding for the key first
    among its causes, and says the body stored another value; nothing is written."""
    model = stale_first_pe()
    before = model.model.SerializeToString()
    with refused(
        "pin-refused",
        error=ExploreError,
        owner="hold",
        message=r"first\.pe: .*the body stored another value for first\.pe, which the bind dropped",
    ) as caught:
        explore_kernel_choices(model, [Hold({"first": {"pe": 3}})])
    (finding,) = [each for each in caught.error.findings if each.code == "pin-refused"]
    assert finding.causes[0].code == "domain-membership"
    assert model.model.SerializeToString() == before


def test_a_hold_taken_on_a_dropped_key_stands() -> None:
    """A hold of a key the bind dropped as stale, at a value the seam takes, stands: the
    key is held at it and recorded, and the stale value is gone."""
    model = stale_first_pe()
    report = explore_kernel_choices(model, [Hold({"first": {"pe": 2}})]).report
    assert choices(model)["first"]["pe"] == 2
    assert (
        report["held"] == {"first": {"pe": 2}} and report["choices"]["first"]["pe"]["by"] == "hold"
    )
    assert search_record(model)[1] == {
        "design": design_digest(model),
        "held": {"first": ["pe"]},
        "objective_fps": None,
    }


class MovesFirst:
    """A strategy that changes a key through the network's own trial, not the seam's ``set``
    (which refuses a held key's change)."""

    strategy = "moves_first"

    def explore(self, seam: Seam, network: Any) -> Any:
        tried = network.trial({("first", "pe"): 2})
        assert tried.accepted
        return tried.network

    def report(self) -> dict[str, object]:
        return {"strategy": self.strategy}


def test_a_held_key_moved_outside_the_seam_is_refused() -> None:
    """A held key not at its held value at the end of the list (an explorer moved it other
    than through ``Seam.set``) refuses the list (``held-not-kept``, owned by the key),
    nothing written."""
    model = cut_model()
    before = model.model.SerializeToString()
    with refused(
        "held-not-kept",
        error=KernelOpError,
        owner="first.pe",
        message="held to 1, but the network has 2",
    ):
        explore_kernel_choices(model, [Hold({"first": {"pe": 1}}), MovesFirst()])
    assert model.model.SerializeToString() == before
    assert search_record(model) == ("none", None)


def test_a_policy_set_value_after_a_refused_restore_is_labelled_its_policy() -> None:
    """The Chain stored with pumping on at SIMD 2, a target of 2e6 frames a second (a budget
    that wants SIMD 1): the fold reopens pumping, ``Seam.restore`` has the engine refuse the
    stored pumping at SIMD 1, and its policy sets it. The report labels SIMD by the entry
    (``target_throughput``) and pumping by the entry and its policy."""
    model = cut_model()
    configured(model, {"first": {**choices(model)["first"], "simd": 2, COMPUTE_PUMPING: True}})
    report = explore_kernel_choices(model, [TargetThroughput(2_000_000)]).report
    first = report["choices"]["first"]
    assert first[COMPUTE_PUMPING]["by"] == "target_throughput, its policy"
    assert first["simd"]["by"] == "target_throughput"
    assert (first["simd"]["value"], first[COMPUTE_PUMPING]["value"]) == (1, False)


def test_a_refold_after_a_retarget_reopens_every_kept_choice_but_a_held_one() -> None:
    """The body explored for a target throughput, then retargeted to another clock, then
    explored again: the fold reopens every kept choice it decides (the same count as
    without the hold, less the held one), refolds them for the new clock and leaves the held
    key at its value. The retarget is done by hand (``write_target``): the target is fixed at
    the cut, so a build never retargets a body in place; this is the witness of what the
    seam does when the platform's facts change under a saved folding."""
    fps = 1_000_000_000 // (4 * 6)
    free, held = cut_model(), cut_model()
    for model in (free, held):
        explore_kernel_choices(model, [TargetThroughput(1_000_000_000 // (5 * 12))])
        write_target(model, resolve_target(board="Ultra96", period_ns=4.0))
    refolded = explore_kernel_choices(free, [TargetThroughput(fps)]).report["strategies"][0]
    around = explore_kernel_choices(
        held, [Hold({"first": {"simd": 1}}), TargetThroughput(fps)]
    ).report["strategies"][1]
    assert choices(free)["first"]["simd"] == 4 and choices(held)["first"]["simd"] == 1
    assert around["reopened"] == refolded["reopened"] - 1
    assert choices(held)["second"]["simd"] == 4 == choices(free)["second"]["simd"]
    assert around["cycles"] == refolded["cycles"] == 6


def test_a_later_entry_reopens_an_earlier_entrys_choice_but_not_a_held_one() -> None:
    """A fold later in the list reopens the choices an earlier fold set and sets them
    again; a held one stays: 6 cycles a frame folds each MatMul to PE 2, 12 cycles to PE 1,
    and the second MatMul's PE, held at 2 between them, stays."""
    model = cut_model()
    explore_kernel_choices(
        model,
        [
            TargetThroughput(1_000_000_000 // (5 * 6)),
            Hold({"second": {"pe": 2}}),
            TargetThroughput(1_000_000_000 // (5 * 12)),
        ],
    )
    saved = choices(model)
    assert saved["first"]["pe"] == 1 and saved["activate"]["pe"] == 1
    assert saved["second"]["pe"] == 2


def int8_matmul() -> ModelWrapper:
    model = matmul_model(annotate=(), infer=False)
    for tensor in ("x", "w"):
        model.set_tensor_datatype(tensor, DataType["INT8"])
    return model.transform(InferKernelTensors())


def test_a_retarget_keeps_the_folding_and_another_core_keeps_it_too() -> None:
    """An INT8 MatMul explored on Ultra96 (DSP48E2: the packed core the one possible),
    retargeted to a VCK190 (DSP58: the INT8 core possible too): the stored core and
    folding bind as they are, nothing stale, nothing open. The other core keeps the
    folding, MatMul's, in play; it puts the packed core's own choices out of play, kept
    (dormant), and opens its own; a hold of it leaves that open for a strategy after it in
    the list, which sets it, reopening the dormant ones, no longer in play, gone."""
    model = int8_matmul()
    explore_kernel_choices(model, [Ranked(Lanes())])
    saved = kernel_op(model, model.graph.node[0]).choices()
    assert saved["compute"] == "packed" and "pe" in saved
    write_target(model, VCK190)
    model = model.transform(InferKernelTensors())
    bound = network(model, model.graph.node)
    assert not bound.stale and bound.undecided() == ()
    assert bound.chosen(("first",)) == {("first", key): value for key, value in saved.items()}
    # The other core: the folding kept in play, the packed core's own choices out of play,
    # kept, the INT8 core's open.
    elsewhere = Seam(VCK190.platform).set(bound, {("first", "compute"): "int8_dsp58"})
    assert isinstance(elsewhere, Accepted) and not elsewhere.cleared
    states = dict(elsewhere.network.states(("first",)))
    assert states[("first", "pe")] == inspection.Set(saved["pe"])
    assert states[("first", "simd")] == inspection.Set(saved["simd"])
    reducer = ("first", "compute.packed.reducer")
    assert states[reducer] == inspection.Dormant(saved["compute.packed.reducer"])
    pumping = ("first", "compute.int8_dsp58.compute_pumping")
    assert set(elsewhere.network.undecided()) == {pumping}
    # Held, the core is set and its own choice left open; explored, it is chosen, the
    # packed core's dormant attributes reopened and gone.
    report = explore_kernel_choices(model, [Hold({"first": {"compute": "int8_dsp58"}})]).report
    assert set(report["open"]) == {"first.compute.int8_dsp58.compute_pumping"}
    explore_kernel_choices(model, [Hold({"first": {"compute": "int8_dsp58"}}), Ranked(Lanes())])
    held = kernel_op(model, model.graph.node[0]).choices()
    assert held["compute"] == "int8_dsp58" and "pe" in held
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
    report = explore_kernel_choices(body, [TargetThroughput(1e6)]).report
    target = report["strategies"][0]
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
        assert held["simd"]["by"] == "target_throughput"


@pytest.mark.slow
def test_tfc_at_a_million_frames_a_second_is_saved_as_the_57_values_it_is_built_with(
    tfc: ModelWrapper,
) -> None:
    """TFC's body explored by target_throughput 1e6, then size_fifos, saves the 57 values
    it was built with when exploration left choices open and a completion policy
    completed them on a copy (``TFC_BUILT``), and the 12 conversion and the cut now store
    beside them, which forcing left unstored (``TFC_STORED``, both): the target
    reopens the kernels' choices, folds and gives the rest back; sizing places no FIFO."""
    model = ModelWrapper(tfc.model.__deepcopy__())
    target = explore_kernel_choices(model, [TargetThroughput(1e6)]).report["strategies"][0]
    report = explore_kernel_choices(model, [SizeFifos()]).report
    found = kernel_choices_config(model)
    assert found == TFC_STORED
    assert sum(map(len, TFC_BUILT.values())) == 57 and sum(map(len, found.values())) == 69
    sizing = report["strategies"][0]
    assert (target["by_policy"], target["unset"]) == ([], {})
    assert (target["cleared"], target["settled"]) == ({}, {})
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


def test_a_json_hold_of_each_value_kind_is_decoded() -> None:
    """A hold stated as JSON is read as each Decision's type, the way a stored choice is
    read back: an int Decision takes an int, a bool Decision a bool (or the 0 or 1 a stored
    form gives it), a str Decision a str; a value of another type is refused by name before
    anything is set. What is held is read back as that type."""
    stated = json.loads(
        '{"first": {"simd": 2, "compute": "packed", "compute.packed.compute_pumping": false},'
        ' "activate": {"deep_pipeline": true, "ram_style": "auto", "use_axilite": 0},'
        ' "levels": {"transport": "fifo", "transport.fifo.depth": 8}}'
    )
    model = cut_model()
    explore_kernel_choices(model, [HoldEntry(stated).strategy()])
    explorer, held = seam_of(model)
    typed = explorer.chosen(held)
    assert typed[("first", "simd")] == 2 and type(typed[("first", "simd")]) is int
    assert typed[("first", "compute")] == "packed"
    assert typed[("activate", "ram_style")] == "auto"
    assert typed[("activate", "deep_pipeline")] is True
    assert typed[("activate", "use_axilite")] is False
    assert typed[("first", "compute.packed.compute_pumping")] is False
    assert typed[("levels", "transport")] == "fifo"
    assert typed[("levels", "transport.fifo.depth")] == 8
    # Held at the decoded values: the same hold again is taken, a bool as 1 as well.
    explore_kernel_choices(model, [HoldEntry({"activate": {"deep_pipeline": 1}}).strategy()])
    for owner, key, value, kind in (
        ("first", "simd", "2", "int"),
        ("first", "simd", True, "int"),
        ("activate", "deep_pipeline", 2, "bool"),
        ("activate", "deep_pipeline", "true", "bool"),
        ("first", "compute", 1, "str"),
    ):
        fresh = cut_model()
        before = choices(fresh)
        with refused(
            "hold-value-invalid",
            error=ExploreError,
            owner="hold",
            message=rf"{owner}\.{key}: {value!r} is no {kind}",
        ):
            explore_kernel_choices(fresh, [HoldEntry({owner: {key: value}}).strategy()])
        assert choices(fresh) == before
