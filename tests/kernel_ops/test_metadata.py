# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's typed graph metadata (qonnx's ``qonnx.core.metadata``): the build target in
``finn.platform``, as ``finn.platform.resolve.resolve_target`` resolves it, and what a build
made of a partition in ``finn.outputs``."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from onnx import StringStringEntryProto, helper
from qonnx.core.metadata import MetadataError
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from support.configure import configured
from support.explore_models import choices
from support.explore_models import kernel_model as open_model
from support.helpers import Lanes
from support.models import TARGET, chain_source, configure_partition, kernel_model
from support.refusals import refused

from finn.core.graph.annotations.outputs import (
    OUTPUT_BITFILE,
    OUTPUT_BITFILE_IP_KEY,
    OUTPUT_BITFILE_MADE_FROM,
    OUTPUT_BITFILE_OPTIONS,
    OUTPUT_HWH,
    OUTPUT_INTERFACES,
    OUTPUT_IP,
    OUTPUT_IP_KEY,
    OUTPUT_IP_OPTIONS,
    OUTPUT_SEARCH,
    OUTPUT_VLNV,
)
from finn.core.graph.annotations.platform import (
    PLATFORM,
    PLATFORM_FIELDS,
    PLATFORM_KEYS,
    read_target,
    write_target,
)
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import design_digest, search_record
from finn.core.reporting.findings import by_code
from finn.core.search import Hold, Ranked
from finn.kernels.explore import KernelExplorer, SizeFifos, TargetThroughput
from finn.platform.resolve import resolve_target
from finn.platform.target import Target
from finn.transformation.kernels import ToKernelOps, explore_kernel_choices
from finn.transformation.kernels.cut import MakeCutChoices


def entries(model: ModelWrapper) -> dict[str, str]:
    return {item.key: item.value for item in model.graph.metadata_props}


def holder(body_name: str = "body") -> ModelWrapper:
    """A model with one node holding a (here empty) subgraph body."""
    body = helper.make_graph([], body_name, [], [])
    node = helper.make_node("Holder", [], [], name="holder", body=body)
    return ModelWrapper(qonnx_make_model(helper.make_graph([node], "outer", [], [])))


def body_of(model: ModelWrapper) -> ModelWrapper:
    wrapper: ModelWrapper = model.make_subgraph_modelwrapper(model.graph.node[0].attribute[0].g)
    return wrapper


#: Ultra96 in the Zynq shell: a target with a board and its part's resources.
ULTRA96 = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")
#: VCK190's part on the ip shell, as a harness states it: no board, no resources (a
#: resolved target always has its part's; a Platform built without a part has none).
_VCK190 = resolve_target(part="xcvc1902-vsva2197-2MP-e-S", period_ns=4.0)
VERSAL = replace(_VCK190, platform=replace(_VCK190.platform, resources=None))


def test_the_target_is_stated_typed_every_key_and_read_back() -> None:
    model = holder()
    write_target(model, TARGET)
    assert read_target(model) == TARGET
    assert set(PLATFORM_KEYS) == {"part", "shell", "board", *PLATFORM_FIELDS}
    stated = entries(model)
    assert stated["finn.platform/@version"] == "1"
    assert stated["finn.platform/part"] == "xczu3eg-sbva484-1-e"
    assert stated["finn.platform/shell"] == "ip"
    assert stated["finn.platform/board"] == "null"
    assert stated["finn.platform/period_ns"] == "5.0"
    assert stated["finn.platform/dsp"] == "DSP48E2"
    assert stated["finn.platform/fabric"] == "ULTRASCALE"
    assert stated["finn.platform/uram"] == "false"
    assert stated["finn.platform/resources"] == (
        '{"bram18":432,"dsp":360,"ff":141120,"lut":70560,"uram":0}'
    )
    # A body opened through its parent reads the parent's; extracted, it carries a copy.
    body = body_of(model)
    assert read_target(body) == TARGET
    body.inherit_metadata(PLATFORM)
    assert read_target(ModelWrapper(body.model)) == TARGET


@pytest.mark.parametrize("target", [ULTRA96, VERSAL], ids=["board", "unknown-resources"])
def test_a_target_round_trips_with_its_board_and_resources_or_none(target: Target) -> None:
    model = holder()
    write_target(model, target)
    assert read_target(ModelWrapper(model.model)) == target
    stated = entries(model)
    if target.board is None:
        assert (stated["finn.platform/board"], stated["finn.platform/resources"]) == (
            "null",
            "null",
        )
    else:
        assert stated["finn.platform/board"] == '"Ultra96"'


@pytest.mark.parametrize("missing", ["dsp", "period_ns", "uram", "shell", "fabric", "resources"])
def test_a_partial_target_is_refused_naming_what_it_misses(missing: str) -> None:
    model = holder()
    write_target(model, TARGET)
    model.delete(PLATFORM_KEYS[missing])
    with refused(
        "target-unrecorded",
        error=KernelOpError,
        owner="read_target",
        message=rf"states no target \(finn.platform: {missing} missing",
    ):
        read_target(model)
    with refused("target-unrecorded", error=KernelOpError, owner="read_target"):
        read_target(holder())


#: The shape before the registry: capabilities a shell budgets (control_ports,
#: memory_ports) or supplies (clk2x, the shell row's since the registry) and an AI Engine flag,
#: no shell, board, fabric or resources.
OLD_SHAPE = {
    "@version": "1",
    "part": "xczu3eg-sbva484-1-e",
    "period_ns": "5.0",
    "dsp": "DSP48E2",
    "uram": "false",
    "uram_init": "false",
    "clk2x": "true",
    "control_ports": "1",
    "memory_ports": "0",
    "aie": "false",
}


def test_the_old_shape_is_refused_naming_the_keys_it_states() -> None:
    model = holder()
    for name, text in OLD_SHAPE.items():
        model.set_metadata_prop(f"finn.platform/{name}", text)
    with pytest.raises(KernelOpError) as unreadable:
        read_target(model)
    assert "stored keys ['aie', 'clk2x', 'control_ports', 'memory_ports'] are not declared" in (
        str(unreadable.value)
    )
    assert "run ToKernelOps to state it again" in str(unreadable.value)
    # Without those keys, it is a partial one: the new keys are named.
    for name in ("aie", "clk2x", "control_ports", "memory_ports"):
        model.graph.metadata_props.remove(
            next(item for item in model.graph.metadata_props if item.key == f"finn.platform/{name}")
        )
    with refused(
        "target-unrecorded",
        error=KernelOpError,
        owner="read_target",
        message="finn.platform: shell, board, fabric, resources missing",
    ) as unstated:
        read_target(model)
    # Two refusals of the reader, each its own code: by_code tells them apart.
    found = (*unreadable.value.findings, *unstated.error.findings)
    assert [(each.owner, each.code) for each in found] == [
        ("read_target", "target-unreadable"),
        ("read_target", "target-unrecorded"),
    ]
    assert len(by_code(found)) == 2


@pytest.mark.parametrize(
    "key, text, expected",
    [
        ("period_ns", "fast", "finn.platform/period_ns: stored 'fast' is not a float"),
        ("resources", '{"lut":1}', "finn.platform/resources: stored '{\"lut\":1}' is not"),
        ("board", '""', "finn.platform/board: stored '\"\"' is not"),
        ("shell", "", "finn.platform/shell: stored '' is not"),
    ],
)
def test_a_malformed_key_is_refused(key: str, text: str, expected: str) -> None:
    model = holder()
    write_target(model, TARGET)
    model.set_metadata_prop(f"finn.platform/{key}", text)
    with pytest.raises(KernelOpError) as malformed:
        read_target(model)
    assert expected in str(malformed.value)


def test_a_target_that_is_not_one_writes_nothing() -> None:
    model = holder()
    with refused(
        "target-dsp-unstated",
        error=KernelOpError,
        owner="write_target",
        message="states its DSP block",
    ):
        write_target(model, replace(TARGET, platform=replace(TARGET.platform, dsp=None)))
    with refused(
        "target-invalid",
        error=KernelOpError,
        owner="write_target",
        message="period_ns: cannot store 0.0",
    ):
        write_target(model, replace(TARGET, platform=replace(TARGET.platform, period_ns=0.0)))
    with refused(
        "target-invalid",
        error=KernelOpError,
        owner="write_target",
        message="shell: cannot store ''",
    ):
        write_target(model, replace(TARGET, shell=""))
    assert entries(model) == {}


def test_conversion_states_the_target_it_is_given() -> None:
    zcu104 = resolve_target(board="ZCU104", period_ns=4.0, shell="pynq")
    assert read_target(chain_source().transform(ToKernelOps(zcu104))) == zcu104


#: A design digest as ``design_digest`` writes it.
DIGEST = "sha256:" + "0123456789abcdef" * 4


def test_what_a_build_made_is_typed_in_finn_outputs() -> None:
    model = holder()
    made = {
        OUTPUT_IP: "/build/stitch/ip",
        OUTPUT_VLNV: "xilinx_finn:finn:partition:1.0",
        OUTPUT_INTERFACES: {"clk": ["ap_clk"], "s_axis": [["s_axis_0", 32]], "axilite": []},
        OUTPUT_IP_KEY: DIGEST,
        OUTPUT_IP_OPTIONS: {"run_synth": False},
        OUTPUT_BITFILE: "/out/bitfile/finn-accel.bit",
        OUTPUT_BITFILE_MADE_FROM: DIGEST,
        OUTPUT_BITFILE_IP_KEY: DIGEST,
        OUTPUT_BITFILE_OPTIONS: {"hw_debug": False},
        OUTPUT_HWH: "/out/bitfile/finn-accel.hwh",
    }
    for key, value in made.items():
        model.set(key, value)
    written = ModelWrapper(model.model.SerializeToString())
    assert {key.name: written.get(key) for key in made} == {
        key.name: value for key, value in made.items()
    }
    # Every entry is the namespace's: no flat key.
    assert all(key.startswith("finn.outputs/") for key in entries(written))
    for key, bad in (
        (OUTPUT_IP, ""),
        (OUTPUT_INTERFACES, ["ap_clk"]),
        (OUTPUT_HWH, ""),
        (OUTPUT_IP_KEY, "sha1:" + "0" * 40),
        (OUTPUT_IP_OPTIONS, ["run_synth"]),
        (OUTPUT_BITFILE_MADE_FROM, DIGEST.upper()),
        (OUTPUT_BITFILE_IP_KEY, ""),
        (OUTPUT_BITFILE_OPTIONS, ["hw_debug"]),
    ):
        with pytest.raises(MetadataError, match=f"{key.entry}: cannot store"):
            model.set(key, bad)


# -- the design digest: what an artifact was made from ----------------------------------


def cut_model() -> ModelWrapper:
    """The Chain as the cut leaves it (``MakeCutChoices``), for Ultra96."""
    return open_model().transform(MakeCutChoices())


def chain() -> ModelWrapper:
    """The Chain as KernelOps for Ultra96 in the Zynq shell, its choices saved."""
    model = kernel_model()
    write_target(model, ULTRA96)
    configure_partition(model)
    return model


def test_the_design_digest_is_stable_across_save_and_load(tmp_path: Path) -> None:
    model = chain()
    digest = design_digest(model)
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64
    assert design_digest(model) == digest
    model.save(str(tmp_path / "chain.onnx"))
    assert design_digest(ModelWrapper(str(tmp_path / "chain.onnx"))) == digest
    assert design_digest(ModelWrapper(model.model.SerializeToString())) == digest


def test_the_design_digest_ignores_what_a_build_made_and_changes_nothing() -> None:
    """finn.outputs is what a build made of the design, not the design: stating an IP,
    its key or a bitfile leaves the digest as it was, and the digest leaves the
    model's own entries in place."""
    model = chain()
    digest = design_digest(model)
    model.set(OUTPUT_IP, "/build/stitch/ip")
    model.set(OUTPUT_IP_KEY, digest)
    model.set(OUTPUT_BITFILE, "/out/bitfile/finn-accel.bit")
    stated = entries(model)
    assert design_digest(model) == digest
    assert entries(model) == stated


def test_the_search_record_is_checked() -> None:
    """The search's record states exactly its design (a digest), the keys it held, by owner,
    and the objective (a positive number or null): one that is not that is refused when it is
    written."""
    model = holder()
    for objective in (None, 1_000_000.0, 3):
        record = {"design": DIGEST, "held": {"first": ["simd"]}, "objective_fps": objective}
        model.set(OUTPUT_SEARCH, record)
        written = ModelWrapper(model.model.SerializeToString())
        assert written.get(OUTPUT_SEARCH) == record
    model.set(OUTPUT_SEARCH, {"design": DIGEST, "held": {}, "objective_fps": None})
    assert all(key.startswith("finn.outputs/") for key in entries(model))
    good: dict[str, object] = {"design": DIGEST, "held": {}, "objective_fps": None}
    for bad in (
        {"design": DIGEST, "held": {}},
        {**good, "extra": 1},
        {**good, "design": DIGEST.upper()},
        {**good, "design": ""},
        {**good, "held": ["first.simd"]},
        {**good, "held": {"first": "simd"}},
        {**good, "held": {"first": [1]}},
        {**good, "objective_fps": 0},
        {**good, "objective_fps": -1.0},
        {**good, "objective_fps": True},
        {**good, "objective_fps": "fast"},
        ["design"],
    ):
        with pytest.raises(MetadataError, match=f"{OUTPUT_SEARCH.entry}: cannot store"):
            model.set(OUTPUT_SEARCH, bad)


def test_the_design_digest_clears_the_search_record() -> None:
    """The search's record is a build's output (``finn.outputs``), not the design: it names
    the design it was made for, so it cannot be part of it, and a hold changes no hardware.
    The record is the body's own while its design is the body's, stale once the body's design
    changes, and none where no search wrote one."""
    model = chain()
    assert search_record(model) == ("none", None)
    digest = design_digest(model)
    record = {"design": digest, "held": {"activate": ["pe"]}, "objective_fps": 1e6}
    model.set(OUTPUT_SEARCH, record)
    assert design_digest(model) == digest
    assert search_record(model) == ("current", record)
    changed_choice(model)
    assert design_digest(model) != digest
    assert search_record(model) == ("stale", None)
    assert model.get(OUTPUT_SEARCH) == record


def test_the_objective_stays_until_an_entry_changes_a_kernel() -> None:
    """A target throughput records its frames a second in the search's record; a later entry
    of the same list that changes no kernel's choice (FIFO sizing, a hold of a value the
    kernel already has) keeps it, and one that changes a kernel's choice, whatever its
    strategy, removes it. A list without a target throughput records none."""
    fps = 1_000_000_000 // (5 * 6)

    def objective(*strategies: KernelExplorer) -> object:
        model = cut_model()
        explore_kernel_choices(model, list(strategies))
        state, record = search_record(model)
        assert state == "current" and record is not None
        return record["objective_fps"]

    assert objective(TargetThroughput(fps)) == fps
    assert objective(TargetThroughput(fps), SizeFifos()) == fps
    # A hold of the folding the target throughput found changes no kernel's choice.
    folded = cut_model()
    explore_kernel_choices(folded, [TargetThroughput(fps)])
    kept = {"first": {"pe": choices(folded)["first"]["pe"]}}
    assert objective(TargetThroughput(fps), Hold(kept)) == fps
    # A later entry that changes a kernel's choice removes it.
    assert objective(TargetThroughput(fps), Hold({"first": {"pe": 1, "simd": 1}})) is None
    assert objective(TargetThroughput(fps), Ranked(Lanes(2))) is None
    assert objective(SizeFifos()) is None
    assert objective() is None


def test_the_design_digest_ignores_orders_that_carry_no_meaning() -> None:
    """The graph's metadata entries and a node's attributes, in another order, are the
    same design."""
    model, reordered = chain(), chain()
    props = list(reordered.graph.metadata_props)[::-1]
    del reordered.graph.metadata_props[:]
    reordered.graph.metadata_props.extend(
        StringStringEntryProto(key=prop.key, value=prop.value) for prop in props
    )
    attributes = list(reordered.graph.node[0].attribute)[::-1]
    del reordered.graph.node[0].attribute[:]
    reordered.graph.node[0].attribute.extend(attributes)
    assert entries(reordered) == entries(model)
    assert design_digest(reordered) == design_digest(model)


def changed_choice(model: ModelWrapper) -> None:
    configured(model, {"activate": {"deep_pipeline": 1}})


def changed_channel(model: ModelWrapper) -> None:
    configured(model, {"x": {"adapter_ram_style": "block"}})


def changed_weight(model: ModelWrapper) -> None:
    weights = next(name for name in ("w1", "w2") if model.get_initializer(name) is not None)
    values = model.get_initializer(weights)
    assert values is not None
    model.set_initializer(weights, np.where(values == 0, 1, 0).astype(values.dtype))


def changed_target(model: ModelWrapper) -> None:
    write_target(model, resolve_target(board="Ultra96", period_ns=4.0, shell="pynq"))


@pytest.mark.parametrize(
    "change",
    [changed_choice, changed_channel, changed_weight, changed_target],
    ids=["choice", "channel", "weight", "target"],
)
def test_the_design_digest_changes_with_the_design(change: object) -> None:
    """A kernel's choice, a channel's, a weight or the target changed makes another IP,
    and another digest."""
    model = chain()
    digest = design_digest(model)
    change(model)  # type: ignore[operator]
    assert design_digest(model) != digest
