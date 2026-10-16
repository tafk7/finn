# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel's stated choices on its tensor (``finn.channel``): the namespace generated
from ``Channel``'s decisions, its reader and its writer checked by the partition's
network, in a partition's body only, and how the cut, exploration and the choices export carry them.

The demarcation's cases: (1) a channel choice stated on a graph with nodes on the host
(the whole graph, before the cut) is refused; (2) a boundary tensor's are the body's
only: the cut moves them into the body and clears the parent's copy; (4) a model of
KernelOps the cut has not made is a body already, and the cut wraps it as one, its
parent stating none (the harness's parity check cuts it so). (3), several partitions,
waits for several partitions designed as one network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from onnx import StringStringEntryProto, TensorProto, helper
from qonnx.core.metadata import MetadataError
from qonnx.core.modelwrapper import ModelWrapper
from support.helpers import Lanes

from finn.core.graph.annotations.channel import (
    CHANNEL,
    CHANNEL_KEYS,
    channel_choices,
    clear_channel_choices,
)
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import KERNEL_OPS_DOMAIN, partition_body
from finn.core.space import inspection
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import network, persist
from finn.custom_op.kernels.shell import save_channels
from finn.kernels.channels import Channel
from finn.kernels.explore import ExploreError, Pinned, Ranked, Rebaseline
from finn.transformation.kernels import explore_kernel_choices, kernel_choices_config
from finn.transformation.kernels.cut import BaselineChoices, CutKernelPartition
from kernel_ops.models import EDGE, kernel_model
from kernel_ops.test_choose import kernel_model as open_model
from kernel_ops.tfc import LANES

FIFO: dict[str, object] = {"transport": "fifo", "transport.fifo.buffer.depth": 8}


def entries(model: ModelWrapper, tensor: str) -> dict[str, str]:
    """The ``finn.channel`` entries ``tensor``'s annotation stores, as text."""
    return {
        entry.key: entry.value
        for annotation in model.graph.quantization_annotation
        if annotation.tensor_name == tensor
        for entry in annotation.quant_parameter_tensor_names
        if entry.key.startswith(f"{CHANNEL.name}/")
    }


def hosted(model: ModelWrapper) -> ModelWrapper:
    """``model`` with a node on the host after its output: y -> Relu ``host`` -> z."""
    (y,) = model.graph.output
    model.graph.value_info.append(y)
    del model.graph.output[:]
    model.graph.output.append(helper.make_tensor_value_info("z", TensorProto.FLOAT, None))
    model.graph.node.append(helper.make_node("Relu", ["y"], ["z"], name="host"))
    return model


def test_the_namespace_is_generated_from_the_channels_decisions() -> None:
    """One key per Decision ``Channel`` declares, typed by its value semantics, a
    selector's values its cases; it follows its tensor into a cut."""
    assert list(CHANNEL_KEYS) == [item.key for item in inspection.decisions(Channel)]
    assert len(CHANNEL_KEYS) == 13
    assert (CHANNEL.name, CHANNEL.version, CHANNEL.follow) == ("finn.channel", 1, True)
    assert CHANNEL_KEYS["transport.fifo.buffer.depth"].encode(8) == "8"
    assert CHANNEL_KEYS["source.memstream.pumped_memory"].encode(True) == "true"
    with pytest.raises(MetadataError, match="cannot store '8': expected an int"):
        CHANNEL_KEYS["transport.fifo.buffer.depth"].encode("8")
    with pytest.raises(MetadataError, match=r"one of \['direct', 'fifo'\]"):
        CHANNEL_KEYS["transport"].encode("dense")


def test_a_channels_choices_are_typed_entries_of_its_tensor() -> None:
    model = kernel_model()
    save_channels(model, {"levels": FIFO})
    assert entries(model, "levels") == {
        "finn.channel/@version": "1",
        "finn.channel/@follow": "true",
        "finn.channel/transport": "fifo",
        "finn.channel/transport.fifo.buffer.depth": "8",
    }
    assert channel_choices(model, "levels") == FIFO
    # Beside qonnx's own annotation of the tensor, which it leaves alone.
    assert model.get_tensor_datatype("levels").name == "UINT2"
    # A node holds its kernel's choices only.
    attributes = {attribute.name for node in model.graph.node for attribute in node.attribute}
    assert not any(name.partition(".")[0] in ("x", "w", "y") for name in attributes)
    assert network(model, model.graph.node).member("levels").transport.buffer.depth == 8


def test_the_writer_is_checked_by_replaying_the_network() -> None:
    """Only the network holds both of a channel's ends: a choice it refuses, a key no
    channel declares, a value of another type and a tensor that is no channel are
    refused, by name, and nothing is written; ``None`` clears a choice."""
    model = kernel_model()
    before = kernel_choices_config(model)
    refused: dict[str, dict[str, dict[str, object]]] = {
        r"levels: refused choices: transport.fifo.buffer.depth: .*domain-membership": {
            "levels": {**FIFO, "transport.fifo.buffer.depth": 1}
        },
        r"x: refused choices: adapter.input_gen_vpc.input_gen.ram_style: inapplicable": {
            "x": {"adapter.input_gen_vpc.input_gen.ram_style": "auto"}
        },
        r"levels: \['memory'\] are not a channel's choices \(finn.channel\)": {
            "levels": {"memory": "auto"}
        },
        r"levels: finn.channel/transport.fifo.buffer.depth: cannot store '8'": {
            "levels": {"transport.fifo.buffer.depth": "8"}
        },
        r"thresholds: no channel of the partition": {"thresholds": EDGE},
    }
    for match, choices in refused.items():
        with pytest.raises(KernelOpError, match=match):
            save_channels(model, {"hidden": FIFO, **choices})
        assert kernel_choices_config(model) == before
    save_channels(model, {"levels": {"transport": None}})
    assert "levels" not in kernel_choices_config(model)


def test_a_graph_with_nodes_on_the_host_states_no_channel_choice(tmp_path: Path) -> None:
    """Case 1: channel choices stated on a graph with a node on the host (the whole graph,
    before the cut) are refused by name, by every reader and writer and by the cut; the
    build's come in through kernel_choices.json (Pinned), applied to the body."""
    model = hosted(kernel_model())
    nodes = [node for node in model.graph.node if node.domain == KERNEL_OPS_DOMAIN]
    outside = r"channel choices \(finn.channel\) are stated in a partition's body only, not"
    hosts = r"on a graph with nodes on the host \(host\)"
    with pytest.raises(KernelOpError, match=rf"x, .*: {outside} {hosts}"):
        network(model, nodes)
    with pytest.raises(KernelOpError, match=f"^x: {outside}"):
        save_channels(model, {"x": EDGE})
    # A node replays its kernel's choices only: its weight channel's are the network's.
    assert kernel_op(model, nodes[0]).verify_node() == []
    with pytest.raises(KernelOpError, match=outside):
        kernel_choices_config(model)
    with pytest.raises(KernelOpError, match=outside):
        model.transform(CutKernelPartition(tmp_path))
    # Stated nowhere, the graph's network replays the nodes' choices as before.
    for tensor in model.tensors_stating(CHANNEL):
        model.clear(CHANNEL, tensor=tensor)
    assert not network(model, nodes).stale


def test_the_cut_moves_a_channels_choices_into_the_body(tmp_path: Path) -> None:
    """Cases 2 and 4: cutting a model of KernelOps, which is a body already, moves its
    channels' choices into the body the cut makes, its boundary's (x, y) too; the parent's copy of
    a boundary tensor states none, and keeps its datatype. The cut keeps every value the
    model states and gives what it leaves open its baseline (an adapter's memory, a
    FIFO's style)."""
    model = kernel_model()
    save_channels(model, {"levels": FIFO})
    stated = kernel_choices_config(model)
    assert {"x", "y", "levels"} <= set(stated)
    parent = model.transform(CutKernelPartition(tmp_path))
    assert parent.tensors_stating(CHANNEL) == []
    assert not any(entries(parent, tensor) for tensor in ("x", "y"))
    assert parent.get_tensor_datatype("x") == model.get_tensor_datatype("x")
    _, body, _ = partition_body(parent)
    config = kernel_choices_config(body)
    assert list(config) == list(stated)
    assert all(config[name].items() >= held.items() for name, held in stated.items())
    assert config["levels"] == {
        **FIFO,
        "adapter.input_gen.input_gen.ram_style": "distributed",
        "transport.fifo.buffer.ram_style": "auto",
    }
    assert not network(body, body.graph.node).stale


def test_baseline_choices_does_to_a_body_made_by_hand_what_the_cut_does(tmp_path: Path) -> None:
    """A body made by hand (the Chain's KernelOps, converted, no cut) states its nodes'
    choices, created at conversion, and no channel's. ``BaselineChoices`` creates each
    channel's at its baseline on its tensor, keeps the nodes', and leaves nothing for
    the exploration to baseline; a node's choice its facts refuse in context (written
    past ``save``) is dropped, named, and re-baselined, as the cut does."""
    model = open_model()
    nodes = kernel_choices_config(model)
    assert list(nodes) == ["first", "activate", "second"] and not model.tensors_stating(CHANNEL)
    baselining = BaselineChoices()
    model = model.transform(baselining)
    assert baselining.dropped == {}
    config = kernel_choices_config(model)
    assert list(config) == ["x", "w1", "first", "hidden", "activate", "levels", "w2", "second", "y"]
    assert {name: config[name] for name in nodes} == nodes
    assert all(held["transport"] == "direct" for name, held in config.items() if name not in nodes)
    assert explore_kernel_choices(model).report["baselined"] == []
    first = kernel_op(model, model.graph.node[0])
    first.set_nodeattr("compute.packed.simd", 2)
    first.set_nodeattr("compute.packed.pe", 3)
    model = model.transform(baselining)
    # Its input buffer's memory, which reads the folding, waits with it: the channel's
    # ends wait on the reopened pe, and its stored memory is replayed once the baseline
    # chooses one, where it still applies.
    assert sorted(baselining.dropped) == ["first.compute.packed.pe"]
    assert "domain-membership" in baselining.dropped["first.compute.packed.pe"]
    first = kernel_op(model, model.graph.node[0])
    assert first.choices() == {
        **nodes["first"],
        "compute.packed.simd": 2,
        "compute.packed.compute_pumping": 0,
    }
    # The cut is the same step: the body it makes from the model holds the same values.
    parent = model.transform(CutKernelPartition(tmp_path))
    assert kernel_choices_config(partition_body(parent)[1]) == kernel_choices_config(model)


def test_the_baseline_strategy_resets_the_bodys_channel_choices_too() -> None:
    """To start over, the ``baseline`` strategy: what the body states, nodes' and
    channels' alike, goes back to its baseline, the values a body stating nothing is
    baselined to."""
    model = kernel_model()
    assert model.tensors_stating(CHANNEL)
    explored = explore_kernel_choices(model, Rebaseline())
    reset = explored.report["strategy"]
    assert reset["strategy"] == "baseline" and reset["changed"]
    cleared = kernel_model()
    for node in cleared.graph.node:
        op = kernel_op(cleared, node)
        op.save(dict.fromkeys(op.choices()))
    clear_channel_choices(cleared)
    explore_kernel_choices(cleared)
    assert kernel_choices_config(model) == kernel_choices_config(cleared)


def test_the_export_names_nodes_and_tensors_and_pinned_reads_it(tmp_path: Path) -> None:
    """kernel_choices.json keeps its shape, {graph name: {key: value}}: the nodes' and
    the tensors' channel choices, in graph order. Pinned commits it on another body, and
    the report names each choice's owner the same way."""
    model = open_model()
    explore_kernel_choices(model, Ranked(Lanes(2)))
    config = kernel_choices_config(model)
    assert list(config) == ["x", "w1", "first", "hidden", "activate", "levels", "w2", "second", "y"]
    assert config["first"]["compute.packed.pe"] == 2
    assert config["w1"]["source.memstream.pumped_memory"] is False
    path = tmp_path / "kernel_choices.json"
    path.write_text(json.dumps(config))
    blank = open_model()
    report = explore_kernel_choices(blank, Pinned(path)).report
    assert kernel_choices_config(blank) == config
    # Every value is held, by its owner as the file names it; what it changed is logged
    # as the pin's.
    assert report["held"] == config
    assert report["choices"]["first"]["compute.packed.pe"] == {"value": 2, "by": "pinned"}
    assert report["choices"]["levels"]["transport"]["value"] == config["levels"]["transport"]
    # A channel's key under its consumer is no Decision of the root: D8's file is
    # refused, named.
    path.write_text(json.dumps({"second": {"x.transport": "direct"}}))
    with pytest.raises(ExploreError, match="second.x.transport: not a Decision of this network"):
        explore_kernel_choices(open_model(), Pinned(path))


def test_the_export_is_in_graph_order_on_tfcs_body(tfc: ModelWrapper) -> None:
    """kernel_choices.json as it reads, on TFC_W2A2's body at 16 lanes: each node's input
    channels not yet written (its weights, the parameter a MatMul owns, among them), the
    node, its output channel; a MultiThreshold's thresholds state no channel."""
    model = ModelWrapper(tfc.model.__deepcopy__())
    explore_kernel_choices(model, Ranked(LANES))
    expected = ["Reshape_0_out0"]
    for layer in range(4):
        threshold, matmul = f"MultiThreshold_{layer}", f"MatMul_{layer}"
        expected += [threshold, f"{threshold}_out0", f"{matmul}_param0", matmul, f"{matmul}_out0"]
    assert list(kernel_choices_config(model)) == expected


def test_the_export_does_not_follow_the_order_annotations_were_written_in() -> None:
    """Two bodies with the same choices, their tensors' annotations written in opposite
    orders, export the same file, in graph order; a tensor that states choices and that
    no node reads or writes comes last, by name."""
    model, reordered = kernel_model(), kernel_model()
    annotations = list(reordered.graph.quantization_annotation)
    del reordered.graph.quantization_annotation[:]
    reordered.graph.quantization_annotation.extend(reversed(annotations))
    assert reordered.tensors_stating(CHANNEL) == model.tensors_stating(CHANNEL)[::-1]
    assert json.dumps(kernel_choices_config(reordered)) == json.dumps(kernel_choices_config(model))
    for each in (model, reordered):
        for name in ("unread_b", "unread_a"):
            each.graph.value_info.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])
            )
            each.set(CHANNEL_KEYS["transport"], "direct", tensor=name)
    exported = kernel_choices_config(reordered)
    assert list(exported)[-2:] == ["unread_a", "unread_b"]
    assert json.dumps(exported) == json.dumps(kernel_choices_config(model))


def test_the_export_refuses_a_node_named_like_a_stating_tensor() -> None:
    """A node and a tensor that states choices share a name: refused, by name, though the
    node holds no choice (the ``{graph name: ...}`` form would name both once)."""
    model = open_model()
    model.set(CHANNEL_KEYS["transport"], "direct", tensor="hidden")
    (activate,) = [node for node in model.graph.node if node.name == "activate"]
    op = kernel_op(model, activate)
    op.save(dict.fromkeys(op.choices()))
    assert not op.choices()
    activate.name = "hidden"
    with pytest.raises(KernelOpError, match="hidden: a node and a tensor are both named hidden"):
        kernel_choices_config(model)


def test_a_body_stated_under_the_nodes_channel_attributes_is_refused_by_name() -> None:
    """A clean break: a channel's choice held on a node (D8's ``x.transport``) is no
    choice of the op, and a key ``Channel`` does not declare on a tensor is no channel
    choice; either way the body is explored again."""
    model = kernel_model()
    model.graph.node[2].attribute.append(helper.make_attribute("x.transport", "direct"))
    with pytest.raises(
        KernelOpError,
        match="second: x.transport is not a choice of MatMul; a channel's choices are its tensor's",
    ):
        network(model, model.graph.node)
    model = kernel_model()
    (annotation,) = [
        each for each in model.graph.quantization_annotation if each.tensor_name == "levels"
    ]
    annotation.quant_parameter_tensor_names.append(
        StringStringEntryProto(key="finn.channel/x.transport", value="direct")
    )
    with pytest.raises(KernelOpError, match=r"levels: .*\['x.transport'\] are not declared.*again"):
        network(model, model.graph.node)


def test_choices_on_a_tensor_that_is_no_channel_are_dropped_and_cleared() -> None:
    """Written past the writer, on Thresholding's table (a fact of its kernel, no
    channel): the network drops them, with why, and persisting clears them."""
    model = kernel_model()
    model.set(CHANNEL_KEYS["transport"], "direct", tensor="thresholds")
    bound = network(model, model.graph.node)
    assert dict(bound.stale) == {
        "thresholds.transport": "stated on a tensor that is no channel of the partition"
    }
    persist(model, bound)
    assert "thresholds" not in model.tensors_stating(CHANNEL)
    assert not network(model, model.graph.node).stale
