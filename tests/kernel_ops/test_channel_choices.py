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
from qonnx.core.datatype import DataType
from qonnx.core.metadata import MetadataError
from qonnx.core.modelwrapper import ModelWrapper
from support.configure import configured, write_raw
from support.explore_models import kernel_model as open_model
from support.helpers import Lanes
from support.models import EDGE, kernel_model, matmul_model
from support.refusals import refused
from support.tfc import LANES

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
from finn.kernels.channels import Channel
from finn.kernels.explore import Defaults, ExploreError, Hold, Ranked
from finn.transformation.kernels import (
    InferKernelTensors,
    explore_kernel_choices,
    kernel_choices_config,
)
from finn.transformation.kernels.cut import CutKernelPartition, MakeCutChoices

FIFO: dict[str, object] = {"transport": "fifo", "transport.fifo.depth": 8}


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
    assert len(CHANNEL_KEYS) == 6
    # A channel's source, mover and output adapter are placed, and its input adapter
    # picked by its steps: no choices. Its input adapter's memory is one choice.
    assert not {"source", "mover", "output_adapter", "adapter"} & set(CHANNEL_KEYS)
    assert [key for key in CHANNEL_KEYS if "adapter" in key] == ["adapter_ram_style"]
    assert (CHANNEL.name, CHANNEL.version, CHANNEL.follow) == ("finn.channel", 1, True)
    assert CHANNEL_KEYS["transport.fifo.depth"].encode(8) == "8"
    assert CHANNEL_KEYS["source.pumped_memory"].encode(True) == "true"
    with pytest.raises(MetadataError, match="cannot store '8': expected an int"):
        CHANNEL_KEYS["transport.fifo.depth"].encode("8")
    with pytest.raises(MetadataError, match=r"one of \['direct', 'fifo'\]"):
        CHANNEL_KEYS["transport"].encode("dense")


def test_a_channels_choices_are_typed_entries_of_its_tensor() -> None:
    """Each choice an entry of its tensor, typed: those set, and those the cut's policies
    make beside them (its adapter's memory, and the FIFO's style)."""
    model = kernel_model()
    configured(model, {"levels": FIFO})
    assert entries(model, "levels") == {
        "finn.channel/@version": "1",
        "finn.channel/@follow": "true",
        "finn.channel/adapter_ram_style": "distributed",
        "finn.channel/transport": "fifo",
        "finn.channel/transport.fifo.depth": "8",
        "finn.channel/transport.fifo.buffer.ram_style": "auto",
    }
    assert channel_choices(model, "levels") == {
        **FIFO,
        "adapter_ram_style": "distributed",
        "transport.fifo.buffer.ram_style": "auto",
    }
    # Beside qonnx's own annotation of the tensor, which it leaves alone.
    assert model.get_tensor_datatype("levels").name == "UINT2"
    # A node holds its kernel's choices only.
    attributes = {attribute.name for node in model.graph.node for attribute in node.attribute}
    assert not any(name.partition(".")[0] in ("x", "w", "y") for name in attributes)
    assert network(model, model.graph.node).member("levels").transport.depth == 8


def test_the_writer_is_checked_by_replaying_the_network() -> None:
    """Only the network holds both sides of a channel: a choice it refuses, a key no
    channel declares, a value of another type and a tensor that is no channel are
    refused, by the seam's code, and nothing is written; a choice is cleared on its tensor (the
    seam has no clearing). (A choice under an adapter not selected is no refusal: it is
    stored, out of play.)"""
    model = kernel_model()
    before = kernel_choices_config(model)
    rejections: list[tuple[str, str, dict[str, dict[str, object]]]] = [
        (
            "domain-membership",
            "transport.fifo.depth",
            {"levels": {**FIFO, "transport.fifo.depth": 1}},
        ),
        ("not-a-decision", "levels.memory", {"levels": {"memory": "auto"}}),
        (
            "request-refused",
            "transport.fifo.depth",
            {"levels": {"transport.fifo.depth": "8"}},
        ),
        ("member-unknown", "configured", {"thresholds": EDGE}),
    ]
    for code, owner, choices in rejections:
        with refused(code, owner=owner):
            configured(model, {"hidden": FIFO, **choices})
        assert kernel_choices_config(model) == before
    model.clear(CHANNEL, tensor="levels")  # every choice it states; the seam has no clearing
    assert "levels" not in kernel_choices_config(model)


def test_a_graph_with_nodes_on_the_host_states_no_channel_choice(tmp_path: Path) -> None:
    """Case 1: channel choices stated on a graph with a node on the host (the whole graph,
    before the cut) are refused by name, by every reader and writer and by the cut; the
    build's come in as holds (``Hold``), applied to the body."""
    model = hosted(kernel_model())
    nodes = [node for node in model.graph.node if node.domain == KERNEL_OPS_DOMAIN]
    outside = r"channel choices \(finn.channel\) are stated in a partition's body only, not"
    hosts = r"on a graph with nodes on the host \(host\)"
    every = ("x", "w1", "w2", "hidden", "levels", "y")
    with refused(
        "channel-choices-outside-body",
        error=KernelOpError,
        message=rf"{outside} {hosts}",
        keys=every,
    ):
        network(model, nodes)
    # The helper no longer asks up front (``persist`` does): this model states channel
    # choices on every tensor, so the bind it starts from refuses them all, as above.
    with refused("channel-choices-outside-body", error=KernelOpError, keys=every):
        configured(model, {"x": EDGE})
    # A node replays its kernel's choices only: its weight channel's are the network's.
    assert kernel_op(model, nodes[0]).verify_node() == []
    with refused("channel-choices-outside-body", error=KernelOpError):
        kernel_choices_config(model)
    with refused("channel-choices-outside-body", error=KernelOpError):
        model.transform(CutKernelPartition(tmp_path))
    # Stated nowhere, the graph's network replays the nodes' choices as before.
    for tensor in model.tensors_stating(CHANNEL):
        model.clear(CHANNEL, tensor=tensor)
    assert not network(model, nodes).stale


def test_the_cut_moves_a_channels_choices_into_the_body(tmp_path: Path) -> None:
    """Cases 2 and 4: cutting a model of KernelOps, which is a body already, moves its
    channels' choices into the body the cut makes, its boundary's (x, y) too; the parent's copy of
    a boundary tensor states none, and keeps its datatype. The cut keeps every value the
    model states; what it leaves open the cut's policies set (here nothing: the
    helper made them, the adapter's memory and the FIFO's style)."""
    model = kernel_model()
    configured(model, {"levels": FIFO})
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
        "adapter_ram_style": "distributed",
        "transport.fifo.buffer.ram_style": "auto",
    }
    assert not network(body, body.graph.node).stale


def test_make_cut_choices_does_to_a_body_made_by_hand_what_the_cut_does(tmp_path: Path) -> None:
    """A body made by hand (the Chain's KernelOps, converted, no cut) states its nodes'
    choices, created at conversion, and no channel's. ``MakeCutChoices`` creates each
    channel's by the cut's policies on its tensor, keeps the nodes', and leaves
    nothing open; a node's choice its facts refuse in context (written raw) is dropped,
    named, and left open for a strategy (emission refuses it until one sets it), as the
    cut does."""
    model = open_model()
    nodes = kernel_choices_config(model)
    assert list(nodes) == ["first", "activate", "second"] and not model.tensors_stating(CHANNEL)
    making = MakeCutChoices()
    model = model.transform(making)
    assert making.dropped == {}
    config = kernel_choices_config(model)
    assert list(config) == ["x", "w1", "first", "hidden", "activate", "levels", "w2", "second", "y"]
    assert {name: config[name] for name in nodes} == nodes
    assert all(held["transport"] == "direct" for name, held in config.items() if name not in nodes)
    assert explore_kernel_choices(model).report["open"] == []
    write_raw(model, "first", "simd", 2)  # past the space, as another tool wrote it
    write_raw(model, "first", "pe", 3)
    model = model.transform(making)
    # Its input buffer's channel waits on the open pe, its stored choices with it.
    assert sorted(making.dropped) == ["first.pe"]
    # Its own domain refuses 3: nothing else stands beside it.
    assert {(f.code, f.owner) for f in making.dropped["first.pe"]} == {
        ("domain-membership", "pe"),
    }
    first = kernel_op(model, model.graph.node[0])
    assert first.choices() == {
        **{key: value for key, value in nodes["first"].items() if key != "pe"},
        "simd": 2,
    }
    # The cut is the same step: the body it makes from the model holds the same values,
    # but for the PE left open, which a body that never had it set the cut sets late, by
    # conversion's policy (first valid, 1), reported.
    cut = CutKernelPartition(tmp_path)
    parent = model.transform(cut)
    assert list(cut.late) == ["first.pe"]
    expected = kernel_choices_config(model)
    expected["first"] = {**expected["first"], "pe": 1}
    assert kernel_choices_config(partition_body(parent)[1]) == expected


def test_a_stale_choice_stays_open_after_an_earlier_member_set_its_choices() -> None:
    """A node's stored choice the bind dropped as stale is left open for a strategy,
    whatever an earlier member of the body had left open for the cut to set: ``first``'s
    cleared SIMD is set late by conversion's policy (and a channel's choices by the cut's),
    ``second``'s PE past its domain stays dropped and open, not set late behind it."""
    model = open_model()
    write_raw(model, "first", "simd", None)
    write_raw(model, "second", "pe", 3)  # past the space, as another tool wrote it
    making = MakeCutChoices()
    model = model.transform(making)
    assert sorted(making.dropped) == ["second.pe"]
    assert sorted(making.late) == ["first.simd"]
    assert "pe" not in kernel_op(model, model.graph.node[-1]).choices()


def test_a_choice_no_value_can_be_set_for_is_left_open_at_the_cut_and_its_others_are_set() -> None:
    """INT32 activations: no core takes them, so a MatMul made by hand has no first valid
    core. The cut leaves ``compute`` open and reports it (``unset``, by key), and still sets
    the node's folding (late) and every channel's choices: the MatMul's other choices were
    all lost when the cut stopped at the first such choice."""
    model = matmul_model(annotate=(), infer=False)
    for name in ("x", "w"):
        model.set_tensor_datatype(name, DataType["INT32"])
    model = model.transform(InferKernelTensors())
    making = MakeCutChoices()
    model = model.transform(making)
    assert list(making.unset) == ["first.compute"]
    (finding,) = making.unset["first.compute"]
    assert (finding.code, finding.owner) == ("choice-no-valid-value", "compute")
    assert sorted(making.late) == ["first.pe", "first.simd"]
    config = kernel_choices_config(model)
    assert config["first"] == {"pe": 1, "simd": 1}
    assert {config[name]["transport"] for name in ("x", "w", "y")} == {"direct"}


def test_the_defaults_strategy_resets_the_bodys_channel_choices_too() -> None:
    """To start over, the ``Defaults`` strategy: what the body states, nodes' and
    channels' alike, goes back to its default, the values the cut gives a body stating
    nothing (``MakeCutChoices``: conversion's policies late, the cut's)."""
    model = kernel_model()
    assert model.tensors_stating(CHANNEL)
    explored = explore_kernel_choices(model, Defaults())
    reset = explored.report["strategy"]
    assert reset["strategy"] == "defaults" and reset["changed"]
    cleared = kernel_model()
    for node in cleared.graph.node:
        for key in list(kernel_op(cleared, node).choices()):
            write_raw(cleared, node.name, key, None)
    clear_channel_choices(cleared)
    cleared = cleared.transform(MakeCutChoices())
    assert kernel_choices_config(model) == kernel_choices_config(cleared)


def test_the_export_names_nodes_and_tensors_and_hold_takes_it() -> None:
    """The choices export keeps its shape, {graph name: {key: value}}: the nodes' and
    the tensors' channel choices, in graph order. ``Hold`` commits it on another body, and
    the report names each choice's owner the same way."""
    model = open_model()
    explore_kernel_choices(model, Ranked(Lanes(2)))
    config = kernel_choices_config(model)
    assert list(config) == ["x", "w1", "first", "hidden", "activate", "levels", "w2", "second", "y"]
    assert config["first"]["pe"] == 2
    assert config["w1"]["source.pumped_memory"] is False
    blank = open_model()
    report = explore_kernel_choices(blank, Hold(config)).report
    assert kernel_choices_config(blank) == config
    # Every value is held, by its owner as the export names it; what it changed is logged
    # as the hold's.
    assert report["held"] == config
    assert report["choices"]["first"]["pe"] == {"value": 2, "by": "held"}
    assert report["choices"]["levels"]["transport"]["value"] == config["levels"]["transport"]
    # A channel's key under its consumer is no Decision of the root: D8's form is
    # refused, named.
    with refused(
        "pin-refused", error=ExploreError, message=r"second\.x\.transport: not a Decision"
    ):
        explore_kernel_choices(open_model(), Hold({"second": {"x.transport": "direct"}}))


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
    for key in list(kernel_op(model, activate).choices()):
        write_raw(model, "activate", key, None)
    assert not kernel_op(model, activate).choices()
    activate.name = "hidden"
    with refused(
        "name-shared",
        error=KernelOpError,
        owner="kernel_choices_config",
        message="a node and a tensor are both named hidden",
        node="hidden",
        tensor="hidden",
    ):
        kernel_choices_config(model)


def test_a_body_stated_under_the_nodes_channel_attributes_is_refused_by_name() -> None:
    """A clean break: a channel's choice held on a node (D8's ``x.transport``) is no
    choice of the op, and a key ``Channel`` does not declare on a tensor is no channel
    choice; either way the body is explored again."""
    model = kernel_model()
    model.graph.node[2].attribute.append(helper.make_attribute("x.transport", "direct"))
    with refused(
        "choice-unknown",
        error=KernelOpError,
        node="second",
        keys=("x.transport",),
        message="is not a choice of MatMul; a channel's choices are its tensor's",
    ):
        network(model, model.graph.node)
    model = kernel_model()
    (annotation,) = [
        each for each in model.graph.quantization_annotation if each.tensor_name == "levels"
    ]
    annotation.quant_parameter_tensor_names.append(
        StringStringEntryProto(key="finn.channel/x.transport", value="direct")
    )
    with refused(
        "channel-choices-unreadable",
        error=KernelOpError,
        message=r"\['x.transport'\] are not declared.*again",
        tensor="levels",
    ):
        network(model, model.graph.node)


def test_a_row_saved_with_the_placements_as_choices_is_refused_by_name() -> None:
    """A clean break: a channel's source, mover and output adapter are placed, not
    chosen, and the source's choices are keyed below it (``source.ram_style``); a tensor
    that stores the old keys (``source``, ``source.memstream.ram_style``) states keys the
    namespace does not declare, so the body is explored again."""
    model = kernel_model()
    (annotation,) = [
        each for each in model.graph.quantization_annotation if each.tensor_name == "w1"
    ]
    annotation.quant_parameter_tensor_names.extend(
        (
            StringStringEntryProto(key="finn.channel/source", value="memstream"),
            StringStringEntryProto(key="finn.channel/source.memstream.ram_style", value="block"),
        )
    )
    with refused(
        "channel-choices-unreadable",
        error=KernelOpError,
        message=r"\['source', 'source.memstream.ram_style'\] are not declared.*again",
        tensor="w1",
    ):
        network(model, model.graph.node)


def test_a_row_saved_with_the_input_adapter_as_a_choice_is_refused_by_name() -> None:
    """A clean break: a channel's input adapter is picked by its steps, not chosen, and
    its memory is the channel's one style (``adapter_ram_style``); a tensor that stores the
    old keys (``adapter``, ``adapter.input_gen.input_gen.ram_style``) states keys the
    namespace does not declare, so the body is explored again."""
    model = kernel_model()
    (annotation,) = [
        each for each in model.graph.quantization_annotation if each.tensor_name == "levels"
    ]
    annotation.quant_parameter_tensor_names.extend(
        (
            StringStringEntryProto(key="finn.channel/adapter", value="input_gen"),
            StringStringEntryProto(
                key="finn.channel/adapter.input_gen.input_gen.ram_style", value="distributed"
            ),
        )
    )
    with refused(
        "channel-choices-unreadable",
        error=KernelOpError,
        message=r"\['adapter', 'adapter.input_gen.input_gen.ram_style'\] are not declared.*again",
        tensor="levels",
    ):
        network(model, model.graph.node)


def test_choices_on_a_tensor_that_is_no_channel_are_dropped_and_cleared() -> None:
    """Written past the writer, on Thresholding's table (a fact of its kernel, no
    channel): the network drops them, with why, and persisting clears them."""
    model = kernel_model()
    model.set(CHANNEL_KEYS["transport"], "direct", tensor="thresholds")
    bound = network(model, model.graph.node)
    ((key, (found,)),) = bound.stale.items()
    assert key == "thresholds.transport"
    assert (found.code, found.owner, dict(found.details)) == (
        "channel-choice-stray",
        "transport",
        {"tensor": "thresholds"},
    )
    persist(model, bound)
    assert "thresholds" not in model.tensors_stating(CHANNEL)
    assert not network(model, model.graph.node).stale


def test_a_channel_choice_out_of_play_is_saved_and_replayed_out_of_play() -> None:
    """A FIFO's depth stored while the channel's transport is open; the cut's policy
    takes ``direct``, the first: the depth is kept out of play (dormant), and saving
    writes it on the tensor, as it writes one in play. The model read again holds it
    dormant, back in play with a FIFO."""
    model = kernel_model()
    model.clear(CHANNEL, tensor="levels")
    configured(model, {"levels": {"transport.fifo.depth": 8}})
    assert entries(model, "levels")["finn.channel/transport.fifo.depth"] == "8"
    assert channel_choices(model, "levels")["transport"] == "direct"
    states = dict(network(model, model.graph.node).states(["levels"]))
    assert states[("levels", "transport.fifo.depth")] == inspection.Dormant(8)
