# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``configured``: choices set on a model the way the compiler sets them (``support.configure``).

The kernels' and the channels' values are set through the seam as a strategy sets them, in
one batch, and persisted (``persist``). What the seam refuses is refused by its own code;
a key forced to the value given is dropped (the compiler never stores a forced Decision),
and one forced to another is refused.
"""

from __future__ import annotations

from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from support.configure import configured, write_raw
from support.explore_models import kernel_model as open_chain
from support.helpers import FULL_DSP58
from support.models import (
    EDGE,
    MATMUL,
    THRESHOLDING,
    WEIGHT_CHANNEL,
    matmul_model,
    thresholding_model,
)
from support.refusals import refused

from finn.core.graph.annotations.channel import channel_choices
from finn.core.graph.annotations.platform import write_target
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import network
from finn.platform.target import Target
from finn.transformation.kernels import InferKernelTensors

FOLDED = {"compute.packed.pe": 2, "compute.packed.simd": 2, "compute.packed.reducer": "tree"}
THRESHOLDS_FOLDED = {"pe": 2, "ram_style": "distributed", "block_stages": 1}
VERSAL = Target(part="a part with UltraRAM it initializes", platform=FULL_DSP58, shell="ip")
"""A target whose MatMul ``compute`` is open (DSP58): nothing forces it."""


def stored(model: ModelWrapper, node: int = 0) -> dict[str, object]:
    """The choices a node stores, by attribute."""
    return kernel_op(model, model.graph.node[node]).choices()


def on_versal(model: ModelWrapper) -> ModelWrapper:
    write_target(model, VERSAL)
    return model.transform(InferKernelTensors())


def hosted(model: ModelWrapper) -> ModelWrapper:
    """``model`` with a node on the host after its output: y -> Relu ``host`` -> z."""
    (y,) = model.graph.output
    model.graph.value_info.append(y)
    del model.graph.output[:]
    model.graph.output.append(helper.make_tensor_value_info("z", TensorProto.FLOAT, None))
    model.graph.node.append(helper.make_node("Relu", ["y"], ["z"], name="host"))
    return model


# -- what it sets ------------------------------------------------------------------------


def test_a_kernels_choices_are_set_and_persisted_on_its_node() -> None:
    model = matmul_model()
    assert configured(model, {"first": FOLDED}) is model
    assert stored(model) == FOLDED
    assert network(model, model.graph.node).member("first").compute.pe == 2


def test_a_channels_choices_are_set_with_its_kernels_in_one_batch() -> None:
    """The weight channel's ends wait on the MatMul's folding: both are one batch, as a
    strategy's is, and the choices persist on the node and on the tensor."""
    model = matmul_model()
    configured(
        model,
        {"first": FOLDED, "w": {"source.memstream.ram_style": "block", "transport": "direct"}},
    )
    assert stored(model)["compute.packed.pe"] == 2
    assert channel_choices(model, "w") == {
        "source.memstream.ram_style": "block",
        "transport": "direct",
    }
    assert network(model, model.graph.node).member("w").source.ram_style == "block"
    # What a later call does not name stays.
    configured(model, {"w": {"transport": "fifo", "transport.fifo.buffer.depth": 8}})
    assert channel_choices(model, "w")["source.memstream.ram_style"] == "block"
    assert channel_choices(model, "w")["transport.fifo.buffer.depth"] == 8
    assert stored(model)["compute.packed.pe"] == 2


def test_the_chain_is_configured_as_the_compiler_states_it() -> None:
    """Every kernel and channel of a partition's body in one call: the same committed
    choices as the model the tests built with ``save`` and ``save_channels``."""
    model = open_chain()
    channels = {"x": EDGE, "w1": WEIGHT_CHANNEL, "hidden": EDGE, "levels": EDGE, "y": EDGE}
    configured(
        model,
        {"first": MATMUL, "activate": THRESHOLDING, "second": MATMUL, **channels},
    )
    bound = network(model, model.graph.node)
    assert not bound.stale
    assert bound.member("first").compute.pe == MATMUL["compute.packed.pe"]
    assert {key for key in bound.chosen() if key[0] in channels} >= {
        ("w1", "transport"),
        ("levels", "transport"),
    }
    assert "compute" not in stored(model)


# -- a forced key ------------------------------------------------------------------------


def test_a_key_forced_to_its_own_value_is_accepted_and_not_stored() -> None:
    """On DSP48E2 ``compute`` is forced to ``packed``, and a Thresholding's ``use_axilite``
    to False: the compiler never stores a forced Decision, so a set-up that lists them
    states what the compiler would."""
    model = matmul_model()
    configured(model, {"first": MATMUL})
    assert "compute" not in stored(model)
    assert stored(model)["compute.packed.pe"] == MATMUL["compute.packed.pe"]
    model = thresholding_model()
    configured(model, {"activate": {**THRESHOLDING, "ram_style": "distributed", "block_stages": 1}})
    assert {"use_axilite", "ultra_stages"}.isdisjoint(stored(model))
    assert stored(model)["pe"] == THRESHOLDING["pe"]


def test_a_key_forced_to_another_value_is_refused() -> None:
    with refused("forced", owner="first.compute"):
        configured(matmul_model(), {"first": {"compute": "int8_dsp58"}})
    with refused("forced", owner="activate.use_axilite"):
        configured(thresholding_model(), {"activate": {"use_axilite": True}})


def test_a_key_that_is_open_on_the_target_is_set() -> None:
    model = on_versal(matmul_model())
    configured(model, {"first": {"compute": "packed", **FOLDED}})
    assert stored(model)["compute"] == "packed"


# -- what the seam refuses ----------------------------------------------------------------


def test_a_key_no_decision_declares_is_refused_by_name() -> None:
    """A typo, and a channel's key given to a node (a channel's keys are its tensor's)."""
    model = matmul_model()
    with refused("not-a-decision", owner="first.compute.packed.typo"):
        configured(model, {"first": {"compute.packed.typo": 1}})
    with refused("not-a-decision", owner="first.w.transport"):
        configured(model, {"first": {"w.transport": "direct"}})
    with refused("not-a-decision", owner="w.memory"):
        configured(model, {"first": FOLDED, "w": {"memory": "auto"}})
    assert stored(model) == {}


def test_a_value_the_decision_does_not_take_is_refused() -> None:
    """On DSP58 ``compute`` is a choice, with two cases; a value of another type or outside
    them is not viable."""
    model = on_versal(matmul_model())
    with refused("case-not-viable", owner="first.compute"):
        configured(model, {"first": {"compute": 1}})
    with refused("case-not-viable", owner="first.compute"):
        configured(model, {"first": {"compute": "dense"}})
    assert stored(model) == {}


def test_a_value_outside_a_domain_is_refused() -> None:
    """3 does not divide the channels: the seam's, when the choice is open
    (``case-not-viable``), and the engine's once the folding is batched with it
    (``domain-membership``)."""
    model = matmul_model()
    with refused("case-not-viable", owner="first.compute.packed.pe"):
        configured(model, {"first": {"compute.packed.pe": 3}})
    configured(model, {"first": FOLDED})
    with refused("domain-membership", owner="compute.packed.pe"):
        configured(model, {"first": {"compute.packed.pe": 3}})
    assert stored(model)["compute.packed.pe"] == 2
    with refused("case-not-viable", owner="activate.pe"):
        configured(thresholding_model(), {"activate": {"pe": 3}})


def test_a_nested_choice_under_a_case_not_selected_is_refused() -> None:
    model = matmul_model()
    with refused("inapplicable", owner="compute.int8_dsp58.pe"):
        configured(model, {"first": {"compute.int8_dsp58.pe": 2}})
    assert stored(model) == {}


def test_a_stored_memory_is_checked_with_the_choice_beside_it() -> None:
    """Block stages beyond what the memory holds is refused once the memory is stored,
    and nothing is written."""
    model = thresholding_model()
    configured(model, {"activate": THRESHOLDS_FOLDED})
    before = stored(model)
    with refused("domain-membership", owner="block_stages") as caught:
        configured(model, {"activate": {"block_stages": 3}})
    assert caught.finding.code == "domain-membership"
    assert stored(model) == before


def test_an_owner_that_is_no_member_is_refused() -> None:
    """A node or a tensor that is not in the network (``Hold``'s own refusal, ``keyed``)."""
    with refused("member-unknown", owner="configured"):
        configured(matmul_model(), {"nothing": {"transport": "direct"}})
    with refused("member-unknown", owner="configured"):
        configured(matmul_model(), {"thresholds": EDGE})


def test_a_channels_value_of_another_type_is_refused() -> None:
    model = matmul_model()
    configured(model, {"first": FOLDED})
    with refused("request-refused", owner="transport.fifo.buffer.depth"):
        configured(model, {"w": {"transport": "fifo", "transport.fifo.buffer.depth": "8"}})
    assert channel_choices(model, "w") == {}


def test_a_channel_whose_ends_wait_on_an_open_kernel_is_refused() -> None:
    """The MatMul is not folded, so the weight channel's ends are open and its choice
    cannot be checked."""
    model = matmul_model()
    with refused("decision-unassigned", owner="compute.packed.pe"):
        configured(model, {"w": {"source.memstream.ram_style": "ultra"}})
    assert channel_choices(model, "w") == {}


def test_a_memory_the_device_cannot_build_is_refused() -> None:
    """Ultra96 has no UltraRAM: batched with the folding the platform's own finding, once
    the folding is stored, the seam's (``case-not-viable``)."""
    ultra = {"w": {"source.memstream.ram_style": "ultra"}}
    model = matmul_model()
    with refused("uram-absent", owner="source.memstream.ram_style"):
        configured(model, {"first": FOLDED, **ultra})
    configured(model, {"first": FOLDED})
    with refused("case-not-viable", owner="w.source.memstream.ram_style"):
        configured(model, ultra)
    assert channel_choices(model, "w") == {}


def test_a_channel_choice_on_a_graph_with_host_nodes_is_refused() -> None:
    """Channel choices are a partition's body's only; a kernel's choices are set."""
    model = hosted(matmul_model())
    configured(model, {"first": FOLDED})
    with refused("channel-choices-outside-body", keys=("w",)):
        configured(model, {"w": WEIGHT_CHANNEL})
    assert channel_choices(model, "w") == {}


# -- clearing ----------------------------------------------------------------------------


def test_none_is_no_value_and_the_seam_has_no_clearing() -> None:
    """Clearing is not supported: ``None`` is refused as a value no case holds, and
    nothing is written. A test that needs a cleared state writes the raw attribute."""
    model = matmul_model()
    configured(model, {"first": FOLDED})
    before = stored(model)
    with refused("request-refused", owner="compute.packed.simd"):
        configured(model, {"first": {"compute.packed.simd": None}})
    assert stored(model) == before


# -- the raw attribute -------------------------------------------------------------------


def test_write_raw_stands_for_a_graph_finn_did_not_write() -> None:
    """A choice as the node's raw attribute, replaced when there, removed for ``None``;
    nothing is checked (a divisor-less ``pe`` is written), the read side finds it."""
    model = matmul_model()
    write_raw(model, "first", "compute.packed.pe", 3)
    write_raw(model, "first", "compute.packed.reducer", "tree")
    assert stored(model) == {"compute.packed.pe": 3, "compute.packed.reducer": "tree"}
    write_raw(model, "first", "compute.packed.pe", 2)
    assert stored(model)["compute.packed.pe"] == 2
    write_raw(model, "first", "compute.packed.pe", None)
    write_raw(model, "first", "compute.packed.simd", None)  # none there: nothing
    assert stored(model) == {"compute.packed.reducer": "tree"}
    assert [attribute.name for attribute in model.graph.node[0].attribute] == [
        "compute.packed.reducer"
    ]
