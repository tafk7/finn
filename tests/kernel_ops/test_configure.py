# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``configured``: choices set on a model the way the compiler sets them (``support.configure``).

The kernels' and the channels' values are set through the seam as a strategy sets them, in
one batch, each channel's other choices then by the interim channel rule (on a partition's
body, as the cut sets them), and persisted (``persist``). What the seam refuses is refused
by its code: the seam's own, or the engine's (a value outside its domain, a rule broken).
Nothing is forced: a set-up names each choice it needs, a core with its folding.
"""

from __future__ import annotations

from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from support.configure import configured, write_raw
from support.explore_models import kernel_model as open_chain
from support.helpers import FULL_DSP58, choose
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
from finn.core.space import inspection
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import network, persist
from finn.custom_op.kernels.shell import in_shell
from finn.platform.target import Target
from finn.transformation.kernels import InferKernelTensors

FOLDED = {
    "compute": "packed",
    "pe": 2,
    "simd": 2,
    "compute.packed.reducer": "tree",
}
THRESHOLDS_FOLDED = {"pe": 2, "ram_style": "distributed", "block_stages": 1}
VERSAL = Target(part="a part with UltraRAM it initializes", platform=FULL_DSP58, shell="ip")
"""A target on which a MatMul's ``compute`` has two possible values (DSP58)."""


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
    """The weight channel's contracts wait on the MatMul's folding: both are one batch, as a
    strategy's is, the channel's other choices set by the interim channel rule, and the
    choices persist on the node and on the tensor."""
    model = matmul_model()
    configured(
        model,
        {"first": FOLDED, "w": {"source.memstream.ram_style": "block", "transport": "direct"}},
    )
    assert stored(model)["pe"] == 2
    assert channel_choices(model, "w") == {
        "source": "memstream",
        "source.memstream.ram_style": "block",
        "source.memstream.pumped_memory": False,
        "transport": "direct",
    }
    assert network(model, model.graph.node).member("w").source.ram_style == "block"
    # What a later call does not name stays.
    configured(model, {"w": {"transport": "fifo", "transport.fifo.depth": 8}})
    assert channel_choices(model, "w")["source.memstream.ram_style"] == "block"
    assert channel_choices(model, "w")["transport.fifo.depth"] == 8
    assert stored(model)["pe"] == 2


def test_the_chain_is_configured_as_the_compiler_states_it() -> None:
    """Every kernel and channel of a partition's body in one call: the network replays
    them whole, none stale, conversion's core stored with them."""
    model = open_chain()
    channels = {"x": EDGE, "w1": WEIGHT_CHANNEL, "hidden": EDGE, "levels": EDGE, "y": EDGE}
    configured(
        model,
        {"first": MATMUL, "activate": THRESHOLDING, "second": MATMUL, **channels},
    )
    bound = network(model, model.graph.node)
    assert not bound.stale
    assert bound.member("first").compute.pe == MATMUL["pe"]
    assert {key for key in bound.chosen() if key[0] in channels} >= {
        ("w1", "transport"),
        ("levels", "transport"),
    }
    assert stored(model)["compute"] == "packed"


# -- a value one possible value is -----------------------------------------------------------


def test_a_choice_with_one_possible_value_is_stored_as_any() -> None:
    """On DSP48E2 ``packed`` is the one possible core, and a Thresholding without a control
    bus takes no AXI-Lite: each is a choice as any, set and stored where a set-up names
    it."""
    model = matmul_model()
    configured(model, {"first": MATMUL})
    assert stored(model)["compute"] == "packed"
    assert stored(model)["pe"] == MATMUL["pe"]
    model = thresholding_model()
    configured(model, {"activate": {**THRESHOLDING, "ram_style": "distributed", "block_stages": 1}})
    assert (stored(model)["use_axilite"], stored(model)["ultra_stages"]) == (0, 0)
    assert stored(model)["pe"] == THRESHOLDING["pe"]


def test_a_whole_folding_is_one_batch() -> None:
    """On the converted Chain, a folding at SIMD 2 with pumping off: one call sets both
    (pumping's rule reads SIMD, set in the same batch)."""
    model = open_chain()
    configured(model, {"first": MATMUL})
    pumping = "compute.packed.compute_pumping"
    assert stored(model)[pumping] == MATMUL[pumping]
    assert stored(model)["compute"] == "packed"
    assert stored(model)["simd"] == MATMUL["simd"]


def test_a_value_a_rule_refuses_is_refused_by_the_rule() -> None:
    """The INT8 core on DSP48E2, AXI-Lite without a control bus: each rule refuses it,
    by its code."""
    with refused("dotp-target", owner="compute.int8_dsp58.core_supported"):
        configured(matmul_model(), {"first": {"compute": "int8_dsp58"}})
    with refused("threshold-control", owner="control_for_writes"):
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
    """On DSP58 ``compute`` is a choice, with two cases; a value of another type, or outside
    them, the engine refuses."""
    model = on_versal(matmul_model())
    with refused("request-refused", owner="compute"):
        configured(model, {"first": {"compute": 1}})
    with refused("domain-membership", owner="compute"):
        configured(model, {"first": {"compute": "dense"}})
    assert stored(model) == {}


def test_a_value_outside_a_domain_is_refused() -> None:
    """3 does not divide the channels: the engine's domain refuses it, whether the core
    it lies under is set or not (membership as if the choice existed)."""
    model = matmul_model()
    with refused("domain-membership", owner="pe"):
        configured(model, {"first": {"pe": 3}})
    configured(model, {"first": FOLDED})
    with refused("domain-membership", owner="pe"):
        configured(model, {"first": {"pe": 3}})
    assert stored(model)["pe"] == 2
    with refused("domain-membership", owner="pe"):
        configured(thresholding_model(), {"activate": {"pe": 3}})


def test_a_nested_choice_under_a_case_not_selected_is_stored_out_of_play() -> None:
    """A choice under a case not selected is accepted and stored, out of play (dormant):
    it comes back in play with its case."""
    model = matmul_model()
    configured(model, {"first": {**FOLDED, "compute.int8_dsp58.compute_pumping": False}})
    assert stored(model)["compute.int8_dsp58.compute_pumping"] == 0  # stored as an int
    bound = network(model, model.graph.node)
    states = dict(bound.states(("first",)))
    assert states[("first", "compute.int8_dsp58.compute_pumping")] == inspection.Dormant(False)
    # The folding is MatMul's: in play whatever the core.
    assert states[("first", "pe")] == inspection.Set(2)


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
    before = channel_choices(model, "w")
    with refused("request-refused", owner="transport.fifo.depth"):
        configured(model, {"w": {"transport": "fifo", "transport.fifo.depth": "8"}})
    assert channel_choices(model, "w") == before


def test_a_channel_whose_contracts_wait_on_an_open_kernel_is_refused() -> None:
    """The MatMul's core is open, so the weight channel's contracts are and its choice cannot
    be checked."""
    model = matmul_model()
    with refused("decision-unassigned", owner="compute"):
        configured(model, {"w": {"source.memstream.ram_style": "ultra"}})
    assert channel_choices(model, "w") == {}


def test_a_memory_the_device_cannot_build_is_refused() -> None:
    """Ultra96 has no UltraRAM: batched with the folding, the style is stored as the
    channel is first bound, and its source, which the interim channel rule then sets,
    can take no value (``choice-no-valid-value``), the memory's rules its causes, the
    style standing in their way; once the channel is made, the style is refused by the
    memory's rules themselves."""
    ultra = {"w": {"source.memstream.ram_style": "ultra"}}
    model = matmul_model()
    with refused(
        "choice-no-valid-value", owner="source", message="uram_present .with source.memstream"
    ) as caught:
        configured(model, {"first": FOLDED, **ultra})
    assert {(cause.owner, cause.code) for cause in caught.finding.causes} == {
        ("source.memstream.uram_present", "uram-absent"),
        ("source.memstream.uram_init_taken", "uram-init"),
    }
    configured(model, {"first": FOLDED})
    before = channel_choices(model, "w")
    with refused("uram-absent", owner="source.memstream.uram_present"):
        configured(model, ultra)
    assert channel_choices(model, "w") == before


def test_a_channel_choice_on_a_graph_with_host_nodes_is_refused() -> None:
    """Channel choices are a partition's body's only; a kernel's choices are set. Named,
    one is refused, and so is every channel the change binds, which the interim channel
    rule gives its choices as it binds it (x and y)."""
    model = hosted(matmul_model())
    configured(model, {"first": FOLDED})
    with refused("channel-choices-outside-body", keys=("x", "w", "y")):
        configured(model, {"w": WEIGHT_CHANNEL})
    assert channel_choices(model, "w") == {}


def test_persist_refuses_a_channel_choice_on_a_graph_with_host_nodes_before_any_write() -> None:
    """The compiler's path, not the helper's: a network of the kernels set at a channel's
    choice and persisted on a graph that is no partition's body is refused by ``persist``
    itself, the one writer, and nothing is written, the kernel's choices included."""
    model = hosted(matmul_model())
    configured(model, {"first": FOLDED})
    before = stored(model)
    nodes = [node for node in model.graph.node if node.name == "first"]
    stated = choose(in_shell(model, nodes), {"w.transport": "direct"})
    with refused("channel-choices-outside-body", keys=("w",)):
        persist(model, stated)
    assert channel_choices(model, "w") == {}
    assert stored(model) == before


# -- clearing ----------------------------------------------------------------------------


def test_none_is_no_value_and_the_seam_has_no_clearing() -> None:
    """Clearing is not supported: ``None`` is refused as a value no case holds, and
    nothing is written. A test that needs a cleared state writes the raw attribute."""
    model = matmul_model()
    configured(model, {"first": FOLDED})
    before = stored(model)
    with refused("request-refused", owner="simd"):
        configured(model, {"first": {"simd": None}})
    assert stored(model) == before


# -- the raw attribute -------------------------------------------------------------------


def test_write_raw_stands_for_a_graph_finn_did_not_write() -> None:
    """A choice as the node's raw attribute, replaced when there, removed for ``None``;
    nothing is checked (a divisor-less ``pe`` is written), the read side finds it."""
    model = matmul_model()
    write_raw(model, "first", "pe", 3)
    write_raw(model, "first", "compute.packed.reducer", "tree")
    assert stored(model) == {"pe": 3, "compute.packed.reducer": "tree"}
    write_raw(model, "first", "pe", 2)
    assert stored(model)["pe"] == 2
    write_raw(model, "first", "pe", None)
    write_raw(model, "first", "simd", None)  # none there: nothing
    assert stored(model) == {"compute.packed.reducer": "tree"}
    assert [attribute.name for attribute in model.graph.node[0].attribute] == [
        "compute.packed.reducer"
    ]
