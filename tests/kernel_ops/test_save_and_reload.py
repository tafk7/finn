# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every kind of stored choice saved and bound again, on a node and on a channel.

A stored value on the graph, after ``persist`` writes the network's choices back and the
network is bound again from the graph (``network``):

- **set**: kept, readable, the same value (``test_graph_network``'s
  ``test_an_explored_network_persists_and_replays_as_itself``, a node's and the channels');
- **out of play** (its selector on another case): kept, dormant, saved and bound again
  dormant (here; ``test_channel_choices``'s
  ``test_a_channel_choice_out_of_play_is_saved_and_replayed_out_of_play``);
- **refused by a changed fact** (outside its domain): dropped as stale, with a finding
  naming the value and the values the choice could take now, left open, and saving clears
  it (here);
- **a rule a changed fact makes false** over stored choices: the bind is refused, naming
  the rule, until the choices are changed (here);
- **its scope left unknown** (a core not stated): a channel whose contracts wait holding its
  choices (here);
- **a key the network does not have**: a node's attribute no choice declares refuses the
  bind, by name (here); a channel's choices on a tensor that is no channel are dropped and
  cleared (``test_channel_choices``'s
  ``test_choices_on_a_tensor_that_is_no_channel_are_dropped_and_cleared``).
"""

from __future__ import annotations

import numpy as np
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from support.configure import configured, write_raw
from support.helpers import FULL_DSP58
from support.models import EDGE, TARGET, WEIGHTS, kernel_model, matmul_model
from support.refusals import refused

from finn.core.graph.annotations.channel import CHANNEL_KEYS, channel_choices
from finn.core.graph.annotations.platform import write_target
from finn.core.graph.errors import KernelOpError
from finn.core.space import inspection
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import Partition, network, persist
from finn.platform.target import Target
from finn.transformation.kernels import InferKernelTensors, kernel_choices_config

FOLDING: dict[str, object] = {
    "compute": "packed",
    "pe": 2,
    "simd": 2,
    "compute.packed.compute_pumping": False,
    "compute.packed.reducer": "tree",
}
"""MatMul's folding, its core with it."""
PUMPING = "compute.int8_dsp58.compute_pumping"
"""A choice of the INT8 core's own, under its case."""
WEIGHT_SOURCE: dict[str, object] = {
    "source.ram_style": "block",
    "source.pumped_memory": False,
    "transport": "direct",
}
"""The weights' channel's choices, as a set-up and the interim channel rule make them."""
VERSAL = Target(part="a part with UltraRAM it initializes", platform=FULL_DSP58, shell="ip")
"""A target with UltraRAM, on which a MatMul's ``compute`` has two possible values."""


def bound(model: ModelWrapper) -> Partition:
    """The network of ``model``'s nodes, bound from the graph."""
    return network(model, model.graph.node)


def node(model: ModelWrapper) -> NodeProto:
    """The MatMul's node, ``first``."""
    found: list[NodeProto] = [each for each in model.graph.node if each.name == "first"]
    (first,) = found
    return first


def stale(network_: Partition) -> dict[str, list[tuple[str, str, dict[str, object]]]]:
    """The network's stale choices by key, each finding's code, owner and details."""
    return {
        key: [(found.code, found.owner, dict(found.details)) for found in findings]
        for key, findings in network_.stale.items()
    }


def holding(network_: Partition, tensor: str) -> dict[str, object]:
    """What channel ``tensor``, unbound, holds, having checked it waits on the MatMul's
    open ``compute``."""
    held = network_.unbound(tensor)
    assert held is not None and held.member == "first"
    assert {(found.code, found.owner) for found in held.result.findings} == {
        ("decision-unassigned", "compute")
    }
    return dict(held.stored)


def retarget(model: ModelWrapper, target: Target) -> ModelWrapper:
    """``model`` built for ``target``, its tensors stated again."""
    write_target(model, target)
    return model.transform(InferKernelTensors())


# -- out of play -------------------------------------------------------------------------


def test_a_node_choice_under_a_case_not_selected_is_kept_out_of_play_and_saved() -> None:
    """``compute.int8_dsp58.compute_pumping`` on a DSP48E2 target, where ``compute`` is
    ``packed``: written raw, as a graph FINN did not write holds it. It is a value of its
    domain under a case not selected: the node's Space and the network keep it, out of play
    (dormant), nothing stale; saving writes it, and bound again it is kept the same."""
    model = configured(matmul_model(), {"first": FOLDING})
    write_raw(model, "first", PUMPING, True)
    point = kernel_op(model, node(model)).point()
    assert dict(inspection.states(point))[PUMPING] == inspection.Dormant(True)
    first = bound(model)
    assert not first.stale and first.member("first").compute.pe == 2
    assert persist(model, first)["first"] == {**FOLDING, PUMPING: True}
    again = bound(model)
    assert not again.stale and again.chosen() == first.chosen()


def test_a_channel_choice_under_an_adapter_not_selected_is_kept_out_of_play_and_saved() -> None:
    """The levels edge's input adapter is ``input_gen``; a memory of the
    ``input_gen_vpc`` chain, written on its tensor past the writer, is out of play: the
    network keeps it, dormant, nothing stale, and saving writes it beside the edge's own
    choices: bound again, the same."""
    model = kernel_model()
    key = "adapter.input_gen_vpc.input_gen.ram_style"
    model.set(CHANNEL_KEYS[key], "auto", tensor="levels")
    first = bound(model)
    assert not first.stale
    assert dict(first.states(["levels"]))[("levels", key)] == inspection.Dormant("auto")
    held = {name: value for (_, name), value in first.chosen(["levels"]).items()}
    assert held[key] == "auto" and persist(model, first)["levels"] == held
    assert channel_choices(model, "levels")[key] == "auto"
    again = bound(model)
    assert not again.stale and again.chosen() == first.chosen()


# -- a fact changed: the choice outside its domain -------------------------------------------


def test_a_node_choice_a_changed_fact_refuses_is_dropped_with_candidates_and_cleared() -> None:
    """The weights' columns change from 4 to 6 under a stored ``pe`` of 4, which no longer
    divides them. Bound again, the network drops ``pe`` as stale: its domain's finding names
    the value stored and the values ``pe`` could take now. Saving clears it and keeps the
    rest; bound again, nothing is stale, ``pe`` is open, the rest reads as stored."""
    model = configured(matmul_model(), {"first": {**FOLDING, "pe": 4}})
    widened = np.concatenate([WEIGHTS, WEIGHTS[:, :2]], axis=1).astype(np.float32)
    model.set_initializer("w", widened)
    model = model.transform(InferKernelTensors())
    first = bound(model)
    ((key, (own,)),) = first.stale.items()
    assert key == "first.pe"
    assert (own.code, own.owner) == ("domain-membership", "pe")
    assert dict(own.details) == {"value": 4, "candidates": (1, 2, 3, 6)}
    assert own.message == "not one of 1, 2, 3, 6"
    kept = {name: value for name, value in FOLDING.items() if name != "pe"}
    assert persist(model, first)["first"] == kept
    assert "pe" not in kernel_choices_config(model)["first"]
    again = bound(model)
    assert not again.stale
    assert ("first", "pe") in again.undecided()
    assert again.chosen(["first"]) == {("first", name): value for name, value in kept.items()}


# -- a fact changed: a rule false over stored choices ----------------------------------------


def test_a_channel_choice_a_rule_a_changed_fact_makes_false_refuses_the_bind() -> None:
    """The weights' memory stored as ``ultra`` on a part with UltraRAM, the model then built
    for one without. Whether the part has UltraRAM is a rule of the memory's, not its
    domain: binding the stored choices together breaks it, so the bind is refused, naming
    each rule by its key once (``network-invalid``), the rules' findings its causes, about
    the channel; nothing is dropped behind the user's back."""
    model = retarget(matmul_model(), VERSAL)
    configured(model, {"first": FOLDING, "w": {**WEIGHT_SOURCE, "source.ram_style": "ultra"}})
    assert channel_choices(model, "w")["source.ram_style"] == "ultra"
    model = retarget(model, TARGET)
    with refused(
        "network-invalid",
        error=KernelOpError,
        message="w: its stored choices are refused: .*uram",
    ) as caught:
        bound(model)
    assert caught.finding.message.count("uram_present") == 1
    assert {(f.code, f.owner, dict(f.details)["tensor"]) for f in caught.finding.causes} == {
        ("uram-absent", "source.uram_present", "w"),
        ("uram-init", "source.uram_init_taken", "w"),
    }
    assert channel_choices(model, "w")["source.ram_style"] == "ultra"


# -- a scope left unknown ------------------------------------------------------------------------


def test_a_channel_whose_contracts_wait_on_a_core_not_stated_holds_its_choices() -> None:
    """The MatMul's ``compute`` not stated (written away, as a graph FINN did not write
    holds it), on DSP58, where it has two possible values: its contracts are unknown, so the
    weights' channel and the input's wait on it, unbound, each holding the choices stored
    on its tensor; none is stale. Saving writes them back on their tensors; bound again,
    the channels wait on ``compute`` holding the same choices."""
    model = configured(matmul_model(), {"first": FOLDING, "x": EDGE, "w": WEIGHT_SOURCE})
    x, w = channel_choices(model, "x"), channel_choices(model, "w")
    model = retarget(model, VERSAL)
    write_raw(model, "first", "compute", None)
    first = bound(model)
    assert not first.stale
    assert (holding(first, "x"), holding(first, "w")) == (x, w)
    assert first.undecided() == (("first", "compute"),)
    written = persist(model, first)
    assert (written["x"], written["w"]) == (x, w)
    again = bound(model)
    assert not again.stale
    assert (holding(again, "x"), holding(again, "w")) == (x, w)
    assert again.chosen(["x", "w"]) == first.chosen(["x", "w"])


# -- a key the network does not have ----------------------------------------------------------


def test_a_node_attribute_no_choice_declares_refuses_the_bind_by_name() -> None:
    """A node's attribute that is no choice of its op is refused where the network reads
    the node, by name (``choice-unknown``): nothing is bound, so nothing is saved, until
    the attribute is gone. A channel's choices on a tensor that is no channel are dropped
    and cleared instead (``test_channel_choices``)."""
    model = configured(matmul_model(), {"first": FOLDING})
    write_raw(model, "first", "compute.packed.typo", 2)
    with refused(
        "choice-unknown",
        error=KernelOpError,
        owner="MatMul",
        node="first",
        keys=("compute.packed.typo",),
    ):
        bound(model)
    write_raw(model, "first", "compute.packed.typo", None)
    again = bound(model)
    assert not again.stale and again.chosen(["first"]) == {
        ("first", key): value for key, value in FOLDING.items()
    }
