# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every kind of stored choice saved and bound again, on a node and on a channel.

A stored value on the graph, after ``persist`` writes the network's choices back and the
network is bound again from the graph (``network``):

- **checked**: kept, readable, the same value (``test_graph_network``'s
  ``test_an_explored_network_persists_and_replays_as_itself``, a node's and the channels');
- **unchecked** (its own check cannot conclude): kept unchecked, the same stored value and
  what it waits on (``test_package``'s
  ``test_an_unchecked_choice_is_saved_and_replayed_as_unchecked``, ``test_channel_choices``'s
  ``test_a_channel_choice_kept_unchecked_is_saved_and_replayed_as_unchecked``);
- **inactive** (its selector settled on another case): one Space refuses it; the network
  drops it as stale, with why, and saving clears it (here);
- **refused by a changed fact** (its own check false): dropped as stale, with a finding
  naming the value (and, for a value outside its domain, the values the Decision could take
  now), and saving clears it (here);
- **its own check true, a rule a changed fact makes false**: kept, committed, the
  configuration refused naming the rule; saving writes it, and bound again it is kept as
  it was (here);
- **left unknown by a changed fact**: kept, not stale: a node's choice unchecked, a
  channel whose ends wait holding its choices (here);
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
    "compute.packed.pe": 2,
    "compute.packed.simd": 2,
    "compute.packed.compute_pumping": False,
    "compute.packed.reducer": "tree",
}
"""MatMul's folding; ``compute`` is forced on ``TARGET`` (DSP48E2), so it is not stored."""
SAVED = {key: value for key, value in FOLDING.items() if key != "compute"}
"""What the node stores of ``FOLDING`` on ``TARGET``."""
WEIGHT_SOURCE: dict[str, object] = {"source.memstream.ram_style": "block", "transport": "direct"}
VERSAL = Target(part="a part with UltraRAM it initializes", platform=FULL_DSP58, shell="ip")
"""A target with UltraRAM, whose MatMul ``compute`` is open (DSP58): nothing forces it."""


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


# -- inactive ----------------------------------------------------------------------------


def test_a_node_choice_its_selector_settled_away_from_is_dropped_and_cleared() -> None:
    """``compute.int8_dsp58.pe`` on a DSP48E2 target, where ``compute`` is forced to
    ``packed``: written raw, as a graph FINN did not write holds it. The node's Space alone
    refuses it by name; the network drops it as stale (``inapplicable``, the value stored
    beside), the rest replays, and saving clears it: bound again, nothing is stale and the
    folding reads as before."""
    model = configured(matmul_model(), {"first": FOLDING})
    write_raw(model, "first", "compute.int8_dsp58.pe", 2)
    with refused(
        "choices-refused",
        error=KernelOpError,
        owner="MatMul",
        node="first",
        keys=("compute.int8_dsp58.pe",),
    ):
        kernel_op(model, node(model)).point()
    first = bound(model)
    assert stale(first) == {
        "first.compute.int8_dsp58.pe": [("inapplicable", "compute.int8_dsp58.pe", {"value": 2})]
    }
    assert first.unchecked() == ()
    assert first.member("first").compute.pe == 2
    assert persist(model, first)["first"] == SAVED
    assert "compute.int8_dsp58.pe" not in {attribute.name for attribute in node(model).attribute}
    again = bound(model)
    assert not again.stale and again.unchecked() == ()
    assert again.chosen() == first.chosen()


def test_a_channel_choice_its_adapter_settled_away_from_is_dropped_and_cleared() -> None:
    """The levels edge's input adapter is forced to ``input_gen`` by its ends; a memory of
    the ``input_gen_vpc`` chain, written on its tensor past the writer, is inactive. The
    network drops it as stale (``inapplicable``, the value beside), and saving clears it
    from the tensor, leaving the edge's own choices: bound again, nothing is stale."""
    model = kernel_model()
    model.set(CHANNEL_KEYS["adapter.input_gen_vpc.input_gen.ram_style"], "auto", tensor="levels")
    first = bound(model)
    key = "adapter.input_gen_vpc.input_gen.ram_style"
    assert stale(first) == {f"levels.{key}": [("inapplicable", key, {"value": "auto"})]}
    assert first.unchecked() == ()
    held = {name: value for (_, name), value in first.chosen(["levels"]).items()}
    assert key not in held and persist(model, first)["levels"] == held
    assert key not in channel_choices(model, "levels")
    again = bound(model)
    assert not again.stale and again.unchecked() == ()
    assert again.chosen() == first.chosen()


# -- a fact changed: the choice's own check false ------------------------------------------


def test_a_node_choice_a_changed_fact_refuses_is_dropped_with_candidates_and_cleared() -> None:
    """The weights' columns change from 4 to 6 under a stored ``pe`` of 4, which no longer
    divides them. Bound again, the network drops ``pe`` as stale: its own finding names the
    value stored and the values ``pe`` could take now, its selector's (``compute``, left
    with no viable case by it) beside. Saving clears it and keeps the rest; bound again,
    nothing is stale, ``pe`` is open, the rest reads as stored."""
    model = configured(matmul_model(), {"first": {**FOLDING, "compute.packed.pe": 4}})
    widened = np.concatenate([WEIGHTS, WEIGHTS[:, :2]], axis=1).astype(np.float32)
    model.set_initializer("w", widened)
    model = model.transform(InferKernelTensors())
    first = bound(model)
    ((key, findings),) = first.stale.items()
    assert key == "first.compute.packed.pe"
    assert {(found.code, found.owner) for found in findings} == {
        ("domain-membership", "compute.packed.pe"),
        ("decision-no-viable-case", "compute"),
    }
    (own,) = [found for found in findings if found.code == "domain-membership"]
    assert dict(own.details) == {"value": 4, "candidates": (1, 2, 3, 6)}
    assert own.message == "not one of 1, 2, 3, 6"
    assert first.unchecked() == ()
    kept = {name: value for name, value in SAVED.items() if name != "compute.packed.pe"}
    assert persist(model, first)["first"] == kept
    assert "compute.packed.pe" not in kernel_choices_config(model)["first"]
    again = bound(model)
    assert not again.stale and again.unchecked() == ()
    assert ("first", "compute.packed.pe") in again.undecided()
    assert again.chosen() == {("first", name): value for name, value in kept.items()}


# -- a fact changed: its own check true, a rule false ----------------------------------------


def test_a_channel_choice_a_rule_a_changed_fact_makes_false_is_kept_and_saved() -> None:
    """The weights' memory stored as ``ultra`` on a part with UltraRAM, the model then built
    for one without. Whether the part has UltraRAM is a rule of the memory's, not its
    domain: the choice's own check holds, so the network keeps it, committed, under its
    source's guard, which the rule leaves with no possible case (the configuration refused,
    ``uram-absent`` among the reasons). Nothing is stale; saving writes the stored ``ultra``
    beside the transport, and bound again the network holds the same."""
    model = retarget(matmul_model(), VERSAL)
    configured(
        model, {"first": FOLDING, "w": {**WEIGHT_SOURCE, "source.memstream.ram_style": "ultra"}}
    )
    model = retarget(model, TARGET)
    first = bound(model)
    key = "source.memstream.ram_style"
    stored = {("w", key): "ultra", ("w", "transport"): "direct"}
    assert not first.stale and first.unchecked() == ()
    assert first.chosen(["w"]) == stored
    validity = inspection.validity(first.member("w"))
    assert validity.status == "refused"
    assert {
        (found.code, found.owner, "uram-absent" in found.message)
        for piece in validity.pieces
        for found in piece.result.findings
    } == {("decision-no-viable-case", "source", True)}
    assert persist(model, first)["w"] == {key: "ultra", "transport": "direct"}
    assert channel_choices(model, "w") == {key: "ultra", "transport": "direct"}
    again = bound(model)
    assert not again.stale and again.unchecked() == ()
    assert again.chosen(["w"]) == stored
    assert inspection.validity(again.member("w")) == validity


# -- a fact changed: the choice's own check unknown ----------------------------------------


def folded_then_moved_to_dsp58() -> ModelWrapper:
    """MatMul folded, and its input's and weights' channels chosen, on ``TARGET``
    (DSP48E2, where ``compute`` is forced, so not stored), the model then built for
    ``VERSAL``, where ``compute`` is open."""
    model = configured(matmul_model(), {"first": FOLDING, "x": EDGE, "w": WEIGHT_SOURCE})
    assert "compute" not in kernel_choices_config(model)["first"]
    return retarget(model, VERSAL)


def test_a_node_choice_a_changed_fact_leaves_unknown_is_kept_unchecked() -> None:
    """On DSP58 ``compute`` is open, so the folding stored under ``packed`` waits on it: each
    key is kept unchecked, its stored value and ``compute`` named, not stale, and
    ``compute`` is the one open choice. Each is committed all the same (a kept commitment
    reads as committed, whatever its read answers), so the network's choices list it.
    Saving writes each by its stored value; bound again, the node holds them unchecked,
    waiting on the same ``compute``."""
    model = folded_then_moved_to_dsp58()
    waiting = tuple(
        (("first", key), inspection.Unchecked(key, value, ("compute",)))
        for key, value in sorted(SAVED.items())
    )
    first = bound(model)
    assert not first.stale
    assert first.unchecked() == waiting
    assert first.chosen(["first"]) == {("first", key): value for key, value in SAVED.items()}
    assert first.undecided() == (("first", "compute"),)
    assert persist(model, first)["first"] == SAVED
    again = bound(model)
    assert not again.stale
    assert again.unchecked() == waiting
    assert inspection.unchecked(again.member("first")) == tuple(item for _, item in waiting)
    assert again.chosen(["first"]) == first.chosen(["first"])


def test_a_channel_whose_ends_a_changed_fact_leaves_unknown_holds_its_choices() -> None:
    """On DSP58 the MatMul's ``compute`` is open, so its ends are unknown: the weights'
    channel and the input's wait on it, unbound, each holding the choices stored on its
    tensor; none is stale, and none is listed unchecked (an unbound channel's choices wait
    with its read; only the node's folding is). Saving writes them back on their tensors;
    bound again, the channels wait on ``compute`` holding the same choices."""
    model = folded_then_moved_to_dsp58()
    first = bound(model)
    assert not first.stale
    assert (holding(first, "x"), holding(first, "w")) == (EDGE, WEIGHT_SOURCE)
    assert {owner for (owner, _), _ in first.unchecked()} == {"first"}
    written = persist(model, first)
    assert (written["x"], written["w"]) == (EDGE, WEIGHT_SOURCE)
    assert (channel_choices(model, "x"), channel_choices(model, "w")) == (EDGE, WEIGHT_SOURCE)
    again = bound(model)
    assert not again.stale
    assert (holding(again, "x"), holding(again, "w")) == (EDGE, WEIGHT_SOURCE)
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
        ("first", key): value for key, value in SAVED.items()
    }
