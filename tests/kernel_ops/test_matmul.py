# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The MatMul KernelOp: facts from the model, the choice schema, persistence and replay.

A node's choice attributes are its own choices, each set by its default policy once the
node is converted (every choice in play, a selector's case included), absent on a node made
by hand until configured (``support.configure``, which sets choices as the compiler does,
never a point); a nested choice applies under its selector's case, waits while the
selector is open, and is kept out of play (dormant) under another case.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import get_domain_opset_version, getCustomOp, op_identity
from support.configure import configured, write_raw
from support.models import TARGET, WEIGHTS, X, matmul_model, schema_digest
from support.op_specs import matmul as spec
from support.refusals import refused

import finn.custom_op.kernels as domain
from finn.core.graph.annotations.channel import channel_choices
from finn.core.graph.annotations.platform import PLATFORM_KEYS
from finn.core.graph.errors import KernelOpError
from finn.core.onnx_exec import execute_onnx
from finn.core.space import inspection
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.conv import Conv
from finn.custom_op.kernels.matmul import MatMul
from finn.custom_op.kernels.network import Partition, network, persist
from finn.custom_op.kernels.thresholding import Thresholding
from finn.harness.reference import converted
from finn.kernels.matmul import MatMulKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.transformation.kernels import InferKernelTensors

FOLDING = {
    "compute": "packed",
    "compute.packed.pe": 2,
    "compute.packed.simd": 2,
    "compute.packed.compute_pumping": False,
    "compute.packed.reducer": "tree",
}


def op(model: ModelWrapper) -> MatMul:
    found = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(found, MatMul)
    return found


def attributes(model: ModelWrapper) -> list[str]:
    return [attribute.name for attribute in model.graph.node[0].attribute]


# -- facts ------------------------------------------------------------------------------


def test_facts_come_from_the_model() -> None:
    model = matmul_model()
    facts = op(model).facts()
    # One kernel, stored or streamed: the node owns the weights when they are an
    # initializer.
    assert facts.space_type is MatMulKernel and op(model).owned() == {"w": "w"}
    arguments = facts.arguments
    # Its extents and datatypes are its tensors': beside them, the platform and the known
    # weights' declared datatype. Known weights are its contents, never a tensor: the
    # weight channel carries what MatMul states over their values' range.
    assert set(arguments) == {"weights_dtype", "platform", "x_tensor", "w_contents"}
    value = IntegerTensorValue.of(tuple(tuple(int(v) for v in row) for row in WEIGHTS))
    assert arguments["w_contents"] == value
    assert op(model).bound().w_carried.element.value_range.span == value.range
    # The output keeps the input's leading axes: the kernel states it at their rank.
    (result,) = op(model).output_tensors().values()
    assert op(model).infer_output_tensors(model) == {"y": result}
    assert result.shape == (1, 3, 4)
    streamed = matmul_model(stored=False)
    assert op(streamed).facts().space_type is MatMulKernel
    assert op(streamed).owned() == {}


def refusal(model: ModelWrapper) -> str:
    with pytest.raises(KernelOpError) as error:
        op(model).facts()
    return str(error.value)


def test_each_missing_or_refused_fact_is_named() -> None:
    assert "x has no datatype annotation" in refusal(matmul_model(annotate=("w",), infer=False))
    unknown = matmul_model()
    unknown.graph.input[0].type.tensor_type.ClearField("shape")
    assert "x has no shape yet (run InferKernelTensors)" in refusal(unknown)
    assert "states no target" in refusal(matmul_model(target=False, infer=False))
    assert "not integers" in refusal(matmul_model(weights=WEIGHTS + 0.5, infer=False))
    assert "annotated INT3 and holds values over [-3, 9]" in refusal(
        matmul_model(weights=np.where(WEIGHTS == 3, 9, WEIGHTS), infer=False)
    )
    assert "x has 5 columns and w 4 rows" in refusal(matmul_model(x_shape=[1, 3, 5], infer=False))


# -- the schema -------------------------------------------------------------------------


def test_the_schema_is_the_kernels_decision_keys() -> None:
    schema = MatMul.schema()
    assert len(schema) == 9
    assert sum(kind == "s" for kind, _ in schema.values()) == 3
    assert schema["compute"] == ("s", ("packed", "int8_dsp58"))
    assert schema["compute.packed.pe"] == ("i", ())
    assert schema["compute.packed.compute_pumping"] == ("i", ())
    assert schema["compute.packed.reducer"] == ("s", ())
    # A channel's keys are its tensor's (finn.channel), never the node's: none sits under
    # a port name, the weights' source neither.
    assert not any(name.partition(".")[0] in ("x", "w", "y") for name in schema)
    assert not any(name.startswith(("memory", "source")) for name in schema)
    types = op(matmul_model()).get_nodeattr_types()
    assert types["compute"] == ("s", False, "", {"packed", "int8_dsp58"})


def test_the_schema_is_pinned_for_its_op_version() -> None:
    """Changing a kernel's keys changes the schema. Unreleased, the digest is re-pinned
    without an op-version bump (clean breaks)."""
    assert (MatMul.op_version, schema_digest(MatMul)) == (1, "6a16c9aa62655d95")


# -- persistence ------------------------------------------------------------------------


def test_configured_choices_are_written_and_replay_reads_them() -> None:
    model = matmul_model()
    configured(model, {"first": FOLDING})
    # Every value set is stored, the core's case too.
    assert attributes(model) == sorted(FOLDING)
    assert op(model).choices()["compute.packed.pe"] == 2
    point = op(model).point()
    assert point.compute.pe == 2
    # A refused choice writes nothing.
    with refused("domain-membership", owner="compute.packed.pe"):
        configured(model, {"first": {"compute.packed.pe": 3}})
    assert op(model).choices()["compute.packed.pe"] == 2
    # A value of another type, one no case holds, and a core this target (DSP48E2) refuses:
    # each refused by its code, the core by its own rule.
    with refused("request-refused", owner="compute"):
        configured(model, {"first": {"compute": 1}})
    with refused("domain-membership", owner="compute"):
        configured(model, {"first": {"compute": "dense"}})
    with refused("dotp-target", owner="compute.int8_dsp58.core_supported"):
        configured(model, {"first": {"compute": "int8_dsp58"}})
    # A channel's choice is no node's: refused, naming the key.
    with refused("not-a-decision", owner="first.w.transport"):
        configured(model, {"first": {"w.transport": "direct"}})
    assert attributes(model) == sorted(FOLDING)


def test_a_graph_with_a_cleared_choice_a_rule_reads_waits_then_is_refused() -> None:
    """The read side of a clear: a graph whose SIMD is gone, its pumping still stored, is
    stated as the raw attributes state it. The pumping is checked by its own domain and
    kept; the rule that pumping needs lanes waits on SIMD, so nothing is refused; once SIMD
    is stated as one lane, the stored choices break the rule, so binding them is refused
    (``verify_node``, ``point``), naming the rule at its origin, its key once, the rule's
    findings the refusal's causes (nothing stands in its way at a bind, which sets every
    stored choice)."""
    model = matmul_model()
    configured(model, {"first": FOLDING})
    write_raw(model, "first", "compute.packed.simd", None)
    assert "compute.packed.simd" not in attributes(model)
    assert op(model).verify_node() == []
    states = dict(inspection.states(op(model).point()))
    assert states["compute.packed.simd"] == inspection.Open()
    assert states["compute.packed.compute_pumping"] == inspection.Set(False)
    # Pumped, it waits all the same; SIMD stated as one lane, the rule refuses it.
    write_raw(model, "first", "compute.packed.compute_pumping", True)
    assert op(model).verify_node() == []
    write_raw(model, "first", "compute.packed.simd", 1)
    (problem,) = op(model).verify_node()
    assert problem.startswith("first: refused choices: compute.packed.pumping_has_lanes: ")
    assert "dotp-pumping: pumping requires SIMD >= 2" in problem
    assert problem.count("pumping_has_lanes") == 1 and "standing" not in problem
    with refused(
        "choices-refused",
        error=KernelOpError,
        message="dotp-pumping",
        keys=("compute.packed.pumping_has_lanes",),
        node="first",
    ) as caught:
        op(model).point()
    assert [(f.code, f.owner, dict(f.details)) for f in caught.finding.causes] == [
        ("dotp-pumping", "compute.packed.pumping_has_lanes", {"node": "first"})
    ]
    write_raw(model, "first", "compute.packed.compute_pumping", None)
    assert op(model).verify_node() == []


def test_persist_checks_every_value_before_it_writes_any(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one writer's own checks (it replays nothing, the network bound each value): a
    key that is no choice of the op and a value of another type than its attribute's, each
    its own code, owned by the op's type, the node in its details; nothing is written, the
    node's other choices included."""
    model = matmul_model()
    configured(model, {"first": FOLDING})
    before = op(model).choices()
    bound = network(model, model.graph.node)
    for forged, code in (
        ("compute.packed.typo", "choice-unknown"),
        ("compute.packed.simd", "choice-value-invalid"),
    ):
        stated = {("first", "compute.packed.pe"): 4, ("first", forged): "two"}
        monkeypatch.setattr(Partition, "chosen", lambda self, names=None, stated=stated: stated)
        with refused(
            code, error=KernelOpError, owner="MatMul", node="first", keys=(forged,)
        ) as caught:
            persist(model, bound)
        assert dict(caught.finding.details)["node"] == op(model).label
    assert op(model).choices() == before


def test_a_kernel_op_error_carries_at_least_one_finding() -> None:
    """No raise can leave its rule unnamed: the error takes its findings, or one
    rejection's (``refused``)."""
    with pytest.raises(TypeError):
        KernelOpError("unnamed")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        KernelOpError("unnamed", findings=())
    rejection = KernelOpError.refused("a-rule", "MatMul", "m: refused", ("pe",), node="m")
    (finding,) = rejection.findings
    assert str(finding) == "MatMul: a-rule: m: refused" and str(rejection) == "m: refused"
    assert dict(finding.details) == {"keys": ("pe",), "node": "m"} and rejection.keys == ("pe",)


def test_the_weight_sources_choices_persist_on_the_weights_and_the_network_replays_them() -> None:
    """The node owns its weight channel, whose choices are stated on the initializer's
    tensor, beside its value, never on the node; the network replays them, the node its
    kernel's alone. The source's one case is stored as the interim channel rule sets it,
    and stating it again changes nothing."""
    model = matmul_model()
    configured(model, {"first": FOLDING})  # its ends known, so its channels bind
    configured(model, {"w": {"source.memstream.ram_style": "block", "transport": "direct"}})
    assert attributes(model) == sorted(FOLDING)
    assert channel_choices(model, "w") == {
        "source": "memstream",
        "source.memstream.ram_style": "block",
        "source.memstream.pumped_memory": False,
        "transport": "direct",
    }
    weights = network(model, model.graph.node).member("w")
    assert weights.source.ram_style == "block" and isinstance(weights.source, MemStreamKernel)
    configured(model, {"w": {"source": "memstream"}})
    assert channel_choices(model, "w")["source"] == "memstream"
    assert network(model, model.graph.node).member("w").source.ram_style == "block"


def test_conversion_stores_every_value_its_policies_set() -> None:
    """Conversion stores each value its policies set on the kernel alone, the core's case
    and its pumping included (each first valid), so a converted node implies nothing: the
    choices of the case not selected are not in play, and not stored."""
    model, _ = converted(spec.matmul(), TARGET)
    (node,) = model.graph.node
    assert kernel_op(model, node).choices() == {
        "compute": "packed",
        "compute.packed.compute_pumping": 0,
        "compute.packed.pe": 1,
        "compute.packed.reducer": "tree",
        "compute.packed.simd": 1,
    }


def test_a_nested_choice_under_a_case_not_selected_is_stored_out_of_play() -> None:
    """A choice under a case is set wherever it sits: under the open core it is stored and
    waits on ``compute``; under a core not selected it is stored and dormant, its rules
    out of play."""
    model = matmul_model()
    configured(model, {"first": {"compute.int8_dsp58.pe": 2}})
    assert attributes(model) == ["compute.int8_dsp58.pe"]
    state = dict(inspection.states(op(model).point()))["compute.int8_dsp58.pe"]
    assert state == inspection.Waiting(2, ("compute",))
    configured(model, {"first": FOLDING})
    assert attributes(model) == sorted({*FOLDING, "compute.int8_dsp58.pe"})
    state = dict(inspection.states(op(model).point()))["compute.int8_dsp58.pe"]
    assert state == inspection.Dormant(2)


def test_a_choice_goes_stale_when_a_fact_changes() -> None:
    model = matmul_model()
    configured(model, {"first": {**FOLDING, "compute.packed.pe": 4}})
    assert op(model).verify_node() == []
    # The weights' columns change from 4 to 6: pe = 4 no longer divides them.
    model.set_initializer("w", np.concatenate([WEIGHTS, WEIGHTS[:, :2]], axis=1).astype(np.float32))
    # The kernel states its results from its inputs, never from the graph's y: the stale
    # choice is refused at once, before inference states y again.
    stale = op(model).verify_node()
    assert len(stale) == 1 and "compute.packed.pe: " in stale[0]
    model = model.transform(InferKernelTensors())
    problems = op(model).verify_node()
    assert len(problems) == 1 and "compute.packed.pe: " in problems[0]
    with pytest.raises(KernelOpError) as error:
        op(model).point()
    assert error.value.keys == ("compute.packed.pe",)


def test_an_unknown_attribute_is_refused() -> None:
    model = matmul_model()
    node = model.graph.node[0]
    # Nor is ``memory``: the weight channel's source owns the memory. Nothing is pinned.
    for name, value in (("PE", 2), ("memory", "memstream")):
        node.attribute.append(helper.make_attribute(name, value))
        with refused("choice-unknown", error=KernelOpError, node="first", keys=(name,)):
            op(model).choices()
        node.attribute.pop()


def test_replay_reads_the_stored_folding_and_verify_node_catches_a_misfolding() -> None:
    model = configured(matmul_model(), {"first": FOLDING})
    assert op(model).point().compute.pe == 2
    # A folding naming only the nested key waits on its selector, open on a node made by
    # hand, and replays once the core is set.
    alone = configured(matmul_model(), {"first": {"compute.packed.pe": 2}})
    state = dict(inspection.states(op(alone).point()))["compute.packed.pe"]
    assert state == inspection.Waiting(2, ("compute",))
    configured(alone, {"first": {"compute": "packed"}})
    assert op(alone).point().compute.pe == 2
    # A misfolding the seam would refuse is stored raw, as a graph FINN did not write holds
    # it: the replay's verification names the key.
    misfolded = matmul_model()
    write_raw(misfolded, "first", "compute.packed.pe", 3)
    assert "compute.packed.pe" in op(misfolded).verify_node()[0]


def test_choices_survive_save_and_load(tmp_path: Path) -> None:
    model = matmul_model()
    configured(model, {"first": FOLDING})
    model.save(str(tmp_path / "model.onnx"))
    loaded = ModelWrapper(str(tmp_path / "model.onnx"))
    assert op(loaded).choices() == op(model).choices()
    assert op(loaded).point().compute.pe == 2


# -- execution and identity -------------------------------------------------------------


@pytest.mark.parametrize("stored", (True, False))
def test_execution_is_onnx_matmul(stored: bool) -> None:
    model = matmul_model(stored=stored)
    feed = {"x": X.astype(np.float32)}
    if not stored:
        feed["w"] = WEIGHTS.astype(np.float32)
    produced = execute_onnx(model, feed)["y"]
    assert np.array_equal(produced, X @ WEIGHTS)


def test_the_domain_resolves_at_its_version_without_a_fallback() -> None:
    model = matmul_model()
    assert model.get_opset_imports()["finn.custom_op.kernels"] == domain.opset_version == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert isinstance(getCustomOp(model.graph.node[0], onnx_opset_version=1), MatMul)
    # Each op class states its identity in its own body: qonnx's rule reads it
    # there, and the domain's opset version is the one the module states.
    for op in (MatMul, Thresholding, Conv):
        assert "op_type" in vars(op) and "op_version" in vars(op)
        assert op_identity(op) == (op.op_type, op.op_version)
    assert op_identity(MatMul) == ("MatMul", MatMulKernel.version)
    assert op_identity(Thresholding) == ("Thresholding", ThresholdingAxiKernel.version)
    assert op_identity(Conv) == ("Conv", MatMulKernel.version)
    assert get_domain_opset_version("finn.custom_op.kernels") == domain.opset_version
    assert domain.__all__ == ["Conv", "MatMul", "Thresholding"]
    assert PLATFORM_KEYS["dsp"].entry in {item.key for item in model.graph.metadata_props}
