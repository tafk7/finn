# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The MatMul KernelOp: facts from the model, the choice schema, persistence and replay.

A node's choice attributes are sparse, absent meaning open; ``save`` takes
choices, never a point, and a forced Decision is never committed, so it is
never written; a nested choice applies under its forced selector.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.registry import get_domain_opset_version, getCustomOp, op_identity

import finn.custom_op.kernels as domain
from finn.custom_op.kernels.base import PLATFORM_KEYS, KernelOpError, channel_choices
from finn.custom_op.kernels.matmul import MatMul
from finn.custom_op.kernels.shell import save_channels
from finn.custom_op.kernels.thresholding import Thresholding
from finn.custom_op.kernels.windowed_matmul import WindowedMatMul
from finn.kernels.matmul import MatMulKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.transformation.general import ApplyConfig
from finn.transformation.kernels import InferKernelTensors
from kernel_ops.models import WEIGHTS, X, matmul_model, schema_digest

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
    # One node root, stored or streamed: the node owns the weights when they are an
    # initializer.
    assert facts.root is MatMul.root() and facts.owned == ("w",)
    formals = facts.formals()
    assert (formals["m"], formals["k"], formals["n"]) == (3, 4, 4)
    # The weights are the weight channel's value, not a formal of MatMul's; the channel's
    # tensor is the initializer's, over its values' range.
    assert "weights" not in formals
    value = IntegerTensorValue.of(tuple(tuple(int(v) for v in row) for row in WEIGHTS))
    assert facts.values() == {"w": value}
    assert facts.inputs()["w"].element.value_range == value.range
    # The output keeps the input's leading axes.
    assert op(model).output_tensors() == {
        "y": ((1, 3, 4), op(model).view("result_tensor").element.dtype)
    }
    streamed = matmul_model(stored=False)
    assert op(streamed).facts().root is MatMul.root()
    assert op(streamed).facts().owned == ()


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


def test_save_writes_choices_and_replay_reads_them() -> None:
    model = matmul_model()
    op(model).save(FOLDING)
    assert attributes(model) == sorted(FOLDING)
    assert op(model).choices()["compute.packed.pe"] == 2
    point = op(model).point()
    assert point.matmul.compute.pe == 2
    # A refused save writes nothing; the refusal names the key.
    with pytest.raises(KernelOpError) as error:
        op(model).save({"compute.packed.pe": 3})
    assert error.value.keys == ("compute.packed.pe",)
    assert op(model).choices()["compute.packed.pe"] == 2
    # None clears a choice.
    op(model).save({"compute.packed.simd": None})
    assert "compute.packed.simd" not in attributes(model)
    with pytest.raises(KernelOpError, match="not an 's' value"):
        op(model).save({"compute": 1})
    with pytest.raises(KernelOpError, match="among"):
        op(model).save({"compute": "dense"})
    # A channel's choice is no node's: refused, naming where it is stated.
    with pytest.raises(KernelOpError, match="are not choices of MatMul; a channel's choices"):
        op(model).save({"w.transport": "direct"})


def test_the_weight_sources_choices_persist_on_the_weights_and_the_node_replays_them() -> None:
    """The node owns its weight channel, whose choices are stated on the initializer's
    tensor, beside its value, never on the node; the node's point replays them. The
    source, its one case forced, is written only when saved on purpose."""
    model = matmul_model()
    save_channels(model, {"w": {"source.memstream.ram_style": "block", "transport": "direct"}})
    assert attributes(model) == []
    assert channel_choices(model, "w") == {
        "source.memstream.ram_style": "block",
        "transport": "direct",
    }
    point = op(model).point()
    assert point.w.source.ram_style == "block" and isinstance(point.w.source, MemStreamKernel)
    save_channels(model, {"w": {"source": "memstream"}})
    assert channel_choices(model, "w")["source"] == "memstream"
    assert op(model).point().w.source.ram_style == "block"


def test_a_forced_choice_is_never_saved() -> None:
    """On DSP48E2 compute = packed is forced; only choices made on purpose reach the
    node, so nothing goes stale on another target."""
    model = matmul_model()
    point = op(model).point({"compute.packed.pe": 2})
    assert point.matmul.compute.pe == 2  # under the forced selector
    op(model).save({"compute.packed.pe": 2})
    assert attributes(model) == ["compute.packed.pe"]


def test_a_nested_choice_under_a_case_not_forced_is_refused_by_name() -> None:
    model = matmul_model()
    with pytest.raises(KernelOpError) as error:
        op(model).save({"compute.int8_dsp58.pe": 2})
    assert error.value.keys == ("compute.int8_dsp58.pe",)
    assert "inapplicable" in str(error.value)


def test_a_choice_goes_stale_when_a_fact_changes() -> None:
    model = matmul_model()
    op(model).save({**FOLDING, "compute.packed.pe": 4})
    assert op(model).verify_node() == []
    # The weights' columns change from 4 to 6: pe = 4 no longer divides them.
    model.set_initializer("w", np.concatenate([WEIGHTS, WEIGHTS[:, :2]], axis=1).astype(np.float32))
    # Until inference states y again, the node root carries the graph's stale y,
    # which the kernel's extents refuse.
    stale = op(model).verify_node()
    assert len(stale) == 1 and "kernel-extents: n is 6 (w axis 1) and 4 (y axis 1)" in stale[0]
    model = model.transform(InferKernelTensors())
    problems = op(model).verify_node()
    assert len(problems) == 1 and "compute.packed.pe: " in problems[0]
    with pytest.raises(KernelOpError) as error:
        op(model).point()
    assert error.value.keys == ("compute.packed.pe",)


def test_a_matmul_no_core_admits_is_refused_while_its_choices_are_open() -> None:
    """INT32 activations: the packed core's DSP input is too narrow and the INT8 core
    needs a DSP58, so the ``compute`` Decision has no viable case on the facts alone.
    Held in float64, as graph preparation widens them: float32 would round their partial
    sums, which the domain step refuses first."""
    model = matmul_model(annotate=(), infer=False)
    for name in ("x", "w"):
        model.set_tensor_datatype(name, DataType["INT32"])
    for info in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        info.type.tensor_type.elem_type = TensorProto.DOUBLE
    model.set_initializer("w", model.get_initializer("w").astype(np.float64))
    (problem,) = op(model.transform(InferKernelTensors())).verify_node()
    assert problem.startswith("first: matmul.compute: decision-no-viable-case: no case is viable: ")
    assert "dotp-activation-width" in problem and "dotp-target" in problem


def test_an_unknown_attribute_is_refused() -> None:
    model = matmul_model()
    node = model.graph.node[0]
    # Nor is ``memory``: the weight channel's source owns the memory. Nothing is pinned.
    for name, value in (("PE", 2), ("memory", "memstream")):
        node.attribute.append(helper.make_attribute(name, value))
        with pytest.raises(KernelOpError, match=f"{name} is not a choice of MatMul"):
            op(model).choices()
        node.attribute.pop()


def test_apply_config_writes_choices_replay_checks() -> None:
    model = matmul_model()
    model = model.transform(
        ApplyConfig({"first": {**FOLDING, "compute.packed.compute_pumping": 0}})
    )
    assert op(model).point().matmul.compute.pe == 2
    # A folding config naming only the nested key replays under the forced selector.
    alone = matmul_model().transform(ApplyConfig({"first": {"compute.packed.pe": 2}}))
    assert op(alone).point().matmul.compute.pe == 2
    refused = matmul_model().transform(ApplyConfig({"first": {"compute.packed.pe": 3}}))
    assert "compute.packed.pe" in op(refused).verify_node()[0]


def test_choices_survive_save_and_load(tmp_path: Path) -> None:
    model = matmul_model()
    op(model).save(FOLDING)
    model.save(str(tmp_path / "model.onnx"))
    loaded = ModelWrapper(str(tmp_path / "model.onnx"))
    assert op(loaded).choices() == op(model).choices()
    assert op(loaded).point().matmul.compute.pe == 2


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
    for op in (MatMul, Thresholding, WindowedMatMul):
        assert "op_type" in vars(op) and "op_version" in vars(op)
        assert op_identity(op) == (op.op_type, op.op_version)
    assert op_identity(MatMul) == ("MatMul", MatMulKernel.version)
    assert op_identity(Thresholding) == ("Thresholding", ThresholdingAxiKernel.version)
    assert op_identity(WindowedMatMul) == ("WindowedMatMul", MatMulKernel.version)
    assert get_domain_opset_version("finn.custom_op.kernels") == domain.opset_version
    assert domain.__all__ == ["MatMul", "Thresholding", "WindowedMatMul"]
    assert PLATFORM_KEYS["dsp"].entry in {item.key for item in model.graph.metadata_props}
