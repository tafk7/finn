# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A node's kernel, bound from its facts, and the member cache.

A KernelOp binds its kernel class from its facts, by value: its formals, its input
tensors at the graph's rank and the values of the initializers it holds. One binding per
node: its outputs are the kernel's statement, not a fact, so there is no second binding
for them. Its choice attributes are its kernel class's decision keys, set at conversion by
each choice's policy, the op's (``KernelOp.policies``), on the kernel alone, with no
channel (``KernelOp.defaulted``).
"""

from __future__ import annotations

import inspect
import json
from contextvars import Context
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qonnx.core.datatype import DataType
from qonnx.core.ranges import TensorRange
from support.models import TARGET, WEIGHTS, matmul_model, thresholding_model
from support.refusals import refused

import finn.kernels.members as members
from finn.core.graph.errors import KernelOpError
from finn.core.graph.facts import integer_tensor
from finn.core.space import DefinitionError, Rejected, design_space, inspection
from finn.custom_op.kernels.base import KernelOp, kernel_op
from finn.custom_op.kernels.conv import Conv
from finn.custom_op.kernels.matmul import MatMul
from finn.custom_op.kernels.thresholding import Thresholding
from finn.dataflow.datatypes import resolve_qonnx_datatype_name as dtype
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.matmul import MatMulKernel
from finn.kernels.members import (
    VERIFY,
    Facts,
    MemberCacheMismatch,
    configured,
    key,
    member_cache,
    typed,
    verification_counts,
    write_counts,
)
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.policies import Defaulted, Fixed, stated
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.transformation.kernels import InferKernelTensors
from finn.transformation.kernels.infer import infer_node

INT3 = dtype("INT3")
X = Tensor((3, 4), ScalarEncoding(INT3))
STORED = tuple(tuple((r + c) % 3 - 1 for c in range(4)) for r in range(4))


def test_a_kernelop_declares_no_formals_its_facts_build_the_arguments() -> None:
    """What a node's kernel is bound with is what its ``facts`` build, op by op: a KernelOp
    has no ``formals`` member to keep beside it (nothing read one)."""
    assert not any(hasattr(op, "formals") for op in (KernelOp, MatMul, Conv, Thresholding))


def test_an_op_made_without_its_model_is_refused_and_points_to_kernel_op() -> None:
    """A node's op made by hand (no model attached) refuses to read the model, by code,
    and its message names the way to get an attached one: the package's ``kernel_op``."""
    model = matmul_model()
    with refused(
        "model-unattached",
        error=KernelOpError,
        owner="MatMul",
        node="first",
        message=r"kernel_op\(model, node\)",
    ):
        MatMul(model.graph.node[0]).model()


def test_conversion_accepts_through_defaulted_alone() -> None:
    """``KernelOp.defaulted`` is the one way a node is accepted: its kernel with its
    default policies applied, or the refusals. ``infer_node`` does not accept, and no
    second method says the same; no node is a network of its own (no boundary channels)."""
    assert not any(hasattr(KernelOp, name) for name in ("acceptance", "accepted", "network"))
    assert not hasattr(KernelOp, "baselined")
    assert "accept" not in inspect.signature(infer_node).parameters


def test_the_schema_is_the_kernel_classs_decision_keys() -> None:
    """A node's choice attributes are its kernel's own decision keys, with no member
    prefix to strip."""
    keys = {item.key for item in inspection.decisions(MatMulKernel)}
    assert set(MatMul.schema()) == keys
    assert set(Thresholding.schema()) == {
        item.key for item in inspection.decisions(ThresholdingAxiKernel)
    }


def test_one_binding_per_node_serves_its_outputs_and_its_point() -> None:
    """Its kernel is bound once from its facts; its outputs are read from that binding,
    and its point is that binding with its stored choices committed."""
    model = matmul_model()
    op = kernel_op(model, model.graph.node[0])
    bound: Any = op.bound()
    (result,) = op.output_tensors().values()
    assert result == bound.y_tensor
    assert op.bound() is bound
    assert kernel_op(model, model.graph.node[0]).bound() is bound  # another op instance
    point: Any = op.point()
    # A node made by hand stores no choice, so its core is open; conversion's policies
    # (first valid) take the packed core.
    assert dict(inspection.states(point))["compute"] == inspection.Open()
    defaulted: Any = op.defaulted()
    assert type(defaulted.point.compute).__name__ == "PackedDotpKernel"


def test_a_held_initializer_is_given_as_contents_never_as_a_tensor() -> None:
    """Known weights are the kernel's contents, with the datatype the graph declares them
    in; streamed weights are a tensor, whose datatype is theirs."""
    stored = kernel_op(matmul_model(), matmul_model().graph.node[0])
    arguments = stored.facts().arguments
    assert set(arguments) == {"weights_dtype", "platform", "x_tensor", "w_contents"}
    assert arguments["x_tensor"] == Tensor((1, 3, 4), ScalarEncoding(INT3))  # whole
    streamed_model = matmul_model(stored=False)
    streamed = kernel_op(streamed_model, streamed_model.graph.node[0])
    arguments = streamed.facts().arguments
    assert set(arguments) == {"platform", "x_tensor", "w_tensor"}
    assert arguments["w_tensor"] == Tensor((4, 4), ScalarEncoding(INT3))


def test_the_views_answer_from_facts() -> None:
    """Without known weights the result is the datatypes' range; with them, their
    columns' (K7), each stated on the result element at the activations' rank."""
    streamed = design_space(
        MatMulKernel(
            platform=TARGET.platform, x_tensor=X, w_tensor=Tensor((4, 4), ScalarEncoding(INT3))
        )
    )
    assert streamed.y_tensor == Tensor((3, 4), ScalarEncoding(dtype("INT8"), TensorRange(-48, 64)))
    owned = design_space(
        MatMulKernel(
            platform=TARGET.platform,
            x_tensor=Tensor((1, 3, 4), ScalarEncoding(INT3)),
            w_contents=STORED,
            weights_dtype=INT3,
        )
    )
    assert owned.y_tensor == Tensor((1, 3, 4), ScalarEncoding(dtype("INT5"), TensorRange(-11, 11)))
    # MatMul reads its weights' declared datatype; the channel carries their range.
    assert owned.weight_tensor == Tensor((4, 4), ScalarEncoding(INT3))
    assert owned.w_carried == Tensor((4, 4), ScalarEncoding(dtype("INT2"), TensorRange(-1, 1)))
    table = (tuple((-9 + c, 1 - c, 8 + 2 * c) for c in range(4)),)
    facts: dict[str, Any] = dict(
        threshold_dtype=dtype("INT8"), thresholds=table, bias=0, platform=TARGET.platform
    )
    edge = Tensor((1, 3, 4), ScalarEncoding(dtype("INT8")))
    alone = design_space(ThresholdingAxiKernel(**facts, input_tensor=edge))
    assert alone.result_dtype.name == "UINT2"
    assert alone.output_tensor.shape == (1, 3, 4)


def test_a_graph_tensor_that_disagrees_with_the_facts_is_refused() -> None:
    wrong = design_space(
        MatMulKernel(
            platform=TARGET.platform,
            x_tensor=Tensor((3, 5), ScalarEncoding(INT3)),
            w_contents=STORED,
            weights_dtype=INT3,
        )
    )
    rejection = wrong.query(MatMulKernel.carried)
    assert isinstance(rejection, Rejected)
    assert {finding.code for finding in rejection.findings} == {"matmul-tensor"}


def test_conversion_sets_each_choice_in_play_by_its_policy_on_the_kernel_alone() -> None:
    """Conversion's answer is the node's kernel, nothing else (no channel): each choice in
    play, in declaration order, at its first valid value: the folding first (MatMul's),
    then the selector's case, then the choices of that case alone; those under the case not
    selected stay absent."""
    model = matmul_model()
    op = kernel_op(model, model.graph.node[0])
    done = op.defaulted()
    assert isinstance(done, Defaulted) and done.fallbacks == ()
    point = done.point
    assert isinstance(point, MatMulKernel)
    assert dict(inspection.states(point)) == {
        "pe": inspection.Set(1),
        "simd": inspection.Set(1),
        "compute": inspection.Set("packed"),
        "compute.packed.compute_pumping": inspection.Set(False),
        "compute.packed.reducer": inspection.Set("tree"),
        "compute.int8_dsp58.compute_pumping": inspection.Absent(),
        "realization": inspection.Absent(),  # a dense MatMul has none
    }
    assert inspection.validity(point).status == "valid"
    thresholding = thresholding_model()
    activated = kernel_op(thresholding, thresholding.graph.node[0]).defaulted()
    assert isinstance(activated, Defaulted)
    activate = activated.point
    assert isinstance(activate, ThresholdingAxiKernel)
    assert {key: state for key, state in inspection.states(activate)} == {
        "pe": inspection.Set(1),
        "use_axilite": inspection.Set(False),
        "deep_pipeline": inspection.Set(False),
        "ultra_stages": inspection.Set(0),
        "ram_style": inspection.Set("distributed"),
        "block_stages": inspection.Set(0),
    }


def test_a_choice_no_value_can_be_set_for_keeps_the_node_on_the_host() -> None:
    """INT32 activations: no core takes them, so the core has no first valid value. The
    answer is one finding owned by the choice's key (``choice-no-valid-value``), naming its
    policy and each value's rule, those rules' findings its causes; with no core chosen,
    no rule of the kernel is false, so the node made by hand is not refused
    (``verify_node``), only open."""
    model = matmul_model(annotate=(), infer=False)
    for name in ("x", "w"):
        model.set_tensor_datatype(name, DataType["INT32"])
    model = model.transform(InferKernelTensors())
    op = kernel_op(model, model.graph.node[0])
    assert op.verify_node() == []
    assert inspection.validity(op.point()).status == "open"
    found = op.defaulted()
    assert isinstance(found, tuple)
    (finding,) = found
    assert (finding.code, finding.owner) == ("choice-no-valid-value", "compute")
    assert dict(finding.details) == {"policy": "first valid"}
    assert "'packed' breaks compute.packed." in finding.message
    assert "'int8_dsp58' breaks compute.int8_dsp58." in finding.message
    assert {"dotp-activation-width", "dotp-target"} <= {cause.code for cause in finding.causes}


@pytest.mark.parametrize("op", [Conv, MatMul, Thresholding], ids=lambda op: op.__name__)
def test_every_op_states_its_policies_and_none_states_one_yet(op: type[KernelOp]) -> None:
    """Every op's policies are well formed, checked when its class was made. None states
    one yet: each choice is first valid, in listed order, as conversion took it before
    policies left the engine."""
    assert dict(op.policies) == {}
    assert stated(op.__qualname__, op.kernel, op.policies) == {}


def test_an_op_s_policy_is_set_at_conversion_wherever_its_class_is_placed() -> None:
    """A policy keyed by a class-level Decision applies to every placement of that class
    below the op's kernel: the packed core's reducer, fixed, under MatMul's ``compute``."""

    class Compressing(MatMul):
        policies = {PackedDotpKernel.reducer: Fixed("compressor")}

    model = matmul_model()
    node = model.graph.node[0]
    found = Compressing(node)
    found.attach_model(model)
    done = found.defaulted()
    assert isinstance(done, Defaulted)
    assert dict(inspection.states(done.point))["compute.packed.reducer"] == (
        inspection.Set("compressor")
    )


def test_an_op_s_policies_are_checked_when_its_class_is_made() -> None:
    """A Decision its kernel never reaches is refused where the class is made, and so is a
    value no policy (``default-policy-undeclared``, ``default-policy-invalid``)."""
    with refused("default-policy-undeclared", error=DefinitionError, message="MemStreamKernel"):

        class Stored(MatMul):
            policies = {MemStreamKernel.ram_style: Fixed("block")}

    with refused("default-policy-invalid", error=DefinitionError, message="Fixed"):

        class Unfixed(Thresholding):
            policies = {ThresholdingAxiKernel.ram_style: "auto"}  # type: ignore[dict-item]


def facts(weights: tuple[tuple[int, ...], ...] = STORED, **changed: object) -> Facts:
    x_tensor = Tensor((3, len(weights)), ScalarEncoding(INT3))
    arguments: dict[str, object] = {
        "platform": TARGET.platform,
        "weights_dtype": INT3,
        "x_tensor": x_tensor,
        "w_contents": IntegerTensorValue.of(weights),
    }
    return Facts(MatMulKernel, arguments | changed)


def test_the_member_cache_keys_a_member_by_its_class_and_arguments_by_value() -> None:
    with member_cache() as cache:
        first = configured(facts()).point
        assert configured(facts()).point is first
        assert (cache.hits, cache.misses) == (1, 1)
        # Other weights are another key: nothing is invalidated, the new arguments miss.
        other = tuple(tuple(-value for value in row) for row in STORED)
        assert configured(facts(other)).point is not first
        assert cache.misses == 2
        # Stored choices are part of the key; the replay starts from the binding's entry.
        chosen = configured(facts(), {"compute": "packed"})
        assert chosen.point is not first and configured(facts()).point is first
        assert configured(facts(), {"compute": "packed"}) is chosen
    # With no cache open (a context without one), nothing is kept.
    assert Context().run(lambda: configured(facts()).point is not configured(facts()).point)


def test_a_key_tells_values_apart_by_their_exact_type() -> None:
    """1, 1.0 and True are equal in Python and three values to the engine."""
    keys = {key(facts(), {"pe": value}, {}, ()) for value in (1, 1.0, True)}
    assert len(keys) == 3
    assert typed((1, (True,))) != typed((1, (1,)))
    assert key(facts(), {}, {}, ()) == key(facts(), {}, {}, ())
    unhashable = Facts(MatMulKernel, {"x_tensor": X, "bad": bytearray(b"unhashable")})
    assert key(unhashable, {}, {}, ()) is None


def test_a_refused_binding_caches_nothing() -> None:
    broken = Facts(MatMulKernel, {"x_tensor": X})  # no platform
    with member_cache() as cache:
        with refused("formal-unsupplied", error=DefinitionError, message=r"^platform\b"):
            configured(broken)
        assert not cache.entries


def test_a_hit_loads_no_held_value() -> None:
    """An initializer is keyed by its digest, so a binding found in the cache never loads
    the integers of the value it was asked with."""
    model = matmul_model()
    first: Any = kernel_op(model, model.graph.node[0]).bound()
    assert first.w_contents == integer_tensor(WEIGHTS)
    again = kernel_op(model, model.graph.node[0])
    value = again.facts().arguments["w_contents"]
    assert again.bound() is first
    assert isinstance(value, IntegerTensorValue) and value._integers is None
    assert value._array is None


def test_the_verifying_mode_configures_each_hit_again_and_names_a_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Counted apart: this test's mismatch is written to its own directory, never the run's.
    monkeypatch.setattr(members, "_COUNTS", dict.fromkeys(verification_counts(), 0))
    monkeypatch.setenv(VERIFY, str(tmp_path))
    before = verification_counts()["member hits verified"]
    with member_cache(verify=True) as cache:
        first = configured(facts())
        assert configured(facts()) is first
        assert verification_counts()["member hits verified"] == before + 1
        # An entry kept under another member's key is found out.
        (found,) = cache.entries
        cache.entries[found] = configured(facts(), {"compute": "packed"})
        with pytest.raises(MemberCacheMismatch, match="the member cache's member hit differs"):
            configured(facts())
    write_counts()
    (written,) = tmp_path.glob("*.json")
    assert json.loads(written.read_text())["mismatches"] == 1


def test_a_fact_changed_is_another_binding() -> None:
    first = kernel_op(matmul_model(), matmul_model().graph.node[0]).bound()
    changed = WEIGHTS.copy()
    changed[0, 0] += 1
    for model in (
        matmul_model(weights=changed),
        matmul_model(x_shape=[1, 6, 4]),
        matmul_model(x_shape=[3, 4]),  # the same rows, another tensor at the graph's rank
        matmul_model(stored=False),
    ):
        assert kernel_op(model, model.graph.node[0]).bound() is not first
    assert kernel_op(matmul_model(), matmul_model().graph.node[0]).bound() is first
    stored = matmul_model().get_initializer("w")
    assert stored is not None and np.array_equal(WEIGHTS, stored)
