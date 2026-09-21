# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An actually uncommitted type stays Unresolved through a graph, without caches."""

import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.dataflow import custom_op
from finn.dataflow._engine import Decided, Unresolved
from finn.dataflow.model.logical.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.model.logical.interface_authoring import PublicOperandDeclaration
from finn.dataflow.model.logical.maps import RectangularDomain
from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp
from finn.dataflow.ops.binding import ChoiceBinding, ImplementationBinding, OperandBinding
from finn.dataflow.ops.mapping import CoordinateMapping
from finn.dataflow.ops.native import (
    FINGERPRINT_ATTRIBUTE,
    SCHEMA_VERSION_ATTRIBUTE,
    NativeAttribute,
    serialize_choices,
)
from finn.dataflow.ops.persistence import assign_dataflow_scope_ids
from finn.dataflow.ops.replay.op import ActivationReplayOp
from finn.dataflow.ops.schema import OpInput, OpOutput
from finn.dataflow.ops.type_context import producer_type
from finn.dataflow.space.declarations import (
    Decision,
    Input,
    Projection,
    Readiness,
    Space,
    Subspace,
    SubspaceChoice,
    derived,
)


class _Interface(Space):
    activation_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    shape = Input(tuple)

    @derived(RectangularDomain, shape=shape)
    def domain(*, shape):
        return RectangularDomain(shape)

    shape_ready = Readiness(properties=(domain,))
    public_domain = Projection(domain, readiness=shape_ready)
    input_ready = Readiness()
    public_input_type = Projection(activation_type, readiness=input_ready)


class _PrecisionInterface(_Interface):
    precision = Decision(int, values=(8, 16))

    @derived(QONNX_DATATYPE_VALUE_SEMANTICS, bits=precision)
    def result_type(*, bits):
        return DataType[f"INT{bits}"]

    type_ready = Readiness(properties=(result_type,))
    public_result_type = Projection(result_type, readiness=type_ready)
    public_operands = (
        PublicOperandDeclaration(
            "activation", "input", _Interface.public_input_type, _Interface.public_domain
        ),
        PublicOperandDeclaration("result", "output", public_result_type, _Interface.public_domain),
    )


class _FixedTypeInterface(_Interface):
    result_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    type_ready = Readiness()
    public_result_type = Projection(result_type, readiness=type_ready)
    public_operands = (
        PublicOperandDeclaration(
            "activation", "input", _Interface.public_input_type, _Interface.public_domain
        ),
        PublicOperandDeclaration("result", "output", public_result_type, _Interface.public_domain),
    )


class _PartialOp(DataflowOp):
    """Test-only type authoring fixture; it makes no logical/codegen claim."""

    family = "test.partial_type"
    activation = OpInput(index=0)
    output = OpOutput(index=0)
    implementation_binding = ImplementationBinding(("kernel",))
    operand_bindings = (
        OperandBinding("activation", "activation", 0, adapter=CoordinateMapping.IDENTITY),
        OperandBinding("output", "result", 0, output=True, adapter=CoordinateMapping.IDENTITY),
    )

    def expected_for(self, source):
        use = self.use_for_source(source)
        datatype = use.operand_type("result")
        return {
            "output": (
                source.operand("activation").shape,
                datatype.value if isinstance(datatype, Decided) else None,
            )
        }


class DecisionTypeOp(_PartialOp):
    kernel = Subspace(
        _PrecisionInterface,
        activation_type=_PartialOp.activation.datatype,
        shape=_PartialOp.activation.shape,
    )
    choice_bindings = (ChoiceBinding("precision", ("kernel",), "precision"),)


class AlternativeTypeOp(_PartialOp):
    @derived(QONNX_DATATYPE_VALUE_SEMANTICS)
    def narrow():
        return DataType["INT8"]

    @derived(QONNX_DATATYPE_VALUE_SEMANTICS)
    def wide():
        return DataType["INT16"]

    kernel = SubspaceChoice(
        {
            "narrow": Subspace(
                _FixedTypeInterface,
                activation_type=_PartialOp.activation.datatype,
                shape=_PartialOp.activation.shape,
                result_type=narrow,
            ),
            "wide": Subspace(
                _FixedTypeInterface,
                activation_type=_PartialOp.activation.datatype,
                shape=_PartialOp.activation.shape,
                result_type=wide,
            ),
        }
    )
    choice_bindings = (ChoiceBinding("implementation", ("kernel",), "case"),)


def _chain(operation_type):
    def tensor(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 4])

    nodes = [
        helper.make_node(
            operation_type.__name__, ["x"], ["middle"], name="producer", domain=DATAFLOW_DOMAIN
        )
    ]
    for index, source in enumerate(("middle", "downstream0")):
        nodes.append(
            helper.make_node(
                "ActivationReplayOp",
                [source],
                [f"downstream{index}"],
                name=f"replay{index}",
                domain=DATAFLOW_DOMAIN,
                neuron_folds=1,
            )
        )
    graph = helper.make_graph(
        nodes,
        "partial-types",
        [tensor("x")],
        [tensor("downstream1")],
        value_info=[tensor("middle"), tensor("downstream0")],
    )
    model = ModelWrapper(
        helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid(DATAFLOW_DOMAIN, 1),
            ],
        )
    )
    model.set_tensor_datatype("x", DataType["INT8"])
    for name in ("middle", "downstream0", "downstream1"):
        model.set_tensor_datatype(name, DataType["INT32"])
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


@pytest.mark.parametrize(
    ("operation_type", "choices"),
    (
        (DecisionTypeOp, {"precision": 16}),
        (AlternativeTypeOp, {"implementation": "wide"}),
    ),
)
def test_unresolved_producer_type_propagates_and_later_choice_resolves(
    monkeypatch, operation_type, choices
):
    monkeypatch.setitem(custom_op, operation_type.__name__, operation_type)
    model = _chain(operation_type)
    before = model.model.SerializeToString(deterministic=True)
    initial = operation_type(model.graph.node[0]).hydrate(model)
    assert isinstance(initial.operand_type("result"), Unresolved)
    if operation_type is DecisionTypeOp:
        assert isinstance(initial.resolve(), Decided)
    else:
        assert isinstance(initial.resolve(), Unresolved)
    old_consumers = [ActivationReplayOp(node).hydrate(model) for node in model.graph.node[1:]]
    for name in ("middle", "downstream0", "downstream1"):
        assert isinstance(producer_type(model, name), Unresolved)
        assert model.get_tensor_datatype(name) == DataType["INT32"]
    for use in old_consumers:
        assert isinstance(use.operand_type("activation"), Unresolved)
        assert isinstance(use.operand_type("result"), Unresolved)
        assert use.source.inputs[0].datatype is None
        assert isinstance(use.source.inputs[0].datatype_answer, Unresolved)
    assert model.model.SerializeToString(deterministic=True) == before

    successor = initial.commit(choices)
    assert successor.operand_type("result") == Decided(DataType["INT16"])
    saved = serialize_choices(successor.root)
    saved[FINGERPRINT_ATTRIBUTE] = NativeAttribute("s", successor.root.local_problem_fingerprint)
    saved[SCHEMA_VERSION_ATTRIBUTE] = NativeAttribute("i", successor.root.schema_version)
    model.graph.node[0].attribute.extend(value.proto(key) for key, value in saved.items())
    after_choice = model.model.SerializeToString(deterministic=True)
    for name in ("middle", "downstream0", "downstream1"):
        assert producer_type(model, name) == Decided(DataType["INT16"])
        assert model.get_tensor_datatype(name) == DataType["INT32"]
    for node in model.graph.node[1:]:
        assert ActivationReplayOp(node).hydrate(model).operand_type("result") == Decided(
            DataType["INT16"]
        )
    assert isinstance(initial.operand_type("result"), Unresolved)
    assert all(isinstance(use.operand_type("result"), Unresolved) for use in old_consumers)
    assert model.model.SerializeToString(deterministic=True) == after_choice
