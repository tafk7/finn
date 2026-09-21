# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One actual graph binding survives Region normalization and private renesting."""

import numpy as np

from finn.dataflow._engine import Absent, Decided
from finn.dataflow.analysis.integer_dot import DotProductBounds
from finn.dataflow.kernels.matmul.base import (
    AccumulationMode,
    ActivationMode,
    MatmulInterface,
    MvauComputationProfile,
)
from finn.dataflow.kernels.matmul.regions import construct_standard_embedded_mvau_region
from finn.dataflow.model.children import KernelChoice
from finn.dataflow.model.kernel import Kernel
from finn.dataflow.model.logical.authoring import NetworkBoundary, RegionDeclaration
from finn.dataflow.model.logical.composition import RegionResult, logical_network
from finn.dataflow.model.logical.interface import OperandExport, OperandTarget
from finn.dataflow.model.logical.interface_authoring import PublicOperandDeclaration
from finn.dataflow.model.logical.network import PositionMap, RegionEndpoint
from finn.dataflow.model.logical.refs import RegionInputRef, RegionOutputRef
from finn.dataflow.model.logical.region import InputInterface
from finn.dataflow.model.physical.authoring import PhysicallyUnsupported
from finn.dataflow.ops.base import DataflowOp
from finn.dataflow.ops.binding import ImplementationBinding, OperandBinding
from finn.dataflow.ops.mapping import CoordinateMapping, Internal, derive_public_operand_mappings
from finn.dataflow.ops.schema import DatatypeAttribute, OpInput, OpOutput
from finn.dataflow.space.declarations import (
    ConstraintGroup,
    Input,
    Subspace,
    constraint,
    declared_members,
    derived,
)

from dataflow.ops.test_dataflow_op import _mvau_model


def _exports(public, logical):
    """Fixture Kernel owns the sole body's private name and weight orientation."""
    network = logical_network(logical)
    assert len(network.nodes) == 1
    node = network.nodes[0]
    operand_id = {"activation": "X", "weights": "W", "result": "Y"}[public.key]
    if public.direction == "input":
        operand = node.region.input(operand_id).operand
        ref = RegionInputRef(node.id, operand_id)
        ports = tuple(item.port for item in node.region.inputs if isinstance(item, InputInterface))
    else:
        ports = tuple(item.port for item in node.region.outputs)
        operand = next(port.operand for port in ports if port.operand.id == operand_id)
        ref = RegionOutputRef(node.id, operand_id)
    if public.key == "weights":
        k, n = public.domain.extents
        mapping = PositionMap.affine(
            public.domain,
            view_extents=(k, n),
            sink=operand.position_domain,
            offset=0,
            coefficients=(1, k),
        )
    else:
        mapping = PositionMap.row_major_reshape(public.domain, operand.position_domain)
    presentations = tuple(
        RegionEndpoint(node.id, port.id) for port in ports if port.operand.id == operand_id
    )
    return OperandExport(public, (OperandTarget(ref, mapping, presentations),))


_PUBLIC_OPERANDS = tuple(
    PublicOperandDeclaration(item.key, item.direction, item.datatype, item.domain, _exports)
    for item in MatmulInterface.public_operands
)
_INPUT_NAMES = tuple(
    name for name, value in declared_members(MatmulInterface) if isinstance(value, Input)
)


class DirectMatrixKernel(Kernel, MatmulInterface):
    """An admitted bare integer matrix Region with fixed one-element lanes."""

    id = "test_direct_matrix"
    physical_unavailable = PhysicallyUnsupported("logical binding fixture has no generator")
    public_operands = _PUBLIC_OPERANDS

    @derived(int)
    def lanes():
        return 1

    region = RegionDeclaration(
        family="test.matrix",
        version="1",
        construct=construct_standard_embedded_mvau_region,
        repetitions=MatmulInterface.repetitions,
        matrix_width=MatmulInterface.matrix_width,
        matrix_height=MatmulInterface.matrix_height,
        activation_element_type=MatmulInterface.activation_type,
        weight_element_type=MatmulInterface.weight_type,
        output_element_type=MatmulInterface.result_type,
        pe=lanes,
        simd=lanes,
    )


def _composite(private_name):
    class CompositeMatrix(Kernel, MatmulInterface):
        id = "test_composite_matrix"
        public_operands = _PUBLIC_OPERANDS
        implementation = KernelChoice(
            Subspace(
                DirectMatrixKernel,
                **{name: getattr(MatmulInterface, name) for name in _INPUT_NAMES},
            ),
            node_id=private_name,
        )
        activation = NetworkBoundary(implementation.input("activation"))
        result = NetworkBoundary(implementation.output("output"))

    return CompositeMatrix


OldMatrix = _composite("old_compute")
RenamedMatrix = _composite("renamed_compute")


class ExtraEnclosure(MatmulInterface):
    renamed = Subspace(
        RenamedMatrix, **{name: getattr(MatmulInterface, name) for name in _INPUT_NAMES}
    )


class MatrixBindingOp(DataflowOp):
    """A test node with a fixed bare MatMul contract and ordinary graph operands."""

    family = "test.public_matrix_binding"
    family_version = "1"
    schema_version = 1
    implementation_binding = ImplementationBinding(("kernel",))
    choice_bindings = ()
    operand_bindings = (
        OperandBinding("activation", "activation", 0, adapter=CoordinateMapping.IDENTITY),
        OperandBinding("weight", "weights", 1, adapter=CoordinateMapping.IDENTITY),
        OperandBinding("output", "result", 0, output=True, adapter=CoordinateMapping.IDENTITY),
    )
    activation = OpInput(index=0, operand="X")
    weight = OpInput(index=1, operand="W")
    output = OpOutput(index=0, operand="Y")
    accumulator_type = DatatypeAttribute(default="INT32", onnx="accDataType")
    output_type = DatatypeAttribute(default="INT32", onnx="outputDataType")

    @constraint(activation_shape=activation.shape, weight_shape=weight.shape)
    def admitted_matrix_shapes(*, activation_shape, weight_shape):
        return (
            len(activation_shape) == len(weight_shape) == 2
            and all(extent > 0 for extent in (*activation_shape, *weight_shape))
            and activation_shape[-1] == weight_shape[0]
        )

    source_accepts = ConstraintGroup(admitted_matrix_shapes)

    @derived(int, shape=activation.shape)
    def repetitions(*, shape):
        return shape[0]

    @derived(int, shape=weight.shape)
    def matrix_width(*, shape):
        return shape[0]

    @derived(int, shape=weight.shape)
    def matrix_height(*, shape):
        return shape[1]

    @derived(MvauComputationProfile)
    def computation_profile():
        return MvauComputationProfile(AccumulationMode.INTEGER, ActivationMode.NONE)

    @derived(DotProductBounds)
    def integer_bounds():
        # This fixture deliberately uses full logical datatype bounds. Initializer
        # values below authenticate value correspondence, not a precision shortcut.
        return Absent()

    kernel = Subspace(
        DirectMatrixKernel,
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        activation_type=activation.datatype,
        weight_type=weight.datatype,
        accumulator_type=accumulator_type,
        output_type=output_type,
        computation_profile=computation_profile,
        integer_bounds=integer_bounds,
    )

    def expected_for(self, source):
        use = self.use_for_source(source)
        datatype, domain = use.operand_type("result"), use.operand_domain("result")
        return {
            "output": (
                domain.value.extents if isinstance(domain, Decided) else None,
                datatype.value if isinstance(datatype, Decided) else None,
            )
        }


_OP_INPUTS = {
    name: (
        MatrixBindingOp.activation.datatype
        if name == "activation_type"
        else MatrixBindingOp.weight.datatype
        if name == "weight_type"
        else getattr(MatrixBindingOp, name)
    )
    for name in _INPUT_NAMES
}


class CompositeBindingOp(MatrixBindingOp):
    kernel = Subspace(OldMatrix, **_OP_INPUTS)


class RenestedBindingOp(MatrixBindingOp):
    kernel = Subspace(ExtraEnclosure, **_OP_INPUTS)
    implementation_binding = ImplementationBinding(("kernel", "renamed"))


def _model():
    model = _mvau_model(repetitions=2, matrix_width=3, matrix_height=2)
    model.graph.node[0].op_type = "MatrixBindingFixtureOp"
    model.set_initializer("weight", np.asarray([[1, -2], [3, 4], [-5, 6]], dtype=np.float32))
    return model


def _mapped(use):
    implementation = use.require_implementation()
    logical = implementation.assess_view("logical").accepted_answer
    assert isinstance(logical, Decided), logical
    network = logical_network(logical.value)
    exports = {}
    for binding in use.operands:
        answer = use.operand_export(binding.role)
        assert isinstance(answer, Decided), answer
        exports[binding.source] = answer.value
    mappings = derive_public_operand_mappings(
        network, use.source, exports, {item.source: item.adapter for item in use.operands}
    )
    # Exercise the actual node consumer as well as its pure mapping service.
    # In particular a direct Region must not need an artificial wrapper Kernel.
    assert use.root.operand_mapping == Decided(mappings)
    return logical.value, mappings


def _body_weights(use, mappings):
    mapping = next(item for item in mappings if item.source_operand == "weight")
    initializer = use.source.operand("weight").initializer_value
    assert initializer is not None
    source_values = initializer.array_copy()
    body_values = np.empty(mapping.semantic_shape, dtype=source_values.dtype)
    for source_position in np.ndindex(source_values.shape):
        body_values[mapping.coordinate_map.mapped(source_position)] = source_values[source_position]
    return mapping, body_values


def test_actual_graph_binding_combines_private_rename_renest_and_weight_transpose():
    model = _model()
    graph_bytes = model.model.SerializeToString(deterministic=True)
    old = CompositeBindingOp(model.graph.node[0]).hydrate(model)
    renamed = RenestedBindingOp(model.graph.node[0]).hydrate(model)
    assert old.operands == renamed.operands == MatrixBindingOp.operand_bindings
    assert old.source == renamed.source
    _, old_maps = _mapped(old)
    _, renamed_maps = _mapped(renamed)
    old_weight, old_values = _body_weights(old, old_maps)
    new_weight, new_values = _body_weights(renamed, renamed_maps)
    assert old_weight.semantic_operand == RegionInputRef("old_compute", "W")
    assert new_weight.semantic_operand == RegionInputRef("renamed_compute", "W")
    assert old_weight.source_shape == new_weight.source_shape == (3, 2)
    assert old_weight.semantic_shape == new_weight.semantic_shape == (2, 3)
    np.testing.assert_array_equal(old_values, [[1, 3, -5], [-2, 4, 6]])
    np.testing.assert_array_equal(new_values, old_values)
    assert isinstance(old_weight.placement, Internal)
    assert isinstance(new_weight.placement, Internal)
    assert model.model.SerializeToString(deterministic=True) == graph_bytes
    assert len(model.graph.node) == 1


def test_actual_dataflow_node_binds_a_direct_region_with_required_unported_weights():
    model = _model()
    use = MatrixBindingOp(model.graph.node[0]).hydrate(model)
    logical, mappings = _mapped(use)
    assert isinstance(logical, RegionResult)
    mapping, body_values = _body_weights(use, mappings)
    assert mapping.semantic_operand == RegionInputRef("root", "W")
    assert mapping.placement == Internal("root", "W")
    assert mapping.unpresented_set.cardinality == 6
    assert use.operand_export("weights").value.targets[0].presentations == ()
    np.testing.assert_array_equal(body_values, [[1, 3, -5], [-2, 4, 6]])
    # Input facts and output inference are the same public contract even before
    # a consumer chooses whether it needs the normalized Network description.
    assert use.operand_domain("weights").value.extents == (3, 2)
    assert use.root.expected_outputs()["output"][0] == (2, 2)
