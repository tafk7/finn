# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical required-operand, request and artifact integrity at actual use boundaries."""

from dataclasses import replace

import pytest
from onnx import helper
from qonnx.core.datatype import DataType

from parked.dataflow.physical_fixture import configure, roots, source_model, template_roots
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.artifacts.abi import Bus
from finn.kernels.artifacts.build import FixedModuleName
from finn.parked.dataflow.kernels.matmul.base import MatmulInterface
from finn.parked.dataflow.model.physical.interface import PhysicalResult
from finn.parked.dataflow.model.logical.interface_authoring import PublicOperandDeclaration
from finn.dataflow.model.logical.region import BeatSequence
from finn.parked.dataflow.ops.base import DataflowOpError
from finn.parked.dataflow.ops.binding import ChoiceBinding, ImplementationBinding
from finn.parked.dataflow.ops.native import NativeAttribute, SCHEMA_VERSION_ATTRIBUTE, serialize_choices
from finn.parked.dataflow.ops.schema import Attribute
from finn.parked.dataflow.ops.model_effects import ModelReadSet
from finn.parked.dataflow.ops.mvau.op import MvauSpace
from finn.parked.dataflow.ops.reconstruction import build_space
from finn.parked.dataflow.ops.physical import (
    authorize_component_use,
    capture_op_physical,
    capture_local_physical,
    install_compiler_physical_component,
    install_physical_component,
    materialize_build_request,
    materialize_local_physical,
    prepare_build_request,
    prepare_local_physical,
    validate_physical_build_association,
)
from finn.kernels.space.declarations import (
    ConstraintGroup,
    Decision,
    Input,
    Projection,
    Readiness,
    Subspace,
    constraint,
    derived,
)


def _point(index=0):
    model, build, _context = source_model()
    operation = configure(
        model.get_customop_wrapper(model.graph.node[index]).set_context(build).space
    )
    return model, build, operation


@pytest.mark.parametrize(
    "field",
    [
        "request_hash",
        "required_binding",
        "tensor",
        "role",
        "port",
        "source_scope",
        "source_reads",
        "incoming_context",
        "occurrence",
        "dependency_hash",
        "local_hash",
    ],
)
def test_altered_meaningful_capture_fields_refuse_before_preparation(field, tmp_path):
    model, build, operation = _point()
    capture = capture_op_physical(operation)
    association = capture.association
    if field == "request_hash":
        association = replace(association, requirements_fingerprint="false")
    elif field == "required_binding":
        association = replace(association, operands=association.operands[:-1])
    elif field in {"tensor", "role", "port"}:
        first = association.operands[0]
        if field == "tensor":
            first = replace(first, tensor="another_tensor")
        elif field == "role":
            first = replace(first, role="weights")
        else:
            first = replace(first, port=None)
        association = replace(association, operands=(first, *association.operands[1:]))
    elif field == "source_scope":
        association = replace(association, scope_id="wrong")
    elif field == "source_reads":
        association = replace(association, source_reads=ModelReadSet())
    elif field == "incoming_context":
        association = replace(association, incoming_context="not-the-frozen-contract")
    elif field == "occurrence":
        capture = replace(
            capture,
            local=replace(capture.local, occurrence_token=capture.local.occurrence_token + 1),
        )
    elif field == "dependency_hash":
        capture = replace(capture, local=replace(capture.local, point_fingerprint="false"))
    else:
        capture = replace(capture, local=replace(capture.local, physical_fingerprint="false"))
    forged = replace(capture, association=association)
    assert validate_physical_build_association(operation, forged, model=model, build=build)
    with pytest.raises(DataflowOpError, match="stale or invalid"):
        prepare_build_request(
            operation,
            forged,
            model=model,
            build=build,
            roots=roots(),
            template_roots=template_roots(),
            blobs=ArtifactStore(tmp_path / "unopened"),
        )


def test_one_built_artifact_is_reusable_by_distinct_valid_node_uses_without_context(tmp_path):
    model, build, left = _point()
    right = configure(model.get_customop_wrapper(model.graph.node[1]).set_context(build).space)
    first, second = capture_op_physical(left), capture_op_physical(right)
    assert first.local.point_fingerprint != second.local.point_fingerprint
    assert first.requirements == second.requirements
    assert first.association.incoming_context is None
    store = ArtifactStore(tmp_path / "shared")
    prepared = prepare_local_physical(
        first.local, roots=roots(), template_roots=template_roots(), blobs=store
    )
    built = materialize_local_physical(prepared, store=store)
    del prepared  # the built result retains its real preparation receipt
    authorized = authorize_component_use(
        right, second, built, model=model, build=build, store=store
    )
    installed = install_compiler_physical_component(
        right, authorized, outer_instance_id="right", model=model, build=build, store=store
    )
    assert installed.component == built.component
    assert installed.association.scope_id == second.association.scope_id
    with pytest.raises(DataflowOpError, match="stale or invalid"):
        authorize_component_use(right, first, built, model=model, build=build, store=store)


def test_supplied_graph_connection_claim_is_checked_at_installation_not_codegen(tmp_path):
    model, build, context = source_model()
    first = context.graph_inputs[0]
    incompatible = replace(
        context,
        graph_inputs=(
            replace(
                first,
                contract=replace(
                    first.contract,
                    beat_sequence=BeatSequence(2, (((0, 1), (0, 0)), ((0, 3), (0, 2)))),
                ),
            ),
            *context.graph_inputs[1:],
        ),
    )
    operation = configure(
        model.get_customop_wrapper(model.graph.node[0])
        .set_context(build, graph_context=incompatible)
        .space
    )
    capture = capture_op_physical(operation)
    store = ArtifactStore(tmp_path / "connection")
    request = prepare_build_request(
        operation,
        capture,
        model=model,
        build=build,
        graph_context=incompatible,
        roots=roots(),
        template_roots=template_roots(),
        blobs=store,
    )
    component = materialize_build_request(
        operation, request, model=model, build=build, graph_context=incompatible, store=store
    )
    with pytest.raises(DataflowOpError, match="graph stream connections"):
        install_physical_component(
            operation,
            request,
            component,
            outer_instance_id="bad_stream",
            model=model,
            build=build,
            graph_context=incompatible,
            store=store,
        )


def test_unconsumed_native_choice_and_output_annotation_cache_do_not_stale_codegen_use():
    class DiagnosticChoice(MvauSpace):
        diagnostic = Decision(bool, values=(False, True))
        diagnostic_tag = Attribute(int, default=0)
        choice_bindings = (
            *MvauSpace.choice_bindings,
            ChoiceBinding("diagnostic", (), "diagnostic"),
        )

    model, build, _context = source_model()
    node = model.graph.node[0]
    configured = configure(build_space(DiagnosticChoice, model, node, build=build)).assign(
        DiagnosticChoice.diagnostic, False
    )
    encoded = serialize_choices(configured)
    encoded[SCHEMA_VERSION_ATTRIBUTE] = NativeAttribute("i", DiagnosticChoice.schema_version)
    node.attribute.extend(value.proto(key) for key, value in encoded.items())
    operation = build_space(DiagnosticChoice, model, node, build=build)
    capture = capture_op_physical(operation)
    assert not any(
        item.field == "diagnostic" for item in capture.association.source_reads.expectations
    )
    attribute = next(item for item in node.attribute if item.name == "diagnostic")
    attribute.CopyFrom(helper.make_attribute("diagnostic", 1))
    node.attribute.append(helper.make_attribute("diagnostic_tag", 17))
    model.set_tensor_datatype(node.output[0], DataType["INT8"])
    model.set_tensor_shape(node.output[0], (99, 99))
    assert not validate_physical_build_association(operation, capture, model=model, build=build)
    current = build_space(DiagnosticChoice, model, node, build=build)
    assert current.answer(DiagnosticChoice.diagnostic).value is True
    assert capture_op_physical(current).requirements == capture.requirements
    physical_key = "kernel__dot_product__simd"
    assert any(item.field == physical_key for item in capture.association.source_reads.expectations)
    folding = next(item for item in node.attribute if item.name == physical_key)
    folding.CopyFrom(helper.make_attribute(physical_key, 1))
    assert validate_physical_build_association(operation, capture, model=model, build=build)


@pytest.mark.parametrize(
    "field", ["requirements_fingerprint", "prepared_fingerprint", "manifest_fingerprint"]
)
def test_tampered_built_artifact_fields_refuse(field, tmp_path):
    model, build, operation = _point()
    capture = capture_op_physical(operation)
    store = ArtifactStore(tmp_path / "built")
    prepared = prepare_local_physical(
        capture.local, roots=roots(), template_roots=template_roots(), blobs=store
    )
    built = materialize_local_physical(prepared, store=store)
    with pytest.raises(DataflowOpError):
        authorize_component_use(
            operation,
            capture,
            replace(built, **{field: "false"}),
            model=model,
            build=build,
            store=store,
        )


def _constant_physical_interface_point():
    """Abstract interface fixture only; no additional generator/profile is emitted."""
    model, build, production = _point()
    original = capture_op_physical(production).local.physical
    requirements = replace(
        original.requirements,
        implementation_id="required-value-interface-fixture",
        abi=replace(
            original.requirements.abi,
            entry_point=FixedModuleName("required_value_fixture"),
            ports=tuple(
                item
                for item in original.requirements.abi.ports
                if not isinstance(item, Bus) or item.name != "in1_V"
            ),
        ),
        contributions=(),
        render_inputs=(),
    )
    interface = PhysicalResult(
        requirements,
        tuple(port for port in original.ports if port.role != "weights"),
        required_values=("weights",),
    )

    class RequiredValueCore(MatmulInterface):
        id = "required-value-interface"
        version = "1"

        @derived(PhysicalResult)
        def result():
            return interface

        ready = Readiness()
        physical = Projection(result, readiness=ready)
        interface_mode = Decision(int, values=(0, 1))
        interface_ready = Decision(bool, values=(False, True))
        interface_present = Decision(bool, values=(False, True))
        domain_requirement = Input(int)

        @constraint(mode=interface_mode)
        def mode_supported(*, mode):
            return mode == 0

        mode_accepts = ConstraintGroup(mode_supported)

        @constraint(policy=domain_requirement)
        def domain_supported(*, policy):
            return policy == 0

        domain_accepts = ConstraintGroup(domain_supported)
        domain_readiness = Readiness(properties=(MatmulInterface.result_domain,))
        public_result_domain = Projection(
            MatmulInterface.result_domain,
            readiness=domain_readiness,
            constraints=domain_accepts,
        )
        interface_readiness = Readiness(
            properties=(MatmulInterface.result_type,),
            decisions=(interface_ready,),
            constraints=MatmulInterface.type_support,
        )
        public_result_type = Projection(
            MatmulInterface.result_type,
            readiness=interface_readiness,
            applicable_if=interface_present,
            constraints=(MatmulInterface.type_support, mode_accepts),
        )
        public_operands = (
            *MatmulInterface.public_operands[:2],
            PublicOperandDeclaration("result", "output", public_result_type, public_result_domain),
        )

    class RequiredValueSpace(MvauSpace):
        implementation_binding = ImplementationBinding(("implementation",))
        interface_binding = None
        choice_bindings = (
            *MvauSpace.choice_bindings,
            ChoiceBinding("interface_mode", ("implementation",), "interface_mode"),
            ChoiceBinding("interface_ready", ("implementation",), "interface_ready"),
            ChoiceBinding("interface_present", ("implementation",), "interface_present"),
        )
        source_type_limit = Attribute(int, default=1)
        domain_requirement = Attribute(int, default=0)

        @constraint(limit=source_type_limit)
        def source_type_supported(*, limit):
            return limit > 0

        type_source_accepts = ConstraintGroup(
            *MvauSpace.type_source_accepts.constraints,
            source_type_supported,
        )
        implementation = Subspace(
            RequiredValueCore,
            repetitions=MvauSpace.repetitions,
            matrix_width=MvauSpace.matrix_width,
            matrix_height=MvauSpace.matrix_height,
            activation_type=MvauSpace.activation.datatype,
            weight_type=MvauSpace.weight.datatype,
            accumulator_type=MvauSpace.accumulator_type,
            output_type=MvauSpace.output_type,
            computation_profile=MvauSpace.profile,
            integer_bounds=MvauSpace.integer_bounds,
            domain_requirement=domain_requirement,
        )

    node = model.graph.node[0]
    operation = build_space(RequiredValueSpace, model, node, build=build).commit_choices(
        {
            "interface_mode": 0,
            "interface_ready": True,
            "interface_present": True,
        }
    )
    encoded = serialize_choices(operation)
    encoded[SCHEMA_VERSION_ATTRIBUTE] = NativeAttribute("i", RequiredValueSpace.schema_version)
    node.attribute.extend(value.proto(key) for key, value in encoded.items())
    return model, build, build_space(RequiredValueSpace, model, node, build=build)


def test_unported_required_value_cannot_disappear_from_the_use_record(tmp_path):
    model, build, operation = _constant_physical_interface_point()
    capture = capture_op_physical(operation)
    weights = next(item for item in capture.association.operands if item.role == "weights")
    assert weights.port is None
    assert weights.tensor == model.graph.node[0].input[1]
    assert not validate_physical_build_association(operation, capture, model=model, build=build)
    missing = replace(
        capture,
        association=replace(
            capture.association,
            operands=tuple(item for item in capture.association.operands if item.role != "weights"),
        ),
    )
    with pytest.raises(DataflowOpError, match="stale or invalid"):
        prepare_build_request(
            operation,
            missing,
            model=model,
            build=build,
            roots=roots(),
            template_roots=template_roots(),
            blobs=ArtifactStore(tmp_path / "not-generated"),
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("accDataType", "INT32"),
        ("outputDataType", "INT32"),
        ("noActivation", 0),
        ("source_type_limit", 0),
        ("domain_requirement", 1),
        ("interface_mode", 1),
        ("interface_ready", None),
        ("interface_present", 0),
    ],
)
def test_interface_only_type_readiness_constraint_and_source_type_reads_are_retained(name, value):
    model, build, operation = _constant_physical_interface_point()
    capture = capture_op_physical(operation)
    assert not any(item.kind in {"problem", "decision"} for item in capture.local.dependencies)
    attributes = {item.field for item in capture.association.source_reads.expectations}
    assert {
        "accDataType",
        "outputDataType",
        "noActivation",
        "binaryXnorMode",
        "source_type_limit",
        "domain_requirement",
        "interface_mode",
        "interface_ready",
        "interface_present",
    } <= attributes
    assert any(
        item.field == "input:2" and item.expected is None
        for item in capture.association.source_reads.expectations
    )  # output type acceptance consumed the absence of optional thresholds
    node = model.graph.node[0]
    retained = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(retained)
    if value is not None:
        node.attribute.append(helper.make_attribute(name, value))
    assert validate_physical_build_association(operation, capture, model=model, build=build)
    # The local codegen query remains constant: these are exclusively the extra
    # interface/source-type premises consumed at the compiler association boundary.
    current = build_space(type(operation), model, node, build=build)
    assert (
        capture_local_physical(current.require_implementation()).dependencies
        == capture.local.dependencies
    )
