# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Acceptance at conversion: ``ToKernelOps`` asks the kernels about each node a pattern
matches, on its inputs as the nodes before it state them. A node the kernels refuse
stays on the host with their codes; a fact the graph does not state is
``fact-unstated``; a contradiction stops the build. After its defaults, each stream of
the node's kernel must meet an open AXIS boundary (its channel's own rules and
policies): a stream no chain of modules joins keeps the node on the host with the
channel's findings."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from onnx import NodeProto, TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from support.models import DOMAIN, TARGET, convert
from support.refusals import refused

import finn.custom_op.kernels as domain
from finn.core.graph.annotations.platform import read_target, write_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.facts import edge_tensor
from finn.core.reporting.findings import Finding
from finn.core.space import (
    Available,
    Inapplicable,
    QueryResult,
    Unresolved,
    design_space,
    inspection,
    reject,
)
from finn.core.space.declarations import MemberRef
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.thresholding import Thresholding
from finn.dataflow.traversal import LevelEnd
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel, ChannelFifo, open_boundary_refusal
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.policies import FirstValid, Fixed, Required, stated
from finn.kernels.port import port_surface
from finn.kernels.sides import Side
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.transformation.kernels import (
    InferKernelTensors,
    Outcome,
    between_kernel_ops,
    kernel_ops_report,
    kernel_ops_summary,
)

GENERAL = "qonnx.custom_op.general"


def model_of(
    nodes: list[NodeProto],
    inputs: dict[str, tuple[list[int], str | None]],
    stored: dict[str, tuple[Any, str | None]],
) -> ModelWrapper:
    """The ``nodes``, ``inputs`` (shape, datatype) and ``stored`` initializers (values,
    datatype; None: unannotated); y the last node's output."""
    infos = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, s) for name, (s, _) in inputs.items()
    ]
    y = helper.make_tensor_value_info(nodes[-1].output[0], TensorProto.FLOAT, None)
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(nodes, "acceptance", infos, [y]),
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(GENERAL, 1)],
        )
    )
    for name, (values, _) in stored.items():
        model.set_initializer(name, np.asarray(values, dtype=np.float32))
    for name, (_, dtype) in {**inputs, **stored}.items():
        if dtype is not None:
            model.set_tensor_datatype(name, DataType[dtype])
    return model


def matmul(
    x: str | None = "INT4", w: str | None = "INT4", weights: Any = np.eye(4), name: str = "mm"
) -> ModelWrapper:
    """x (3, 4) -> MatMul ``name`` with the initializer w -> y."""
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name=name)
    return model_of([node], {"x": ([3, 4], x)}, {"w": (weights, w)})


def multithreshold_node(x: str, t: str, y: str, name: str = "mt") -> NodeProto:
    return helper.make_node(
        "MultiThreshold", [x, t], [y], name=name, domain=GENERAL, out_dtype="UINT2"
    )


SORTED = [[-1.5, 0.5, 2.5]] * 4
UNSORTED = [[2.5, -1.5, 0.5]] * 4  # count(x >= t) is any order's; the kernel's RTL needs one


def multithreshold(x: str = "INT4", t: str = "FLOAT32", table: Any = SORTED) -> ModelWrapper:
    """x (3, 4) -> MultiThreshold ``mt`` with the (4, 3) initializer t -> y (UINT2)."""
    return model_of([multithreshold_node("x", "t", "y")], {"x": ([3, 4], x)}, {"t": (table, t)})


def rejected(model: ModelWrapper) -> list[tuple[str, str]]:
    """The (owner, code) of each finding of the one node, which stays on the host as it
    was."""
    (source,) = model.graph.node
    converted, conversion = convert(model)
    assert list(converted.graph.node) == [source]
    (outcome,) = conversion.outcomes
    assert outcome.op is None
    return [(f.owner, f.code) for f in outcome.findings if f.kind.value == "rejection"]


def test_an_integer_matmul_and_multithreshold_convert() -> None:
    for model, op in ((matmul(), "MatMul"), (multithreshold(), "Thresholding")):
        assert convert(model)[1].outcomes == (Outcome((model.graph.node[0].name,), op),)


#: Each core's ports' rules on float operands: the element their hardware takes.
FLOAT_PORTS = [
    (f"compute.{core}.{port}.element_taken", "dtype-family")
    for core in ("int8_dsp58", "packed")
    for port in ("w", "x")
]


def test_a_float_matmul_stays_on_the_host_with_the_kernels_codes() -> None:
    # Each core refuses with the same finding, reported once, and each core's ports by
    # their own rules (every rule of a present Space counts); float weights that are not
    # integers are no contradiction of FLOAT32.
    for weights in (np.eye(4), np.full((4, 4), 0.5)):
        found = rejected(matmul("FLOAT32", "FLOAT32", weights))
        assert found == [*FLOAT_PORTS, ("result_range", "matmul-arithmetic")]


def test_an_int20_matmul_stays_on_the_host_with_each_cores_code() -> None:
    """INT20 activations on a DSP48E2: the packed core's DSP input is too narrow, and
    the INT8 core needs a DSP58. Each case of the ``compute`` Decision says why."""
    assert sorted(rejected(matmul("INT20", "INT20"))) == [
        ("compute.int8_dsp58.core_supported", "dotp-target"),
        ("compute.packed.core_supported", "dotp-activation-width"),
    ]


def test_an_int32_matmul_stays_on_the_host_with_each_cores_code() -> None:
    """INT32 activations on a DSP48E2: the kernels refuse the node, each case of the
    ``compute`` Decision by its code. Its sums pass 2**24, which no longer matters:
    FINN executes the ONNX it covers with exact integers."""
    assert sorted(rejected(matmul("INT32", "INT32"))) == [
        ("compute.int8_dsp58.core_supported", "dotp-target"),
        ("compute.packed.core_supported", "dotp-activation-width"),
    ]


def test_a_float_multithreshold_stays_on_the_host_with_the_kernels_codes() -> None:
    assert rejected(multithreshold("FLOAT32", "FLOAT32")) == [
        ("table_supported", "threshold-type"),
        ("types_supported", "dtype-family"),
    ]


def test_an_unannotated_input_is_an_unstated_fact() -> None:
    (source,) = matmul(x=None).graph.node
    _, conversion = convert(matmul(x=None))
    (outcome,) = conversion.outcomes
    assert outcome.op is None
    ((kind, owner, code, tensor),) = [
        (f.kind.value, f.owner, f.code, dict(f.details)["tensor"]) for f in outcome.findings
    ]
    assert (kind, owner, code, tensor) == ("limitation", "MatMul", "fact-unstated", "x")


def test_a_contradiction_still_stops_the_conversion() -> None:
    with refused(
        "initializer-out-of-datatype",
        error=KernelOpError,
        owner="admitted",
        node="mm",
        tensor="w",
        message="is annotated INT2 and holds values over",
    ):
        convert(matmul(w="INT2", weights=np.full((4, 4), 3.0)))


def test_a_refused_trial_leaves_the_node_as_inference_leaves_it() -> None:
    """Unsorted thresholds: the kernels refuse their order. The node stays, its thresholds
    as they were, and the model is what stating the target and inferring every node would
    make of it."""
    model = multithreshold(table=UNSORTED)
    expected = model.transform(_Stated()).transform(InferKernelTensors())
    converted, conversion = convert(model)
    (outcome,) = conversion.outcomes
    assert [f.code for f in outcome.findings] == ["threshold-order"]
    assert converted.get_tensor_datatype("t").name == "FLOAT32"
    table = converted.get_initializer("t")
    assert table is not None and np.array_equal(table, np.asarray(UNSORTED, dtype=np.float32))
    assert converted.model.SerializeToString() == expected.model.SerializeToString()


class _Stated(InferKernelTensors):
    """The target stated and the domain imported, as ``ToKernelOps`` does first."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        write_target(model, TARGET)
        model.set_opset_import(DOMAIN, 1)
        return model, False


def test_conversion_reads_the_types_the_nodes_before_state() -> None:
    """``h`` carries a stale INT20 annotation, which the second MatMul's cores refuse.
    Conversion states ``h`` from the first MatMul's kernel (its weights' columns) before
    it asks the kernels about the second, so both convert."""
    assert ("compute.packed.core_supported", "dotp-activation-width") in rejected(
        matmul("INT20", "INT4")
    )
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"], name="a"),
        helper.make_node("MatMul", ["h", "w"], ["y"], name="b"),
    ]
    model = model_of(nodes, {"x": ([3, 4], "INT4")}, {"w": (np.eye(4), "INT4")})
    model.set_tensor_datatype("h", DataType["INT20"])
    converted, conversion = convert(model)
    assert [outcome.op for outcome in conversion.outcomes] == ["MatMul", "MatMul"]
    assert converted.get_tensor_datatype("h").name == "INT4"


def test_a_matmul_after_a_relu_reads_its_exact_type() -> None:
    """qonnx types Relu(INT8) by its range, UINT7, so the MatMul after it converts (its
    rule before typed the Relu FLOAT32, which the kernels refused: matmul-arithmetic)."""
    nodes = [
        helper.make_node("Relu", ["x"], ["r"], name="relu"),
        helper.make_node("MatMul", ["r", "w"], ["y"], name="mm"),
    ]
    model = model_of(nodes, {"x": ([3, 4], "INT8")}, {"w": (np.eye(4), "INT4")})
    converted, conversion = convert(model)
    assert [outcome.op for outcome in conversion.outcomes] == [None, "MatMul"]
    assert converted.get_tensor_datatype("r").name == "UINT7"


def test_a_refused_node_between_kernel_ops_is_a_host_node_there() -> None:
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"], name="a"),
        multithreshold_node("h", "t", "levels"),
        helper.make_node("MatMul", ["levels", "w"], ["y"], name="b"),
    ]
    model = model_of(
        nodes, {"x": ([3, 4], "INT4")}, {"w": (np.eye(4), "INT4"), "t": (UNSORTED, "FLOAT32")}
    )
    converted, conversion = convert(model)
    assert [outcome.op for outcome in conversion.outcomes] == ["MatMul", None, "MatMul"]
    assert between_kernel_ops(converted) == ("mt",)
    report = kernel_ops_report(converted, conversion.outcomes)
    assert report["on_host"] == report["between_kernel_ops"] == ["mt"]
    assert kernel_ops_summary(conversion.outcomes)[1:] == [
        "ToKernelOps:   threshold-order (rejection) 1: mt"
    ]


# -- the stream check ------------------------------------------------------------------------


class MarkedThresholding(Thresholding):
    """Test-only: Thresholding whose input closes a frame every 4 beats and every 6, two
    marker levels: an input generator can make neither, so no chain of modules joins an
    open boundary to it (``adapter-steps``). It takes Thresholding's place in the domain."""

    op_type = "MarkedThresholding"

    def streamed(self, point: Kernel) -> tuple[tuple[str, Side | tuple[Finding, ...], bool], ...]:
        found = []
        for tensor, side, output in super().streamed(point):
            if not output and isinstance(side, Side) and side.stream is not None:
                sequence = replace(side.stream.sequence, markers=(LevelEnd(4), LevelEnd(6)))
                side = replace(side, stream=replace(side.stream, sequence=sequence))
            found.append((tensor, side, output))
        return tuple(found)


def in_place_of_thresholding(monkeypatch: pytest.MonkeyPatch, op: type[Thresholding]) -> None:
    """``op``, a test-only Thresholding, takes Thresholding's place in the domain."""
    monkeypatch.setattr(domain, op.__name__, op, raising=False)
    monkeypatch.setattr(
        domain,
        "__all__",
        [op.__name__ if name == "Thresholding" else name for name in domain.__all__],
    )


def test_a_stream_no_chain_joins_keeps_the_node_on_the_host_with_the_channel_s_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """x (3, 4) is 12 beats; the input's two marker levels are what no chain of modules
    makes from what an open boundary presents. The kernel takes its defaults, so its own
    rules refuse nothing: the channel's do, by their owners in the channel, about x."""
    assert convert(multithreshold())[1].outcomes == (Outcome(("mt",), "Thresholding"),)
    in_place_of_thresholding(monkeypatch, MarkedThresholding)
    found = rejected(multithreshold())
    assert ("input_chain", "adapter-steps") in found
    assert {code for _, code in found} == {"adapter-steps", "guard-refused"}
    _, conversion = convert(multithreshold())
    (outcome,) = conversion.outcomes
    assert {dict(each.details)["tensor"] for each in outcome.findings} == {"x"}
    (steps,) = [each for each in outcome.findings if each.code == "adapter-steps"]
    assert (
        steps.message
        == "no FinnLib chain carries out the steps: no input_gen level closes every 4 beats"
    )


#: The channel's policies with its transport fixed to the FIFO: its depth is required.
THROUGH_A_FIFO = stated(
    "Channel",
    Channel,
    {
        Channel.transport: Fixed("fifo"),
        ChannelFifo.depth: Required(),
        MemStreamKernel.ram_style: FirstValid(order=MemStreamKernel.ram_preference),
        Channel.adapter_ram_style: FirstValid(order=Channel.adapter_ram_preference),
    },
)


def test_a_stream_joined_through_a_fifo_whose_depth_is_open_converts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validity left open by a required choice only is no refusal: with the transport a
    FIFO, the check's channel waits on its depth, which sizing sets, and the node converts
    with no finding. (An open boundary presents what its other side presents, so no channel
    of a real kernel refuses ``direct`` where a FIFO is accepted: the policy is fixed here.)"""
    monkeypatch.setattr(Channel, "policies", THROUGH_A_FIFO)
    model = multithreshold()
    converted, conversion = convert(model)
    assert conversion.outcomes == (Outcome(("mt",), "Thresholding"),)
    op = kernel_op(converted, converted.graph.node[0])
    point = op.point()
    platform = read_target(converted).platform
    for tensor, side, output in op.streamed(point):
        assert isinstance(side, Side)
        boundary = Side.boundary("m_axis_0" if output else "s_axis_0")
        channel = design_space(
            Channel(
                tensor=edge_tensor(converted, tensor, op.label),
                producer=side if output else boundary,
                consumer=boundary if output else side,
                platform=platform,
            )
        )
        assert open_boundary_refusal(side, channel.tensor, platform, output=output) == ()
        done = channel.with_choices(transport="fifo")
        assert inspection.validity(done).status == "open"
        assert dict(inspection.states(done))["transport.fifo.depth"] == inspection.Open()


class OpenPeThresholding(Thresholding):
    """Test-only: Thresholding whose policies leave the folding ``pe`` open (required), so
    the kernel taken at its defaults has both its sides waiting on it."""

    op_type = "OpenPeThresholding"
    policies = {ThresholdingAxiKernel.pe: Required()}


def test_a_stream_whose_side_waits_keeps_the_node_on_the_host_with_the_side_s_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The kernel's own rules refuse nothing and no choice lacks a value, but with ``pe``
    left open neither stream's side (``x_side``, ``y_side``) is known: the stream check
    cannot meet them with a boundary, and the node stays on the host with the side's
    findings, each about its tensor, rather than converting with the stream unchecked."""
    assert convert(multithreshold())[1].outcomes == (Outcome(("mt",), "Thresholding"),)
    in_place_of_thresholding(monkeypatch, OpenPeThresholding)
    model = multithreshold()
    (source,) = model.graph.node
    converted, conversion = convert(model)
    assert list(converted.graph.node) == [source]
    (outcome,) = conversion.outcomes
    assert outcome.op is None
    assert {(each.owner, each.code, dict(each.details)["tensor"]) for each in outcome.findings} == {
        ("pe", "decision-unassigned", "x"),
        ("pe", "decision-unassigned", "y"),
    }


def test_only_a_port_that_is_not_placed_is_skipped_by_the_stream_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``KernelOp.streamed`` skips a port whose side is ``Inapplicable`` (the point does not
    place it); a side that is ``Unresolved`` or ``Rejected`` is given as its findings, which
    refuse the check; an ``Available`` one as the side, labelled below the node's name."""
    converted, _ = convert(multithreshold())
    op = kernel_op(converted, converted.graph.node[0])
    point = op.point()
    known = point.query(cast(Any, port_surface(type(point), "input").side))
    assert isinstance(known, Available) and isinstance(known.value, Side)
    waits = Unresolved(reject("decision-unassigned", "pe is open", owner="pe").findings)
    refused = reject("channel-form", "no form", owner="x")
    original_query = type(point).query

    def streamed(**answers: QueryResult[Side]) -> Any:
        # Keep the actual kernel and native Port references; intercept only the two
        # sides under test, delegating every other query to the real implementation.
        observed: list[str] = []

        def answer(receiver: Kernel, request: Any) -> QueryResult[Any]:
            if (
                receiver is point
                and isinstance(request, MemberRef)
                and request.member.name == "side"
            ):
                for port, value in answers.items():
                    if [node.name for node in request.path] == [port]:
                        observed.append(port)
                        return value
            return original_query(receiver, request)

        with monkeypatch.context() as control:
            control.setattr(type(point), "query", answer)
            result = op.streamed(point)
        assert observed == ["input", "output"]
        return result

    assert streamed(input=Inapplicable(), output=known) == (("y", known.value.under("mt"), True),)
    assert streamed(input=known, output=Inapplicable()) == (("x", known.value.under("mt"), False),)
    assert streamed(input=Inapplicable(), output=Inapplicable()) == ()
    assert streamed(input=waits, output=known) == (
        ("x", waits.findings, False),
        ("y", known.value.under("mt"), True),
    )
    assert streamed(input=known, output=refused) == (
        ("x", known.value.under("mt"), False),
        ("y", refused.findings, True),
    )
