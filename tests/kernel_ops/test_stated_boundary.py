# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What the build states of an ``ip`` partition's boundary (``DataflowBuildConfig.boundary``),
from the cut to the IP.

The cut resolves each port the build states to the tensor it carries and writes the
statements on the partition's body, by tensor (``finn.boundary``,
``finn.core.graph.annotations.boundary``); the graph layer reads them where it makes a
boundary side, on a shell whose row offers no mover (``ip``): the side carries the build's
statement (``Outside``, its source ``BUILD``). FIFO sizing reads its pace, the partition's
frame is no shorter than its span (the edge channel's cycles), the edge channel
presents its lanes (a width step, the port's AXIS width the stated word's), and what it
leaves out is as the partition presents it: an output as produced. A slice reads the
statement of an edge that is a partition's port, by its tensor. On the Chain
(``support.models.kernel_model``), cut as a build cuts it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from qonnx.core.modelwrapper import ModelWrapper
from support.models import kernel_model
from support.packaging import PackagedByStub
from support.refusals import refused
from support.tfc import ULTRA96

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.build_dataflow_steps import (
    step_create_dataflow_partition,
    step_size_fifos,
    step_verify_partition,
)
from finn.core.graph.annotations.boundary import (
    BOUNDARY,
    STATEMENTS,
    read_boundary,
    write_boundary,
)
from finn.core.graph.annotations.platform import write_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import partition_body
from finn.custom_op.kernels.network import network, partition_ports
from finn.custom_op.kernels.shell import InShell, in_shell
from finn.kernels.fifo_sizing import Pattern, sides
from finn.kernels.movers import iodma_hls
from finn.kernels.sides import BUILD, ROW_MAJOR, BeatTimes, Outside, Steady
from finn.toolchain.launcher import Toolchain
from finn.transformation.kernels import PackagePartition
from finn.transformation.kernels.cut import CutKernelPartition

#: The Chain's input, stated a beat every other cycle and nothing else; its output stated
#: four lanes a beat, one pass, its three beats at the end of a twelve-cycle frame.
X_STATED = {"pace": {"every": 2}}
Y_STATED = {"lanes": 4, "passes": 1, "pace": {"times": [9, 10, 11], "span": 12}}


def build(tmp_path: Path, boundary: dict[str, Any]) -> DataflowBuildConfig:
    """An ``ip`` build for Ultra96's part at 5 ns stating ``boundary``, its output in
    ``tmp_path``."""
    target = {"period_ns": 5.0, "part": "xczu3eg-sbva484-1-e"}
    stated = {"output_dir": str(tmp_path), "target": target, "boundary": boundary}
    return DataflowBuildConfig.from_dict(stated)


def cut(tmp_path: Path, boundary: dict[str, Any], **options: Any) -> ModelWrapper:
    """The Chain's KernelOps cut as the build cuts them, ``boundary`` stated, its FIFOs
    sized (``step_size_fifos``): the parent graph."""
    cfg = build(tmp_path, boundary)
    parent = step_create_dataflow_partition(kernel_model(**options), cfg)
    return step_size_fifos(parent, cfg)


def body_of(parent: ModelWrapper) -> InShell:
    """The partition of the parent's body, in its shell."""
    _, body, _ = partition_body(parent)
    return in_shell(body, body.graph.node)


def sized(tmp_path: Path) -> dict[str, tuple[int, int | None, str]]:
    """Each boundary channel's FIFO as step_size_fifos proposed it: its depth, its least
    words and why."""
    report = json.loads((tmp_path / "report" / "step_size_fifos.json").read_text())
    channels = report["strategy"]["channels"]
    return {
        name: (channels[name]["depth"], channels[name]["least"], channels[name]["why"])
        for name in ("x", "y")
    }


def period(tmp_path: Path) -> int:
    """The bottleneck period step_size_fifos sized at."""
    report = json.loads((tmp_path / "report" / "step_size_fifos.json").read_text())
    return cast(int, report["strategy"]["period"])


def test_the_build_s_statements_are_stored_by_tensor_their_pace_as_given() -> None:
    """``finn.boundary`` holds the build's statements by tensor, each field as stated: a
    steady beat stays the shorthand. Written whole (nothing stated, nothing stored); a
    mover's statement is never stored, and a stored one not well formed is refused, by
    name."""
    model = kernel_model()
    assert read_boundary(model) == {}
    statements = {
        "x": Outside(BUILD, pace=Steady(2)),
        "y": Outside(BUILD, ROW_MAJOR, 1, 4, BeatTimes((9, 10, 11), 12)),
    }
    write_boundary(model, statements)
    assert read_boundary(model) == statements
    assert (model.get(STATEMENTS) or {})["x"] == {
        "order": None,
        "passes": None,
        "lanes": None,
        "pace": {"every": 2},
    }
    write_boundary(model, {})
    assert model.namespace(BOUNDARY) == {}
    with refused(
        "boundary-invalid", error=KernelOpError, message=r"^x: the model stores the build's"
    ):
        write_boundary(model, {"x": iodma_hls(128).states()})
    model.set_metadata_prop("finn.boundary/@version", "1")
    model.set_metadata_prop("finn.boundary/statements", json.dumps({"x": {"pace": {"every": 0}}}))
    with refused("boundary-unreadable", error=KernelOpError, message="every 0"):
        read_boundary(model)


def test_the_cut_writes_the_statements_on_the_body_and_refuses_a_tensor_no_port_carries(
    tmp_path: Path,
) -> None:
    """The cut writes the statements it is given, by tensor, on the body it cuts, before it
    makes the channels; the parent graph states none. A tensor no port carries is refused,
    naming the partition's ports, before anything is cut."""
    model = kernel_model()
    assert partition_ports(model) == {"x": "s_axis_0", "y": "m_axis_0"}
    stated = {"x": Outside(BUILD, pace=Steady(2))}
    parent = model.transform(CutKernelPartition(tmp_path, stated))
    _, body, _ = partition_body(parent)
    assert read_boundary(body) == stated and read_boundary(parent) == {}
    assert partition_ports(body) == dict(in_shell(body, body.graph.node).boundary)
    with refused(
        "boundary-port-unknown",
        error=KernelOpError,
        message=r"^hidden: no port of the partition carries it, so the build states nothing "
        r"there; its ports are s_axis_0 \(x\), m_axis_0 \(y\)$",
    ):
        kernel_model().transform(CutKernelPartition(tmp_path / "other", {"hidden": stated["x"]}))
    assert not (tmp_path / "other").exists()


def test_an_ip_partition_is_sized_from_the_paces_its_build_states(tmp_path: Path) -> None:
    """The input stated a beat every other cycle: its supply is that beat at the six beats
    its edge presents, which direct absorbs. The output stated three beats at the end of
    the frame: its acceptance is those beat times, and step_size_fifos places a FIFO of
    the three words they keep waiting. Unstated, both are not modelled."""
    bound = body_of(cut(tmp_path, {"s_axis_0": X_STATED, "m_axis_0": Y_STATED}))
    x, y = bound.member("x"), bound.member("y")
    assert x.producer.outside == Outside(BUILD, pace=Steady(2))
    assert y.consumer.outside == Outside(BUILD, None, 1, 4, BeatTimes((9, 10, 11), 12))
    supply, _ = sides(x, bound.neighbours("x"))
    assert supply == Pattern((0, 2, 4, 6, 8, 10), 12)
    _, accept = sides(y, bound.neighbours("y"))
    assert accept == Pattern((9, 10, 11), 12)
    assert sized(tmp_path) == {
        "x": (0, 0, "direct absorbs it"),
        "y": (2, 3, "least 3 words"),
    }
    cut(tmp_path / "unstated", {})
    assert sized(tmp_path / "unstated") == {
        "x": (0, None, "a boundary: not modelled"),
        "y": (0, None, "a boundary: not modelled"),
    }


@pytest.mark.parametrize(("port", "tensor"), [("s_axis_0", "x"), ("m_axis_0", "y")])
def test_a_stated_pace_slower_than_the_members_bounds_the_partition_s_frame(
    tmp_path: Path, port: str, tensor: str
) -> None:
    """The Chain's members take 12 cycles a frame at most; a port stated a beat every third
    cycle spans 18 over its six beats, an input's supply or an output's acceptance. The
    partition runs no faster than its outside: the stated channel's cycles are 18, and
    step_size_fifos sizes every channel at that period, the stated one from its pace
    (direct absorbs it); the build's FIFO check passes. Unstated, the period is 12."""
    stated = {port: {"pace": {"every": 3}}}
    parent = step_verify_partition(cut(tmp_path, stated), build(tmp_path, stated))
    channel = body_of(parent).member(tensor)
    assert channel.boundary.form.beats == 6 and channel.cycles == 18
    assert period(tmp_path) == 18
    assert sized(tmp_path)[tensor] == (0, 0, "direct absorbs it")
    cut(tmp_path / "unstated", {})
    assert period(tmp_path / "unstated") == 12


def test_stated_lanes_place_a_width_step_and_the_port_takes_the_stated_word(
    tmp_path: Path,
) -> None:
    """The output stated four lanes a beat, where the Chain produces two: the edge channel
    places a width step (a ``vpc``) before its transport, and its AXIS port is the stated
    word, four INT5 lanes padded to 24 bits (16 unstated), three beats a frame. The
    interface description states the port as stated, its lanes, beats and TDATA; it
    reports no pace, so nothing expands one."""
    parent = step_verify_partition(
        cut(tmp_path, {"m_axis_0": Y_STATED}), build(tmp_path, {"m_axis_0": Y_STATED})
    )
    y = body_of(parent).member("y")
    assert y.output_adapting and [stage.kind for stage in y.output_adapter.realization] == ["vpc"]
    assert (y.boundary.form.lanes, y.boundary.form.beats) == (4, 3)
    assert y.boundary.transport.data_width == 24
    unstated = body_of(cut(tmp_path / "unstated", {})).member("y")
    assert not unstated.output_adapting and unstated.boundary.transport.data_width == 16
    _, body, _ = partition_body(parent)
    stub = cast(Toolchain, PackagedByStub())
    body.transform(PackagePartition("sdp_1", directory=tmp_path / "ip", toolchain=stub))
    described = json.loads((tmp_path / "ip" / "interface.json").read_text())
    (output,) = [stream for stream in described["streams"] if stream["name"] == "m_axis_0"]
    assert (output["lanes"], output["beats"], output["tdata"]) == (4, 3, 24)
    assert output["order"]["row_major"] is True and "pace" not in output


def test_an_output_stated_only_its_pace_is_as_produced(tmp_path: Path) -> None:
    """A statement of a pace alone leaves the order, the passes and the lanes as the
    partition presents them: the output's port is the one it produces, nothing placed."""
    stated = body_of(cut(tmp_path, {"m_axis_0": {"pace": {"every": 2}}})).member("y")
    unstated = body_of(cut(tmp_path / "unstated", {})).member("y")
    assert stated.consumer.outside == Outside(BUILD, pace=Steady(2))
    assert stated.boundary == unstated.boundary and not stated.output_adapting


def test_a_slice_reads_the_statement_of_an_edge_that_is_a_partition_port(
    tmp_path: Path,
) -> None:
    """The Chain with its w2 streamed: the partition's ports are x, w2 and y. A slice of
    its second MatMul alone names w2 its first input port, its other input the frozen
    neighbour's; it reads the statement of w2 by its tensor all the same."""
    stated = {"s_axis_1": {"pace": {"every": 1}}}
    parent = cut(tmp_path, stated, second_weights=False)
    _, body, _ = partition_body(parent)
    assert partition_ports(body) == {"x": "s_axis_0", "w2": "s_axis_1", "y": "m_axis_0"}
    (second,) = [node for node in body.graph.node if node.name == "second"]
    sliced = network(body, [second], name="second")
    assert dict(sliced.boundary) == {"w2": "s_axis_0", "y": "m_axis_0"}
    w2 = sliced.member("w2")
    assert w2.producer.label == "s_axis_0"
    assert w2.producer.outside == Outside(BUILD, pace=Steady(1))
    assert sliced.member("y").consumer.outside is None


def test_a_build_statement_on_a_boundary_offered_a_mover_is_refused_where_the_graph_reads_it(
    tmp_path: Path,
) -> None:
    """The build refuses a statement on a shell whose row offers a mover before its first
    step; a model annotated by hand that states one there is refused where the graph
    makes the side, naming the port and the mover."""
    model = kernel_model()
    write_target(model, ULTRA96)
    write_boundary(model, {"x": Outside(BUILD, pace=Steady(2))})
    with refused(
        "boundary-stated-twice",
        error=KernelOpError,
        message=r"^chain: the model's boundary statements state s_axis_0 \(x\), and the "
        r"boundary is offered an iodma_hls mover, which states each side it is offered on: "
        r"a build statement would be a second source$",
    ):
        in_shell(model, model.graph.node, name="chain")
