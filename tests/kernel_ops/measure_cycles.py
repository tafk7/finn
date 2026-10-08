# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Cycles measured in XSim against the schedules' prediction, on the Chain and TFC_W2A2.

``python -m kernel_ops.measure_cycles OUT [--frames N] [--only chain|tfc] [--work DIR]``
(from ``tests``, FinnLib
and Vivado selected) builds each partition as the kernel path builds it (the
Chain's KernelOp nodes, ``kernel_ops.models``; TFC_W2A2 at 16 lanes,
``kernel_ops.tfc``) and measures, with ``finn.harness.rtl.measure``,
``N`` frames streamed back to back, never stalled:

- the **stitched** partition, its root's latency, interval and total, and each
  layer between the hop into it and the hop out of it;
- each layer **alone**, the node as a one-node partition (its own adapter and
  memories), so that its interval is its own and not its slowest neighbour's.

Each layer's prediction is its schedule's ``beat_count`` (decision K10), beside
FINN's ``get_exp_cycles`` for the same folding (``MVAU``, ``Thresholding``). The
tables are printed, and written to ``OUT/cycles.json`` and ``OUT/cycles.md``.
Every frame's outputs are checked against ``execute_onnx`` of the partition's
nodes, as the XSim tests check them.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.registry import getCustomOp

from finn.custom_op.kernels.partition import member
from finn.custom_op.kernels.shell import ShellRoot, shell_root
from finn.dataflow.traversal import Traversal
from finn.harness.rtl import Words, link_stream, measure, pack
from finn.harness.toolchain import print_identity
from finn.transformation.kernels.package import boundary_facts


@dataclass(frozen=True)
class Layer:
    """One node's prediction and measurements (cycles, both ends counted)."""

    name: str
    op: str
    folding: str
    beats: int
    """The schedule's beat count: the prediction."""
    finn: int
    """FINN's ``get_exp_cycles`` for the same folding."""
    stitched_span: int
    """In the stitched partition, the steady frame's first beat into the layer to its last
    beat out."""
    stitched_busy: int
    """In the stitched partition, the steady frame's first to last beat out of the layer."""
    alone_interval: int
    """Alone, the steady state's interval."""
    alone_latency: int
    """Alone, the steady frame's latency."""
    alone_first: int
    """Alone, the first frame's latency."""


def schedule_of(root: ShellRoot, node: NodeProto) -> Any:
    kernel = getattr(root.point.partition, member(node.name))
    return kernel.compute.schedule if node.op_type == "MatMul" else kernel.schedule


def finn_cycles(node: NodeProto, schedule: Any) -> int:
    """FINN's ``get_exp_cycles`` of its own op at this folding."""
    domain = "finn.custom_op.fpgadataflow"
    if node.op_type == "MatMul":
        m, n, k = schedule.order
        finn = helper.make_node(
            "MVAU",
            ["x", "w"],
            ["y"],
            domain=domain,
            MW=schedule.extent(k),
            MH=schedule.extent(n),
            SIMD=schedule.factor(k),
            PE=schedule.factor(n),
            numInputVectors=[schedule.extent(m)],
            inputDataType="INT8",
            weightDataType="INT8",
            outputDataType="INT32",
        )
    else:
        *rows, c = schedule.order
        finn = helper.make_node(
            "Thresholding",
            ["x", "t"],
            ["y"],
            domain=domain,
            NumChannels=schedule.extent(c),
            PE=schedule.factor(c),
            numInputVectors=[schedule.extent(row) for row in rows],
            inputDataType="INT8",
            weightDataType="INT8",
            outputDataType="INT8",
        )
    op: Any = getCustomOp(finn)
    return int(op.get_exp_cycles())


def folding(node: NodeProto, schedule: Any) -> str:
    if node.op_type == "MatMul":
        _, n, k = schedule.order
        return f"pe {schedule.factor(n)}, simd {schedule.factor(k)}"
    return f"pe {schedule.factor(schedule.order[-1])}"


def words(form: Traversal, values: NDArray[Any], bits: int) -> list[int]:
    """The beats ``form`` presents of ``values``, each packed lane zero lowest."""
    flat = values.reshape(form.shape)
    return [pack([int(flat[position]) for position in beat], bits) for beat in form.positions()]


def boundary_words(
    model: ModelWrapper, root: ShellRoot, context: Mapping[str, Any], label: str
) -> tuple[dict[str, Words], dict[str, Words]]:
    """Each boundary port's words of one frame, inputs then outputs, from ``context``."""
    inputs, outputs = boundary_facts(model, root.point, root.boundary, label)
    found: tuple[dict[str, Words], dict[str, Words]] = ({}, {})
    for side, facts in zip(found, (inputs, outputs)):
        for each in facts:
            ends = getattr(root.point, member(each["tensor"])).endpoints
            end = ends.source if ends.source_owner is None else ends.sink
            bits = each["element_bits"]
            side[each["port"]] = (
                words(end.form, context[each["tensor"]], bits),
                bits * each["lanes"],
            )
    return found


def measure_partition(
    model: ModelWrapper,
    feed: dict[str, NDArray[Any]],
    directory: Path,
    *,
    frames: int,
    label: str,
) -> dict[str, Any]:
    """The stitched partition of ``model``'s nodes and each node alone, measured."""
    context = execute_onnx(model, feed, return_full_exec_context=True)
    nodes = list(model.graph.node)
    root = shell_root(model, nodes, name=label)
    inputs, outputs = boundary_words(model, root, context, label)
    print(f"== {label}: stitched", flush=True)
    stitched = measure(
        root.point.module, directory / "stitched", inputs=inputs, outputs=outputs, frames=frames
    )
    # The hop out of each layer, by the layer's leaves: its member and what lies under it.
    links = root.point.module.fragment.links
    hops: list[str] = []
    for node in nodes:
        name = member(node.name)
        out = [
            link
            for link in links
            if link.source.instance is not None
            and (link.source.instance == name or link.source.instance.startswith(name + "."))
            and not (link.sink.instance or "").startswith(name + ".")
            and link.sink.instance != name
        ]
        if len(out) != 1:
            raise ValueError(f"{label}: {node.name} has {len(out)} hops out")
        hops.append(link_stream(out[0]) or next(iter(outputs)))
    into = [next(iter(inputs)), *hops[:-1]]
    layers = []
    recorded = {"stitched": dict(stitched.beats)}
    for index, node in enumerate(nodes):
        schedule = schedule_of(root, node)
        alone_root = shell_root(model, [node], name=node.name)
        alone_in, alone_out = boundary_words(model, alone_root, context, node.name)
        print(f"== {label}: {node.name} alone", flush=True)
        alone = measure(
            alone_root.point.module,
            directory / node.name,
            inputs=alone_in,
            outputs=alone_out,
            frames=frames,
        )
        recorded[node.name] = dict(alone.beats)
        layers.append(
            Layer(
                name=node.name,
                op=node.op_type,
                folding=folding(node, schedule),
                beats=schedule.beat_count,
                finn=finn_cycles(node, schedule),
                stitched_span=stitched.span(into[index], hops[index])[-1],
                stitched_busy=stitched.busy(hops[index])[-1],
                alone_interval=alone.interval(),
                alone_latency=alone.latencies[-1],
                alone_first=alone.latencies[0],
            )
        )
    return {
        "label": label,
        "frames": frames,
        "latencies": list(stitched.latencies),
        "intervals": list(stitched.intervals()),
        "total": stitched.total,
        "bottleneck": max(layer.beats for layer in layers),
        "beats_sum": sum(layer.beats for layer in layers),
        "finn_sum": sum(layer.finn for layer in layers),
        "layers": [asdict(layer) for layer in layers],
        "streams": {
            name: {
                "beats": stitched.per_frame(name),
                "busy": stitched.busy(name)[-1],
                "interval": stitched.interval(name),
                "first": stitched.of(name)[0][0] - stitched.of(next(iter(inputs)))[0][0],
            }
            for name in stitched.beats
        },
        # Every handshake's cycle, by simulation and stream.
        "recorded": recorded,
    }


def table(result: dict[str, Any]) -> str:
    """One partition's result as Markdown."""
    lines = [
        f"### {result['label']} ({result['frames']} frames back to back, never stalled)",
        "",
        f"Stitched: latency per frame {result['latencies']}; intervals {result['intervals']};"
        f" total {result['total']}. Predicted: bottleneck {result['bottleneck']}, sum of the"
        f" layers' beats {result['beats_sum']} (FINN {result['finn_sum']}).",
        "",
        "| layer | op | folding | beats | FINN | stitched span | stitched busy out"
        " | alone II | alone latency | alone first frame |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in result["layers"]:
        lines.append(
            f"| {layer['name']} | {layer['op']} | {layer['folding']} | {layer['beats']}"
            f" | {layer['finn']} | {layer['stitched_span']} | {layer['stitched_busy']}"
            f" | {layer['alone_interval']} | {layer['alone_latency']} | {layer['alone_first']} |"
        )
    lines += [
        "",
        "| stitched stream | beats/frame | busy | II | first beat |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stream in result["streams"].items():
        lines.append(
            f"| {name} | {stream['beats']} | {stream['busy']} | {stream['interval']}"
            f" | {stream['first']} |"
        )
    return "\n".join(lines)


def chain_case() -> tuple[ModelWrapper, dict[str, NDArray[Any]]]:
    from kernels import chain  # noqa: PLC0415

    from kernel_ops.models import configure_partition, kernel_model  # noqa: PLC0415

    model = kernel_model()
    configure_partition(model)
    return model, {"x": np.array(chain.X, dtype=np.float32)}


def tfc_case(directory: Path) -> tuple[ModelWrapper, dict[str, NDArray[Any]]]:
    from kernel_ops.tfc import SHAPE, partitioned  # noqa: PLC0415

    directory.mkdir(parents=True, exist_ok=True)
    _, _, body = partitioned(directory)
    image = np.random.default_rng(3).integers(0, 256, size=SHAPE).astype(np.float32)
    return body, {body.graph.input[0].name: image.reshape(1, -1)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--only", choices=("chain", "tfc"))
    parser.add_argument("--work", type=Path, help="the simulations' directory (OUT/work)")
    args = parser.parse_args(argv)
    print_identity()
    out: Path = args.out.resolve()
    work: Path = (args.work or out / "work").resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    if args.only in (None, "chain"):
        model, feed = chain_case()
        results.append(
            measure_partition(model, feed, work / "chain", frames=args.frames, label="chain")
        )
    if args.only in (None, "tfc"):
        model, feed = tfc_case(work / "tfc-model")
        results.append(
            measure_partition(model, feed, work / "tfc", frames=args.frames, label="tfc_w2a2")
        )
    (out / "cycles.json").write_text(json.dumps(results, indent=2) + "\n")
    text = "\n\n".join(table(result) for result in results) + "\n"
    (out / "cycles.md").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
