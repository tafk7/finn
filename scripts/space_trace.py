# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Trace where named Spaces' checks read, in several configurations, and write it as JSON.

A prototype's runner (``finn.core.space._trace``). Each Space is traced in more than one
configuration (with nothing committed, and with its baselines committed, at least) and
the traces joined: every read names the configurations that showed it. One JSON file a
Space set, and ``index.json`` with every Space's counts:

- ``pipeline``: the toy Pipeline (``tests/core/space/_pipeline_support.py``): nothing,
  its baselines, a few chosen cores, and each step of the seed 6, step 23 sequence, with
  that sequence's states, the burst's own check after it, and a new commit of the
  same value (``repro``);
- ``dotp``: the two dotp cores, placed as their conformance cases place them;
- ``matmul``: MatMul's node root, placed as its conformance case places it;
- ``tfc``: every kernel of TFC's network, each its KernelOp's own root: nothing chosen,
  the baselines stored at conversion, and the choices exploration stores at 16 lanes
  (``ExploreKernelChoices(Ranked(Lanes(16)))``);
- ``chain8``: stage 7's synthetic chain of eight kernels and its channels as one Space,
  nothing committed and the network's baselines committed (the engine pass's profile
  reader, ``probe_profile.py``).

usage: PYTHONPATH=src:tests python scripts/space_trace.py OUT_DIR [SET ...]
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from finn.core.space import Space, _trace, design_space, inspection, occurrence
from finn.core.space import _runtime as runtime

Traced = tuple[str, Space]


def baselined(point: Space) -> Space:
    """``point`` with every open choice that has viable cases and is not required at its
    first viable case the engine accepts, in rank order (as ``Seam.settle`` does with no
    prior value, for a Space that is not a network)."""
    infos = inspection.decisions(point)
    owned = {item.key: item.reference for item in infos}
    required = {item.key for item in infos if item.required}
    skipped: set[str] = set()
    while True:
        pending = [
            item
            for item in inspection.viable(point)
            if item.cases and item.key not in required and item.key not in skipped
        ]
        if not pending:
            return point
        choice = pending[0]
        for case in choice.cases:
            result = point.try_with_choices({owned[choice.key]: case})
            if result.accepted:
                point = result.instance
                break
        else:
            skipped.add(choice.key)


def committed(point: Space, choices: dict[str, object]) -> Space:
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    result = point.try_with_choices({owned[key]: value for key, value in choices.items()})
    assert result.accepted, choices
    return result.instance


def traced(name: str, about: str, configurations: Iterable[Traced]) -> dict[str, Any]:
    """One Space's joined trace, its relations and counts."""
    started = time.perf_counter()
    traces = [_trace.trace(point, configuration) for configuration, point in configurations]
    joined = _trace.join(traces)
    return {
        "space": name,
        "about": about,
        "configurations": joined["configurations"],
        "summary": _trace.summary(joined),
        "relations": _trace.relations(joined),
        "joined": joined,
        "seconds": round(time.perf_counter() - started, 2),
    }


# -- the toy Pipeline -------------------------------------------------------------------


def pipeline() -> list[dict[str, Any]]:
    from core.space._pipeline_support import SEED_6_STEP_23, Pipeline  # noqa: PLC0415

    point = design_space(Pipeline(width=10))
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    configurations: list[Traced] = [
        ("nothing", point),
        ("baselines", baselined(point)),
        ("u0 packed", committed(point, {"u0.compute": "packed"})),
        ("u0 stub", committed(point, {"u0.compute": "stub"})),
        (
            "u0, u1, u2 packed, probe on",
            committed(
                point,
                {
                    "u0.compute": "packed",
                    "u1.compute": "packed",
                    "u2.compute": "packed",
                    "probe.enabled": True,
                },
            ),
        ),
    ]
    steps = []
    current = point
    for number, (values, clears) in enumerate(SEED_6_STEP_23, 1):
        changes = [current.field(owned[key]).clear() for key in clears]
        result = current.try_with_choices({owned[k]: v for k, v in values.items()}, *changes)
        assert result.accepted, (values, clears)
        current = result.instance
        name = f"seed 6 step 23, after step {number}"
        configurations.append((name, current))
        steps.append(
            {
                "step": number,
                "commits": {key: repr(value) for key, value in values.items()},
                "clears": list(clears),
                "accepted": result.accepted,
                "configuration": name,
            }
        )
    # After the sequence: the burst stays committed, and its own check waits.
    snapshot = occurrence.state(current)
    burst = snapshot.linked.keys["l1.burst"]
    check = runtime.membership(snapshot, burst, 1)
    cleared = current.try_with_choices({}, current.field(owned["l1.burst"]).clear())
    again = cleared.instance.try_with_choices({owned["l1.burst"]: 1})
    repro = {
        "sequence": steps,
        "burst_committed": repr(snapshot.assignments.get(burst)),
        "burst_check_after": {
            "outcome": type(check.result).__name__,
            "codes": sorted({finding.code for finding in getattr(check.result, "findings", ())}),
            "owners": sorted({finding.owner for finding in getattr(check.result, "findings", ())}),
        },
        "new_commit_of_the_same_value": {
            "accepted": again.accepted,
            "refused": [
                {
                    "owner": item.owner,
                    "result": type(item.result).__name__,
                    "codes": sorted(
                        {finding.code for finding in getattr(item.result, "findings", ())}
                    ),
                }
                for item in again.outcomes
                if item.status == "refused"
            ],
        },
    }
    found = traced(
        "Pipeline",
        "the toy Pipeline (width 10): three units, two links, a probe and a meter",
        configurations,
    )
    found["repro"] = repro
    return [found]


# -- the dotp cores and MatMul ----------------------------------------------------------


def dotp() -> list[dict[str, Any]]:
    from finn.kernels.dotp import Int8Dsp58DotpKernel, PackedDotpKernel  # noqa: PLC0415
    from finn.platform.capabilities import DspBlock  # noqa: PLC0415
    from support.kernel_specs.base import placed  # noqa: PLC0415
    from support.kernel_specs.dotp import dotp as case  # noqa: PLC0415

    found = []
    for name, space_type, dsp, bits in (
        ("PackedDotpKernel", PackedDotpKernel, DspBlock.DSP48E2, 4),
        ("Int8Dsp58DotpKernel", Int8Dsp58DotpKernel, DspBlock.DSP58, 8),
    ):
        point = placed(case(space_type, dsp, bits))
        found.append(
            traced(
                name,
                f"{name} placed as its conformance case places it ({dsp.name}, INT{bits})",
                [("nothing", point), ("baselines", baselined(point))],
            )
        )
    return found


def matmul() -> list[dict[str, Any]]:
    from finn.platform.capabilities import DspBlock  # noqa: PLC0415
    from support.helpers import full_platform  # noqa: PLC0415
    from support.kernel_specs.base import placed  # noqa: PLC0415
    from support.kernel_specs.matmul import matmul as case  # noqa: PLC0415

    spec = case()
    point = placed(spec)
    return [
        traced(
            "MatMulKernel",
            "MatMul's node root, placed as its conformance case places it (INT4, DSP48E2), "
            "and on a DSP58 platform",
            [
                ("nothing", point),
                ("baselines", baselined(point)),
                ("compute packed", placed(spec, {"compute": "packed"})),
                ("on DSP58: nothing", placed(spec, platform=full_platform(DspBlock.DSP58))),
                (
                    "on DSP58: baselines",
                    baselined(placed(spec, platform=full_platform(DspBlock.DSP58))),
                ),
                (
                    "on DSP58: the INT8 core, then baselines",
                    baselined(
                        placed(
                            spec,
                            {"compute": "int8_dsp58"},
                            platform=full_platform(DspBlock.DSP58),
                        )
                    ),
                ),
            ],
        )
    ]


# -- TFC's network ----------------------------------------------------------------------


def tfc() -> list[dict[str, Any]]:
    from finn.custom_op.kernels.base import kernel_op  # noqa: PLC0415
    from finn.transformation.kernels import ExploreKernelChoices  # noqa: PLC0415
    from finn.kernels.explore import Ranked  # noqa: PLC0415
    from support.models import DOMAIN  # noqa: PLC0415
    from support.tfc import LANES, cut, streamlined  # noqa: PLC0415

    work = Path(tempfile.mkdtemp(prefix="space-trace-tfc-"))
    source = streamlined(work / "tfc")
    _, body, _ = cut(source, work / "cut")
    explored = body.transform(ExploreKernelChoices(Ranked(LANES)))
    found = []
    for node, after in zip(body.graph.node, explored.graph.node):
        if node.domain != DOMAIN:
            continue
        assert after.name == node.name
        op, done = kernel_op(body, node), kernel_op(explored, after)
        found.append(
            traced(
                f"{node.name} ({node.op_type})",
                f"TFC's {node.name}, its KernelOp's own root ({type(op).kernel.__name__})",
                [
                    ("nothing", op.bound()),
                    ("baselines (stored at conversion)", op.point()),
                    ("explored at 16 lanes", done.point()),
                ],
            )
        )
    return found


# -- stage 7's chain of eight kernels -----------------------------------------------------


def chain8() -> list[dict[str, Any]]:
    import numpy as np  # noqa: PLC0415
    from onnx import TensorProto, helper  # noqa: PLC0415
    from qonnx.core.datatype import DataType  # noqa: PLC0415
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415
    from qonnx.util.basic import qonnx_make_model  # noqa: PLC0415

    from finn.core.space import composite  # noqa: PLC0415
    from finn.custom_op.kernels.base import kernel_op  # noqa: PLC0415
    from finn.custom_op.kernels.shell import in_shell  # noqa: PLC0415
    from finn.kernels.channels import Channel  # noqa: PLC0415
    from finn.kernels.configure import commit  # noqa: PLC0415
    from finn.kernels.explore import Seam  # noqa: PLC0415
    from finn.kernels.network import Fact, Interface  # noqa: PLC0415
    from finn.platform.resolve import resolve_target  # noqa: PLC0415
    from finn.transformation.kernels import InferKernelTensors, ToKernelOps  # noqa: PLC0415

    width, rows, layers = 16, 4, 4

    # Stage 7's ``mlp``, as the engine pass's profile reader copies it.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, width])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    nodes, current = [], "x"
    for i in range(layers):
        hidden, levels = f"h{i}", ("y" if i == layers - 1 else f"a{i}")
        nodes.append(helper.make_node("MatMul", [current, f"w{i}"], [hidden], name=f"mm{i}"))
        nodes.append(
            helper.make_node(
                "MultiThreshold",
                [hidden, f"t{i}"],
                [levels],
                name=f"mt{i}",
                domain="qonnx.custom_op.general",
                out_dtype="UINT2",
                out_bias=0.0,
            )
        )
        current = levels
    graph = helper.make_graph(nodes, "mlp", [x], [y])
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            producer_name="walkthrough",
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("qonnx.custom_op.general", 1),
            ],
        )
    )
    model.set_tensor_datatype("x", DataType["UINT2"])
    for i in range(layers):
        w = np.array(
            [[(3 * n + 2 * k + i) % 7 - 3 for n in range(width)] for k in range(width)],
            dtype=np.float32,
        )
        model.set_initializer(f"w{i}", w)
        model.set_tensor_datatype(f"w{i}", DataType["INT3"])
        t = np.array([[-9 + c % 5, 1 - c % 3, 8 + c % 4] for c in range(width)], dtype=np.float32)
        model.set_initializer(f"t{i}", t)
        model.set_tensor_datatype(f"t{i}", DataType["INT8"])

    target = resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0)
    model = model.transform(ToKernelOps(target)).transform(InferKernelTensors())
    members = list(model.graph.node)
    net = in_shell(model, members)
    base = Seam(target.platform).baseline(net)
    declared = {}
    for node in members:
        facts = kernel_op(model, node).facts()
        declared[node.name] = facts.space_type(**facts.arguments)
    channels = {}
    for name in net.channels:
        given = {}
        for fact, value in net.wiring(name).facts.items():
            if isinstance(value, (Interface, Fact)):
                value = getattr(declared[value.member], value.view)
            given[fact] = value
        channels[name] = Channel(**given)
    whole = composite("chain_8", {**channels, **declared})
    point = design_space(whole())
    stored = {f"{owner}.{key}": value for (owner, key), value in base.chosen().items()}
    return [
        traced(
            "chain_8",
            "stage 7's synthetic chain: four MatMuls and four thresholdings (eight kernels) "
            "and their channels, as one Space (channels declared first)",
            [("nothing", point), ("baselines (the network's)", commit(point, stored))],
        )
    ]


SETS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "pipeline": pipeline,
    "dotp": dotp,
    "matmul": matmul,
    "tfc": tfc,
    "chain8": chain8,
}


def main() -> None:
    out = Path(sys.argv[1])
    names = sys.argv[2:] or list(SETS)
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "scripts", "tests"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    for name in names:
        started = time.perf_counter()
        spaces = SETS[name]()
        document = {
            "set": name,
            "finn_commit": commit,
            "uncommitted_changes": dirty,
            "private_reads": list(_trace.PRIVATE_READS),
            "spaces": spaces,
        }
        (out / f"{name}.json").write_text(json.dumps(document, indent=1, default=str) + "\n")
        print(f"{name}: {len(spaces)} spaces, {time.perf_counter() - started:.1f} s", flush=True)
        for space in spaces:
            print(f"  {space['space']}: {json.dumps(space['summary'])}", flush=True)
    index = {}
    for path in sorted(out.glob("*.json")):
        if path.name == "index.json":
            continue
        document = json.loads(path.read_text())
        index[document["set"]] = {
            "finn_commit": document["finn_commit"],
            "uncommitted_changes": document["uncommitted_changes"],
            "spaces": {space["space"]: space["summary"] for space in document["spaces"]},
        }
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
