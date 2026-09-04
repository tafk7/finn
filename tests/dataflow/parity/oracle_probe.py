# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What the previous implementation says about a node, as JSON.

Executed **inside the oracle worktree**, with the oracle's ``src`` on
``PYTHONPATH`` and nothing of this stack importable.  That is the whole design:
the two implementations share a Python installation and a qonnx, and share no
FINN module at all, so a parity result cannot be produced by one of them
answering for both.

It reads ``{"fixtures": [...], "build": {...}}`` on stdin and writes one JSON
object on stdout.  Values are encoded structurally -- datatypes by name, enums
by value, dataclasses by field -- because the comparison is between two stacks
whose Python objects are different classes with the same meaning.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, fields, is_dataclass
from typing import Any


@dataclass
class _Config:
    """The shape of FINN's build configuration the oracle's accessors read."""

    synth_clk_period_ns: float = 5.0
    fpga_part: str = "xczu3eg-sbva484-1-e"

    def _resolve_fpga_part(self) -> str:
        return self.fpga_part


def _encode(value: Any) -> Any:
    """One structural spelling both stacks can be compared in."""

    import numpy  # noqa: PLC0415

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    name = type(value).__name__
    if name == "DataType" or hasattr(value, "allowed") and hasattr(value, "bitwidth"):
        return {"datatype": str(value)}
    if isinstance(value, numpy.ndarray):
        return {"array": value.tolist()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if hasattr(value, "value") and hasattr(type(value), "__members__"):
        return {"enum": value.value}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": type(value).__name__,
            "fields": {item.name: _encode(getattr(value, item.name)) for item in fields(value)},
        }
    return {"repr": repr(value)}


def _model(spec: dict[str, Any]) -> Any:
    import numpy  # noqa: PLC0415
    from onnx import TensorProto, helper  # noqa: PLC0415
    from qonnx.core.datatype import DataType  # noqa: PLC0415
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415
    from qonnx.util.basic import qonnx_make_model  # noqa: PLC0415

    rows = int(spec["repetitions"])
    width = int(spec["matrix_width"])
    height = int(spec["matrix_height"])
    fused = not bool(spec["no_activation"])
    inputs = ["activation", "weight"] + (["threshold"] if fused else [])
    node = helper.make_node(
        "MvauDataflowOp",
        inputs,
        ["output"],
        name="mvau0",
        domain="finn.custom_op.dataflow",
        dataflow_scope_id="parity_scope",
        noActivation=int(bool(spec["no_activation"])),
        binaryXnorMode=int(bool(spec["binary_xnor"])),
        accDataType=str(spec["accumulator_type"]),
        ActVal=int(spec["activation_bias"]),
        dataflow_source_nodes=str(spec["source_nodes"]),
    )
    value_info = [
        helper.make_tensor_value_info("activation", TensorProto.FLOAT, [rows, width]),
        helper.make_tensor_value_info("weight", TensorProto.FLOAT, [width, height]),
    ]
    if fused:
        thresholds = spec["thresholds"] or []
        value_info.append(
            helper.make_tensor_value_info(
                "threshold", TensorProto.FLOAT, [height, len(thresholds[0])]
            )
        )
    graph = helper.make_graph(
        [node],
        "parity",
        value_info,
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [rows, height])],
    )
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("finn.custom_op.dataflow", 1),
            ],
        )
    )
    model.set_tensor_datatype("activation", DataType[str(spec["activation_type"])])
    model.set_tensor_datatype("weight", DataType[str(spec["weight_type"])])
    model.set_tensor_datatype("output", DataType[str(spec["output_type"])])
    if spec["weights"] is not None:
        model.set_initializer("weight", numpy.array(spec["weights"], dtype=numpy.float32))
    if fused:
        model.set_tensor_datatype("threshold", DataType[str(spec["threshold_type"])])
        model.set_initializer("threshold", numpy.array(spec["thresholds"], dtype=numpy.float32))
    return model


def _derived(operation: Any, context: Any) -> dict[str, Any]:
    from finn.dataflow.design import Engine  # noqa: PLC0415

    point = operation.hydrate_dataflow_point(context)
    engine = Engine()
    answers: dict[str, Any] = {}
    for name in (
        "matrix_width",
        "matrix_height",
        "repetitions",
        "computation_profile",
        "source_description",
        "effective_narrow_weights",
    ):
        answer = engine.query_property(point, f"semantic.mvau.{name}")
        answers[name] = _encode(getattr(answer, "value", None))
    return answers


def _execution(spec: dict[str, Any], model: Any, operation: Any) -> Any:
    import numpy  # noqa: PLC0415

    activation = numpy.array(spec["activation"], dtype=numpy.float32)
    weights = spec["weights"]
    if weights is None:
        return None
    context = {
        "activation": activation,
        "weight": numpy.array(weights, dtype=numpy.float32),
        "output": numpy.zeros(
            (int(spec["repetitions"]), int(spec["matrix_height"])), dtype=numpy.float32
        ),
    }
    if not bool(spec["no_activation"]):
        context["threshold"] = numpy.array(spec["thresholds"], dtype=numpy.float32)
    operation.execute_node(context, model.graph)
    return context["output"].tolist()


def _context(base: Any, config: Any, *, runtime_writable: bool, contract: Any) -> Any:
    """The oracle's build view, with the range contract it reads by ``getattr``.

    ``runtime_weight_range_contract`` is not a field of the oracle's context --
    its accessor reads it off whatever object it is handed -- so supplying it
    means extending the dataclass rather than passing a keyword.
    """

    if contract is None:
        return base(config, runtime_writable_weights=runtime_writable)

    @dataclass(frozen=True)
    class _WithContract(base):  # type: ignore[misc, valid-type]
        runtime_weight_range_contract: Any = None

    return _WithContract(
        config,
        runtime_writable_weights=runtime_writable,
        runtime_weight_range_contract=bool(contract),
    )


def main() -> int:
    request = json.load(sys.stdin)
    build = request["build"]

    from finn.dataflow.ops.mvau.contracts import MVAUProblem, MVAUSourceDescription  # noqa: PLC0415
    from finn.dataflow.ops.mvau.op import MVAUDataflowBuildContext  # noqa: PLC0415

    results: dict[str, Any] = {
        "fields": {
            "MVAUProblem": [item.name for item in fields(MVAUProblem)],
            "MVAUSourceDescription": [item.name for item in fields(MVAUSourceDescription)],
        },
        "fixtures": {},
    }
    for spec in request["fixtures"]:
        model = _model(spec)
        operation = model.get_customop_wrapper(model.graph.node[0])
        config = _Config(
            synth_clk_period_ns=float(build["synth_clk_period_ns"]),
            fpga_part=str(build["fpga_part"]),
        )
        context = _context(
            MVAUDataflowBuildContext,
            config,
            runtime_writable=bool(spec["runtime_writable"]),
            contract=spec["runtime_range_contract"],
        )
        entry: dict[str, Any] = {}
        entry["problem"] = {
            str(path): _encode(value) for path, value in operation.problem_instance(context).items()
        }
        entry["derived"] = _derived(operation, context)
        entry["execution"] = _execution(spec, model, operation)
        entry["verify_node"] = list(operation.verify_node() or [])
        results["fixtures"][spec["name"]] = entry

    json.dump(results, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
