# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The MVAU nodes both stacks are asked about, as data.

Data rather than code because the *same* description has to build the same graph
in two interpreters that share no imports.  A helper function could not cross
that boundary; a dict can.

The set is chosen to cover every combination the old three-valued computation
profile could and could not name -- see ``PROFILE_ROWS`` -- plus the facts whose
ownership moved: the weight initializer's values (narrowness), the runtime-
writable flag, and the range contract.
"""

from __future__ import annotations

from typing import Any

#: The build facts both stacks are given.  ``fpga_part`` is the oracle's way of
#: naming a DSP generation; the new stack takes the generation directly, and the
#: two are pinned to the same device family here so the comparison is about the
#: projection rather than about part-number parsing.
BUILD = {
    "fpga_part": "xczu3eg-sbva484-1-e",
    "target_dsp": "DSP48E2",
    "synth_clk_period_ns": 5.0,
}


def fixture(
    name: str,
    *,
    repetitions: int = 2,
    matrix_width: int = 4,
    matrix_height: int = 2,
    activation_type: str = "INT8",
    weight_type: str = "INT8",
    output_type: str = "INT16",
    accumulator_type: str = "INT16",
    no_activation: bool = True,
    binary_xnor: bool = False,
    activation_bias: int = 0,
    source_nodes: str = "",
    weights: list[list[float]] | None = None,
    thresholds: list[list[float]] | None = None,
    threshold_type: str = "INT16",
    runtime_writable: bool = False,
    runtime_range_contract: bool | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "repetitions": repetitions,
        "matrix_width": matrix_width,
        "matrix_height": matrix_height,
        "activation_type": activation_type,
        "weight_type": weight_type,
        "output_type": output_type,
        "accumulator_type": accumulator_type,
        "no_activation": no_activation,
        "binary_xnor": binary_xnor,
        "activation_bias": activation_bias,
        "source_nodes": source_nodes,
        "weights": weights,
        "thresholds": thresholds,
        "threshold_type": threshold_type,
        "runtime_writable": runtime_writable,
        "runtime_range_contract": runtime_range_contract,
    }


#: ``integer`` weights that avoid INT8's minimum, so narrowness is *derivable*
#: and both stacks must derive the same answer.
_NARROW = [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]]
#: The same matrix with one element at the minimum, which forbids narrowing.
_WIDE = [[1.0, -128.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]]

SOURCE_FIXTURES = (
    fixture("plain_integer", weights=_NARROW),
    fixture("wide_weights", weights=_WIDE),
    fixture("no_initializer"),
    fixture(
        "fused_threshold",
        no_activation=False,
        weights=_NARROW,
        thresholds=[[10.0], [20.0]],
        output_type="UINT2",
    ),
    fixture(
        "xnor_popcount",
        binary_xnor=True,
        activation_type="BINARY",
        weight_type="BINARY",
        weights=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        output_type="UINT8",
        accumulator_type="UINT8",
    ),
    fixture(
        "bipolar_operands",
        activation_type="BIPOLAR",
        weight_type="BIPOLAR",
        weights=[[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]],
        output_type="INT8",
        accumulator_type="INT8",
    ),
    fixture("runtime_writable", weights=_NARROW, runtime_writable=True),
    fixture(
        "runtime_writable_with_contract",
        weights=_NARROW,
        runtime_writable=True,
        runtime_range_contract=True,
    ),
    fixture("fused_provenance", weights=_NARROW, source_nodes="mul0,add0"),
)


#: The six combinations of the two axes, and what the old enum could name.
#:
#: Parity for these is against the oracle's **execution**, not its profile
#: string: three of the six were reachable at execution time and had no profile
#: value, which is exactly why the old enum is *represented by* the new pair
#: rather than equal to one of its values.
PROFILE_ROWS = (
    ("integer+none", "ACCUMULATOR_INTEGER", "INTEGER", "NONE", "plain_integer"),
    ("xnor+none", "BIPOLAR_XNOR_ACCUMULATOR", "XNOR_POPCOUNT", "NONE", "xnor_popcount"),
    ("bipolar+none", None, "BIPOLAR_POPCOUNT", "NONE", "bipolar_operands"),
    ("integer+threshold", "FUSED_THRESHOLD", "INTEGER", "MULTITHRESHOLD", "fused_threshold"),
    (
        "xnor+threshold",
        "FUSED_THRESHOLD",
        "XNOR_POPCOUNT",
        "MULTITHRESHOLD",
        "xnor_fused_threshold",
    ),
    (
        "bipolar+threshold",
        "FUSED_THRESHOLD",
        "BIPOLAR_POPCOUNT",
        "MULTITHRESHOLD",
        "bipolar_fused_threshold",
    ),
)

#: The two the collapsed enum could not represent, as executable fixtures.
PROFILE_FIXTURES = (
    *SOURCE_FIXTURES,
    fixture(
        "xnor_fused_threshold",
        no_activation=False,
        binary_xnor=True,
        activation_type="BINARY",
        weight_type="BINARY",
        weights=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        thresholds=[[2.0], [3.0]],
        threshold_type="UINT8",
        accumulator_type="UINT8",
        output_type="UINT2",
    ),
    fixture(
        "bipolar_fused_threshold",
        no_activation=False,
        activation_type="BIPOLAR",
        weight_type="BIPOLAR",
        weights=[[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]],
        thresholds=[[0.0], [1.0]],
        threshold_type="INT8",
        accumulator_type="INT8",
        output_type="UINT2",
    ),
)


def by_name(name: str) -> dict[str, Any]:
    for item in PROFILE_FIXTURES:
        if item["name"] == name:
            return item
    raise KeyError(name)


def activation_values(spec: dict[str, Any]) -> list[list[float]]:
    """One deterministic activation per fixture, in that fixture's alphabet."""

    rows = int(spec["repetitions"])
    width = int(spec["matrix_width"])
    datatype = str(spec["activation_type"])
    if datatype == "BINARY":
        return [[float((index + lane) % 2) for lane in range(width)] for index in range(rows)]
    if datatype == "BIPOLAR":
        return [
            [1.0 if (index + lane) % 2 == 0 else -1.0 for lane in range(width)]
            for index in range(rows)
        ]
    return [[float(index + lane + 1) for lane in range(width)] for index in range(rows)]


__all__ = [
    "BUILD",
    "PROFILE_FIXTURES",
    "PROFILE_ROWS",
    "SOURCE_FIXTURES",
    "activation_values",
    "by_name",
    "fixture",
]
