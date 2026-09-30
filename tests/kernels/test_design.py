# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A Design: two MatMuls and a thresholding on its streams, fused or not.

Each MatMul sits on the design's streams through its boundary inputs and
presents there what its own boundary stream presents. Fused, it is one module
the design nests; unfused, its modules and streams join the design's module,
each boundary stream spliced with the design stream it sits on. Both compute
``thresholds(x @ W1) @ W2`` in XSim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Inapplicable, Rejected, design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.composite import Composite, Design
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel, exact_result_dtype
from finn.kernels.streams import Stream
from finn.kernels.target import DspBlock
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernels.helpers import settled
from kernels.xsim import pack, requires_xsim, stream_through

ROOT = Path(__file__).resolve().parents[2]
ROWS, INPUTS, HIDDEN, OUTPUTS, PE, SIMD = 3, 4, 4, 4, 2, 2
A = W = DataType["INT3"]
H = exact_result_dtype(INPUTS, A, W)
T = DataType["UINT2"]
Y = exact_result_dtype(HIDDEN, T, W)
THRESHOLDS = (tuple((-9 + c, 1 - c, 8 + 2 * c) for c in range(HIDDEN)),)
W1 = tuple(tuple((3 * n + 2 * k) % 7 - 3 for n in range(HIDDEN)) for k in range(INPUTS))
W2 = tuple(tuple((2 * n + 5 * k) % 7 - 3 for n in range(OUTPUTS)) for k in range(HIDDEN))
X = tuple(tuple((5 * r + 3 * k) % 8 - 4 for k in range(INPUTS)) for r in range(ROWS))


def matmul(k: int, n: int, dtype: Any, weights: Any, **streams: Stream) -> MatMulKernel:
    return MatMulKernel(
        m=ROWS,
        n=n,
        k=k,
        activation_dtype=dtype,
        weights_dtype=W,
        target_dsp=DspBlock.DSP48E2,
        target_period_ns=5.0,
        weights=weights,
        **streams,
    )


class Chain(Design):
    x = Stream(tensor=Tensor((ROWS, INPUTS), ScalarEncoding(A)), port="in0_V")
    hidden = Stream(tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(H)))
    levels = Stream(tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(T)))
    y = Stream(tensor=Tensor((ROWS, OUTPUTS), ScalarEncoding(Y)), port="out0_V")
    first = matmul(INPUTS, HIDDEN, A, W1, x_stream=x, y_stream=hidden)
    activate = ThresholdingAxiKernel(
        input_dtype=H,
        threshold_dtype=H,
        thresholds=THRESHOLDS,
        bias=0,
        pe=PE,
        depth_trigger_bram=0,
        depth_trigger_uram=0,
        input_stream=hidden,
        output_stream=levels,
    )
    second = matmul(HIDDEN, OUTPUTS, T, W2, x_stream=levels, y_stream=y)


def chain(*, fused: bool) -> Chain:
    choices: dict[str, object] = {"activate.use_axilite": False, "activate.deep_pipeline": False}
    for layer in ("first", "second"):
        choices |= {
            f"{layer}.fused": fused,
            f"{layer}.memory": "memstream",
            f"{layer}.weight_stream.transport": "direct",
        }
    point = settled(commit(design_space(Chain()), choices))
    # The Decisions inside the subspaces just selected, each keyed by its owner.
    nested: dict[str, object] = {}
    for layer in ("first", "second"):
        nested |= {
            f"{layer}.compute.packed.pe": PE,
            f"{layer}.compute.packed.simd": SIMD,
            f"{layer}.compute.packed.compute_pumping": False,
            f"{layer}.memory.memstream.ram_style": "auto",
            f"{layer}.memory.memstream.pumped_memory": False,
        }
    return settled(commit(point, nested))


def instances(point: Chain) -> list[str]:
    return [item.instance_id for item in point.structure.structure.instances]


def test_a_fused_matmul_is_one_nested_module_on_the_design_streams():
    point = chain(fused=True)
    names = instances(point)
    assert names[:3] == ["u_first", "u_activate", "u_second"]
    # The design's streams connect directly: each MatMul presents its boundary.
    assert all(stream.plan.steps == () for stream in (point.x, point.hidden, point.levels))
    first = point.structure.structure.instances[0].requirements
    assert first.abi.entry_point.value.startswith("finn_matmul_memstream__")  # type: ignore[union-attr]
    assert {port.name for port in first.abi.ports} >= {"in0_V", "out0_V"}
    # The design's own ports are its boundary streams.
    top = {port.name for port in point.structure.structure.top_abi.ports}
    assert top == {"ap_clk", "ap_rst_n", "in0_V", "out0_V"}


def test_an_unfused_matmul_joins_the_design_module_spliced_on_its_streams():
    point = chain(fused=False)
    names = instances(point)
    assert {
        "u_first_compute_packed",
        "u_first_memory_memstream",
        "u_first_activations_input_gen",
        "u_activate",
        "u_second_compute_packed",
        "u_second_memory_memstream",
        "u_second_activations_input_gen",
    } == set(names)
    # It exports its parts, not a module; each boundary names its internal stream.
    assert isinstance(point.first.query(Composite.module_export), Inapplicable)
    assert dict(point.first.parts.boundaries)["x_stream"] == "activations"


def test_a_composite_on_a_stream_of_another_tensor_is_refused():
    class Misplaced(Design):
        x = Stream(tensor=Tensor((ROWS, INPUTS + 2), ScalarEncoding(A)), port="in0_V")
        y = Stream(tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(H)), port="out0_V")
        first = matmul(INPUTS, HIDDEN, A, W1, x_stream=x, y_stream=y)

    point = design_space(Misplaced())
    refused = point.first.query(Composite.seated)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"composite-tensor"}


def test_a_matmul_on_a_stream_of_another_element_is_refused():
    """The result type MatMul states (its core's ``result_dtype``) meets the stream's."""

    class Widened(Design):
        x = Stream(tensor=Tensor((ROWS, INPUTS), ScalarEncoding(A)), port="in0_V")
        y = Stream(tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(DataType["INT32"])), port="out0_V")
        first = matmul(INPUTS, HIDDEN, A, W1, x_stream=x, y_stream=y)

    refused = design_space(Widened()).first.query(Composite.seated)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"composite-tensor"}


@requires_xsim
@pytest.mark.parametrize("fused", (True, False))
def test_the_design_computes_in_xsim_fused_or_not(tmp_path: Path, fused: bool) -> None:
    hidden = [
        [sum(X[r][k] * W1[k][n] for k in range(INPUTS)) for n in range(HIDDEN)] for r in range(ROWS)
    ]
    levels = [
        [sum(t <= hidden[r][c] for t in THRESHOLDS[0][c]) for c in range(HIDDEN)]
        for r in range(ROWS)
    ]
    y = [
        [sum(levels[r][k] * W2[k][n] for k in range(HIDDEN)) for n in range(OUTPUTS)]
        for r in range(ROWS)
    ]
    a_bits, y_bits = A.bitwidth(), Y.bitwidth()
    stream_through(
        chain(fused=fused).structure.requirements,
        tmp_path,
        inputs={
            "in0_V": (
                [
                    pack(X[r][f : f + SIMD], a_bits)
                    for r in range(ROWS)
                    for f in range(0, INPUTS, SIMD)
                ],
                SIMD * a_bits,
            )
        },
        outputs={
            "out0_V": (
                [
                    pack(y[r][f : f + PE], y_bits)
                    for r in range(ROWS)
                    for f in range(0, OUTPUTS, PE)
                ],
                PE * y_bits,
            )
        },
    )


def writable(*, fused: bool) -> Any:
    class Rewritable(Design):
        x = Stream(tensor=Tensor((ROWS, INPUTS), ScalarEncoding(A)), port="in0_V")
        y = Stream(tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(H)), port="out0_V")
        first = MatMulKernel(
            m=ROWS,
            n=HIDDEN,
            k=INPUTS,
            activation_dtype=A,
            weights_dtype=W,
            target_dsp=DspBlock.DSP48E2,
            target_period_ns=5.0,
            weights=W1,
            writable_weights=True,
            x_stream=x,
            y_stream=y,
        )

    choices = {
        "first.fused": fused,
        "first.memory": "memstream",
        "first.weight_stream.transport": "direct",
    }
    point = settled(commit(design_space(Rewritable()), choices))
    return settled(
        commit(
            point,
            {
                "first.compute.packed.pe": PE,
                "first.compute.packed.simd": SIMD,
                "first.compute.packed.compute_pumping": False,
                "first.memory.memstream.ram_style": "auto",
                "first.memory.memstream.pumped_memory": False,
            },
        )
    )


def test_an_unfused_composite_exports_its_control_bus_under_its_name():
    point = writable(fused=False)
    top = {port.name for port in point.structure.structure.top_abi.ports}
    assert "first_s_axilite" in top


def test_a_fused_composite_with_a_control_bus_is_refused():
    refused = writable(fused=True).first.query(Composite.module_export)
    assert isinstance(refused, Rejected)
    assert "composite-control" in {finding.code for finding in refused.findings}
