# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The generated hardware against its KernelOp's oracle, at the op's boundary (H1).

PRINCIPLES 8: the KernelOp's ``execute_node`` is the computational reference; a kernel
declares none. ``finn.harness.ops.check_parity`` builds the hardware a one-node model
generates at a chosen point and checks it in XSim against the op's ``execute_node`` on
the same inputs: random integers of each input's datatype, seeded, and its extremes.

The points are drawn from the node's network (``finn.harness.points.covering``
of the op's own kernel Decisions): each case of each Decision once, each
ordered domain at its extremes; the rest (transports, adapter memories) at the
baselines the cut gives them (``CutKernelPartition``), as packaging replays them. A
handful of simulations an op, stalled, and the first point of each op free running too
(conformance runs both; one point an op each way, not every point twice).

**The platform as an axis** (P3): the points above are on the Ultra96 target that
``kernel_ops.models`` builds for (DSP48E2, no UltraRAM). One point an op is on a Versal
row (``finn.harness.reference.platform_rows``: the Versal AI Core family on ``ip``):
MatMul on the INT8 DSP58 core with its activations' replay buffer (``input_gen``) in
UltraRAM; Thresholding there too, its tables out of UltraRAM, which on Versal takes no
initial contents (``uram-init``). XSim simulates the DSP58 primitive (unisims); an
UltraRAM is the RTL's ``RAM_STYLE`` attribute, which simulates as the behavioural
memory: the mapping itself is synthesis's to show.

- **Thresholding**: a row a channel, its thresholds typed narrower than the input
  (so the RTL saturates the input's extremes to the thresholds' type), the least of
  them -16, a negative bias; and one row shared by every channel, at one point.
  Against ``execute_node`` itself. The RTL's saturation would count a threshold at the
  type's minimum (INT5's -16) for inputs below it: the op's facts type the table with
  a value below its least threshold (INT6), and the kernel refuses the case
  (``threshold-saturation``).
- **MatMul**: INT8 activations and weights at their extremes, against ``execute_node``
  itself: exact integer arithmetic, as the hardware's accumulator is, on the oracle's
  integer inputs (``test_matmul_execute_node_is_exact_beyond_float32``). The ONNX side,
  which FINN executes with exact integers too, is ``tests/kernel_ops/test_reference.py``'s.
- **A wide MatMul**: the spec's INT8 layer over k 2304 with spread weights
  (``int8-k2304``), at one lean point: PE 3 (every output channel at once) and SIMD 16,
  so each sum builds over 144 accumulation steps in an INT26 accumulator, 576 compute
  cycles in all. Beside the extremes' rows, channel 0's greatest and least sums, each odd
  and past 2**24, where float32 would round them.

The specs of the kernels these ops bind are ``kernels.specs.matmul`` and
``kernels.specs.thresholding``.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from kernels.helpers import choose
from kernels.xsim import requires_xsim
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.graph.annotations.platform import write_target
from finn.core.onnx_exec import execute_onnx
from finn.core.rtlsim_pacing import FREE
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import network, persist
from finn.custom_op.kernels.shell import configured_network
from finn.harness.ops import Disagreement, check_parity
from finn.harness.points import Covering, covering
from finn.harness.reference import platform_rows
from finn.platform.capabilities import DspBlock
from finn.platform.target import Target
from finn.transformation.kernels import InferKernelTensors
from finn.transformation.kernels.cut import BaselineChoices
from finn.util.draws import Integers, boundary_inputs, graph_inputs
from kernel_ops.models import TARGET, matmul_model, thresholding_model
from kernel_ops.specs.matmul import SPREAD

# -- the models ------------------------------------------------------------------------

ROWS, K, N = 4, 8, 4
# Columns at the weights' extremes (all greatest, all least) and two mixed ones.
WEIGHTS = np.array(
    [[127, -128, (5 * k) % 17 - 8, 127 if k % 2 else -128] for k in range(K)], dtype=np.int64
)


def matmul(target: Target = TARGET, weights: Any = WEIGHTS, rows: int = ROWS) -> ModelWrapper:
    """x (1, 4, 8) INT8 -> MatMul ``first`` with w (8, 4) INT8, an initializer -> y, built
    for ``target``; or x (1, rows, k) with ``weights`` (k, n)."""
    k = np.shape(weights)[0]
    model = matmul_model(weights=weights, x_shape=[1, rows, k], annotate=(), infer=False)
    write_target(model, target)
    for name in ("x", "w"):
        model.set_tensor_datatype(name, DataType["INT8"])
    return model.transform(InferKernelTensors())


# A row a channel, three thresholds each, -16 and 15 among them: the kernel's table INT6.
THRESHOLDS = np.array([[-16, -3, 2], [-9, 0, 15], [-4, -4, 7], [-1, 6, 14]])
SHARED = np.array([[-7, 0, 9]])
BIAS = -2


def thresholding(table: Any = THRESHOLDS, target: Target = TARGET) -> ModelWrapper:
    """x (3, 4) INT8 -> Thresholding ``activate`` with t (C, 3), bias -2 -> y, built for
    ``target``."""
    return thresholding_model(thresholds=table, bias=BIAS, target=target)


MODELS: dict[str, tuple[Callable[..., ModelWrapper], str]] = {
    "matmul": (matmul, "first"),
    "thresholding": (thresholding, "activate"),
}


def points(build: Any, member: str) -> Covering:
    """The op's covering points: its own kernel's Decisions, its kernel bound alone; each
    key its node's, ``member``."""
    model = build()
    return covering(kernel_op(model, model.graph.node[0]).bound())


COVERING = {name: points(*MODELS[name]) for name in MODELS}
PARITY = [
    pytest.param(
        name,
        index,
        id=f"{name}-" + ",".join(f"{k.rsplit('.', 1)[-1]}={v}" for k, v in point.items()),
    )
    for name, found in COVERING.items()
    for index, point in enumerate(found.points)
]


# A Versal row (``platform_rows``: the Versal AI Core family on ip, its rule's sample
# part xcvc1902), one point an op:
# MatMul on the INT8 DSP58 core, its activations' replay buffer in UltraRAM. Thresholding
# keeps its memories out of UltraRAM in this point (``ultra_stages`` 0), though Versal's
# UltraRAM takes its tables as initial contents (``uram_init``).
VERSAL = platform_rows(TARGET.platform.period_ns)["versalaicore", "ip"]
ON_VERSAL: dict[str, Mapping[str, object]] = {
    "matmul": {
        "first.compute": "int8_dsp58",
        "first.compute.int8_dsp58.pe": 2,
        "first.compute.int8_dsp58.simd": 4,
        "first.compute.int8_dsp58.compute_pumping": False,
        "x.adapter.input_gen.input_gen.ram_style": "ultra",
    },
    "thresholding": {"activate.pe": 2, "activate.ultra_stages": 0},
}


def chosen(name: str, index: int) -> ModelWrapper:
    """The model with the covering point's choices persisted on its node."""
    build, member = MODELS[name]
    point = COVERING[name].points[index]
    return persisted(build(), {f"{member}.{key}": value for key, value in point.items()})


def persisted(model: ModelWrapper, choices: Mapping[str, object]) -> ModelWrapper:
    """``model`` with ``choices`` (``member.key``) committed in its network, persisted."""
    persist(model, choose(network(model, model.graph.node), choices))
    return model


def seed(label: str) -> int:
    return zlib.crc32(label.encode())


# -- a wide MatMul ---------------------------------------------------------------------

#: A lean point of the wide layer: all three output channels at once, 16 products a
#: cycle, so 144 accumulation steps a sum and 576 compute cycles in all, where SIMD 1
#: takes 9 216 and the covering's full SIMD (2 304 lanes) would make a slow elaboration.
WIDE_POINT: Mapping[str, object] = {
    "first.compute.packed.pe": 3,
    "first.compute.packed.simd": 16,
    "first.compute.packed.reducer": "tree",
    "first.compute.packed.compute_pumping": False,
}


def wide() -> ModelWrapper:
    """x (1, 4, 2304) INT8 -> MatMul ``first`` with the spec's spread weights (2304, 3)
    INT8 -> y (INT26), at ``WIDE_POINT``."""
    return persisted(matmul(weights=SPREAD, rows=4), WIDE_POINT)


def wide_inputs(model: ModelWrapper) -> dict[str, Integers]:
    """The seeded draw with the extremes in rows 0 and 1; rows 2 and 3 channel 0's
    greatest and least sums, each made odd, if it is not, by stepping the first
    activation one toward zero (its weight is odd)."""
    inputs = boundary_inputs(model, seed("wide"))
    rows = inputs["x"][0]
    up = SPREAD[:, 0] > 0
    for row, (positive, negative) in ((2, (127, -128)), (3, (-128, 127))):
        rows[row] = np.where(up, positive, negative)
        if int(rows[row] @ SPREAD[:, 0]) % 2 == 0:
            rows[row, 0] -= np.sign(rows[row, 0])
    return inputs


# -- offline ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_parity_points_cover_the_ops_kernel(name: str) -> None:
    found = COVERING[name]
    assert not found.missed and 2 <= len(found.points) <= 4, found


def test_the_points_take_each_case_and_the_extremes() -> None:
    taken: dict[str, set[object]] = {}
    for point in COVERING["matmul"].points:
        for key, value in point.items():
            taken.setdefault(key.rsplit(".", 1)[-1], set()).add(value)
    assert taken["pe"] == {1, 4} and taken["simd"] == {1, 8}
    assert taken["reducer"] == {"tree", "compressor"}
    assert taken["compute_pumping"] == {False, True}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_inputs_are_the_datatypes_extremes_and_seeded(name: str) -> None:
    model = MODELS[name][0]()
    first, again = boundary_inputs(model, 7), boundary_inputs(model, 7)
    (tensor,) = graph_inputs(model)
    rows = first[tensor].reshape(-1, first[tensor].shape[-1])
    assert (rows[0] == -128).all() and (rows[1] == 127).all()
    assert np.array_equal(first[tensor], again[tensor])
    assert not np.array_equal(first[tensor], boundary_inputs(model, 8)[tensor])


def test_matmul_execute_node_is_the_integer_product_on_the_parity_inputs() -> None:
    model = matmul()
    inputs = boundary_inputs(model, seed("matmul"))
    assert np.array_equal(execute_onnx(model, inputs)["y"], inputs["x"] @ WEIGHTS)


def test_matmul_execute_node_is_exact_beyond_float32() -> None:
    """With INT16 operands over k = 1024 the sums pass 2**24, where a float32 product
    rounds: the reference is the integer product the hardware computes."""
    k = 1024
    weights = np.full((k, 1), 32767, dtype=np.int64)
    weights[1::2] = 32765
    model = matmul_model(weights=weights, x_shape=[1, 1, k], annotate=(), infer=False)
    for tensor in ("x", "w"):
        model.set_tensor_datatype(tensor, DataType["INT16"])
    x = np.full((1, 1, k), 16383, dtype=np.int64)
    x[..., 1::3] = 16381
    exact = x @ weights
    assert abs(exact).max() > 2**24
    assert not np.array_equal(np.matmul(x.astype(np.float32), weights.astype(np.float32)), exact)
    assert np.array_equal(execute_onnx(model, {"x": x})["y"], exact)


def test_the_wide_matmul_execute_node_is_exact_on_its_parity_inputs() -> None:
    """The wide layer's reference on its parity inputs is the integer product: channel 0's
    sums in rows 2 and 3 odd and past 2**24 either way, where float32 rounds them, and the
    output annotated INT26, the accumulator the hardware carries."""
    model = wide()
    inputs = wide_inputs(model)
    x = inputs["x"]
    assert (x[0, 0] == -128).all() and (x[0, 1] == 127).all()
    assert x.min() >= -128 and x.max() <= 127
    exact = x @ SPREAD
    sums = exact[0, 2:, 0]
    assert sums[0] > 2**24 and sums[1] < -(2**24) and (sums % 2 == 1).all(), sums
    assert (sums.astype(np.float32).astype(np.int64) != sums).all()
    assert model.get_tensor_datatype("y") == DataType["INT26"]
    assert np.array_equal(execute_onnx(model, inputs)["y"], exact)


def test_an_oracle_value_the_boundary_cannot_present_is_a_disagreement(tmp_path: Path) -> None:
    """Refused before anything simulates: no Vivado needed."""
    model = thresholding()
    inputs = boundary_inputs(model, 1)
    with pytest.raises(Disagreement, match="does not hold"):
        check_parity(
            model, tmp_path, inputs=inputs, reference=lambda _: {"y": inputs["x"] * 0 + 99}
        )
    with pytest.raises(Disagreement, match="not integers"):
        check_parity(model, tmp_path, inputs=inputs, reference=lambda _: {"y": inputs["x"] * 0.5})


# -- in XSim ---------------------------------------------------------------------------


@requires_xsim
@pytest.mark.parametrize(("name", "index"), PARITY)
def test_the_hardware_computes_what_its_kernel_op_computes(
    name: str, index: int, tmp_path: Path
) -> None:
    model = chosen(name, index)
    inputs = boundary_inputs(model, seed(name))
    check_parity(model, tmp_path, inputs=inputs, label=name)


@requires_xsim
@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_hardware_computes_what_its_kernel_op_computes_free_running(
    name: str, tmp_path: Path
) -> None:
    """One covering point an op never stalled, as conformance runs free and stalled:
    the first, each Decision's first case."""
    model = chosen(name, 0)
    check_parity(model, tmp_path, inputs=boundary_inputs(model, seed(name)), pacing=FREE)


def test_the_versal_points_take_its_dsp58_core_and_ultraram() -> None:
    """Offline: on the Versal row, MatMul's point is admitted with the INT8 DSP58 core and
    its replay buffer in UltraRAM; Thresholding's tables may be too, since Versal's
    UltraRAM takes initial contents (``uram_init``)."""
    assert VERSAL.platform.dsp is DspBlock.DSP58 and VERSAL.platform.uram
    assert VERSAL.platform.uram_init
    model = persisted(matmul(VERSAL), ON_VERSAL["matmul"]).transform(BaselineChoices())
    point, _ = configured_network(model, "versal")
    assert type(point.network.member("first").compute).__name__ == "Int8Dsp58DotpKernel"
    styles = {
        label: dict(leaf.parameters).get("RAM_STYLE")
        for label, leaf in point.module.fragment.instances
    }
    assert [label for label, style in styles.items() if style == '"ultra"'] == [
        "x.adapter.input_gen.input_gen"
    ], styles
    model = thresholding(target=VERSAL)
    assert choose(network(model, model.graph.node), {"activate.ultra_stages": 1})


@requires_xsim
@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_hardware_computes_what_its_kernel_op_computes_on_versal(
    name: str, tmp_path: Path
) -> None:
    build, _ = MODELS[name]
    model = persisted(build(target=VERSAL), ON_VERSAL[name])
    check_parity(model, tmp_path, inputs=boundary_inputs(model, seed(name)), label=name)


@requires_xsim
def test_a_wide_matmul_computes_what_its_kernel_op_computes(tmp_path: Path) -> None:
    """The INT8 layer over k 2304, its sums past 2**24, at ``WIDE_POINT``."""
    model = wide()
    check_parity(model, tmp_path, inputs=wide_inputs(model), label="wide")


@requires_xsim
def test_a_shared_row_computes_what_its_kernel_op_computes(tmp_path: Path) -> None:
    """One row for every channel (C = 1), at PE 2: the table bound once, each lane a copy."""
    model = thresholding(SHARED)
    persisted(model, {"activate.pe": 2})
    check_parity(model, tmp_path, inputs=boundary_inputs(model, seed("shared")), label="shared")
