# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""LayerNorm: over the last axis, without a scale or a bias, on FinnLib's HLS
``layernorm``. No KernelOp reaches it yet, so its reference is test-side: ``layernorm``,
in float64, which its conformance cases and its numeric sweep
(``kernels.sweeps.layernorm_numeric``) both state.

The hardware's statistics are binary32, so its results are compared within a tolerance,
not bit for bit. Its error is mostly the mean's rounding, carried into every element of
the row and divided by ``sqrt(var + epsilon)``: it grows with the row's magnitude against
its spread. So the tolerance is ``TOLERANCE`` per unit of the row's condition
(``condition``: its root mean square over ``sqrt(var + epsilon)``, at least 1), absolute
(``bounds``). ``binary32_core`` models the core's arithmetic in its order: on the cases'
and the sweep's rows it reaches about a fiftieth of each bound, and its tests hold it
within a tenth (``tests/kernels/test_layernorm.py``), a margin for what the model leaves
out. It is provisional: whether an approximating kernel states an error bound or is
checked to a tolerance is not decided.

The conformance harness takes one absolute tolerance a case: its random rows
(``FLOAT_BOUND`` either side of zero) are about as large as their spread, so it is
``TOLERANCE`` itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from qonnx.core.datatype import DataType

from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.layernorm import LayerNormKernel
from kernels.helpers import FULL_DSP48E2
from kernels.specs.base import TEST_SIDE, KernelSpec, Probe, SweepCases, placed

#: The absolute tolerance its results are compared within, per unit of a row's
#: condition (provisional: at least ten times what the core's binary32 statistics reach).
TOLERANCE = 1e-5
#: ONNX LayerNormalization's default epsilon, and the core's.
EPSILON = 1e-5


def _rows(x: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """``x`` as the hardware reads it, in float64: each value a binary32."""
    return np.asarray(x).astype(np.float32).astype(np.float64)


def layernorm(x: npt.NDArray[Any], epsilon: float) -> npt.NDArray[np.float64]:
    """``(x - mean) / sqrt(var + epsilon)`` over the last axis, in float64, of ``x`` and
    ``epsilon`` as the hardware reads them: each a binary32."""
    values, eps = _rows(x), float(np.float32(epsilon))
    centred = values - values.mean(axis=-1, keepdims=True)
    variance = (centred * centred).mean(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore"):  # a constant row with epsilon 0: no value (NaN)
        found: npt.NDArray[np.float64] = centred / np.sqrt(variance + eps)
    return found


def condition(x: npt.NDArray[Any], epsilon: float) -> npt.NDArray[np.float64]:
    """Each row's root mean square over ``sqrt(var + epsilon)``, at least 1 (and 1 for a
    row of zeros with epsilon 0), kept as an axis: how far the rounding of its statistics
    reaches into its results."""
    values, eps = _rows(x), float(np.float32(epsilon))
    square = (values * values).mean(axis=-1, keepdims=True)
    variance = values.var(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):  # 0/0 for a zero row, epsilon 0
        found: npt.NDArray[np.float64] = np.fmax(1.0, np.sqrt(square / (variance + eps)))
    return found


def bounds(x: npt.NDArray[Any], epsilon: float) -> npt.NDArray[np.float64]:
    """The absolute tolerance of each result: ``TOLERANCE`` times its row's condition."""
    found: npt.NDArray[np.float64] = np.broadcast_to(
        TOLERANCE * condition(x, epsilon), np.shape(x)
    ).copy()
    return found


def _tree(lanes: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """The sum of the last axis's lanes, in binary32, as a heap-ordered tree adds them:
    lane ``i`` a leaf at ``n - 1 + i``, each inner node ``i`` the sum of ``2i+1`` and
    ``2i+2``."""
    n = lanes.shape[-1]
    tree: list[npt.NDArray[np.float32]] = [lanes[..., 0]] * (2 * n - 1)
    for i in range(n):
        tree[n - 1 + i] = lanes[..., i]
    for i in reversed(range(n - 1)):
        tree[i] = (tree[2 * i + 1] + tree[2 * i + 2]).astype(np.float32)
    return tree[0]


def _mean(values: npt.NDArray[np.float32], simd: int) -> npt.NDArray[np.float32]:
    """The mean over the last axis as the core takes it: each beat's lanes a tree, the
    beats accumulated in order, then divided by the row's length, all in binary32."""
    n = values.shape[-1]
    beats = values.reshape(*values.shape[:-1], n // simd, simd)
    total = np.zeros(values.shape[:-1], dtype=np.float32)
    for beat in range(n // simd):
        total = (total + _tree(beats[..., beat, :])).astype(np.float32)
    found: npt.NDArray[np.float32] = (total / np.float32(n)).astype(np.float32)[..., None]
    return found


def binary32_core(x: npt.NDArray[Any], simd: int, epsilon: float) -> npt.NDArray[np.float32]:
    """A model of the core's arithmetic, not a reference: its statistics in binary32 in
    its order (``_mean`` of the row, then of the centred values' squares),
    ``1 / sqrt(var + epsilon)`` rounded at each step, times each centred value. Denormals
    are kept, which the hardware's operators may flush; ``sqrt`` and the division are
    rounded once each, which a fused reciprocal square root may not be."""
    values = np.asarray(x).astype(np.float32)
    centred = (values - _mean(values, simd)).astype(np.float32)
    variance = _mean((centred * centred).astype(np.float32), simd)
    root = np.sqrt((variance + np.float32(epsilon)).astype(np.float32)).astype(np.float32)
    scale = (np.float32(1.0) / root).astype(np.float32)
    found: npt.NDArray[np.float32] = (scale * centred).astype(np.float32)
    return found


def case(
    shape: tuple[int, ...], dtype: str = "FLOAT32", epsilon: float = EPSILON
) -> dict[str, Any]:
    """LayerNorm of a ``dtype`` tensor of ``shape`` over its last axis, with ``epsilon``."""
    return dict(
        space_type=LayerNormKernel,
        inputs={"input_channel": Tensor(shape, ScalarEncoding(DataType[dtype]))},
        outputs={"output_channel": shape},
        reference=lambda input_channel: {"output_channel": layernorm(input_channel, epsilon)},
        facts={"platform": FULL_DSP48E2, "epsilon": epsilon},
        tolerance=TOLERANCE,
    )


def layernorm_rows() -> dict[str, Any]:
    """Rows of eight FLOAT32 values over two leading axes: SIMD 1, 4 and 8."""
    return case((2, 3, 8))


def layernorm_epsilon() -> dict[str, Any]:
    """Rows of twelve, with epsilon 1e-3: SIMD 1, 4 and 12."""
    return case((4, 12), epsilon=1e-3)


SPEC = KernelSpec(
    kernel=LayerNormKernel,
    reference=TEST_SIDE,
    cases={"layernorm_rows": layernorm_rows, "layernorm_epsilon": layernorm_epsilon},
    space="layernorm_rows",
    probes=(
        Probe(
            "an integer input: the mean stage adds its lanes in their own type",
            lambda: placed(case((2, 8), "INT8")),
            frozenset({"dtype-family"}),
        ),
        Probe(
            "a float other than FLOAT32: the top's lanes are binary32",
            lambda: placed(case((2, 8), "FLOAT16")),
            frozenset({"dtype-family"}),
        ),
        Probe(
            "an infinite epsilon",
            lambda: placed(layernorm_rows(), epsilon=float("inf")),
            frozenset({"layernorm-epsilon"}),
        ),
        Probe(
            "an epsilon beyond binary32",
            lambda: placed(layernorm_rows(), epsilon=1e300),
            frozenset({"layernorm-epsilon"}),
        ),
        Probe(
            "a NaN epsilon",
            lambda: placed(layernorm_rows(), epsilon=float("nan")),
            frozenset({"layernorm-epsilon"}),
        ),
    ),
    sweeps=(SweepCases("kernels.sweeps.layernorm_numeric", ("CASES",), ("sweep-layernorm",)),),
    unit=("tests/kernels/test_layernorm.py",),
)

__all__ = [
    "EPSILON",
    "SPEC",
    "TOLERANCE",
    "binary32_core",
    "bounds",
    "case",
    "condition",
    "layernorm",
    "layernorm_epsilon",
    "layernorm_rows",
]
