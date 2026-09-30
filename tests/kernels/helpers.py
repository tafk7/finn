# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Test harness: configure a kernel node from its formals, then commit choices by key.

Facts are the root node's typed formals; a missing required one is refused at
the node call. Choices use the stable decision keys ``inspection`` reports."""

from collections.abc import Callable, Mapping
from typing import TypeVar

from finn.core.space import Constraint, Space, design_space
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.gemm import Form
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.streams import ADAPTER_RAM_STYLES, Stream
from finn.core.space.results import Available, QueryResult
from finn.kernels.configure import commit, settle, undecided

T = TypeVar("T")
S = TypeVar("S", bound=Space)


def point_for(kernel: Callable[..., S], facts: Mapping[str, object], **choices: object) -> S:
    return commit(design_space(kernel(**facts)), choices)


def value(answer: QueryResult[T]) -> T:
    assert isinstance(answer, Available), answer
    return answer.value


def assess(point: Space, condition: Constraint) -> QueryResult[bool]:
    return point.inspect(condition).result


def placed_dotp(
    family: Callable[..., S],
    *,
    activation_dtype: QONNXDataType,
    weights_dtype: QONNXDataType,
    result_dtype: QONNXDataType,
    pe: object = None,
    simd: object = None,
    compute_pumping: object = False,
    rows: int = 1,
    outputs: int | None = None,
    reduction: int | None = None,
    **facts: object,
) -> S:
    """A dot-product core between three boundary streams, its folding factors committed.

    The core takes its extents from the streams: ``outputs`` (N) defaults to PE
    and ``reduction`` (K) to SIMD, one fold each. A folding factor left ``None``
    stays open.
    """
    form = facts.get("form", Form.DENSE)
    n = outputs if outputs is not None else (pe if isinstance(pe, int) and pe > 0 else 1)
    k = reduction if reduction is not None else (simd if isinstance(simd, int) and simd > 0 else 1)
    x_shape = (rows, k, n) if form is Form.DEPTHWISE else (rows, k)

    class Placed(Space):
        x = Stream(tensor=Tensor(x_shape, ScalarEncoding(activation_dtype)), port="in0_V")
        w = Stream(tensor=Tensor((k, n), ScalarEncoding(weights_dtype)), port="in1_V")
        y = Stream(tensor=Tensor((rows, n), ScalarEncoding(result_dtype)), port="out0_V")
        compute = family(x_stream=x, w_stream=w, y_stream=y, result_dtype=result_dtype, **facts)

    choices = {
        key: value
        for key, value in (
            ("compute.pe", pe),
            ("compute.simd", simd),
            ("compute.compute_pumping", compute_pumping),
        )
        if value is not None
    }
    point = design_space(Placed())
    placed = commit(point, choices)
    return placed.compute


def settled(point: S, ram_style: str = "auto") -> S:
    """Settle every Decision over kernels; each adapter input_gen's memory takes ``ram_style``."""
    point = settle(point).point
    styles = undecided(point, ADAPTER_RAM_STYLES)
    return commit(point, dict.fromkeys(styles, ram_style)) if styles else point
