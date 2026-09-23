# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A bank of exact integer dot products and its PE/SIMD design space.

    X: (repetitions, matrix_width)
    W: (matrix_height, matrix_width)
    Y: (repetitions, matrix_height)
    Y[r, m] = sum_k X[r, k] * W[m, k]

All values admitted by each input datatype are allowed, including the most
negative signed value. Results are exact, with the smallest sufficient signed
output dtype. PE and SIMD change grouping and presentation, never this equation.
The input vector is presented again for each output fold; no replay storage is
implied. Weights describe one matrix, presented again for each repetition.
"""

from collections.abc import Sequence

from finn.dataflow.analysis.integer_dot import ExactIntegerDot, IntegerRange
from finn.dataflow.model.kernel_base import Kernel
from finn.dataflow.model.logical.contract_authoring import (
    Count,
    Final,
    InputContract,
    LocalContract,
    OperandDeclaration,
    OutputContract,
    Presentation,
)
from finn.dataflow.model.logical.contract_expressions import ExactQuotient, Index, Schedule, integer
from finn.dataflow.model.logical.datatype_domains import Integer
from finn.dataflow.model.logical.datatype_semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.model.logical.datatypes import (
    DatatypeError,
    QONNXDataType,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.dataflow.space import Decision, Input, derived, divisors_of, reject


class DotProduct(Kernel):
    """One logical definition; query ``contract`` for its assessed schedule/interface."""

    id = "exact_integer_dot_product"
    version = "1"

    repetitions = Input(int)
    matrix_width = Input(int)
    matrix_height = Input(int)
    activation_dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    weights_dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)

    @derived(
        ExactIntegerDot,
        activation=activation_dtype,
        weight=weights_dtype,
        terms=matrix_width,
    )
    def arithmetic(*, activation: QONNXDataType, weight: QONNXDataType, terms: int) -> object:
        try:
            return ExactIntegerDot(
                IntegerRange(*ordinary_integer_bounds(activation)),
                IntegerRange(*ordinary_integer_bounds(weight)),
                terms,
            )
        except (DatatypeError, ValueError) as error:
            return reject("dot-product-arithmetic", str(error))

    @derived(QONNX_DATATYPE_VALUE_SEMANTICS, arithmetic=arithmetic)
    def result_type(*, arithmetic: ExactIntegerDot) -> QONNXDataType:
        return resolve_qonnx_datatype_name(f"INT{arithmetic.result_bits}")

    pe = Decision(int, domain=divisors_of(matrix_height))
    simd = Decision(int, domain=divisors_of(matrix_width))
    nf_count = ExactQuotient(matrix_height, pe)
    sf_count = ExactQuotient(matrix_width, simd)
    rep, nf, sf = Index("rep", repetitions), Index("nf", nf_count), Index("sf", sf_count)
    p, s = Index("p", pe), Index("s", simd)
    work = Schedule((rep, nf, sf))

    X = OperandDeclaration("X", activation_dtype, (repetitions, matrix_width), admits=Integer())
    W = OperandDeclaration("W", weights_dtype, (matrix_height, matrix_width), admits=Integer())
    Y = OperandDeclaration("Y", result_type, (repetitions, matrix_height))
    contract = LocalContract(
        work,
        inputs=(
            InputContract(
                "activation",
                selection=X[rep, sf * simd + s].over(s),
                requirements=Count(work),
                presentation=Presentation(work.axes, (s,)),
            ),
            InputContract(
                "weight",
                selection=W[nf * pe + p, sf * simd + s].over(p, s),
                requirements=Count(work),
                presentation=Presentation(work.axes, (p, s)),
            ),
        ),
        outputs=(
            OutputContract(
                "output",
                selection=Y[rep, nf * pe + p].over(p),
                availability=Final(work.fix(sf=integer(sf_count) - 1)),
                presentation=Presentation((rep, nf), (p,)),
            ),
        ),
        counting="one requirement occurrence per member of each fold group",
        conditions=(nf_count.condition, sf_count.condition),
    )

    def reference(
        self, activation: Sequence[Sequence[int]], weights: Sequence[Sequence[int]]
    ) -> tuple[tuple[int, ...], ...]:
        """Evaluate the defining equation without selecting PE, SIMD or a target."""
        if self.repetitions <= 0 or self.matrix_height <= 0:
            raise ValueError("dot-product tensor extents must be positive")
        if len(activation) != self.repetitions or len(weights) != self.matrix_height:
            raise ValueError("operand shapes differ from the declared dot-product tensors")
        arithmetic = self.arithmetic
        return tuple(tuple(arithmetic.evaluate(x, w) for w in weights) for x in activation)


__all__ = ["DotProduct"]
