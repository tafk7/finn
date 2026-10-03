# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""PROBE (stream source, not for merge): a stub second source for a stream's known value.

It stands for a future fetcher, one that streams a value from memory-mapped
memory (DDR, HBM) through a memory port, so that a stream's ``source`` has two
candidates. It presents what a memstream presents (one pass of the consumer's
order, cyclic) and refuses on a platform fact: a platform with no memory port
(``fetch-port``). It binds no real module.
"""

from __future__ import annotations

from collections.abc import Mapping

from finn.core.space import ConstraintGroup, Param, Rejected, constraint, derived, reject
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.schedule import Index
from finn.dataflow.stream import Stream
from finn.dataflow.traversal import BeatSequence, Repetition, Traversal
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import Contribution
from finn.kernels.base import Kernel
from finn.kernels.datatypes.semantics import (
    INTEGER_TENSOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
    integer_range,
)
from finn.kernels.port import AxiStreamPort
from finn.kernels.target import Platform

LANE = Index("lane")


class FetchStubKernel(Kernel):
    id = "probe.fetch"
    version = 1
    rtl_module = "probe_fetch_stub"

    dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    form: Traversal = Param()
    contents: IntegerTensor = Param(semantics=INTEGER_TENSOR)
    sets: int = Param(default=1)
    set_stream: Stream = Param(required=False)
    staged: bool = Param(default=False)
    platform: Platform = Param(default=Platform())

    @derived
    def value_range(self) -> tuple[int, ...]:
        least, greatest = integer_range(self.contents)
        return (least, greatest)

    @constraint
    def port_available(self) -> bool | Rejected:
        if self.platform.memory_ports < 1:
            return reject("fetch-port", "a fetcher needs a memory port, and the platform has none")
        return True

    @constraint
    def single_set(self) -> bool | Rejected:
        if self.sets > 1:
            return reject("fetch-sets", "this fetcher streams one set")
        return True

    admission = ConstraintGroup(port_available, single_set)

    @derived
    def output_sequence(self) -> BeatSequence:
        return BeatSequence(self.form, Repetition.CYCLIC)

    @derived
    def word_factors(self) -> dict[Index, int]:
        return {LANE: self.form.lanes}

    output = AxiStreamPort(
        name="m_axis_0",
        endpoint=Endpoint.INITIATOR,
        staged=staged,
        sequence=output_sequence,
        dtype=dtype,
        value_range=value_range,
        lanes=(LANE,),
        factors=word_factors,
    )

    def parameters(self) -> Mapping[str, int | str]:
        return {}

    def sources(self) -> tuple[Contribution, ...]:
        return ()


__all__ = ["FetchStubKernel"]
