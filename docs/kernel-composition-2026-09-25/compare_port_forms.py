# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the adopted bound port form with a handwritten contained-scalar form.

Run from the FINN checkout with ``PYTHONPATH=src:deps/qonnx/src``. Both forms
admit the same port under a dynamic bound. The contained form works, but each
policy shape needs its own port subclass, and a dynamic bound must be forwarded
through that subclass as another Param. The bound form places the production
``integer_scalar`` beside an unmodified ``AxiStreamPort``.
"""

from qonnx.core.datatype import DataType

from finn.core.space import (
    Param,
    Rejected,
    Space,
    Subspace,
    View,
    compile_space,
    constraint,
    default_semantics,
    derived,
    inspection,
    reject,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.scalar import IntegerScalar, Scalar, integer_scalar
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS as QONNX
from finn.kernels.datatypes.values import qonnx_datatype_width
from finn.kernels.physical.axi_stream import AxiStream, axi_stream


class ContainedPort(Space):
    """A port that owns its scalar; the base admits any positive-width dtype."""

    name = Param(str)
    endpoint = Param(Endpoint)
    last = Param(bool)
    lanes = Param(int)
    dtype = Param(QONNX)
    element = Subspace(Scalar, dtype=dtype)

    @derived
    def payload_bits(self) -> int:
        return qonnx_datatype_width(self.dtype) * self.lanes

    @constraint
    def lanes_valid(self) -> bool | Rejected:
        return True if self.lanes > 0 else reject("interface-lanes", "lanes must be positive")

    @derived(semantics=default_semantics(AxiStream))
    def candidate(self) -> AxiStream:
        encoding = self.element.encoding()
        return AxiStream(
            self.name, encoding.dtype, self.lanes, endpoint=self.endpoint, last=self.last
        )

    stream = View(candidate, constraints=(lanes_valid,))


class BoundedContainedPort(ContainedPort):
    """One subclass per policy shape; the dynamic bound is forwarded as a Param."""

    limit = Param(int)
    element = integer_scalar(ContainedPort.dtype, Integer(1, limit))


class Contained(Space):
    limit = Param(int, required=False)
    dtype = Param(QONNX)
    values = Subspace(
        BoundedContainedPort,
        name="values",
        endpoint=Endpoint.TARGET,
        last=False,
        lanes=2,
        dtype=dtype,
        limit=limit,
    )


class Bound(Space):
    limit = Param(int, required=False)
    dtype = Param(QONNX)
    values_type = integer_scalar(dtype, Integer(1, limit))
    values = axi_stream("values", 2, Endpoint.TARGET, values_type)


def report(label: str, point: Space, scalar: IntegerScalar, stream_query: object) -> None:
    admission = scalar.inspect(IntegerScalar.admission)
    rules = {
        key.rsplit(".", 1)[-1]: type(value).__name__ for key, value in admission.results.items()
    }
    stats = inspection.statistics(compile_space(type(point)))
    print(f"{label}: rules={rules} stream={type(stream_query).__name__}", end=" ")
    print(f"declarations={stats.authored_declarations} scopes={stats.scopes}")


def main() -> None:
    for dtype, limit in (("TERNARY", None), ("INT3", None), ("INT3", 8), ("INT9", 8)):
        facts = {"dtype": DataType[dtype]} | ({} if limit is None else {"limit": limit})
        contained, bound = Contained(**facts), Bound(**facts)
        print(f"-- dtype={dtype} limit={limit}")
        report("contained", contained, contained.values.element, contained.values.stream.query())
        report("bound    ", bound, bound.values_type, bound.values.stream.query())
        assert contained.values.payload_bits == bound.values.payload_bits


if __name__ == "__main__":
    main()
