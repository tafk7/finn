# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Cyclic on-chip delivery of a constant integer operand, reusable by any consumer.

The consumer supplies the operand values and the beat ``form`` in which it reads
them; the kernel packs the image in that form and streams it cyclically from an
initialized ROM. Its ``output`` view is a stream contract, so a parent connects
it to a consumer port through a stream net instead of wiring pins. The
image is embedded in the build requirements; there is no initialization file.

The same kernel serves a matrix tile walk (MVAU/VVAU weights), a chunked or
replicated tile (tiled MVU), a channel vector (elementwise parameters) or any
other traversal, so the consumer's order is produced directly with no adapter.
A parent may place it inside an operation kernel or beside one; the contract is
the same either way.
"""

from finn.core.space import Decision, Param, Port, Rejected, derived, reject, view
from finn.kernels.artifacts.requirements import ModuleBuildRequirements
from finn.kernels.base import Kernel
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.scalar import integer_scalar
from finn.kernels.datatypes.semantics import (
    INTEGER_TENSOR,
    INTEGER_VECTOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerVector,
)
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.kernels.physical.contract import STREAM_CONTRACT, StreamContract
from finn.kernels.physical.forms import TRAVERSAL, Repetition, pack
from finn.kernels.streaming import (
    CYCLIC_ROM_STYLES,
    cyclic_stream_interface,
    cyclic_stream_requirements,
)
from finn.core.space import default_semantics
from finn.kernels.streams import MODULE, STREAM


class CyclicDelivery(Kernel):
    id = "finn.cyclic_delivery"
    version = "1"

    dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    element = integer_scalar(dtype, Integer())
    form = Param(TRAVERSAL)
    values = Param(INTEGER_TENSOR)
    rom_style = Decision(str, values=CYCLIC_ROM_STYLES)

    @derived(semantics=INTEGER_VECTOR)
    def image(self) -> IntegerVector | Rejected:
        encoding = self.element.encoding()
        low, high = ordinary_integer_bounds(encoding.dtype)
        try:
            words = pack(self.form, self.values, encoding.bits)
        except ValueError as error:
            return reject("cyclic-values", str(error))
        if any(not low <= value <= high for value in _leaves(self.values)):
            return reject(
                "cyclic-values",
                f"every value must be an integer admitted by {encoding.datatype_name}",
            )
        return words

    @view(semantics=STREAM_CONTRACT)
    def output(self) -> StreamContract:
        encoding = self.element.encoding()
        form = self.form
        return StreamContract(
            cyclic_stream_interface(word_bits=form.lanes * encoding.bits),
            encoding,
            form,
            Repetition.CYCLIC,
        )

    @view(semantics=default_semantics(ModuleBuildRequirements))
    def build_requirements(self) -> ModuleBuildRequirements:
        image = self.image
        return cyclic_stream_requirements(
            word_bits=self.output().payload_bits,
            depth=len(image),
            image=image,
            rom_style=self.rom_style,
        )

    output_stream = Port(STREAM, "out", offer=output)
    exports = {MODULE: build_requirements}


def _leaves(values: object) -> tuple[int, ...]:
    if type(values) is int:
        return (values,)
    assert isinstance(values, tuple)
    return tuple(leaf for item in values for leaf in _leaves(item))


__all__ = ["CyclicDelivery"]
