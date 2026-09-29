# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A ROM streaming a constant integer operand cyclically, for any consumer.

The consumer supplies the operand ``contents`` and the beat ``form`` in which
it reads them; the kernel packs the image in that form and streams it
cyclically from an initialized ROM. Its ``output`` port presents the stream
contract, so a parent connects it to a consumer through a stream instead of
wiring pins: supplied with a ``Stream`` node (``output_stream=weights``), it
is the stream's producer, on FinnLib's native ``odat``/``ovld``/``ordy``. The
image is embedded in the build requirements; there is no initialization file.

The same kernel serves a matrix tile walk (matmul weights), a chunked or
replicated tile (tiled MVU), a channel vector (elementwise parameters) or any
other traversal, so the consumer's order is produced directly with no adapter.
A parent may place it inside an operation kernel or beside one; the contract is
the same either way. A ROM is read-only and holds one operand: it refuses
contents that must be ``writable`` at run time, or several ``sets``.
"""

from __future__ import annotations

from collections.abc import Mapping

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    reject,
)
from finn.dataflow.datatypes import QONNXDataType, ordinary_integer_bounds
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import (
    BEAT_SEQUENCE,
    TRAVERSAL,
    BeatSequence,
    Repetition,
    Traversal,
    pack,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.base import CLOCKING, NATIVE_CLOCKING, Clocking, Kernel
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.scalar import integer_scalar
from finn.kernels.datatypes.semantics import (
    INTEGER_TENSOR,
    INTEGER_VECTOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
    IntegerVector,
)
from finn.kernels.port import GivenPort
from finn.kernels.streams import Stream

# UltraRAM is not offered: bitstream initialization of UltraRAM is not
# available on every supported device family.
CYCLIC_ROM_STYLES = ("auto", "distributed", "block")


class RomKernel(Kernel):
    """The ``cyclic_stream`` module: the image, from reset, word after word, wrapping.

    The image is the packed ``INIT_DATA`` parameter, word zero in the low bits,
    so the contents are part of the build identity. The registered output
    holds its word until ready; reset discards a pending word. ``rom_style``
    is the synthesis attribute: ``auto`` leaves inference to the tool,
    ``distributed`` requests LUT ROM and ``block`` block RAM.
    """

    id = "cyclic_stream"
    version = "1"
    module = "cyclic_stream"

    dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    element = integer_scalar(dtype, Integer())
    form: Traversal = Param(semantics=TRAVERSAL)
    contents: IntegerTensor = Param(semantics=INTEGER_TENSOR)
    # What the consumer needs of its memory: a ROM refuses either.
    writable: bool = Param(default=False)
    sets: int = Param(default=1)
    # The stream it drives, when a parent places it beside a consumer.
    output_stream: Stream = Param(required=False)
    rom_style: str = Decision(values=CYCLIC_ROM_STYLES)

    @constraint
    def read_only(self) -> bool | Rejected:
        if self.writable:
            return reject("rom-writable", "a ROM cannot be rewritten at run time")
        if self.sets != 1:
            return reject("rom-sets", "a ROM holds one operand")
        return True

    @constraint
    def carried(self) -> bool | Rejected:
        """The stream it drives, when placed, carries the element it stores."""
        if not self.present(RomKernel.output_stream):
            return True
        return stored_element(self.output_stream.tensor.element, self.element.encoding)

    admission = ConstraintGroup(read_only, carried)

    @derived(semantics=INTEGER_VECTOR)
    def image(self) -> IntegerVector | Rejected:
        encoding = self.element.encoding
        low, high = ordinary_integer_bounds(encoding.dtype)
        try:
            words = pack(self.form, self.contents, encoding.bits)
        except ValueError as error:
            return reject("cyclic-values", str(error))
        if any(not low <= value <= high for value in _leaves(self.contents)):
            return reject(
                "cyclic-values",
                f"every value must be an integer admitted by {encoding.datatype_name}",
            )
        return words

    @derived(semantics=BEAT_SEQUENCE)
    def sequence(self) -> BeatSequence:
        return BeatSequence(self.form, Repetition.CYCLIC)

    output = GivenPort(
        name="output",
        endpoint=Endpoint.INITIATOR,
        stream=output_stream,
        sequence=sequence,
        idle_dtype=dtype,
        idle_lanes=sequence.form.lanes,
        signals=("odat", "ovld", "ordy"),
        clock="clk",
        reset="rst",
    )

    @derived(semantics=CLOCKING)
    def clocking(self) -> Clocking:
        return NATIVE_CLOCKING

    def parameters(self) -> Mapping[str, int | str]:
        bits, image = self.output.transport.data_width, self.image
        packed = sum(word << (index * bits) for index, word in enumerate(image))
        return {
            "DEPTH": len(image),
            "INIT_DATA": f"{bits * len(image)}'h{packed:x}",
            "ROM_STYLE": f'"{self.rom_style}"',
            "W": bits,
        }

    def sources(self) -> tuple[CopiedSource, ...]:
        return (CopiedSource("kernels", "cyclic_stream.sv", provides=("module:cyclic_stream",)),)


def stored_element(carried: ScalarEncoding, stored: ScalarEncoding) -> bool | Rejected:
    """A memory's output stream carries the element the memory stores."""
    if carried != stored:
        return reject(
            "memory-element",
            f"the stream carries {carried.datatype_name}, the memory stores {stored.datatype_name}",
        )
    return True


def _leaves(values: object) -> tuple[int, ...]:
    if type(values) is int:
        return (values,)
    assert isinstance(values, tuple)
    return tuple(leaf for item in values for leaf in _leaves(item))


__all__ = ["CYCLIC_ROM_STYLES", "RomKernel", "stored_element"]
