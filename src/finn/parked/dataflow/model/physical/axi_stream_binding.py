# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional adapters between AXIS streams and logical composition bindings."""

from finn.dataflow.model.logical.region import DataflowRegion
from finn.kernels.physical.axi_stream import AxiStream
from finn.parked.dataflow.model.physical.interface import KernelStreamBinding, region_ports
from finn.kernels.physical.layout import PeriodicLast


def axi_stream_from_port(
    name: str, region: DataflowRegion, port_id: str, *, last: bool = False
) -> AxiStream:
    """Derive a stream's dtype, beat size and direction from a logical port."""
    for port, endpoint in region_ports(region):
        if port.id == port_id:
            return AxiStream(
                name,
                port.operand.element_type,
                port.beat_sequence.elements_per_beat,
                endpoint=endpoint,
                last=last,
            )
    raise ValueError(f"no logical stream port {port_id!r}")


def bind_axi_stream(
    stream: AxiStream,
    region: DataflowRegion,
    port_id: str,
    *,
    framing: PeriodicLast | None = None,
) -> KernelStreamBinding:
    """Check logical agreement and produce the composer's binding value."""
    expected = axi_stream_from_port(stream.name, region, port_id, last=stream.last)
    if stream != expected:
        raise ValueError("AXIS dtype, elements per beat or direction disagrees with logical port")
    if stream.last != (framing is not None):
        raise ValueError("tlast requires explicit framing; unframed streams have no tlast")
    port = next(port for port, _ in region_ports(region) if port.id == port_id)
    if framing is not None and port.beat_sequence.beat_count % framing.period_beats:
        raise ValueError("framing period must divide the logical pass")
    return KernelStreamBinding(port_id, stream.name, stream.payload, framing)


__all__ = ["axi_stream_from_port", "bind_axi_stream"]
