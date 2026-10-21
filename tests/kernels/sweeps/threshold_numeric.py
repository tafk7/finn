# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Numeric XSI conformance of thresholding_axi's threshold rows, initial and written.

Each case places ``ThresholdingAxiKernel`` between two boundary channels and
streams every INT4 value through it, free and stalled. Its table is one row
shared by every channel (C = 1, PE above it: each lane keeps a copy of the
row), or a row a channel. With AXI-Lite, the kernel presents its control bus and
the run first writes a new table through it: the writes the kernel declares for
that table (its register map, ``ThresholdingAxiKernel.register_map``), into
hardware built with the initial one. Every output word shows that the write
reached the lanes reading it: for a shared row, one write reaches every lane.
With one threshold (N = 1) the wrapper's configuration address is the AXI-Lite
word select alone (no address bit of its own). The expected words come from the
table each case leaves in the memories, packed here in the boundaries' row-major
order, PE lanes a beat; words that differ are decoded as an order
(``finn.harness.orders``). Run with Vivado selected (FinnLib is the ``finnlib``
resource); each simulation runs in a fresh process.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, built, choose, with_direct_transports

from finn.core.rtlsim_bench import materialize
from finn.core.rtlsim_pacing import FREE, STALLED
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import pack, vector_major
from finn.harness.orders import Stream, decode
from finn.harness.toolchain import print_identity
from finn.kernels.artifacts.module import declared_registers
from finn.kernels.channels import Channel
from finn.kernels.network import Interface, Network, wire
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.toolchain.launcher import host_toolchain
from kernels.sweeps.rtl_transport import drive_observed

ELEMENT = DataType["INT4"]
PIXELS, CHANNELS = 4, 8
# Every INT4 value, twice, in an order that puts each value in several lanes.
VALUES = np.array([(5 * i + 3) % 16 - 8 for i in range(PIXELS * CHANNELS)]).reshape(
    PIXELS, CHANNELS
)


@dataclass(frozen=True)
class Case:
    label: str
    initial: tuple[tuple[int, ...], ...]  # the table's rows: one, or one a channel
    written: tuple[tuple[int, ...], ...] | None = None  # through AXI-Lite, when given
    pe: int = 4


def _rows(count: int, thresholds: tuple[int, ...], step: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(t + step * c for t in thresholds) for c in range(count))


CASES = (
    Case("shared", ((-3, 0, 2),)),
    Case("shared_axilite", ((-3, 0, 2),), ((-6, -1, 5),)),
    Case("shared_axilite_n1", ((0,),), ((-2,),)),
    Case("channels_axilite", _rows(CHANNELS, (-7, -6, -5), 1), _rows(CHANNELS, (-1, 1, 3), -1)),
)


def placed(case: Case) -> Network:
    """The kernel between boundary channels ``s_axis_0`` and ``m_axis_0``, presenting its
    control bus at ``s_axilite`` when the case writes its table."""
    count = len(case.initial[0])
    result = ScalarEncoding(DataType[f"UINT{count.bit_length()}"])
    x = Tensor((PIXELS, CHANNELS), ScalarEncoding(ELEMENT))
    y = Tensor((PIXELS, CHANNELS), result)
    facts: dict[str, Any] = dict(
        threshold_dtype=ELEMENT,
        thresholds=(case.initial,),
        bias=0,
        input_tensor=x,
        platform=FULL_DSP48E2,
    )
    if case.written is not None:
        facts["control"] = "s_axilite"
    network = Network(
        {
            "x": wire(
                tensor=x,
                port="s_axis_0",
                consumer=Interface("activate", "input_contract"),
                platform=FULL_DSP48E2,
            ),
            "y": wire(
                tensor=y,
                port="m_axis_0",
                producer=Interface("activate", "output_contract"),
                platform=FULL_DSP48E2,
            ),
            "activate": ThresholdingAxiKernel(**facts),
        }
    )
    return with_direct_transports(
        choose(
            network,
            {
                "activate.pe": case.pe,
                "activate.use_axilite": case.written is not None,
                "activate.deep_pipeline": False,
                "activate.ram_style": "distributed",
                "activate.block_stages": 0,
                "activate.ultra_stages": 0,
            },
        )
    )


def stem(case: Case) -> str:
    """A case's module's stem: ``thresholds_<label>``."""
    return f"thresholds_{case.label}"


def run(case: Case, evidence: Path) -> None:
    point = placed(case)
    table = np.array(case.written if case.written is not None else case.initial)
    if len(table) == 1:
        table = np.repeat(table, CHANNELS, axis=0)  # the reference's row a channel
    levels = (VALUES[..., None] >= table).sum(axis=-1)
    result_bits = point.member("activate").result_dtype.bitwidth()
    form = vector_major((PIXELS, CHANNELS), case.pe)
    for name in ("x", "y"):
        ends = point.member(name).query(Channel.sides).value
        assert form in (ends.sink.form, ends.source.form), (case.label, name, ends)
    stimulus = list(pack(form, VALUES.ravel().tolist(), ELEMENT.bitwidth()))
    expected = list(pack(form, levels.ravel().tolist(), result_bits))
    top, sources, data = materialize(
        built(point, stem(case)), evidence / case.label, toolchain=host_toolchain()
    )
    # The writes a kernel holding the written table declares, into the hardware built
    # with the initial one: the same addresses (the same shape and PE), other words.
    written = placed(replace(case, initial=case.written)) if case.written is not None else None
    configuration = declared_registers(built(written, stem(case))) if written is not None else {}
    assert set(configuration) == ({"s_axilite"} if case.written is not None else set())
    mask = (1 << (case.pe * result_bits)) - 1
    for stalled in (False, True):
        measured = drive_observed(
            top,
            sources,
            {"s_axis_0": stimulus},
            {"m_axis_0": len(expected)},
            {},
            pacing=STALLED if stalled else FREE,
            directory=evidence / case.label / ("stalled" if stalled else "free"),
            compiled=evidence / case.label / "compiled",
            data_files=data,
            registers=configuration,
        )
        actual = [word & mask for word in measured["outputs"]["m_axis_0"]]
        if actual != expected:
            decoded = decode(
                {"m_axis_0": Stream(form, result_bits)},
                {"m_axis_0": actual},
                inputs={"s_axis_0": Stream(form, ELEMENT.bitwidth())},
                values={"s_axis_0": VALUES},
                reference=lambda read: {"m_axis_0": (read["s_axis_0"][..., None] >= table).sum(-1)},
            )
            found = decoded.message if decoded else f"{actual} != {expected}"
            raise AssertionError(f"{case.label} stalled={stalled}: {found}")
        print(f"PASS {case.label} stalled={stalled}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case.label for case in CASES])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.output or Path(tempfile.mkdtemp(prefix="threshold-evidence-"))
    print_identity()
    print(f"Evidence: {directory}", flush=True)
    for case in CASES:
        if args.case in (None, case.label):
            run(case, directory)


if __name__ == "__main__":
    main()
