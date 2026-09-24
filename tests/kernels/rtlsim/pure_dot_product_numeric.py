# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded arithmetic conformance of the physical dotp component.

Run explicitly on a machine with XSI; this is not part of ordinary unit tests.
Packing and transport reuse the existing independent RTL test helpers. The
caller proves a result width from its own reduction bound, then supplies
that dtype to the physical component. No logical kernel or contract is needed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import cast

import numpy as np  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from kernels.rtlsim.dotp_support import (
    Case,
    _activation_beats,
    _unpack,
    _weight_beats,
    _wrapper,
)
from kernels.helpers import point_for, value
from kernels.rtlsim.rtl_transport import drive_observed
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.target import DspBlock
from finn.kernels.resources import resource_root


@dataclass(frozen=True)
class Configuration:
    label: str
    target: DspBlock
    width: int
    pe: int
    simd: int
    activation: str
    weight: str
    pumping: bool = False
    segment: int = 0
    repetitions: int = 4


CASES = (
    Configuration("tiny", DspBlock.DSP48E1, 1, 1, 1, "INT2", "INT2"),
    Configuration("packed_e1", DspBlock.DSP48E1, 8, 2, 4, "INT3", "INT3"),
    Configuration("packed_e2_pumped", DspBlock.DSP48E2, 6, 2, 3, "UINT3", "INT3", True),
    Configuration("packed_dsp58", DspBlock.DSP58, 8, 2, 4, "INT3", "INT3"),
    Configuration("int8_signed", DspBlock.DSP58, 8, 2, 4, "INT8", "INT8"),
    Configuration("int8_unsigned", DspBlock.DSP58, 8, 4, 4, "UINT8", "INT8"),
    Configuration("int8_odd_pumped", DspBlock.DSP58, 6, 2, 3, "UINT8", "INT8", True),
    Configuration("int8_segmented_pumped", DspBlock.DSP58, 14, 2, 7, "UINT8", "INT8", True, 1),
    Configuration("signed9", DspBlock.DSP58, 6, 2, 3, "INT9", "INT8"),
    Configuration("unsigned17", DspBlock.DSP48E2, 4, 2, 2, "UINT17", "INT2"),
    Configuration("signed18", DspBlock.DSP48E2, 4, 2, 2, "INT18", "INT2"),
    Configuration("unsigned23", DspBlock.DSP58, 4, 2, 2, "UINT23", "INT2"),
    Configuration("int8_segmented", DspBlock.DSP58, 12, 2, 6, "INT8", "INT8", False, 1),
)


STRESS_CASES = (
    Configuration("single_e1", DspBlock.DSP48E1, 1, 1, 1, "INT2", "INT2", repetitions=32),
    Configuration("one_beat_e1", DspBlock.DSP48E1, 4, 2, 4, "INT3", "INT3", repetitions=32),
    Configuration("one_beat_dsp58", DspBlock.DSP58, 4, 2, 4, "INT3", "INT3", repetitions=32),
    Configuration(
        "one_beat_softvec_pumped", DspBlock.DSP58, 2, 2, 2, "INT3", "INT3", True, repetitions=32
    ),
    Configuration("one_beat_int8", DspBlock.DSP58, 1, 2, 1, "INT8", "INT8", repetitions=32),
    Configuration(
        "one_beat_int8_pumped", DspBlock.DSP58, 3, 2, 3, "UINT8", "INT8", True, repetitions=32
    ),
    Configuration(
        "one_beat_int8_segmented", DspBlock.DSP58, 24, 2, 24, "INT8", "INT8", False, 1, 32
    ),
    Configuration(
        "one_beat_segmented_pumped", DspBlock.DSP58, 7, 2, 7, "UINT8", "INT8", True, 1, 32
    ),
)


def run(configuration: Configuration, evidence: Path, *, backpressure_ticks: int = 5) -> None:
    c = configuration
    a_type, w_type = DataType[c.activation], DataType[c.weight]
    # Independent caller-side bound: cover every endpoint product and every
    # partial sum in a reduction. Grow a signed encoding until the range fits.
    endpoints = [
        int(a) * int(w) * c.width
        for a in (a_type.min(), a_type.max())
        for w in (w_type.min(), w_type.max())
    ]
    result_bits = 1
    while min(endpoints) < -(1 << (result_bits - 1)) or max(endpoints) >= (1 << (result_bits - 1)):
        result_bits += 1
    result_type = DataType[f"INT{result_bits}"]
    point = cast(
        DotpAxiKernel,
        point_for(
            DotpAxiKernel,
            dict(
                pe=c.pe,
                simd=c.simd,
                activation_dtype=a_type,
                weights_dtype=w_type,
                result_dtype=result_type,
                target_dsp=c.target,
                segment_length=c.segment,
            ),
            compute_pumping=c.pumping,
        ),
    )
    module = value(point.build_requirements().accepted_result)
    case = Case(
        c.label,
        c.target,
        c.repetitions,
        c.width,
        4,
        c.pe,
        c.simd,
        c.activation,
        c.weight,
        result_type.name,
        pumping=c.pumping,
    )
    rng = np.random.RandomState(37)
    activations = rng.randint(
        int(a_type.min()), int(a_type.max()) + 1, size=(c.repetitions, c.width), dtype=np.int64
    )
    weights = rng.randint(
        int(w_type.min()), int(w_type.max()) + 1, size=(c.width, 4), dtype=np.int64
    )
    activations[0, :] = int(a_type.min())
    activations[1, :] = int(a_type.max())
    weights[:, 0] = int(w_type.min())
    weights[:, 1] = int(w_type.max())
    expected = activations @ weights
    assert tuple(
        tuple(sum(a * b for a, b in zip(x, w)) for w in weights.T.tolist())
        for x in activations.tolist()
    ) == tuple(map(tuple, expected.tolist()))
    stimulus = {
        "in0": _activation_beats(case, activations, a_type.bitwidth()),
        "in1": _weight_beats(case, weights, w_type.bitwidth()),
    }
    # Exercise ignored high input padding, rather than supplying only zero fill.
    for name, stream in (("in0", point.activation), ("in1", point.weights)):
        padding = ((1 << stream.carrier_bits) - 1) ^ ((1 << stream.payload_bits) - 1)
        stimulus[name] = [beat | (padding if j % 2 else 0) for j, beat in enumerate(stimulus[name])]
    roots = {
        "finnlib": Path(os.environ["FINNLIB_ROOT"]),
        "kernels": resource_root(),
    }
    sources = []
    for source in module.contributions:
        assert isinstance(source, CopiedSource)
        sources.append(str(roots[source.root] / source.path))
    case_directory = evidence / c.label
    case_directory.mkdir(parents=True, exist_ok=False)
    top = "pure_dotp_" + c.label
    wrapper = case_directory / (top + ".sv")
    wrapper.write_text(_wrapper(top, case, dict(module.parameters)))
    for stalled in (False, True):
        measured = drive_observed(
            top,
            [*sources, str(wrapper)],
            {name + "_V": beats for name, beats in stimulus.items()},
            {"out0_V": case.repetitions * case.neuron_folds},
            {},
            stalls=stalled,
            input_stalls=False,
            backpressure_ticks=backpressure_ticks,
            directory=case_directory / ("stalled" if stalled else "free"),
        )
        actual = _unpack(case, measured["outputs"]["out0_V"], result_type.bitwidth())
        if not np.array_equal(actual, expected):
            raise AssertionError(
                f"{c.label} stalled={stalled}: {actual.tolist()} != {expected.tolist()}"
            )
        print(f"PASS {c.label} stalled={stalled} result={result_type.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    available = (*CASES, *STRESS_CASES)
    parser.add_argument("--case", choices=[case.label for case in available])
    parser.add_argument("--stress", action="store_true", help="run sustained one-beat reductions")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    evidence = arguments.output or Path(tempfile.mkdtemp(prefix="pure-dotp-evidence-"))
    print(f"Evidence: {evidence}", flush=True)
    selected: tuple[Configuration, ...] = STRESS_CASES if arguments.stress else CASES
    if arguments.case is not None:
        selected = tuple(case for case in available if case.label == arguments.case)
    for case in selected:
        run(case, evidence, backpressure_ticks=32 if case in STRESS_CASES else 5)
    print(f"PASS {len(selected)} configurations, two transport modes each")


if __name__ == "__main__":
    main()
