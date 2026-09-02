# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel-only numerical evidence for the declarative ReplayBuffer Kernel.

The obligation is exactly the Region's: ``R x SF`` beats in, ``R x NF x SF``
beats out, each compact beat presented once per neuron fold, in order, with
nothing dropped or duplicated when the consumer stalls.  The golden sequence is
read off the resolved Region rather than restated here, so the fixture cannot
agree with the RTL about something the Region does not say.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow._engine import Decided, Engine
from finn.dataflow.design.region import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.model.compiler import _Ref, _compile_space
from finn.dataflow.model.declarations import Decision, Problem, Space, divisors_of
from finn.dataflow.model.kernel import configure_kernel
from finn.dataflow.model.replay_buffer import FINNLIB_SOURCES, ReplayBufferKernel
from finn.dataflow.spec_algebra import assemble_specs

from dataflow.rtlsim.rtl_transport import drive

PASS, FAIL = 0, 1

BOUND_NAMES = (
    "repetitions",
    "matrix_width",
    "matrix_height",
    "activation_type",
    "pe",
    "simd",
)


@dataclass(frozen=True)
class Case:
    label: str
    repetitions: int
    matrix_width: int
    matrix_height: int
    pe: int
    simd: int
    activation: str

    @property
    def neuron_folds(self) -> int:
        return self.matrix_height // self.pe

    @property
    def synapse_folds(self) -> int:
        return self.matrix_width // self.simd


CASES = (
    # One neuron fold: the buffer is an identity and must still pass everything.
    Case("identity", 1, 8, 4, 4, 2, "INT8"),
    Case("identity_repeated", 3, 8, 4, 4, 2, "INT8"),
    # Several neuron folds: each compact beat is presented NF times.
    Case("two_folds", 1, 8, 4, 2, 2, "INT8"),
    Case("three_folds", 2, 8, 6, 2, 2, "INT8"),
    Case("wide_lanes", 2, 12, 8, 4, 3, "INT4"),
    Case("unsigned", 2, 8, 4, 2, 2, "UINT8"),
    Case("single_lane", 1, 4, 4, 2, 1, "INT8"),
)
CASES_BY_LABEL = {case.label: case for case in CASES}


class Harness(Space):
    """The Design's stand-in: PE and SIMD are Region-visible, so it owns them."""

    repetitions = Problem(int)
    matrix_width = Problem(int)
    matrix_height = Problem(int)
    activation_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS)

    pe = Decision(int, domain=divisors_of(matrix_height))
    simd = Decision(int, domain=divisors_of(matrix_width))


def record_identity() -> None:
    root = Path(os.environ["FINN_ROOT"])
    finnlib = Path(os.environ["FINNLIB_ROOT"])
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    library = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=finnlib,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    library_dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=finnlib,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(f"FINN: {head} ({'dirty' if dirty else 'clean'})")
    print(f"FinnLib: {library} ({'dirty' if library_dirty else 'clean'})")


def _configure(case: Case) -> ReplayBufferKernel:
    harness = _compile_space(Harness, "fixture", problem_namespace="problem.replay")
    kernel = _compile_space(
        ReplayBufferKernel,
        "fixture.replay_buffer",
        {name: cast("_Ref[object]", harness.member(name)) for name in BOUND_NAMES},
        _allow_problem=False,
    )
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, kernel.spec))),
        {
            "problem.replay.repetitions": case.repetitions,
            "problem.replay.matrix_width": case.matrix_width,
            "problem.replay.matrix_height": case.matrix_height,
            "problem.replay.activation_type": DataType[case.activation],
        },
    )
    point = engine.commit_assignments(
        point, {"fixture.pe": case.pe, "fixture.simd": case.simd}
    ).point
    answer = configure_kernel(engine, kernel, point)
    if not isinstance(answer, Decided):
        raise AssertionError(f"{case.label} did not configure: {answer.findings}")
    return cast(ReplayBufferKernel, answer.value)


def _encode(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def _pack(
    beats: tuple[tuple[tuple[int, ...], ...], ...],
    values: np.ndarray,
    width: int,
) -> list[int]:
    """Pack one Region beat sequence into RTL words, lane 0 in the low bits."""

    packed = []
    for beat in beats:
        word = 0
        for lane, position in enumerate(beat):
            word |= _encode(int(values[position]), width) << (lane * width)
        packed.append(word)
    return packed


def run_one(case: Case) -> int:
    kernel = _configure(case)
    region = kernel.resolved_region
    datatype = DataType[case.activation]
    width = datatype.bitwidth()
    generator = np.random.RandomState(zlib.crc32(case.label.encode()) % (2**31))
    activations = generator.randint(
        int(datatype.min()), int(datatype.max()) + 1, size=(case.repetitions, case.matrix_width)
    )

    stimulus = _pack(
        region.input_interface("activation_in").port.beat_sequence.beats, activations, width
    )
    expected = _pack(
        region.output_interface("activation_out").port.beat_sequence.beats, activations, width
    )
    if len(stimulus) != case.repetitions * case.synapse_folds:
        print(f"{case.label}: compact beat count disagrees with the folding")
        return FAIL
    if len(expected) != case.repetitions * case.neuron_folds * case.synapse_folds:
        print(f"{case.label}: expanded beat count disagrees with the folding")
        return FAIL

    top = f"replay_kernel_{case.label}"
    finnlib = Path(os.environ["FINNLIB_ROOT"])
    sources = [str(finnlib / path) for path in FINNLIB_SOURCES]
    ok = True
    for stalls in (False, True):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory) / f"{top}.sv"
            wrapper.write_text(_wrapper(top, dict(kernel.parameters)))
            measured = drive(
                top,
                [*sources, str(wrapper)],
                {"in0": stimulus},
                len(expected),
                stalls=stalls,
            )
        match = measured == expected
        print(f"{case.label} {'stalled' if stalls else 'free'}: {'PASS' if match else 'FAIL'}")
        if not match:
            print("expected", expected)
            print("measured", measured)
        ok &= match
    return PASS if ok else FAIL


def _wrapper(name: str, parameters: dict[str, bool | int | float | str]) -> str:
    width = int(cast(int, parameters["W"]))
    parameter_text = ",\n        ".join(f".{key}({parameters[key]})" for key in sorted(parameters))
    return f"""module {name}(
    input logic ap_clk, input logic ap_rst_n,
    input logic [{width - 1}:0] in0_V_tdata,
    input logic in0_V_tvalid, output logic in0_V_tready,
    output logic [{width - 1}:0] out0_V_tdata,
    output logic out0_V_tvalid, input logic out0_V_tready
);
    replay_buffer #(
        {parameter_text}
    ) dut (
        .clk(ap_clk), .rst(!ap_rst_n),
        .idat(in0_V_tdata), .ivld(in0_V_tvalid), .irdy(in0_V_tready),
        .odat(out0_V_tdata), .olast(), .ofin(),
        .ovld(out0_V_tvalid), .ordy(out0_V_tready)
    );
endmodule
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES_BY_LABEL))
    arguments = parser.parse_args()
    record_identity()
    selected = CASES if arguments.case is None else (CASES_BY_LABEL[arguments.case],)
    results = tuple(run_one(case) for case in selected)
    ok = all(result == PASS for result in results)
    print(f"RESULT: {'PASS' if ok else 'FAIL'} ({len(selected)} cases, two modes each)")
    return PASS if ok else FAIL


if __name__ == "__main__":
    raise SystemExit(main())
