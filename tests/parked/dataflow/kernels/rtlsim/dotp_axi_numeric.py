# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel-only numerical evidence for the declarative DotpAxi Kernel."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import cast

import numpy as np  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from kernels.rtlsim.rtl_transport import drive
from kernels.rtlsim.dotp_support import (
    Case,
    _activation_beats,
    _unpack,
    _weight_beats,
    _wrapper,
)
from finn.kernels._engine import Decided, Engine
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.space.compiler import _Ref, _compile_space
from finn.kernels.space.declarations import Decision, Problem, Space, divisors_of
from finn.parked.dataflow.kernels.dotp_axi import DotpAxiKernel, FINNLIB_SOURCES
from finn.parked.dataflow.kernels.matmul.base import DspBlock
from finn.parked.dataflow.model import kernel_physical
from finn.kernels.space.spec_algebra import assemble_specs

PASS, FAIL = 0, 1
CLOCK_PERIOD_NS = 4.0


CASES = (
    Case(
        "identity", DspBlock.DSP58, 1, 4, 4, 2, 2, "INT8", "INT8", "INT32", "counting", "identity"
    ),
    Case("signed_random", DspBlock.DSP58, 2, 8, 4, 2, 2, "INT8", "INT8", "INT32"),
    Case("signed_random_softvec", DspBlock.DSP48E2, 2, 8, 6, 3, 2, "INT8", "INT8", "INT32"),
    Case(
        "unsigned_activations", DspBlock.DSP58, 2, 8, 4, 2, 2, "UINT8", "INT8", "INT32", "extremes"
    ),
    Case(
        "unsigned_activations_softvec",
        DspBlock.DSP48E2,
        2,
        8,
        4,
        2,
        2,
        "UINT8",
        "INT8",
        "INT32",
        "extremes",
    ),
    Case(
        "all_negative",
        DspBlock.DSP58,
        2,
        8,
        4,
        2,
        2,
        "INT8",
        "INT8",
        "INT32",
        "negative",
        "negative",
    ),
    Case(
        "minimum_signed_weight",
        DspBlock.DSP58,
        2,
        8,
        4,
        2,
        2,
        "INT8",
        "INT8",
        "INT32",
        "full",
        "extremes",
    ),
    Case(
        "narrow_weights", DspBlock.DSP58, 2, 8, 4, 2, 2, "INT8", "INT8", "INT32", "full", "narrow"
    ),
    Case(
        "narrow_weights_softvec",
        DspBlock.DSP48E2,
        2,
        8,
        4,
        2,
        2,
        "INT8",
        "INT8",
        "INT32",
        "full",
        "narrow",
    ),
    Case("dsp48e1", DspBlock.DSP48E1, 2, 8, 4, 2, 2, "INT8", "INT8", "INT32"),
    Case(
        "dsp48e1_minimum_weight",
        DspBlock.DSP48E1,
        2,
        8,
        4,
        2,
        2,
        "INT8",
        "INT8",
        "INT32",
        "full",
        "extremes",
    ),
    Case(
        "dsp48e1_narrow", DspBlock.DSP48E1, 2, 8, 4, 2, 2, "INT8", "INT8", "INT32", "full", "narrow"
    ),
    Case(
        "dsp48e1_frames", DspBlock.DSP48E1, 3, 4, 4, 2, 2, "INT4", "INT8", "INT16", "full", "narrow"
    ),
    Case(
        "dsp48e1_fixture5_point",
        DspBlock.DSP48E1,
        2,
        8,
        4,
        2,
        2,
        "INT4",
        "INT8",
        "INT16",
        "full",
        "narrow",
    ),
    Case("dsp58_frames", DspBlock.DSP58, 3, 4, 4, 2, 2, "INT4", "INT8", "INT16", "full", "narrow"),
    Case("pumped", DspBlock.DSP58, 2, 8, 4, 2, 4, "INT8", "INT8", "INT32", pumping=True),
)
CASES_BY_LABEL = {case.label: case for case in CASES}


def record_identity() -> None:
    root = Path(os.environ["FINN_ROOT"])
    finnlib = Path(os.environ["FINNLIB_ROOT"])
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    library = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=finnlib,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(f"FINN: {head} ({'dirty' if dirty else 'clean'})")
    print(f"FinnLib: {library}")


class Harness(Space):
    repetitions = Problem(int)
    matrix_width = Problem(int)
    matrix_height = Problem(int)
    activation_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS)
    weight_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS)
    accumulator_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS)
    output_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS)
    narrow_weights = Problem(bool)
    target_dsp = Problem(DspBlock)
    clock_period_ns = Problem(float)

    # PE and SIMD are Region-visible, so a Kernel owns them; this fixture is
    # the Kernel's stand-in for a Kernel-only run.
    pe = Decision(int, domain=divisors_of(matrix_height))
    simd = Decision(int, domain=divisors_of(matrix_width))


def _filled(
    kind: str, datatype: object, shape: tuple[int, ...], rng: np.random.RandomState
) -> np.ndarray:
    minimum, maximum = int(datatype.min()), int(datatype.max())  # type: ignore[attr-defined]
    if kind == "identity":
        return np.eye(shape[0], shape[1], dtype=np.float32)
    if kind == "counting":
        return (np.arange(int(np.prod(shape))) + 1).reshape(shape).astype(np.float32)
    if kind == "negative":
        values = rng.randint(minimum, 0, size=shape)
    elif kind == "narrow":
        values = rng.randint(minimum + 1, maximum + 1, size=shape)
    else:
        values = rng.randint(minimum, maximum + 1, size=shape)
    if kind == "extremes":
        flat = values.reshape(-1)
        flat[0], flat[-1] = minimum, maximum
    return values.astype(np.float32)


def _configure(case: Case, weights: np.ndarray):
    root = _compile_space(Harness, "fixture", problem_namespace="problem.dotp")
    kernel = _compile_space(
        DotpAxiKernel,
        "fixture.dotp_axi",
        {
            name: cast("_Ref[object]", root.member(name))
            for name in (
                "repetitions",
                "matrix_width",
                "matrix_height",
                "activation_type",
                "weight_type",
                "accumulator_type",
                "output_type",
                "narrow_weights",
                "target_dsp",
                "clock_period_ns",
                "pe",
                "simd",
            )
        },
        _allow_problem=False,
    )
    weight_type = DataType[case.weight]
    narrow = not bool(np.any(weights == int(weight_type.min())))
    problem = {
        "problem.dotp.repetitions": case.repetitions,
        "problem.dotp.matrix_width": case.matrix_width,
        "problem.dotp.matrix_height": case.matrix_height,
        "problem.dotp.activation_type": DataType[case.activation],
        "problem.dotp.weight_type": weight_type,
        "problem.dotp.accumulator_type": DataType[case.accumulator],
        "problem.dotp.output_type": DataType[case.accumulator],
        "problem.dotp.narrow_weights": narrow,
        "problem.dotp.target_dsp": case.target,
        "problem.dotp.clock_period_ns": CLOCK_PERIOD_NS,
    }
    engine = Engine()
    point = engine.start(engine.validate(assemble_specs((root.spec, kernel.spec))), problem)
    point = engine.commit_assignments(
        point,
        {
            "fixture.pe": case.pe,
            "fixture.simd": case.simd,
            "fixture.dotp_axi.compute_pumping": case.pumping,
        },
    ).point
    answer = kernel_physical(engine, kernel, point).accepted_answer
    if not isinstance(answer, Decided):
        raise AssertionError(f"{case.label} did not configure: {answer.findings}")
    return answer.value


def run_one(case: Case) -> int:
    rng = np.random.RandomState(zlib.crc32(case.label.encode()) % (2**31))
    activations = _filled(
        case.activation_values,
        DataType[case.activation],
        (case.repetitions, case.matrix_width),
        rng,
    )
    weights = _filled(
        case.weight_values,
        DataType[case.weight],
        (case.matrix_width, case.matrix_height),
        rng,
    )
    expected = activations.astype(np.int64) @ weights.astype(np.int64)
    kernel = _configure(case, weights)
    parameters = dict(kernel.parameters)
    accumulator_width = DataType[case.accumulator].bitwidth()
    activation_width = DataType[case.activation].bitwidth()
    weight_width = DataType[case.weight].bitwidth()
    limit = 1 << (accumulator_width - 1)
    if not (-limit <= expected.min() and expected.max() < limit):
        print(f"{case.label}: golden exceeds accumulator range")
        return FAIL

    top = f"dotp_kernel_{case.label}"
    finnlib = Path(os.environ["FINNLIB_ROOT"])
    sources = [str(finnlib / path) for path in FINNLIB_SOURCES]
    stimulus = {
        "in0": _activation_beats(case, activations, activation_width),
        "in1": _weight_beats(case, weights, weight_width),
    }
    expected_beats = case.repetitions * case.neuron_folds
    ok = True
    for stalls in (False, True):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory) / f"{top}.sv"
            wrapper.write_text(_wrapper(top, case, parameters))
            measured_beats = drive(
                top,
                [*sources, str(wrapper)],
                stimulus,
                expected_beats,
                stalls=stalls,
            )
        measured = _unpack(case, measured_beats, accumulator_width)
        match = np.array_equal(measured, expected)
        print(f"{case.label} {'stalled' if stalls else 'free'}: {'PASS' if match else 'FAIL'}")
        if not match:
            print("expected", expected.tolist())
            print("measured", measured.tolist())
        ok &= match
    return PASS if ok else FAIL


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
