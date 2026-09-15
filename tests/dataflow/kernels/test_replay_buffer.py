# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""KD6: ReplayBuffer is independently authored and matches the old authority."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow._engine import Decided, Engine, QualifiedPath
from finn.dataflow.artifacts.abi import ComponentABI, Reset, Signal
from finn.dataflow.artifacts.formats import _descriptor
from finn.dataflow.artifacts.rtl import Declined, check_abi
from finn.dataflow.artifacts.store import ArtifactStore
from finn.dataflow.space.dataflow_value_semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.space.compiler import _Ref, _compile_space
from finn.dataflow.space.declarations import Decision, Problem, Space, divisors_of
from finn.dataflow.kernels.kernel import ModuleBuildRequirements, kernel_physical, kernel_dataflow
from finn.dataflow.artifacts.build import (
    prepare_module_build,
    module_source_derivation,
    materialize_module_sources,
    portable_module_component,
)
from finn.dataflow.kernels.replay_buffer import (
    FINNLIB_ROOT,
    FINNLIB_SOURCES,
    ReplayBufferKernel,
)
from finn.dataflow.ops.mvau import regions as mvau_regions
from finn.dataflow.ops.mvau.regions import (
    construct_activation_replay_region as baseline_region,
)
from finn.dataflow.model.region_validation import validate_region
from finn.dataflow.space.spec_algebra import assemble_specs

BOUND_NAMES = (
    "repetitions",
    "matrix_width",
    "matrix_height",
    "activation_type",
    "pe",
    "simd",
)


class Harness(Space):
    """The Design's stand-in: it owns every Region-visible choice."""

    repetitions = Problem(int)
    matrix_width = Problem(int)
    matrix_height = Problem(int)
    activation_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS)

    pe = Decision(int, domain=divisors_of(matrix_height))
    simd = Decision(int, domain=divisors_of(matrix_width))


def _compile() -> tuple[object, object]:
    harness = _compile_space(Harness, "replay_test", problem_namespace="problem.replay")
    kernel = _compile_space(
        ReplayBufferKernel,
        "replay_test.kernel",
        {name: cast("_Ref[object]", harness.member(name)) for name in BOUND_NAMES},
        _allow_problem=False,
    )
    return harness, kernel


def _configure_values(
    *,
    repetitions: int = 2,
    matrix_width: int = 8,
    matrix_height: int = 4,
    activation: str = "INT8",
    pe: int = 2,
    simd: int = 2,
) -> tuple[ModuleBuildRequirements, object]:
    harness, kernel = _compile()
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, kernel.spec))),  # type: ignore[attr-defined]
        {
            "problem.replay.repetitions": repetitions,
            "problem.replay.matrix_width": matrix_width,
            "problem.replay.matrix_height": matrix_height,
            "problem.replay.activation_type": DataType[activation],
        },
    )
    point = engine.commit_assignments(point, {"replay_test.pe": pe, "replay_test.simd": simd}).point
    answer = kernel_physical(engine, kernel, point).accepted_answer  # type: ignore[arg-type]
    assert isinstance(answer, Decided), answer
    logical = kernel_dataflow(engine, kernel, point).accepted_answer
    assert isinstance(logical, Decided)
    return answer.value, logical.value


def _configure(**kwargs):
    return _configure_values(**kwargs)[0]


def _logical(**kwargs):
    return _configure_values(**kwargs)[1]


def _finnlib_root() -> Path:
    """Return the caller-selected FinnLib checkout, or FINN's pinned default."""

    override = os.environ.get("FINNLIB_ROOT")
    if override:
        return Path(override).resolve()
    return (Path(__file__).parents[3] / "deps/finnlib").resolve()


#: One and several repetitions, neuron folds, synapse folds; several legal
#: PE/SIMD divisors; signed and unsigned activations the RTL supports.
MATRIX = (
    (1, 4, 4, "INT8", 4, 4),
    (1, 4, 4, "INT8", 1, 1),
    (3, 8, 4, "INT8", 2, 2),
    (2, 8, 6, "INT8", 3, 4),
    (2, 12, 8, "INT4", 4, 3),
    (2, 8, 4, "UINT8", 2, 2),
    (1, 8, 8, "UINT4", 8, 8),
    (4, 6, 6, "INT16", 2, 3),
)


def _abi(requirements):
    abi = requirements.abi
    return ComponentABI(abi.entry_point.value, abi.ports, abi.parameters, abi.clock_alignments)


@pytest.mark.parametrize(
    ("repetitions", "matrix_width", "matrix_height", "activation", "pe", "simd"), MATRIX
)
def test_replay_kernel_binds_the_region_constructor_inputs(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation: str,
    pe: int,
    simd: int,
) -> None:
    expected = baseline_region(
        repetitions, matrix_width, matrix_height, DataType[activation], pe, simd
    )
    region = _logical(
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        activation=activation,
        pe=pe,
        simd=simd,
    )
    assert region == expected
    assert not validate_region(region).issues


@pytest.mark.parametrize(
    ("repetitions", "matrix_width", "matrix_height", "activation", "pe", "simd"), MATRIX
)
def test_replay_parameters_restate_the_folding(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation: str,
    pe: int,
    simd: int,
) -> None:
    configured = _configure(
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        activation=activation,
        pe=pe,
        simd=simd,
    )
    assert dict(configured.parameters) == {
        "LEN": matrix_width // simd,
        "REP": matrix_height // pe,
        "W": simd * DataType[activation].bitwidth(),
    }


def test_replay_owns_no_decision_at_all() -> None:
    _harness, kernel = _compile()
    assert {str(p) for p in kernel.extension.imported_decisions} == {
        "replay_test.pe",
        "replay_test.simd",
    }
    assert kernel.spec.decisions == ()  # type: ignore[attr-defined]


def test_replay_declares_one_region_family() -> None:
    assert (ReplayBufferKernel.region.family, ReplayBufferKernel.region.version) == (
        "mvau.activation_replay",
        "2",
    )


def test_replay_is_an_identity_at_one_neuron_fold_and_still_a_region() -> None:
    configured = _configure(matrix_height=4, pe=4)
    region = _logical(matrix_height=4, pe=4)
    inside = region.input_interface("activation_in").port.beat_sequence
    outside = region.output_interface("activation_out").port.beat_sequence
    assert inside == outside
    assert dict(configured.parameters)["REP"] == 1


def test_several_neuron_folds_multiply_the_output_beats() -> None:
    region = _logical(repetitions=2, matrix_width=8, matrix_height=6, pe=2, simd=2)
    compact = region.input_interface("activation_in").port.beat_sequence
    expanded = region.output_interface("activation_out").port.beat_sequence
    assert compact.beat_count == 2 * 4
    assert expanded.beat_count == 2 * 3 * 4
    assert compact.image_set.cardinality == 2 * 8
    assert expanded.image_set.cardinality == 2 * 3 * 8
    assert region.input("X").operand.shape == (2, 8)
    assert region.output_interface("activation_out").port.operand.shape == (6, 8)


def test_replay_sources_and_abi_are_exact() -> None:
    configured = _configure(simd=2, activation="INT8")
    assert {source.root for source in configured.contributions} == {FINNLIB_ROOT}
    assert tuple(source.path for source in configured.contributions) == FINNLIB_SOURCES
    assert configured.abi.entry_point.value == "replay_buffer"
    assert set(_abi(configured).physical_names()) == {
        "clk",
        "rst",
        "idat",
        "ivld",
        "irdy",
        "odat",
        "ovld",
        "ordy",
        "olast",
        "ofin",
    }
    widths = {
        member.physical: member.width
        for port in configured.abi.ports
        if hasattr(port, "signals")
        for member in port.signals  # type: ignore[union-attr]
    }
    assert widths["idat"] == 16
    assert widths["odat"] == 16
    assert configured.abi.parameters == (("LEN", "4"), ("REP", "2"), ("W", "16"))


@pytest.mark.parametrize(("matrix_height", "pe"), ((4, 4), (4, 2)))
def test_replay_reset_is_synchronous_in_identity_and_buffered_descriptors(
    matrix_height: int, pe: int
) -> None:
    configured = _configure(matrix_height=matrix_height, pe=pe)
    reset = next(
        port for port in configured.abi.ports if isinstance(port, Signal) and port.name == "rst"
    )
    assert reset.role == Reset(active_low=False, synchronous=True, synchronous_to=("clk",))
    encoded = _descriptor.encode(_abi(configured))
    assert b'"synchronous":true' in encoded
    assert _descriptor.decode(encoded) == _abi(configured)


def test_replay_abi_agrees_with_the_selected_finnlib_rtl() -> None:
    configured = _configure()
    root = _finnlib_root()
    sources = tuple(root / path for path in FINNLIB_SOURCES)
    if any(not path.is_file() for path in sources):
        pytest.skip("FinnLib is not fetched; set FINNLIB_ROOT or run fetch-repos.sh")
    result = check_abi(
        _abi(configured),
        sources,
        "replay_buffer",
        configured.abi.parameters,
    )
    assert not isinstance(result, Declined)
    assert result == ()


def test_replay_refuses_folding_it_cannot_realize() -> None:
    class LooseHarness(Space):
        repetitions = Problem(int)
        matrix_width = Problem(int)
        matrix_height = Problem(int)
        activation_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS)
        pe = Decision(int, values=(3,))
        simd = Decision(int, values=(3,))

    harness = _compile_space(LooseHarness, "loose", problem_namespace="problem.loose")
    kernel = _compile_space(
        ReplayBufferKernel,
        "loose.kernel",
        {name: cast("_Ref[object]", harness.member(name)) for name in BOUND_NAMES},
        _allow_problem=False,
    )
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, kernel.spec))),
        {
            "problem.loose.repetitions": 1,
            "problem.loose.matrix_width": 8,
            "problem.loose.matrix_height": 4,
            "problem.loose.activation_type": DataType["INT8"],
        },
    )
    point = engine.commit_assignments(point, {"loose.pe": 3, "loose.simd": 3}).point
    answer = kernel_physical(engine, kernel, point).accepted_answer
    assert not isinstance(answer, Decided)
    assert "kernel-region-refused" in {finding.code for finding in answer.findings}
    assert QualifiedPath("semantic.loose.kernel.region") in {
        finding.path for finding in answer.findings
    }


def test_replay_source_closure_completes_and_round_trips_through_store(
    tmp_path: Path,
) -> None:
    root = _finnlib_root()
    if not (root / FINNLIB_SOURCES[0]).is_file():
        pytest.skip("FinnLib is not fetched; set FINNLIB_ROOT or run fetch-repos.sh")
    kernel = _configure()
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        kernel, roots={FINNLIB_ROOT: root}, blobs=store, template_roots=()
    )
    derivation = module_source_derivation(prepared)
    published = materialize_module_sources(prepared, store)
    found = store.lookup(derivation)
    assert found == published
    assert found.files == FINNLIB_SOURCES


def test_replay_packages_into_a_portable_component(tmp_path: Path) -> None:
    root = _finnlib_root()
    if not (root / FINNLIB_SOURCES[0]).is_file():
        pytest.skip("FinnLib is not fetched; set FINNLIB_ROOT or run fetch-repos.sh")
    kernel = _configure()
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        kernel, roots={FINNLIB_ROOT: root}, blobs=store, template_roots=()
    )
    component = portable_module_component(prepared, materialize_module_sources(prepared, store))
    assert component.entry_point == "replay_buffer"
    assert component.abi == _abi(kernel)
    assert tuple(path for path, _content in component.files) == FINNLIB_SOURCES
    del tmp_path


def test_the_replay_kernel_declares_the_one_semantic_constructor_authority() -> None:
    """The Kernel-local copy is gone; the declaration names `ops.mvau.regions`."""

    assert ReplayBufferKernel.region.construct is mvau_regions.construct_activation_replay_region
