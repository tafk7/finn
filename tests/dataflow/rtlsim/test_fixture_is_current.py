# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Everything fixture 5 does up to the point where Vivado is needed.

The RTL comparison itself only runs inside the container behind ``xelab``, so
nothing in the ordinary suite touched this harness -- and a pool rename left it
importing a symbol that no longer existed for two commits without anything
noticing.

But the harness now drives the *production* path: it resolves a real
``MvauDataflowOp``, elaborates through the declared provider, and takes the
Verilog and the source manifest from the compiler.  All of that is plain
Python, so all of it can be a normal test.  What is left for the container is
only ``xelab`` and the bit comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
import pytest

from dataflow.rtlsim import composed_mvau_equiv as fixture
from dataflow.rtlsim.rtl_transport import random_word
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.ops.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.ops.mvau.hardware.binding import verify_manifest
from finn.dataflow.ops.mvau.hardware.dotp_axi import FINNLIB_SOURCES


@pytest.mark.parametrize("config", fixture.CONFIGS, ids=lambda item: item.label)
def test_every_configuration_builds_what_it_will_simulate(config: fixture.Config) -> None:
    """The whole compiler path, short of touching the filesystem for FinnLib."""

    built = fixture.decomposed_requirements(config)

    design = MVAU_DESIGN_INVENTORY.inventory.declaration(DotProductDesign.id)
    declared = {
        item.name
        for placement in design.placements
        if placement.name in {"compute", "replay"}
        for kernel in placement.candidates
        for item in kernel.parameters
    }
    assert {name for name, _ in built.parameters} == declared

    # The generated top instantiates both cores with the declared values, and
    # the geometry never appears: no MW, no MH.
    assert f"module {built.top_module_name}" in built.wrapper_source
    assert "replay_buffer #(" in built.wrapper_source
    assert "dotp_axi #(" in built.wrapper_source
    assert ".MW(" not in built.wrapper_source and ".MH(" not in built.wrapper_source
    for name, value in built.parameters:
        assert f".{name}({int(value) if isinstance(value, bool) else value})" in (
            built.wrapper_source
        ), name

    assert built.target_fpga_part == config.fpga_part
    assert len(built.finnlib_sources) == len(FINNLIB_SOURCES)


def test_the_manifest_resolves_against_this_checkout() -> None:
    """Skipped where FinnLib is not fetched; the harness needs it, this does not.

    Naming the sources is a statement about the design and is checked above.
    Whether they are *present* depends on ``fetch-repos.sh`` having run, which
    a plain unit-test environment has no reason to have done.
    """

    built = fixture.decomposed_requirements(fixture.CONFIGS[0])
    if any(not Path(path).is_file() for path in built.finnlib_sources):
        pytest.skip("FinnLib is not fetched; set FINNLIB_ROOT or run fetch-repos.sh")
    verify_manifest(built.source_dependencies)


def test_the_folding_reaches_the_rtl_as_the_point_decided_it() -> None:
    config = fixture.CONFIGS_BY_LABEL["repetitions_softvec"]
    values = fixture.declared_parameters(config)

    assert values["PE"] == config.pe
    assert values["SIMD"] == config.simd
    # The replay buffer's three parameters are the folding restated: it is told
    # what to feed, it does not decide it.
    assert values["LEN"] == config.synapse_folds
    assert values["REP"] == config.neuron_folds
    assert values["W"] == config.simd * config.activation_bits


def test_the_target_family_selects_the_core_version() -> None:
    """DSP48E2 is VERSION 2, DSP58 is VERSION 3, read from the point."""

    assert fixture.declared_parameters(fixture.CONFIGS_BY_LABEL["softvec"])["VERSION"] == 2
    assert fixture.declared_parameters(fixture.CONFIGS_BY_LABEL["packed"])["VERSION"] == 3


def test_the_synthesis_path_is_the_one_that_is_built() -> None:
    """``FORCE_BEHAVIORAL`` is a debug aid and must never be on by default."""

    for config in fixture.CONFIGS:
        assert fixture.declared_parameters(config)["FORCE_BEHAVIORAL"] == 0


# -- the stimulus keeps the promises the design point makes --------------------


def test_the_weight_stimulus_honours_the_declared_narrow_weights() -> None:
    """The defect DSP48E1 exposed, as a check rather than a comment.

    ``NARROW_WEIGHTS`` is derived from the model's weight initializer and tells
    the core the datatype minimum will never arrive.  The streamed weights were
    an independent random draw that could carry it anyway, so the point made a
    promise the stimulus did not keep -- and ``mvu.sv``, which packs only
    ``WEIGHT_WIDTH-1`` magnitude bits under that promise, diverged from a core
    that tolerates it.  Two DUTs differing on a value neither is obliged to
    handle is not a result about either.
    """

    for config in fixture.CONFIGS:
        narrow = bool(fixture.declared_parameters(config)["NARROW_WEIGHTS"])
        assert narrow, f"{config.label}: this check assumes the narrow promise"
        generator = np.random.RandomState(0)
        minimum = -(2 ** (config.weight_bits - 1))
        mask = (1 << config.weight_bits) - 1
        for _ in range(64):
            beat = fixture._random_weight_beat(generator, config, narrow=narrow)
            for lane in range(config.pe * config.simd):
                raw = (beat >> (lane * config.weight_bits)) & mask
                value = raw - (1 << config.weight_bits) if raw > (mask >> 1) else raw
                assert value != minimum, config.label


def test_a_random_word_is_random_in_every_byte() -> None:
    """``bytes()`` on an ``int64`` numpy array reads its buffer, not its values.

    Seven bytes in eight were zero, so a 32-bit weight beat was one random low
    byte and 24 zero bits -- three of four lanes stuck at zero. Nothing failed,
    because both DUTs were fed the same impoverished stream, which is exactly
    why a fixture that only compares two implementations cannot notice.
    """

    generator = np.random.RandomState(0)
    words = [random_word(generator, 64) for _ in range(64)]
    for byte in range(8):
        seen = {(word >> (byte * 8)) & 0xFF for word in words}
        assert len(seen) > 1, f"byte {byte} never varies"
