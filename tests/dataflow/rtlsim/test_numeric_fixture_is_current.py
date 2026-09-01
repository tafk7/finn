# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Everything fixture 8 does up to the point where Vivado is needed.

The same reason ``test_fixture_is_current`` exists for fixture 5: a harness
that only runs behind ``xelab`` rots between runs, and a rename once left that
one importing a symbol that had not existed for two commits.

There is a second reason here.  Fixture 8 is the first thing in this migration
that has to *interpret* a beat -- fixture 5 fed two DUTs the same opaque words
and compared them, so no ordering it used could be wrong.  Reading a matrix
into streams and back is arithmetic of its own, and a packing error is
indistinguishable from an RTL error when all you have is a mismatch.

So the packing is checked here, against a Python statement of what the core
does.  That is deliberately *not* a second oracle for the RTL: it proves that
pack and unpack are inverse under the documented lane layout, which is the part
that lives in the fixture.  Whether the RTL implements that layout is what the
container run answers, and nothing here can.
"""

from __future__ import annotations

import zlib

import numpy as np  # type: ignore[import-not-found]
import pytest

from dataflow.rtlsim import composed_mvau_numeric as fixture
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.ops.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.ops.mvau.lane_packing import a_datapath_width, pack_lanes


def _stimulus(case: fixture.Case) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.RandomState(zlib.crc32(case.label.encode()) % (2**31))
    return fixture._weights(case, generator), fixture._activations(case, generator)


@pytest.mark.parametrize("case", fixture.CASES, ids=lambda item: item.label)
def test_every_case_builds_what_it_will_simulate(case: fixture.Case) -> None:
    """The whole compiler path, short of touching the filesystem for FinnLib."""

    weights, _activations = _stimulus(case)
    built = fixture.requirements_for(case, fixture._model(case, weights))

    design = MVAU_DESIGN_INVENTORY.inventory.declaration(DotProductDesign.id)
    declared = {
        item.name
        for placement in design.placements
        if placement.name in {"compute", "replay"}
        for kernel in placement.candidates
        for item in kernel.parameters
    }
    assert {name for name, _ in built.parameters} == declared
    assert built.target_fpga_part == case.fpga_part

    values = dict(built.parameters)
    assert values["PE"] == case.pe
    assert values["SIMD"] == case.simd
    assert values["LEN"] == case.synapse_folds
    assert values["REP"] == case.neuron_folds


@pytest.mark.parametrize("case", fixture.CASES, ids=lambda item: item.label)
def test_the_golden_is_the_operation_and_fits_the_accumulator(case: fixture.Case) -> None:
    """A wrapped golden would measure overflow while claiming to measure arithmetic.

    Checked in the fixture too, where it aborts the case; checked here because
    a case added with too narrow an accumulator should fail in the ordinary
    suite rather than after a container start and an ``xelab``.
    """

    weights, activations = _stimulus(case)
    model = fixture._model(case, weights)
    expectation = fixture.golden(model, activations, weights)

    assert expectation.shape == (case.repetitions, case.matrix_height)
    width = int(dict(fixture.requirements_for(case, model).parameters)["ACCU_WIDTH"])  # type: ignore[arg-type]
    limit = 1 << (width - 1)
    assert -limit <= expectation.min() and expectation.max() < limit


def _core(
    case: fixture.Case,
    weights: list[int],
    activations: list[int],
    widths: dict,
    *,
    signed_activations: bool,
) -> list[int]:
    """What ``dotp_axi`` is documented to do with the beats it is given.

    A statement of the *lane layout*, not a model of the core: it multiplies and
    accumulates in the order the packed-array declarations imply, so that pack
    followed by this followed by unpack must reproduce the matrix product.  If
    it did not, the fixture's own arithmetic would be wrong before the RTL ever
    ran.

    ``signed_activations`` is a parameter and not an assumption because that is
    exactly what ``SIGNED_ACTIVATIONS`` is: the bus carries the same bits either
    way and the dial says how to read them.  Writing it in as always-signed made
    the ``UINT8`` cases fail here before a simulator was ever involved, which is
    the split this module exists to make.
    """

    activation_width, weight_width, accumulator_width = (
        widths["ACTIVATION_WIDTH"],
        widths["WEIGHT_WIDTH"],
        widths["ACCU_WIDTH"],
    )
    output = []
    index = 0
    for repetition in range(case.repetitions):
        for _neuron in range(case.neuron_folds):
            lanes = [0] * case.pe
            for synapse in range(case.synapse_folds):
                weight_beat = weights[index]
                index += 1
                # The replay: the same activation beat is reused for every
                # neuron fold, which is why the index is not ``index``.
                activation_beat = activations[repetition * case.synapse_folds + synapse]
                for lane in range(case.pe):
                    for element in range(case.simd):
                        offset = (lane * case.simd + element) * weight_width
                        weight = fixture._decode(
                            (weight_beat >> offset) & ((1 << weight_width) - 1), weight_width
                        )
                        raw = (activation_beat >> (element * activation_width)) & (
                            (1 << activation_width) - 1
                        )
                        activation = (
                            fixture._decode(raw, activation_width) if signed_activations else raw
                        )
                        lanes[lane] += weight * activation
            beat = 0
            for lane in range(case.pe):
                beat |= fixture._encode(lanes[lane], accumulator_width) << (
                    lane * accumulator_width
                )
            output.append(beat)
    return output


@pytest.mark.parametrize("case", fixture.CASES, ids=lambda item: item.label)
def test_packing_and_unpacking_reproduce_the_matrix_product(case: fixture.Case) -> None:
    """The fixture's own arithmetic, checked before a simulator is involved.

    A failure here is a bug in this fixture.  A failure in the container with
    this passing is a bug in the RTL or in the layout the comment above
    ``activation_beats`` reads out of it -- which is the distinction that makes
    a mismatch actionable instead of a mystery.
    """

    weights, activations = _stimulus(case)
    model = fixture._model(case, weights)
    values = dict(fixture.requirements_for(case, model).parameters)
    widths = {
        name: int(values[name]) for name in ("ACTIVATION_WIDTH", "WEIGHT_WIDTH", "ACCU_WIDTH")
    }  # type: ignore[arg-type]

    beats = _core(
        case,
        fixture.weight_beats(case, weights, widths["WEIGHT_WIDTH"]),
        fixture.activation_beats(case, activations, widths["ACTIVATION_WIDTH"]),
        widths,
        signed_activations=bool(values["SIGNED_ACTIVATIONS"]),
    )
    assert len(beats) == case.repetitions * case.neuron_folds
    measured = fixture.unpack_output(case, beats, widths["ACCU_WIDTH"])
    expectation = fixture.golden(model, activations, weights).astype(np.int64)
    assert measured.tolist() == expectation.tolist()


def test_the_identity_case_needs_no_arithmetic_to_read() -> None:
    """The harness-trust case is only useful if its answer is obvious.

    With an identity weight matrix the output is the activation, so a reader
    can check the expected value by eye.  If a change ever made this case
    non-trivial it would stop serving its purpose silently.
    """

    case = fixture.CASES_BY_LABEL["identity"]
    weights, activations = _stimulus(case)
    assert np.array_equal(weights, np.eye(case.matrix_width, dtype=np.float32))
    assert len(set(activations.flatten().tolist())) == activations.size
    model = fixture._model(case, weights)
    assert np.array_equal(fixture.golden(model, activations, weights), activations)


def test_every_case_says_what_it_discriminates() -> None:
    """A case with no stated reason is a case nobody can decide to remove."""

    for case in fixture.CASES:
        assert case.why, case.label
        assert case.activation_values in fixture.STIMULUS_KINDS, case.label
        assert case.weight_values in fixture.STIMULUS_KINDS, case.label
    assert len({case.label for case in fixture.CASES}) == len(fixture.CASES)


# -- the matrix moves the dials it claims to ------------------------------------


def _dials(case: fixture.Case) -> dict[str, object]:
    weights, _activations = _stimulus(case)
    return dict(fixture.requirements_for(case, fixture._model(case, weights)).parameters)


def test_the_unsigned_cases_actually_send_something_above_the_signed_maximum() -> None:
    """``SIGNED_ACTIVATIONS`` off is only tested by a value that needs it.

    A ``UINT8`` sample confined to 0..127 is indistinguishable from an ``INT8``
    one, so the case would pass while measuring nothing.  ``extremes`` plants
    the maximum for exactly this reason; here is the assertion that says so.
    """

    for label in ("unsigned_activations", "unsigned_activations_softvec"):
        case = fixture.CASES_BY_LABEL[label]
        _weights, activations = _stimulus(case)
        assert activations.max() > 127, label
        assert _dials(case)["SIGNED_ACTIVATIONS"] is False, label


def test_the_signed_cases_still_declare_signed_activations() -> None:
    """The other side of the same dial, so "it is always false" cannot pass."""

    for case in fixture.CASES:
        if case.activation.startswith("UINT"):
            continue
        assert _dials(case)["SIGNED_ACTIVATIONS"] is True, case.label


def test_the_narrow_and_wide_weight_cases_are_a_pair() -> None:
    """Both weight paths, and they are different paths.

    ``minimum_signed_weight`` contains -128 so the packing cannot assume a
    narrow range; ``narrow_weights`` excludes it so it can.  Same geometry,
    same types, one bit apart in the parameter table -- which is what makes the
    pair evidence about the rule rather than about two unrelated runs.
    """

    wide = fixture.CASES_BY_LABEL["minimum_signed_weight"]
    narrow = fixture.CASES_BY_LABEL["narrow_weights"]
    wide_weights, _ = _stimulus(wide)
    narrow_weights, _ = _stimulus(narrow)
    assert wide_weights.min() == -128
    assert narrow_weights.min() > -128

    assert _dials(wide)["NARROW_WEIGHTS"] is False
    assert _dials(narrow)["NARROW_WEIGHTS"] is True
    differing = {name for name, value in _dials(wide).items() if _dials(narrow).get(name) != value}
    assert differing == {"NARROW_WEIGHTS"}


def test_the_all_negative_case_is_all_negative() -> None:
    """Otherwise it is the random case with a different name.

    Every operand negative and therefore every product positive: a datapath
    that read the operands as unsigned would agree with a correct one on
    non-negative inputs and cannot agree here.
    """

    case = fixture.CASES_BY_LABEL["all_negative"]
    weights, activations = _stimulus(case)
    assert weights.max() < 0
    assert activations.max() < 0


def test_every_dsp_generation_is_represented_in_the_matrix() -> None:
    """All three, including the one that had never been run.

    ``VERSION`` 1, 2 and 3 are DSP48E1, DSP48E2 and DSP58.  A matrix that ran
    only two of them would say nothing about the third, and DSP48E1 is the
    third: it was in the part table from the first fixture and in no
    configuration until Phase 6e.
    """

    versions = {int(_dials(case)["VERSION"]) for case in fixture.CASES}  # type: ignore[arg-type]
    assert versions == {1, 2, 3}


def test_the_dsp48e1_pair_differs_only_in_the_narrow_promise() -> None:
    """Phase 4's correction, set up so that a failure names what is wrong.

    The pre-Phase-4 rule was "DSP48E1 requires the narrow-weight promise", so
    ``dsp48e1_minimum_weight`` is the configuration it refused.  Both are
    covered now; if the RTL disagrees, the pair says whether the family or the
    promise is the reason.
    """

    wide = fixture.CASES_BY_LABEL["dsp48e1_minimum_weight"]
    narrow = fixture.CASES_BY_LABEL["dsp48e1_narrow"]
    assert _dials(wide)["NARROW_WEIGHTS"] is False
    assert _dials(narrow)["NARROW_WEIGHTS"] is True
    assert _dials(wide)["VERSION"] == 1
    differing = {name for name, value in _dials(wide).items() if _dials(narrow).get(name) != value}
    assert differing == {"NARROW_WEIGHTS"}


def test_the_frame_boundary_pair_differs_only_in_the_dsp_generation() -> None:
    """The diagnostic pair, set up so its answer means something.

    Built when fixture 5 failed on DSP48E1 from the second repetition while
    passing on every other generation.  Fixture 5 compares two DUTs and cannot
    say which is right; these ask arithmetic the same question on both
    generations, so a DSP48E1 failure with DSP58 passing would point at the
    family and both failing would point at the frame boundary.  Both passed,
    which is what sent the search to fixture 5's stimulus.  Anything else about
    them differing would make the comparison say nothing.
    """

    first = fixture.CASES_BY_LABEL["dsp48e1_frames"]
    second = fixture.CASES_BY_LABEL["dsp58_frames"]
    for field in (
        "repetitions",
        "matrix_width",
        "matrix_height",
        "pe",
        "simd",
        "activation",
        "weight",
        "accumulator",
        "activation_values",
        "weight_values",
    ):
        assert getattr(first, field) == getattr(second, field), field
    assert first.target is not second.target
    assert first.repetitions >= 3, "a defect at the second frame should show at the third too"


def test_the_non_narrow_dsp48e1_weights_really_do_pack() -> None:
    """The lane arithmetic, so a coverage refusal is a finding and not a skip.

    ``pack_lanes`` is the model of ``sliceLanes()`` Phase 4 derived the rule
    from.  It is *not* the fixture's expectation -- that comes from
    ``execute_node`` -- but it is what says this case is meant to be buildable,
    so a coverage refusal here would be the rule contradicting itself before
    any tool ran.
    """

    packing = pack_lanes(
        a_width=a_datapath_width(1), weight_width=8, activation_width=8, narrow_weights=False
    )
    assert packing.fits, packing
    assert packing.lanes >= 2
