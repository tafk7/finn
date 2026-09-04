# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The old three-valued profile against the new pair, in all six combinations.

Not the three that happen to correspond.  The oracle's enum had three values and
its *execution* distinguished six cases, so three combinations were reachable at
run time with no profile value to name them:

* a bipolar popcount, which its profile called ``accumulator_integer``;
* an XNOR popcount followed by a threshold, and
* a bipolar popcount followed by a threshold, both of which its profile called
  ``fused_threshold``.

The middle one is the defect this correction was made for: collapsing the two
axes into one exclusive enum made "popcount **and** threshold" unrepresentable,
and the value that won was the threshold -- so the node computed a plain matrix
product and thresholded that, silently.

Parity is therefore against the oracle's **execution**, which was right in all
six cases, rather than against its profile string, which was not.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataflow.parity.correspondence import _PROFILE_CORRESPONDENCE, _enum_value
from dataflow.parity.fixtures import (
    BUILD,
    PROFILE_FIXTURES,
    PROFILE_ROWS,
    activation_values,
    by_name,
)
from dataflow.parity.local import bound, execute
from dataflow.parity.oracle import oracle_report

SPECS = tuple({**item, "activation": activation_values(item)} for item in PROFILE_FIXTURES)


def _report() -> dict[str, Any]:
    return oracle_report(SPECS, BUILD)


def _spec(name: str) -> dict[str, Any]:
    return {**by_name(name), "activation": activation_values(by_name(name))}


@pytest.mark.parametrize(
    ("label", "old_value", "accumulation", "activation", "fixture_name"),
    PROFILE_ROWS,
    ids=[row[0] for row in PROFILE_ROWS],
)
def test_each_of_the_six_combinations_maps_and_executes_alike(
    label: str,
    old_value: str | None,
    accumulation: str,
    activation: str,
    fixture_name: str,
) -> None:
    del label
    spec = _spec(fixture_name)
    _model, occurrence = bound(spec, BUILD)
    profile = occurrence.profile
    assert profile.accumulation.name == accumulation
    assert profile.activation.name == activation

    entry = dict(_report()["fixtures"][fixture_name])
    oracle_profile = _enum_value(entry["derived"]["computation_profile"])
    if old_value is None:
        # The row the oracle's enum could not name at all: it reported the
        # integer value while executing a popcount.
        assert oracle_profile == "accumulator_integer"
        assert profile.accumulation.name == "BIPOLAR_POPCOUNT"
    else:
        from finn.dataflow.ops.mvau.computation import AccumulationMode, ActivationMode  # noqa: PLC0415

        assert oracle_profile == old_value.lower()
        expected = _PROFILE_CORRESPONDENCE[oracle_profile]
        pair = (
            AccumulationMode[accumulation].value,
            ActivationMode[activation].value,
        )
        assert pair in expected, (
            f"{oracle_profile} does not correspond to {pair}; the mapping table and the "
            "fixtures disagree"
        )

    # The claim that matters: the same numbers, computed by two stacks that
    # named this case differently.
    assert entry["execution"] is not None, "the fixture supplies a matrix to execute"
    assert execute(spec) == entry["execution"]


def test_the_two_collapsed_rows_are_present_and_are_not_the_integer_path() -> None:
    """The rows the old enum could not represent, named explicitly.

    Without this, the six-row table could quietly become a five-row one and
    every remaining row would still pass.
    """

    collapsed = {
        row[4] for row in PROFILE_ROWS if row[3] == "MULTITHRESHOLD" and row[2] != "INTEGER"
    }
    assert collapsed == {"xnor_fused_threshold", "bipolar_fused_threshold"}

    import numpy  # noqa: PLC0415
    import qonnx.custom_op.general.xnorpopcount as xnor  # noqa: PLC0415
    from qonnx.custom_op.general.multithreshold import multithreshold  # noqa: PLC0415

    for name in sorted(collapsed):
        spec = _spec(name)
        _model, occurrence = bound(spec, BUILD)
        assert occurrence.profile.accumulation.name != "INTEGER"
        assert occurrence.profile.fuses_activation

        # The reference, written out rather than inferred: a popcount, and then
        # a threshold over *that*.  The collapsed enum produced a plain matrix
        # product here, so the two accumulations are compared as well -- if they
        # agreed on these operands the fixture would not be testing anything.
        activation = numpy.array(spec["activation"], dtype=numpy.float32)
        weight = numpy.array(spec["weights"], dtype=numpy.float32)
        if spec["binary_xnor"]:
            popcount = xnor.xnorpopcountmatmul(activation, weight)
        else:
            popcount = xnor.xnorpopcountmatmul((activation + 1) / 2, (weight + 1) / 2)
        plain = numpy.matmul(activation, weight)
        assert not numpy.array_equal(popcount, plain), (
            f"{name}: the popcount and the plain product agree on these operands, so the "
            "fixture cannot show which one was used"
        )
        expected = multithreshold(
            popcount, numpy.array(spec["thresholds"], dtype=numpy.float32), 1, 0
        )
        assert execute(spec) == [list(map(float, row)) for row in expected.tolist()]


def test_every_old_value_is_covered_by_the_rows() -> None:
    named = {row[1] for row in PROFILE_ROWS if row[1] is not None}
    assert named == {
        "ACCUMULATOR_INTEGER",
        "BIPOLAR_XNOR_ACCUMULATOR",
        "FUSED_THRESHOLD",
    }
    assert len(PROFILE_ROWS) == 6
