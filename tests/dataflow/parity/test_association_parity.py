# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""L6: the retired source description, field by field.

``MVAUSourceDescription`` is **retired**, not reimplemented.  It was a wrapper
holding eight facts that the source reading and ``SourceAssociation`` now own
between them, and restoring it for structural parity would be restoring a
container for the sake of the comparison.

So the claim here is the harder one: every field it carried is still produced,
and produces the same value.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataflow.parity.correspondence import (
    ADDITIONS,
    DESCRIPTION_TABLE,
    MARKS,
    PROBLEM_TABLE,
    comparable,
    delegated,
    encode,
)
from dataflow.parity.fixtures import BUILD, SOURCE_FIXTURES, activation_values
from dataflow.parity.local import bound
from dataflow.parity.oracle import oracle_report

SPECS = tuple({**item, "activation": activation_values(item)} for item in SOURCE_FIXTURES)


def _report() -> dict[str, Any]:
    return oracle_report(SPECS, BUILD)


def test_the_table_covers_every_description_field() -> None:
    declared = set(_report()["fields"]["MVAUSourceDescription"])
    covered = {item.oracle for item in DESCRIPTION_TABLE}
    assert declared == covered, {
        "missing from the table": sorted(declared - covered),
        "named by the table but not by the oracle": sorted(covered - declared),
    }


def test_every_description_entry_carries_exactly_one_mark() -> None:
    for item in DESCRIPTION_TABLE:
        assert item.mark in MARKS
        assert item.detail


def test_the_problem_field_delegates_here_and_this_is_here() -> None:
    """The delegation is real: the named test exists and is this one."""

    names = {item.compared_in for item in delegated(PROBLEM_TABLE)}
    assert names == {"test_association_parity"}
    assert __name__.endswith("test_association_parity")


def test_the_wrapper_itself_is_retired() -> None:
    """Nothing in this stack reintroduces the container under any spelling."""

    import finn.dataflow.ops.association as association  # noqa: PLC0415

    assert not hasattr(association, "MVAUSourceDescription")
    assert not hasattr(association, "SourceDescription")
    assert hasattr(association, "SourceAssociation")


CASES = tuple((spec["name"], item) for spec in SPECS for item in comparable(DESCRIPTION_TABLE))


@pytest.mark.parametrize(
    ("fixture_name", "entry"), CASES, ids=[f"{name}-{item.oracle}" for name, item in CASES]
)
def test_each_description_field_agrees_with_the_oracle(fixture_name: str, entry: Any) -> None:
    spec = next(item for item in SPECS if item["name"] == fixture_name)
    expected = entry.read_oracle(dict(_report()["fixtures"][fixture_name]))
    _model, occurrence = bound(spec, BUILD)
    observed = entry.read_local(occurrence)
    assert entry.agrees(expected, observed), (
        f"{entry.oracle} ({entry.mark}: {entry.detail}) on {fixture_name}: "
        f"oracle {expected!r}, this stack {observed!r}"
    )


def test_the_association_carries_the_same_identities_where_a_network_resolves() -> None:
    """The other half: what the association says agrees with the source reading.

    Only where a Network resolves, and that restriction is itself a difference
    worth stating: the oracle's description was a source projection that existed
    whether or not a Design applied, while an association is read off a selected
    Network.  A fused-threshold node has no applicable Design here, so it has no
    association -- and the fields the description carried are still produced,
    which is what the table above compares.
    """

    from finn.dataflow.kernels.dotp_axi import DotpAxiKernel  # noqa: PLC0415
    from finn.dataflow.ops.mvau.designs.base import WeightedDotProductDesign  # noqa: PLC0415
    from finn.dataflow._engine import Decided  # noqa: PLC0415

    spec = next(item for item in SPECS if item["name"] == "fused_provenance")
    _model, occurrence = bound(spec, BUILD)
    chosen = occurrence.design.select("dot_product").root
    for declaration, value in (
        (WeightedDotProductDesign.pe, 2),
        (WeightedDotProductDesign.simd, 2),
    ):
        chosen = chosen.design.alternative("dot_product").assign(declaration, value).root
    kernel = chosen.design.alternative("dot_product").kernel("compute")
    assert isinstance(kernel, Decided)
    chosen = kernel.value.assign(DotpAxiKernel.compute_pumping, False).root

    answer = chosen.association
    assert isinstance(answer, Decided)
    association = answer.value
    assert association.scope_id == chosen.binding.node_identity
    assert association.origin_nodes == ("mul0", "add0")
    assert {item.operand: item.tensor for item in association.operands} == {
        "activation": "activation",
        "weight": "weight",
        "output": "output",
    }
    assert encode(association.origin_nodes) == ["mul0", "add0"]


def test_the_additions_are_recorded_rather_than_left_out() -> None:
    """A table listing only correspondences would read as though nothing was added."""

    assert len(ADDITIONS) >= 3
    for item in ADDITIONS:
        assert " -- " in item, "each addition states what it is and why the oracle had none"
