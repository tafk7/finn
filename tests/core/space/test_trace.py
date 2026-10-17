# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The Space trace prototype (``finn.core.space._trace``) on the toy Pipeline: the relations
known by hand.

- a link's ``burst`` check reads the upstream unit's core through ``upstream_packed``
  (a requirement reading another choice), the lowest Space holding both the Pipeline;
- the gearbox case's check reads the upstream core, outside its candidate;
- the packed core's case check reads only its width, a fact;
- the second unit's packed ``pe`` has a domain over a width that follows the first
  unit's choices (the leftover case).
"""

from __future__ import annotations

from typing import Any

from core.space._pipeline_support import Pipeline
from finn.core.space import _trace, design_space


def joined() -> dict[str, Any]:
    point = design_space(Pipeline(width=10))
    packed = point.with_choices({Pipeline.u0.compute: "packed"})
    return _trace.join([_trace.trace(point, "nothing"), _trace.trace(packed, "u0 packed")])


def decision(found: dict[str, Any], key: str) -> dict[str, Any]:
    item: dict[str, Any] = next(each for each in found["decisions"] if each["key"] == key)
    return item


def test_a_bursts_check_reads_the_upstream_core_through_upstream_packed() -> None:
    found = joined()
    for link, unit in (("l0", "u0"), ("l1", "u1")):
        burst = decision(found, f"{link}.burst")
        fact = burst["own_check"]["requires[0]"]
        assert fact["kind"] == "requirement"
        assert fact["requirement"]["named"] == ["2", "4"]
        assert list(fact["choices"]) == [f"{unit}.compute"]
        read = fact["choices"][f"{unit}.compute"]
        assert read["path"] == [
            f"{link}.burst",
            f"{link}.upstream_packed",
            f"{unit}.packed",
            f"{unit}.compute",
        ]
        assert read["holds_both"] == {"scope": "(root)", "space_type": "Pipeline"}
        # Its fact is read for the case it does not name, in every configuration.
        assert fact["unnamed_cases"]["1"]["read_in"] == ["nothing", "u0 packed"]


def test_the_gearbox_case_reads_the_upstream_core_outside_its_candidate() -> None:
    gearbox = decision(joined(), "l0.adapter")["case_checks"]["gearbox"]
    assert gearbox["candidate"] == {"scope": "l0.adapter.gearbox", "space_type": "Gearbox"}
    read = gearbox["choices"]["u0.compute"]
    assert read["inside"] is False
    assert read["path"][-3:] == ["l0.adapter.gearbox.upstream_packed", "u0.packed", "u0.compute"]


def test_the_packed_case_reads_only_its_width() -> None:
    packed = decision(joined(), "u0.compute")["case_checks"]["packed"]
    assert packed["admission"] == "u0.compute.packed.admission"
    assert "choices" not in packed
    assert list(packed["facts"]) == ["width"]


def test_the_second_units_pe_domain_follows_the_first_units_choices() -> None:
    found = joined()
    extent = decision(found, "u1.compute.packed.pe")["own_check"]["extent"]
    assert extent["kind"] == "domain"
    assert set(extent["choices"]) == {"u0.compute", "u0.compute.packed.pe"}
    assert extent["choices"]["u0.compute.packed.pe"]["configurations"] == ["u0 packed"]
    leftover = {
        (row["choice"], row["other"])
        for row in _trace.relations(found)
        if row["relation"] == "leftover"
    }
    assert ("u1.compute.packed.pe", "u0.compute.packed.pe") in leftover
    # The first unit's pe reads only the Pipeline's width.
    first = decision(found, "u0.compute.packed.pe")["own_check"]["extent"]
    assert "choices" not in first
    assert list(first["facts"]) == ["width"]
