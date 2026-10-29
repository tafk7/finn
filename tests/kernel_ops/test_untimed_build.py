# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Real configured modules and packaging do not request cycle estimates."""

from __future__ import annotations

from typing import Any

import pytest
from support.models import configure_partition, kernel_model

import finn.custom_op.kernels.shell as shell
from finn.core.graph.errors import KernelOpError
from finn.core.space import Available, QueryResult, reject
from finn.custom_op.kernels.network import PRODUCER
from finn.kernels.artifacts.build import netlist
from finn.kernels.base import Timed
from finn.kernels.partition import Partition


def test_configuration_and_emission_do_not_read_refused_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = kernel_model()
    configure_partition(model)
    expected, boundary = shell.configured_network(model, "with-existing-timing")
    query = Partition.query
    reads: list[str] = []

    def untimed(self: Partition, name: str, member: object) -> QueryResult[Any]:
        if member is Timed.cycles:
            reads.append(name)
            return reject("timed-cycles", "no cycle estimate supplied")
        return query(self, name, member)

    monkeypatch.setattr(Partition, "query", untimed)
    actual, actual_boundary = shell.configured_network(model, "untimed")
    assert reads == []
    assert actual_boundary == boundary
    assert actual.module == expected.module
    assert netlist(actual.module, "untimed") == netlist(expected.module, "untimed")
    with pytest.raises(KernelOpError, match="cycle characterization unavailable"):
        _ = actual.cycles
    assert reads == list(actual.network.members)
    assert isinstance(actual.network.module_of(PRODUCER), Available)
