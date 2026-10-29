# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Real configured modules and packaging do not request cycle estimates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from qonnx.core.modelwrapper import ModelWrapper
from support.models import choose_memories, configure_partition, kernel_model

import finn.builder.build_dataflow_testbench as testbench
import finn.core.rtlsim_exec as rtlsim
import finn.custom_op.kernels.shell as shell
from finn.core.graph.annotations.outputs import OUTPUT_VERIFIED
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import design_digest
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.rtlsim_exec import XSim
from finn.core.space import Available, QueryResult, Unresolved, reject
from finn.custom_op.kernels.network import PRODUCER
from finn.kernels.artifacts.build import netlist
from finn.kernels.base import Timed
from finn.kernels.explore import Seam
from finn.kernels.partition import Partition
from finn.toolchain.launcher import Selection, Toolchain
from finn.transformation.kernels.cut import CutKernelPartition


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


def _unverified_model() -> ModelWrapper:
    """Create real cut/memory choices, without using the verification helper."""
    model = kernel_model()
    made = shell.make_cut_choices(model)
    assert not (made.stale or made.unset)
    choose_memories(model)
    assert model.get(OUTPUT_VERIFIED) is None
    return model


@pytest.mark.parametrize("timing", ("refused", "waiting"))
@pytest.mark.parametrize("previously_verified", (False, True))
def test_held_verification_does_not_stamp_unchecked_timing(
    timing: str,
    previously_verified: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _unverified_model()
    if previously_verified:
        shell.verify_partition(model, "known-timing")
        assert model.get(OUTPUT_VERIFIED) == design_digest(model)
    previous = model.get(OUTPUT_VERIFIED)
    point, _ = shell.configured_network(model, "untimed")
    members = list(point.network.members)
    query = Partition.query
    reads: list[str] = []

    def unavailable(self: Partition, name: str, member: object) -> QueryResult[Any]:
        if member is Timed.cycles:
            reads.append(name)
            if timing == "refused":
                return reject("timed-cycles", "cycle estimate unavailable")
            return Unresolved(
                (
                    Finding(
                        FindingKind.BLOCKER,
                        "timed-cycles",
                        "cycles",
                        "cycle estimate not supplied",
                    ),
                )
            )
        return query(self, name, member)

    def must_not_check(self: object, network: Partition) -> None:
        raise AssertionError("unavailable cycles reached the timed FIFO check")

    monkeypatch.setattr(Partition, "query", unavailable)
    monkeypatch.setattr(Seam, "check_fifos", must_not_check)
    with pytest.raises(KernelOpError) as refused:
        shell.verify_partition(model, "unknown-timing")
    assert reads == members
    assert len(refused.value.findings) == len(members)
    assert {finding.code for finding in refused.value.findings} == {"timed-cycles"}
    assert model.get(OUTPUT_VERIFIED) == previous


def test_known_cycle_verification_keeps_its_period_and_design_marker() -> None:
    model = _unverified_model()
    expected, _ = shell.configured_network(model, "known-timing")
    period = expected.cycles
    configured, checked = shell.verify_partition(model, "known-timing")
    assert configured.module == expected.module
    assert checked.period == period and period > 0
    assert model.get(OUTPUT_VERIFIED) == design_digest(model)


@pytest.mark.parametrize("consumer", ("testbench", "xsim"))
@pytest.mark.parametrize("timing", ("refused", "waiting"))
def test_held_timed_actions_refuse_before_reference_or_directory_work(
    consumer: str,
    timing: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _unverified_model()
    shell.verify_partition(model, "known-timing")
    target = tmp_path / "timed-action"
    toolchain = Toolchain(Selection(), {})
    parent = (
        model.transform(CutKernelPartition(tmp_path / "partition")) if consumer == "xsim" else None
    )
    point, _ = shell.configured_network(model, "known-timing")
    members = list(point.network.members)
    query = Partition.query
    reads: list[str] = []

    def unavailable(self: Partition, name: str, member: object) -> QueryResult[Any]:
        if member is Timed.cycles:
            reads.append(name)
            if timing == "refused":
                return reject("timed-cycles", "cycle estimate unavailable")
            return Unresolved(
                (
                    Finding(
                        FindingKind.BLOCKER,
                        "timed-cycles",
                        "cycles",
                        "cycle estimate not supplied",
                    ),
                )
            )
        return query(self, name, member)

    def must_not_act(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("unavailable timing reached reference, input, or hardware work")

    monkeypatch.setattr(Partition, "query", unavailable)
    with pytest.raises(KernelOpError, match="^cycle characterization unavailable: ") as refused:
        if consumer == "testbench":
            monkeypatch.setattr(testbench, "execute_onnx", must_not_act)
            monkeypatch.setattr(testbench, "stream_bench", must_not_act)
            testbench.write_testbench(model, target, {}, toolchain=toolchain)
        else:
            assert parent is not None
            monkeypatch.setattr(rtlsim, "_integers", must_not_act)
            monkeypatch.setattr(XSim, "_run_directory", must_not_act)
            XSim(directory=target, toolchain=toolchain).run(
                parent.graph.node[0],
                parent.make_empty_exec_context(),
                parent,
            )
    assert reads == members
    expected_local = Finding(
        FindingKind.REJECTION if timing == "refused" else FindingKind.BLOCKER,
        "timed-cycles",
        "" if timing == "refused" else "cycles",
        "cycle estimate unavailable" if timing == "refused" else "cycle estimate not supplied",
    )
    expected = tuple(point.network.located(name, expected_local) for name in members)
    assert type(refused.value) is KernelOpError
    assert len(refused.value.findings) == len(members)
    assert set(refused.value.findings) == set(expected)
    assert str(refused.value) == "cycle characterization unavailable: " + "; ".join(
        finding.subject_line() for finding in expected
    )
    assert not target.exists()
