# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Independent real-network descriptions require an explicit complete build boundary."""

from __future__ import annotations

import pytest
from support.configure import core_in_shell
from support.helpers import root_facts
from support.models import configure_partition, kernel_model
from support.refusals import refused

import finn.custom_op.kernels.shell as shell
from finn.core.graph.errors import KernelOpError
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import (
    Available,
    Param,
    QueryResult,
    Rejected,
    Space,
    Unresolved,
    constraint,
    derived,
    design_space,
    inspection,
    occurrence,
    reject,
)
from finn.kernels.base import Kernel
from finn.kernels.dataflow_core import Bound, DataflowCoreKernel


@pytest.mark.parametrize("condition", ("open-rule", "refused-rule", "open-presence", "absent"))
def test_real_netlist_does_not_certify_unrelated_rule_or_presence(
    condition: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = kernel_model()
    configure_partition(model)
    original = core_in_shell(model)

    class Optional(Space):
        value: int = Param(default=1)

    def local_rule() -> QueryResult[bool]:
        if condition == "open-rule":
            return Unresolved(
                (Finding(FindingKind.BLOCKER, "external-input", "unrelated", "not supplied"),)
            )
        if condition == "refused-rule":
            return reject("unrelated-rule", "independent root rule refused")
        return Available(True)

    def presence() -> QueryResult[bool]:
        return (
            Unresolved(
                (Finding(FindingKind.BLOCKER, "external-input", "unrelated", "not supplied"),)
            )
            if condition == "open-presence"
            else Available(False)
        )

    enabled = derived(presence)
    expanded = type(
        "WithIndependentRule",
        (type(original.point),),
        {
            "__module__": __name__,
            "unrelated": constraint(local_rule),
            "enabled": enabled,
            "optional": Optional(when=enabled),
        },
    )
    choices = {
        key: state.value
        for key, state in inspection.states(original.point)
        if isinstance(state, (inspection.Set, inspection.Waiting, inspection.Dormant))
    }
    point = design_space(expanded(**root_facts(original.point)))
    refs = {item.key: item.reference for item in inspection.decisions(point)}
    report = occurrence.bind(point, {refs[key]: value for key, value in choices.items()})
    assert not any(outcome.status == "dropped" for outcome in report.outcomes)
    # The real graph, members, bindings and saved choices; only a root rule added.
    bound = shell.shell_composition(report.instance, shell.body_row(model))
    module = bound.answer(DataflowCoreKernel.module)
    if condition == "refused-rule":
        # A partition a rule refuses has no channel's hops (no view of a refused design point
        # answers), so no module: refused once, by the rule. Its members' own views answer.
        assert isinstance(module, Rejected)
        assert [(each.code, each.owner) for each in module.findings] == [
            ("unrelated-rule", "unrelated")
        ]
    else:
        assert isinstance(module, Available)
    assert all(
        isinstance(bound.query(name, Kernel.frame_cycles), Available) for name in bound.members
    )
    assert not bound.undecided()
    monkeypatch.setattr(shell, "core_point", lambda *_: Bound(bound.point, {}, {}))
    if condition == "absent":
        configured = shell.configured_network(model, "independent")
        assert configured.network.point is bound.point
    elif condition == "refused-rule":
        with refused("unrelated-rule", error=KernelOpError, owner="unrelated") as caught:
            shell.configured_network(model, "independent")
        # A rule of the partition's own, no member's.
        assert str(caught.error).startswith("independent: refused: ")
    else:
        with refused("design-incomplete", error=KernelOpError, node="independent"):
            shell.configured_network(model, "independent")
