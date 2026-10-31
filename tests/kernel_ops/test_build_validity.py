# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Independent real-network descriptions require an explicit complete build boundary."""

from __future__ import annotations

from copy import copy

import pytest
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
    inspection,
    occurrence,
    reject,
)
from finn.kernels.base import Kernel
from finn.kernels.composer import Composed


@pytest.mark.parametrize("condition", ("open-rule", "refused-rule", "open-presence", "absent"))
def test_real_netlist_does_not_certify_unrelated_rule_or_presence(
    condition: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = kernel_model()
    configure_partition(model)
    original = shell.in_shell(model, model.graph.node)

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
    composed = Composed(
        expanded, original.composed.names, original.composed.facts, name=original.name
    )
    choices = {
        key: state.value
        for key, state in inspection.states(original.point)
        if isinstance(state, (inspection.Set, inspection.Waiting, inspection.Dormant))
    }
    point = composed.design_space()
    refs = {item.key: item.reference for item in inspection.decisions(point)}
    report = occurrence.bind(point, {refs[key]: value for key, value in choices.items()})
    assert not any(outcome.status == "dropped" for outcome in report.outcomes)
    bound = copy(original)
    # Preserve the real graph, members, bindings and saved choices; add only a root rule.
    bound._composed = composed
    bound._point = report.instance
    bound._memo = {}
    module = bound.module_of()
    if condition == "refused-rule":
        # A partition a rule refuses has no channel's hops, so no module: refused once, by the
        # rule. Whether it should keep them is pending a decision.
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
    monkeypatch.setattr(shell, "in_shell", lambda *_: bound)
    if condition == "absent":
        configured, _ = shell.configured_network(model, "independent")
        assert configured.network is bound
    elif condition == "refused-rule":
        with refused("unrelated-rule", error=KernelOpError, owner="unrelated"):
            shell.configured_network(model, "independent")
    else:
        with refused("design-incomplete", error=KernelOpError, node="independent"):
            shell.configured_network(model, "independent")
