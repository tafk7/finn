# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""States of the new core (``finn.core.space._state``) for the engine's own tests: a root
declaration's design space as a state, a choice's node by key, a change by key, and a
refusal's codes."""

from __future__ import annotations

from collections.abc import Mapping

from finn.core.space import Inapplicable, Rejected, Space, Unresolved
from finn.core.space._assign import Assigned, assign
from finn.core.space._state import State, bind, prepare
from finn.core.space.compiler import compile_model, root_parameters, root_record
from finn.core.space.edits import Change
from finn.core.space.ir import NodeId


def start(declaration: Space) -> State:
    """The design space of a root declaration, as a state: its facts bound, every choice
    open."""
    record = root_record(declaration)
    model = compile_model(record.space_type)
    facts = root_parameters(model, record)
    return bind(model, prepare(model.linked, {i: (v, "parameter") for i, v in facts.items()}))


def key(current: State, name: str) -> NodeId:
    return current.linked.keys[name]


def change(
    current: State,
    choices: Mapping[str, object] | None = None,
    facts: Mapping[str, object] | None = None,
    clear: tuple[str, ...] = (),
) -> Assigned:
    """``current`` with ``choices`` set and ``clear`` cleared (by key), and ``facts``."""
    nodes = current.linked.nodes
    requests: list[Change[object]] = [
        Change(current, nodes[key(current, name)].scope, key(current, name), value)
        for name, value in (choices or {}).items()
    ]
    requests += [
        Change(current, nodes[key(current, name)].scope, key(current, name), remove=True)
        for name in clear
    ]
    given = {key(current, name): value for name, value in (facts or {}).items()}
    return assign(current, given, requests)


def codes(result: object) -> list[tuple[str, str]]:
    assert isinstance(result, (Inapplicable, Rejected, Unresolved))
    return [(finding.code, finding.owner) for finding in result.findings]
