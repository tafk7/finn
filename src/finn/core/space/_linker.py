# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Link a Space class into one owned node table, phase by phase.

Every declared node becomes a scope; every member a node of the evaluation
graph. A Decision over nodes becomes a selector decision, one guarded scope
per candidate, and a ``select`` node per member read through it; ``Members``
becomes a ``members`` node over its children's exports.

The phases pass one table (``_table``): ``_allocate`` places the scopes and
reserves every member's node, ``_lower`` gives each node its payload
(references resolved by ``_names``), the nodes are ordered by their known
dependencies, ``_check`` validates their semantics and ``_collapse`` bypasses
forwarding aliases. The table is then frozen into the ``LinkedModel``.
"""

from __future__ import annotations

from ._allocate import allocate
from ._check import check
from ._collapse import collapse
from ._configuration import Space
from ._lower import lower
from ._ordering import dependency_order
from .errors import DefinitionError
from .ir import LinkedModel

_OWNER = "space.linker"


def link_space(space_type: type[Space]) -> LinkedModel:
    table = allocate(space_type)
    lower(table)
    order = dependency_order(
        tuple(node.dependencies for node in table.nodes),
        tuple(node.key for node in table.nodes),
    )
    check(table, order)
    forward = collapse(table.nodes, order)
    choices = tuple(choice.freeze() for choice in table.choice_drafts)
    nodes = tuple(table.nodes)
    keys = {node.key: node.index for node in nodes}
    if len(keys) != len(nodes):
        raise DefinitionError.broken("node-names-collide", _OWNER, "generated node names collide")
    return LinkedModel(
        nodes,
        table.scopes,
        order,
        tuple(node.index for node in nodes if node.kind == "param"),
        tuple(node.index for node in nodes if node.kind == "decision"),
        keys,
        choices,
        frozenset(table.editable_aliases),
        forward,
    )
