# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""One declaration's model with and without collapsed forwarding, compiled afresh.

Collapse rewrites evaluation edges only, so both models have the same nodes, scopes and
keys. The collapsed model is a model's structure, as compiled (``_structure``), its nodes
collapsed as they are resolved; the plain one is the same structure's nodes as resolved
before collapse, in one table (``_structure.materialize(..., collapsed=False)``), read as a
compiled form only: a table built so has no design space value, and is not evaluated.
"""

from __future__ import annotations

from typing import TypeVar, cast

from finn.core.space import Model, Space
from finn.core.space._structure import Structure, materialize, modes
from finn.core.space._template import link_template
from finn.core.space.compiler import root_parameters, root_record
from finn.core.space.ir import Linked
from finn.core.space.occurrence import opened

S = TypeVar("S", bound=Space)


def compiled(node: Space, *, collapsed: bool) -> Model[Space]:
    """The model of ``node``'s Space class compiled afresh (no model cache), collapsed as
    compiled, or its plain table."""
    record = root_record(node)
    if collapsed:
        permuted, eager = modes()
        structure = Structure(link_template(record.space_type), permuted=permuted, eager=eager)
        return Model(record.space_type, cast(Linked, structure), structure.lock)
    structure = Structure(link_template(record.space_type))
    return Model(record.space_type, materialize(structure, collapsed=False))


def open_space(node: S) -> S:
    """The design space of ``node`` on its model compiled afresh, collapsed."""
    model = compiled(node, collapsed=True)
    return cast(S, opened(model, root_parameters(model, root_record(node))))
