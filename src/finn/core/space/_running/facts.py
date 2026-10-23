# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The facts' preparation, and a design space opened on them.

- ``prepare``: values as a design space holds them: the entire request recognized by each
  node's semantics (each value with the role it is supplied in) before any snapshot
  adapter is invoked, then each frozen; ``prepare_values``, the same with one role for
  all;
- ``open``: the design space of a compiled model over its facts (each a root formal the
  model exposes; every required one given), every choice open: its facts layer's design
  space with nothing stored (``value.FactsLayer.bare``).

A node is read for its own fields (key, semantics, owner): its header, as the compiled
model states it (``Linked.header``); whether a formal is required, as its structure
resolves it (``raw_nodes``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .._failure import driver_only
from ..errors import RequestError
from ..ir import Linked, NodeId
from ..semantics import recognize, unrecognized
from ..semantics import snapshot as snapshot_value
from .value import DesignSpace, FactsLayer

if TYPE_CHECKING:
    from .._configuration import Space
    from ..compiler import Model


def prepare(linked: Linked, pending: Mapping[NodeId, tuple[object, str]]) -> dict[NodeId, object]:
    """Values as a design space holds them: the entire request recognized by each node's
    semantics (each value with the role it is supplied in) before any snapshot adapter is
    invoked, then each frozen."""
    for index, (value, role) in pending.items():
        header = linked.header(index)
        assert header.semantics is not None
        if not recognize(header.semantics, value, owner=header.owner, role=f"{role} recognition"):
            raise RequestError(f"{header.key}: {unrecognized(header.semantics)}")
    prepared: dict[NodeId, object] = {}
    for index, (value, role) in pending.items():
        header = linked.header(index)
        assert header.semantics is not None
        prepared[index] = snapshot_value(
            header.semantics, value, owner=header.owner, role=f"{role} snapshot"
        )
    return prepared


def prepare_values(
    linked: Linked,
    pending: Mapping[NodeId, object],
    role: str,
) -> dict[NodeId, object]:
    """Recognize every value before any is snapshot, then snapshot each."""
    return prepare(linked, {index: (value, role) for index, value in pending.items()})


def open(model: Model[Space], parameters: Mapping[NodeId, object]) -> DesignSpace:
    """The design space of ``model`` over ``parameters`` (by root formal), every choice
    open: all root inputs checked (each exposed, every required one given), recognized and
    frozen before any evaluation."""
    driver_only("configuration binding")
    structure = model.structure
    if structure is None:
        raise RequestError(
            "a table built by hand has no design space value: compile a Space class "
            "(compile_model) to open one"
        )
    for index in parameters:
        if index not in structure.parameters:
            raise RequestError(f"{structure.header(index).key} is not an exposed parameter")
    missing = [
        structure.header(index).key
        for index in structure.parameters
        if structure.raw_nodes[index].required and index not in parameters
    ]
    if missing:
        raise RequestError(f"missing required parameters: {', '.join(missing)}")
    frozen = prepare_values(model.linked, parameters, "parameter")
    return FactsLayer.of(model, frozen).bare


__all__ = ["open", "prepare", "prepare_values"]
