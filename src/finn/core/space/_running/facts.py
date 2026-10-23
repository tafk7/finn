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

An address is read for its own fields (key, semantics, owner, whether a formal is
required): its member record, as its scope reads it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .._failure import driver_only
from ..errors import RequestError
from ..semantics import recognize, unrecognized
from ..semantics import snapshot as snapshot_value
from .value import DesignSpace, FactsLayer

if TYPE_CHECKING:
    from .._configuration import Space
    from .._scopes import Address
    from ..compiler import Model


def prepare(pending: Mapping[Address, tuple[object, str]]) -> dict[Address, object]:
    """Values as a design space holds them: the entire request recognized by each node's
    semantics (each value with the role it is supplied in) before any snapshot adapter is
    invoked, then each frozen."""
    for index, (value, role) in pending.items():
        semantics = index.semantics
        assert semantics is not None
        if not recognize(semantics, value, owner=index.owner, role=f"{role} recognition"):
            raise RequestError(f"{index.key}: {unrecognized(semantics)}")
    prepared: dict[Address, object] = {}
    for index, (value, role) in pending.items():
        semantics = index.semantics
        assert semantics is not None
        prepared[index] = snapshot_value(
            semantics, value, owner=index.owner, role=f"{role} snapshot"
        )
    return prepared


def prepare_values(pending: Mapping[Address, object], role: str) -> dict[Address, object]:
    """Recognize every value before any is snapshot, then snapshot each."""
    return prepare({index: (value, role) for index, value in pending.items()})


def open(model: Model[Space], parameters: Mapping[Address, object]) -> DesignSpace:
    """The design space of ``model`` over ``parameters`` (by root formal), every choice
    open: all root inputs checked (each exposed, every required one given), recognized and
    frozen before any evaluation."""
    driver_only("configuration binding")
    exposed = model.scopes.parameters
    for index in parameters:
        if index not in exposed:
            raise RequestError(f"{index.key} is not an exposed parameter")
    missing = [index.key for index in exposed if index.required and index not in parameters]
    if missing:
        raise RequestError(f"missing required parameters: {', '.join(missing)}")
    frozen = prepare_values(parameters, "parameter")
    return FactsLayer.of(model, frozen).bare


__all__ = ["open", "prepare", "prepare_values"]
