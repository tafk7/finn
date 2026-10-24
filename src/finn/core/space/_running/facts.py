# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The facts' preparation, and a design space opened on them.

- ``prepare``: values as a design space holds them: the entire request type-checked by
  each node's annotation (each value with the role it is supplied in), then each in its
  canonical form (``semantics.canonical``: ``-0.0`` is ``0.0``; NaN and a mutable value are
  refused, naming the input); ``prepare_values``, the same with one role for all;
- ``open``: the design space of a compiled model over its facts (each a root formal the
  model exposes; every required one given, every other one its default where not given,
  ``value.FactsLayer.of``), every choice open: its facts layer's design space with nothing
  stored (``value.FactsLayer.bare``).

An address is read for its own fields (key, semantics, owner, whether a formal is
required, its default): its member record, as its scope reads it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .._failure import driver_only
from ..errors import RequestError
from ..semantics import NotAValue, canonical, recognize, unrecognized
from .value import DesignSpace, FactsLayer

if TYPE_CHECKING:
    from .._configuration import Space
    from .._scopes import Address
    from ..compiler import Model


def prepare(pending: Mapping[Address, tuple[object, str]]) -> dict[Address, object]:
    """Values as a design space holds them: the entire request type-checked by each node's
    annotation (each value with the role it is supplied in), then each in its canonical
    form; a ``RequestError`` naming the input where a value is not its type, or no value."""
    for index, (value, role) in pending.items():
        semantics = index.semantics
        assert semantics is not None
        if not recognize(semantics, value, owner=index.owner, role=f"{role} recognition"):
            raise RequestError(f"{index.key}: {unrecognized(semantics)}")
    prepared: dict[Address, object] = {}
    for index, (value, _role) in pending.items():
        try:
            prepared[index] = canonical(value)
        except NotAValue as refusal:
            raise RequestError(f"{index.key}: {refusal}") from None
    return prepared


def prepare_values(pending: Mapping[Address, object], role: str) -> dict[Address, object]:
    """Type-check every value, then take each in its canonical form."""
    return prepare({index: (value, role) for index, value in pending.items()})


def open(model: Model[Space], parameters: Mapping[Address, object]) -> DesignSpace:
    """The design space of ``model`` over ``parameters`` (by root formal), every choice
    open: all root inputs checked (each exposed, every required one given), type-checked
    and in their canonical form before any evaluation; each other formal not given takes
    its default in its facts layer (``value.FactsLayer.of``)."""
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
