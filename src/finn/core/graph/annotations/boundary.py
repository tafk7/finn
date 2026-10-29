# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What the build states of a partition's boundary, stored in a model's graph metadata
(the ``finn.boundary`` namespace), by tensor.

The build states its partition's boundary by port (``DataflowBuildConfig.boundary``); the
cut resolves each port to the tensor it carries and writes the statements here, before the
partition's body is cut, so the body carries them. Keyed by tensor, a statement is read
wherever a selection's boundary side is that tensor's (``finn.custom_op.kernels.network``):
the whole partition's port, or a slice's edge that is one, whatever the slice names its
own ports. Each statement is the build's (``finn.kernels.sides.Outside``, its source
``BUILD``), its pace kept as given: a ``Steady`` shorthand stays one. An absent pace
is stored as the explicit empty object, leaving timing unstated.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from qonnx.core.metadata import JSON, MetadataError, Namespace

from finn.core.graph.errors import KernelOpError
from finn.kernels.sides import BUILD, BeatTimes, Outside, Steady

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

BOUNDARY = Namespace("finn.boundary", version=1, inherit=True)
"""What the build states of the partition's boundary: ``statements``, each by the tensor
of the port it states. A partition's body opened through its parent reads the parent's.
``read_boundary`` and ``write_boundary`` are its only reader and writer."""

#: A statement's fields as stored: an object of these, its pace an object too.
FIELDS = ("order", "passes", "lanes", "pace")


def _statements(value: object) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(stated, dict)
            and set(stated) <= set(FIELDS)
            and isinstance(stated.get("pace"), dict)
            for stated in value.values()
        )
    )


STATEMENTS = BOUNDARY.key(
    "statements",
    JSON,
    check=_statements,
    expect="an object of a statement by tensor, each an object of order, passes, lanes and "
    "its pace (an object)",
)


def _stored(outside: Outside) -> dict[str, Any]:
    """A statement as stored: its fields, its pace as given."""
    pace = outside.pace
    stored_pace: dict[str, Any] = (
        {"every": pace.every}
        if isinstance(pace, Steady)
        else {"times": list(pace.times), "span": pace.span}
        if isinstance(pace, BeatTimes)
        else {}
    )
    return {
        "order": outside.order,
        "passes": outside.passes,
        "lanes": outside.lanes,
        "pace": stored_pace,
    }


def _read(tensor: str, stored: Mapping[str, Any]) -> Outside:
    """The build's statement stored for ``tensor``; an empty pace leaves timing unstated."""
    pace = dict(stored["pace"])
    made: Steady | BeatTimes | None
    if not pace:
        made = None
    elif set(pace) == {"every"}:
        made = Steady(pace["every"])
    elif set(pace) == {"times", "span"} and isinstance(pace["times"], list):
        made = BeatTimes(tuple(pace["times"]), pace["span"])
    else:
        raise ValueError(f"{tensor}: a pace is every k cycles or beat times over a span")
    try:
        return Outside(BUILD, stored.get("order"), stored.get("passes"), stored.get("lanes"), made)
    except ValueError as error:
        raise ValueError(f"{tensor}: {error}") from None


def read_boundary(model: ModelWrapper) -> dict[str, Outside]:
    """The build's statements from the model's ``finn.boundary`` metadata, by tensor;
    none where it states none. Refuses a namespace not one, by name
    (``boundary-unreadable``)."""
    try:
        stated = model.namespace(BOUNDARY).get("statements", {})
        return {tensor: _read(tensor, each) for tensor, each in stated.items()}
    except (MetadataError, ValueError) as error:
        raise KernelOpError.refused(
            "boundary-unreadable",
            "read_boundary",
            f"the model's boundary statements are not ones: {error}",
        ) from error


def write_boundary(model: ModelWrapper, statements: Mapping[str, Outside]) -> None:
    """Store the build's ``statements``, by tensor, in the model's ``finn.boundary``
    metadata, the namespace's whole: what it stated before is cleared, and none stated
    leaves it empty. Refuses a statement whose source is not the build
    (``boundary-invalid``): a mover states its side from the shell's row, never from the
    model."""
    others = sorted(tensor for tensor, each in statements.items() if each.source != BUILD)
    if others:
        raise KernelOpError.refused(
            "boundary-invalid",
            "write_boundary",
            f"{', '.join(others)}: the model stores the build's statements only",
        )
    model.clear(BOUNDARY)
    if statements:
        model.set(
            STATEMENTS, {tensor: _stored(each) for tensor, each in sorted(statements.items())}
        )


__all__ = ["BOUNDARY", "FIELDS", "STATEMENTS", "read_boundary", "write_boundary"]
