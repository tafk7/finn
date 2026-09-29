# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Composite kernels: kernels of kernels, wired through their streams.

A composite places kernels, Decisions over kernels and the streams between
them; ``structure`` wires its children's modules (``finn.kernels.streams.netlist``)
into one generated module whose ports are its boundary streams. A ``Design``
is a composite at the top.

A composite may itself be placed in another. It then sits on its parent's
streams through reference inputs, each paired in ``boundaries`` with the
internal stream that is its boundary there (``{"x_stream": "activations"}``),
and exports, under ``PORT``, that stream's boundary contract for each: the
parent's stream plans and adapts to the composite as to any kernel. Its
``fused`` Decision says what it becomes in its parent:

- fused, one module (``MODULE``), which the parent nests as a child; a fused
  composite exposing a control bus is refused (``composite-control``), as
  its parent does not export a child's bus yet;
- otherwise its parts (``PARTS``): the parent places its modules, streams,
  tie-offs and control buses in its own module, each boundary stream spliced
  with the parent's stream it sits on, each bus exported as
  ``<composite>_<port>``.

The two are the same wiring with a different module hierarchy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from finn.core.space import (
    Decision,
    Members,
    Rejected,
    View,
    constraint,
    derived,
    reject,
    view,
)
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.kernels.base import MODULE, MODULE_REQUIREMENTS, TIEOFFS, Kernel
from finn.kernels.control import EXPORTED
from finn.kernels.artifacts.requirements import ModuleBuildRequirements
from finn.kernels.streams import (
    COMPOSED,
    CONNECTION,
    PARTS,
    PARTS_SEMANTICS,
    Composed,
    Parts,
    merge_parts,
    netlist,
)


class Composite(Kernel):
    """Its children's modules wired through its streams: one module, or its parent's parts."""

    id = "finn.composite"
    version = "1"

    # Reference input -> the internal stream that is the composite's boundary on it.
    boundaries: ClassVar[Mapping[str, str]] = {}

    modules = Members(MODULE)
    streams = Members(CONNECTION)
    tied = Members(TIEOFFS)
    controls = Members(EXPORTED)
    flattened = Members(PARTS)

    def stem(self) -> str:
        """The generated module's name stem."""
        return "finn_" + type(self).__name__.lower()

    def producer_identity(self) -> ProducerIdentity:
        return ProducerIdentity("finn." + type(self).__name__.lower(), "1")

    @derived
    def placed(self) -> bool:
        """Whether it sits on a stream of a parent."""
        family = type(self)
        return any(self.present(getattr(family, reference)) for reference in family.boundaries)

    fused: bool = Decision(values=(True, False), when=placed)

    @derived
    def parted(self) -> bool:
        return not self.fused

    @constraint
    def seated(self) -> bool | Rejected:
        """Each parent stream it sits on carries its boundary stream's tensor."""
        for reference, stream in type(self).boundaries.items():
            if not self.present(getattr(type(self), reference)):
                continue
            outer, inner = getattr(self, reference).tensor, getattr(self, stream).tensor
            if outer != inner:
                return reject(
                    "composite-tensor",
                    f"{reference} carries {outer.shape} {outer.element.datatype_name}; "
                    f"the boundary {stream} carries {inner.shape} {inner.element.datatype_name}",
                )
        return True

    @view(
        semantics=COMPOSED,
        requires=(Kernel.admission, seated, modules, streams, tied, controls, flattened),
    )
    def structure(self) -> Composed | Rejected:
        return netlist(
            self.modules,
            self.streams,
            self.tied,
            self.controls,
            self.flattened,
            module=self.stem(),
            producer=self.producer_identity(),
        )

    @view(semantics=MODULE_REQUIREMENTS, requires=(structure,))
    def build_requirements(self) -> ModuleBuildRequirements:
        return self.structure.requirements

    @constraint
    def uncontrolled(self) -> bool | Rejected:
        """A fused composite exposes no control bus: its parent does not export one yet."""
        if any(item.value for item in self.controls):
            return reject(
                "composite-control",
                "a fused composite's control bus is not exported through its parent; "
                "place it unfused",
            )
        return True

    # What it exports to a parent: its module when fused, its parts otherwise.
    module_export = View(build_requirements, when=fused, requires=(uncontrolled,))

    @view(
        semantics=PARTS_SEMANTICS,
        requires=(Kernel.admission, seated, modules, streams, tied, controls, flattened),
        when=parted,
    )
    def parts(self) -> Parts | Rejected:
        try:
            modules, streams, tieoffs, controls = merge_parts(
                list(self.modules),
                list(self.streams),
                list(self.tied),
                list(self.controls),
                self.flattened,
            )
        except ValueError as error:
            return reject("stream-composition", str(error))
        return Parts(
            tuple(modules),
            tuple(streams),
            tuple(tieoffs),
            tuple(controls),
            tuple(type(self).boundaries.items()),
        )

    exports = {MODULE: module_export, PARTS: parts}


class Design(Composite):
    """A composite at the top: its boundary streams are its module's ports."""

    id = "finn.design"
    version = "1"


__all__ = ["Composite", "Design"]
