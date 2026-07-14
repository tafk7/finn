############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""The op-identity / implementation seam.

This module defines the irreducible interfaces that cross the boundary between
an op's *identity* (what it computes, its derived ``design_point``) and an
*implementation* (a composed backend that turns that design point into build
artifacts). The whole thesis of the dataflow-kernel backend rests on one
property: :meth:`Implementation.emit` is **hermetic** — it is written against
nothing but the derived ``design_point``, the extracted ``ParamBundle``, and
resolved knob values. It never touches ``onnx_node``, a ``ModelWrapper``, a
base class, or the filesystem. If a backend cannot be written that way, the
seam leaks and we learn it cheaply.

Codegen is *typed*, not string-surgery. A :class:`Template` declares its
parameters; :meth:`Template.render` fails loudly on a missing or unknown
binding — replacing the untyped ``$KEY$`` ``str.replace`` mechanism whose
silent no-op-on-rename is a documented defect in both prior systems.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from .derivation import KernelDesignPoint


# =============================================================================
# ParamBundle — the tensor data the design_point deliberately does NOT carry
# =============================================================================


@dataclass(frozen=True)
class ParamBundle:
    """Immutable parameter/initializer data extracted from the graph.

    A ``design_point`` carries only shapes and datatypes, never tensor values
    (verified against the derivation layer). The identity extracts constant
    initializers — e.g. the threshold matrix — into this frozen bundle so that
    :meth:`Implementation.emit` has the data it needs without reaching into a
    ``ModelWrapper``. This is the concrete mechanism that keeps ``emit``
    hermetic while still able to bake weights into params files.
    """

    tensors: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.tensors[name]

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def get(self, name: str, default: Any = None) -> Any:
        return self.tensors.get(name, default)


# =============================================================================
# SelectionContext — the device-aware context precondition() gates on
# =============================================================================


@dataclass(frozen=True)
class SelectionContext:
    """Everything an :meth:`Implementation.precondition` needs to decide
    feasibility. This is the seam neither prior system had: the prototype's
    constraint was ``Callable[[Kernel], bool]`` with no device info, and
    Brainsmith ported FINN's centralized device ladder. Here feasibility is a
    per-implementation predicate over an injected, device-aware context.
    """

    fpgapart: str
    design_point: "KernelDesignPoint"
    params: ParamBundle | None = None
    #: Resolved nodeattr map available at selection time (kernel params + knobs).
    #: A backend's precondition may consult declared knobs (e.g. a URAM request)
    #: alongside ``fpgapart`` — but never touches the graph to get them.
    config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_versal(self) -> bool:
        """True for AMD Versal parts (``xcv*`` / ``xqv*``).

        Mirrors the prefix rule finn.util.basic.is_versal uses; the adapter
        should prefer that helper when running inside full FINN.
        """
        p = self.fpgapart.lower()
        return p.startswith(("xcv", "xqv")) or p.startswith("xcvm") or p.startswith("xcvc")


# =============================================================================
# Typed codegen — Template replaces untyped $KEY$ str.replace
# =============================================================================

_TOKEN_RE = re.compile(r"\$([A-Z][A-Z0-9_]*)\$")


class TemplateError(Exception):
    """Raised when a template's bindings do not exactly match its slots."""


@dataclass(frozen=True)
class Template:
    """A code template with declared, validated parameter slots.

    Slots are ``$NAME$`` tokens (a delimiter proven not to collide with HLS C++
    or Verilog across FINN's history). Unlike the prior ``str.replace`` loop,
    :meth:`render` enforces an exact match between the slots present in the text
    and the bindings supplied:

    * a slot with no binding is an error (no silently-unfilled ``$NAME$``);
    * a binding with no slot is an error (catches a renamed token — the exact
      silent-no-op defect flagged in both prior systems).

    Binding values are scalars (``int``/``float``/``str`` → ``str(value)``) or
    ``lines`` (a sequence of ``str`` → joined with newlines).
    """

    text: str

    @property
    def slots(self) -> frozenset[str]:
        return frozenset(_TOKEN_RE.findall(self.text))

    def render(self, bindings: Mapping[str, Any]) -> str:
        slots = self.slots
        keys = frozenset(bindings)

        missing = slots - keys
        if missing:
            raise TemplateError(f"template slots with no binding: {sorted(missing)}")
        unknown = keys - slots
        if unknown:
            raise TemplateError(
                f"bindings with no matching slot (renamed/typo'd token?): {sorted(unknown)}"
            )

        out = self.text
        for name, value in bindings.items():
            out = out.replace(f"${name}$", _stringify(value))
        return out


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    return str(value)


# =============================================================================
# Artifacts — the four kinds every backend emits, as typed data
# =============================================================================


@dataclass(frozen=True)
class GeneratedFile:
    """A generated top file (``.cpp`` / ``.v``): a template + typed bindings.
    Content is produced lazily so a binding error surfaces at write time with
    a clear message, not as a corrupt file."""

    filename: str
    template: Template
    bindings: Mapping[str, Any]

    def content(self) -> str:
        return self.template.render(self.bindings)


@dataclass(frozen=True)
class DataFile:
    """A parameter/weight data file (``thresh.h``, ``*.dat``) whose content is
    computed imperatively from the ParamBundle + folding — the escape hatch for
    logic a template cannot express."""

    filename: str
    content: str


@dataclass(frozen=True)
class StaticFile:
    """A declarative reference to an in-tree shared HDL/library resource copied
    verbatim (e.g. ``thresholding.sv``). Resolved as a packaged resource by the
    adapter, not by ``emit`` — no ``FINN_ROOT`` / ``os.listdir`` globbing."""

    package: str
    resource: str


@dataclass(frozen=True)
class IPICommands:
    """Structured Vivado block-design (IPI) commands. For the embedded first cut
    these are trivial instantiation lines; the decoupled-mode stitch god-method
    is explicitly out of scope."""

    commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class Artifacts:
    """The complete, filesystem-free output of :meth:`Implementation.emit`.
    The adapter — not ``emit`` — writes these to a code-gen directory."""

    generated: tuple[GeneratedFile, ...] = ()
    data_files: tuple[DataFile, ...] = ()
    static_files: tuple[StaticFile, ...] = ()
    ipi: IPICommands = field(default_factory=IPICommands)


# =============================================================================
# Implementation — the composed backend (stateless strategy, selected by name)
# =============================================================================


class Implementation(ABC):
    """A composed backend for one op kind in one language.

    Implementations are **stateless** strategy objects registered in the pool
    and resolved **by name** from the ``implementation`` nodeattr — nothing is
    pickled. Per-node configuration lives in the ``design_point`` (derived) and
    in resolved knob values (from nodeattrs); an implementation stores neither.
    """

    #: Unique name; the value stored in the ``implementation`` nodeattr.
    name: str
    #: The op kind this backend implements, e.g. ``"Thresholding"``.
    op_kind: str
    #: Code-generation language, ``"hls"`` or ``"rtl"``.
    language: str
    #: Selection tie-break; lower is preferred (mirrors the prototype's scheme).
    priority: int = 0
    #: Backend-specific knob *declarations* (nodeattr tuples), e.g.
    #: ``{"ram_style": ("s", False, "distributed")}``. The adapter registers
    #: these as nodeattrs; their resolved *values* arrive in ``emit``'s
    #: ``config`` map alongside the op's kernel params.
    knob_specs: Mapping[str, tuple] = {}

    @abstractmethod
    def precondition(self, ctx: SelectionContext) -> bool:
        """Return True iff this backend can build the node in the given device
        context. This is the hard feasibility half of selection (⊥ preference).
        """

    @abstractmethod
    def emit(
        self,
        design_point: "KernelDesignPoint",
        params: ParamBundle,
        config: Mapping[str, Any],
    ) -> Artifacts:
        """Produce build artifacts from **only** these three inputs.

        * ``design_point`` — the derived shape/width/folding view.
        * ``params`` — extracted initializer tensors (keeps ``get_initializer``
          out of codegen).
        * ``config`` — the adapter-resolved nodeattr map: kernel params
          (``act_val``, ``num_steps``) and backend knobs (``ram_style``, …),
          plus interface datatype strings. Plain values, resolved once by the
          adapter — the one boundary that touches the node.

        The hermeticity contract is about *access*, not signature width: ``emit``
        reads config as passed-in data and **never touches the graph** — no
        ``onnx_node``/``get_nodeattr`` on a live node, no ``ModelWrapper``/
        ``get_initializer``, no ``getCustomOp().set_nodeattr`` write-back, no
        filesystem. Those three couplings are the diamond, the graph-reach, and
        the prototype's bridge respectively; keeping them out of ``emit`` is what
        proves the seam holds.
        """
