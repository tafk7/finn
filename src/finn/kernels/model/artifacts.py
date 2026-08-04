############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Typed codegen artifacts — the flat emit vocabulary.

``emit(point, context) -> Artifacts`` produces build artifacts as pure data; the
adapter (not emit) writes them to a code-gen directory. Codegen is *typed*, not
string-surgery: a :class:`Template` declares its slots and :meth:`Template.render`
fails loudly on a missing or unknown binding — replacing the untyped ``$KEY$``
``str.replace`` mechanism whose silent no-op-on-rename is a documented defect in
baseline FINN codegen (matrixvectoractivation_rtl.py's ``template.replace`` loop).

Harvested (mechanism only) from the superseded ``finn/src/finn/kernels`` spike and
rebuilt flat against this engine's ``Point``/``Context`` — no structured
``design_point.inputs[...]``, no separate ``ParamBundle`` (our ``Context`` already
carries initializer VALUES as frozen data, so it is the hermetic param source).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .ports import Port

_TOKEN_RE = re.compile(r"\$([A-Z][A-Z0-9_]*)\$")


class TemplateError(Exception):
    """Raised when a template's bindings do not exactly match its slots."""


@dataclass(frozen=True)
class Template:
    """A code template with declared, validated parameter slots.

    Slots are ``$NAME$`` tokens (``$`` + UPPERCASE) — a delimiter proven not to
    collide with HLS C++ or Verilog. Verilog ``$clog2(...)`` is deliberately NOT a
    slot (lowercase, no trailing ``$NAME$`` form). Unlike the prior ``str.replace``
    loop, :meth:`render` enforces an exact match between slots and bindings:

    * a slot with no binding is an error (no silently-unfilled ``$NAME$``);
    * a binding with no slot is an error (catches a renamed/typo'd token — the exact
      silent-no-op defect in the baseline mechanism).

    Binding values are scalars (``int``/``float``/``str`` -> ``str(value)``) or a
    sequence of ``str`` (joined with newlines).
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


# -- typed value-binding (F4) --------------------------------------------------
#
# `Template.render` validates slot↔binding NAMES but not VALUE TYPES — so a width
# slot could silently receive a Verilog range string, the layernorm `[31:0]`
# silent-wrongness class. `RtlModule` adds a TYPE per slot; `bind` type-checks the
# values against it and returns the same `{str: str}` dict `Template.render` consumes.
# Each typed value's `.render()` produces EXACTLY the string the untyped binding did
# (`BitWidth(14) -> "14"`, `Bool(True) -> "1"`), so rendered bytes are unchanged.


class BindError(TemplateError):
    """Raised when a value binding does not match its :class:`RtlModule` schema —
    a missing slot, an extra value, or a value whose type is not the declared one."""


@dataclass(frozen=True)
class BitWidth:
    """A bit-width slot (an ``ap_int``/``ACCU_WIDTH`` count). Renders as the decimal int."""

    value: int

    def render(self) -> str:
        return str(int(self.value))


@dataclass(frozen=True)
class Dim:
    """A geometry extent / fold dial (``MW``, ``PE``, ``VERSION``, …). Decimal int."""

    value: int

    def render(self) -> str:
        return str(int(self.value))


@dataclass(frozen=True)
class Bool:
    """A boolean flag slot (``SIGNED_ACTIVATIONS``, ``NARROW_WEIGHTS``). Renders ``1``/``0``."""

    value: bool

    def render(self) -> str:
        return "1" if self.value else "0"


@dataclass(frozen=True)
class Raw:
    """An escape hatch for a slot that is a free-form string, not geometry — a Verilog
    module name or an ``AP_INT_MAX_W`` knob. Renders the string verbatim."""

    value: str

    def render(self) -> str:
        return self.value


# The set of concrete typed-value classes a schema may declare and `bind` accepts.
TypedValue = (BitWidth, Dim, Bool, Raw)


@dataclass(frozen=True)
class RtlModule:
    """The TYPE signature of a :class:`Template` — the second half of the artifact
    contract. Where ``Template`` declares slot NAMES, ``RtlModule`` declares each
    slot's TYPE (one of :data:`TypedValue`), so :func:`bind` catches a value whose
    type is wrong before it reaches the template. Owned by the artifact (defined next
    to its ``Template``), not the ``Backend`` — softvec + packed share one wrapper,
    hence one schema (design pitch §5)."""

    name: str
    params: Mapping[str, type]


def bind(module: RtlModule, values: Mapping[str, Any]) -> dict[str, str]:
    """Type-check ``values`` against ``module``'s schema and return the stringified
    ``{slot: str}`` dict :meth:`Template.render` consumes. Raises :class:`BindError`
    on a missing slot, an extra value, or a value whose type is not the declared one."""
    schema = frozenset(module.params)
    given = frozenset(values)

    missing = schema - given
    if missing:
        raise BindError(f"{module.name}: schema slots with no value: {sorted(missing)}")
    extra = given - schema
    if extra:
        raise BindError(f"{module.name}: values with no schema slot: {sorted(extra)}")

    out: dict[str, str] = {}
    for name, expected in module.params.items():
        value = values[name]
        if type(value) is not expected:
            raise BindError(
                f"{module.name}.{name}: expected {expected.__name__}, "
                f"got {type(value).__name__}"
            )
        out[name] = value.render()
    return out


@dataclass(frozen=True)
class GeneratedFile:
    """A generated top file (``.cpp`` / ``.v``): a template + typed bindings.
    Content is produced lazily so a binding error surfaces at render time with a
    clear message, not as a corrupt file."""

    filename: str
    template: Template
    bindings: Mapping[str, Any]

    def content(self) -> str:
        return self.template.render(self.bindings)


@dataclass(frozen=True)
class DataFile:
    """A parameter/weight data file (``*.dat`` / ``thresh.h``) whose content is
    computed imperatively from the ``Context`` param VALUES + folding — the escape
    hatch for logic a template cannot express."""

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
    these are trivial instantiation lines; the decoupled-mode stitch is out of
    scope."""

    commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class Artifacts:
    """The complete, filesystem-free output of ``emit``. The adapter — not emit —
    writes these to a code-gen directory.

    ``ports`` is the block's declared port taxonomy (:class:`Port`) — the surface the
    composition stitch (``emit/stitch.py``) binds against. A single-cell emit
    declares its own ports; :meth:`merge` unions the artifacts of two composed cells
    (compute + parameter delivery) so the resolver can wire them."""

    generated: tuple[GeneratedFile, ...] = ()
    data_files: tuple[DataFile, ...] = ()
    static_files: tuple[StaticFile, ...] = ()
    ipi: IPICommands = field(default_factory=IPICommands)
    ports: tuple[Port, ...] = ()

    def merge(self, other: "Artifacts") -> "Artifacts":
        """Union two cells' artifacts (files + static + IPI commands + ports),
        preserving order (self first). Used by ``emit_composed`` to gather both
        halves of a composed kernel before the stitch wires their ports. Ports are
        NOT deduplicated — each cell owns a distinct port surface; the resolver
        distinguishes them by cell membership, not by identity."""
        return Artifacts(
            generated=self.generated + other.generated,
            data_files=self.data_files + other.data_files,
            static_files=self.static_files + other.static_files,
            ipi=IPICommands(self.ipi.commands + other.ipi.commands),
            ports=self.ports + other.ports,
        )
