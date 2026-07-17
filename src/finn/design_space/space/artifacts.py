############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
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
    writes these to a code-gen directory."""

    generated: tuple[GeneratedFile, ...] = ()
    data_files: tuple[DataFile, ...] = ()
    static_files: tuple[StaticFile, ...] = ()
    ipi: IPICommands = field(default_factory=IPICommands)
