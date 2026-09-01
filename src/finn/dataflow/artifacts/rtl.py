# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The RTL checker: refusal only, and it never supplies a value.

The declaration is authoritative.  Generated RTL is generated *from* the ABI,
so parsing it back would be circular, and the ABI is the packaging contract --
it must not move whenever the RTL moves.  What a checker adds is that **a
declaration nothing checks is a second authority waiting to disagree**, and the
tree already contains the disagreement: ``in0_V_TDATA`` in the physical model
against ``in0_V_tdata`` in the generated text.

So this module has exactly two powers.  It can **decline** -- say it could not
establish anything -- and it can **refuse** a declaration the source
contradicts.  It cannot fill a field in.  ``no build-time path derives a Kernel
from RTL``: discovery would make a grammar version bump able to change which
designs exist.

Three things measured against real sources rather than assumed (the A0 gate):

* **Resolved widths need a parameter binding.**  Neither ``replay_buffer`` nor
  ``dotp_axi`` gives its parameters defaults, so slang cannot elaborate either
  as a top level unbound.  The binding comes from the declaration, which is the
  right direction -- the checker is told what to check against.
* **Vendor primitives have no source in any closure we compile.**  ``DSP58``
  and its relatives are black-boxed, which cannot change the top's own ports.
* **FinnLib places ``\\`default_nettype`` inside module bodies.**  Illegal per
  LRM 22.8, tolerated by Vivado, rejected by slang, and present in 66 of the
  155 files we compile.  It is tolerated here by an explicit list rather than
  by relaxing the error filter, because the two are different: one names what
  is forgiven and why, the other forgives whatever turns up.

One thing slang will not do for us: an **undeclared parameter override is
silently ignored**.  So the override set is checked against the declared
parameters here, or a typo'd binding would pass as a match.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pyslang  # type: ignore[import-not-found]
from pyslang import ast, syntax

from finn.dataflow.artifacts.abi import ComponentABI, Direction, ObservedPort, check_against_rtl

#: Diagnostics that cannot bear on ports or parameters, and are therefore not
#: grounds to decline.  Each entry is a decision with a reason, not a filter.
TOLERATED_DIAGNOSTICS = frozenset(
    {
        # FinnLib writes `default_nettype between the port list and the body.
        # Illegal per LRM 22.8 and unable to change a port width either way.
        "DiagCode(DirectiveInsideDesignElement)",
    }
)

_DIRECTIONS = {
    "ArgumentDirection.In": Direction.IN,
    "ArgumentDirection.Out": Direction.OUT,
    "ArgumentDirection.InOut": Direction.INOUT,
}


@dataclass(frozen=True)
class ExtractedModule:
    """What the source says, as far as the checker could establish it."""

    name: str
    ports: tuple[ObservedPort, ...]
    parameters: tuple[tuple[str, int], ...]
    local_parameters: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Declined:
    """The checker could not establish anything, and says so.

    Declining is **permitted and expected**.  A checker that guessed rather
    than declined would be supplying a value, which is the one thing it may
    never do.  The honest scope of the guarantee is whatever it does not
    decline, which is why the decline rate is measured rather than described.
    """

    reason: str
    details: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.reason}: " + "; ".join(self.details[:4])


Extraction = Union[ExtractedModule, Declined]


def _integer(value: pyslang.ConstantValue) -> int | None:
    inner = value.value
    if not isinstance(inner, pyslang.SVInt) or inner.hasUnknown:
        return None
    return int(inner)


def _report(compilation: ast.Compilation, diagnostics: list[pyslang.Diagnostic]) -> tuple[str, ...]:
    engine = pyslang.DiagnosticEngine(compilation.sourceManager)
    client = pyslang.TextDiagnosticClient()
    engine.addClient(client)
    for diagnostic in diagnostics:
        engine.issue(diagnostic)
    return tuple(line for line in client.getString().splitlines() if line)


def extract(
    files: Sequence[Path], top: str, parameters: Sequence[tuple[str, str]] = ()
) -> Extraction:
    """Elaborate ``top`` and report its ports, widths and parameter values.

    ``parameters`` is the binding the declaration supplies.  Without it a
    module whose parameters have no defaults cannot be elaborated at all, and
    the checker declines rather than inventing widths.
    """

    options = ast.CompilationOptions()
    options.topModules = {top}
    # A vendor primitive has no source in any closure we compile.  Treating it
    # as a black box is the only way to reach the top's own ports, and it
    # cannot change them.
    options.flags = ast.CompilationFlags.IgnoreUnknownModules
    options.paramOverrides = [f"{name}={value}" for name, value in parameters]
    compilation = ast.Compilation(pyslang.Bag([options]))

    for path in files:
        compilation.addSyntaxTree(syntax.SyntaxTree.fromFile(str(path)))

    diagnostics = compilation.getAllDiagnostics()
    errors = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.isError() and str(diagnostic.code) not in TOLERATED_DIAGNOSTICS
    ]
    if errors:
        return Declined("elaboration failed", _report(compilation, errors))

    instances = [
        instance for instance in compilation.getRoot().topInstances if instance.name == top
    ]
    if not instances:
        return Declined("no such top-level module", (top,))
    body = instances[0].body

    ports: list[ObservedPort] = []
    for port in body.portList:
        if type(port).__name__ != "PortSymbol":
            # An interface port carries no resolved width of its own.
            return Declined("unsupported port kind", (f"{port.name}: {type(port).__name__}",))
        direction = _DIRECTIONS.get(str(port.direction))
        if direction is None:
            return Declined("unsupported port direction", (f"{port.name}: {port.direction}",))
        width = getattr(port.type, "bitstreamWidth", None)
        if not isinstance(width, int) or width <= 0:
            return Declined("unresolved port width", (f"{port.name}: {port.type}",))
        ports.append(ObservedPort(port.name, direction, width))

    declared: list[tuple[str, int]] = []
    local: list[tuple[str, int]] = []
    for member in body:
        if type(member).__name__ != "ParameterSymbol":
            continue
        value = _integer(member.value)
        if value is None:
            return Declined("non-integer parameter", (f"{member.name}: {member.value}",))
        (local if member.isLocalParam else declared).append((member.name, value))

    supplied = {name for name, _ in parameters}
    unknown = sorted(supplied - {name for name, _ in declared})
    if unknown:
        # slang ignores an override for a parameter that does not exist, so a
        # typo would otherwise pass as agreement.
        return Declined(
            "parameter binding names something the module does not declare", tuple(unknown)
        )

    return ExtractedModule(top, tuple(ports), tuple(declared), tuple(local))


def check_abi(
    abi: ComponentABI,
    files: Sequence[Path],
    top: str,
    parameters: Sequence[tuple[str, str]] = (),
) -> tuple[str, ...] | Declined:
    """Refuse a declaration the source contradicts, or decline.

    Returns the empty tuple when the declaration and the source agree, a
    non-empty tuple of refusals when they do not, and ``Declined`` when the
    checker could not establish either.  Three outcomes, because collapsing
    "agrees" and "could not tell" is how a guarantee becomes a claim.
    """

    extracted = extract(files, top, parameters)
    if isinstance(extracted, Declined):
        return extracted
    return check_against_rtl(abi, extracted.ports)


def check_symbols(modules: Sequence[tuple[str, ExtractedModule]]) -> tuple[str, ...]:
    """Two files defining one module name with different ports.

    The source-level half of §5.2's rule.  ``sources.merge_closures`` catches
    the declared case from ``provides``; this catches it where nobody declared
    anything, which for checked-in third-party RTL is the normal situation.
    """

    seen: dict[str, tuple[str, ExtractedModule]] = {}
    issues: list[str] = []
    for origin, module in modules:
        previous = seen.get(module.name)
        if previous is None:
            seen[module.name] = (origin, module)
            continue
        if previous[1].ports != module.ports:
            issues.append(
                f"{previous[0]} and {origin} both define {module.name!r} with different "
                "ports; isolate them by library or refuse"
            )
    return tuple(issues)


__all__ = [
    "TOLERATED_DIAGNOSTICS",
    "Declined",
    "Extraction",
    "ExtractedModule",
    "check_abi",
    "check_symbols",
    "extract",
]
