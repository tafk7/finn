# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The RTL checker: refusal only, and it never supplies a value.

The declaration is authoritative.  Generated RTL is generated *from* the ABI,
so parsing it back would be circular, and the ABI is the packaging contract --
it must not move whenever the RTL moves.  What a checker adds is that **a
declaration nothing checks is a second authority waiting to disagree**: a pin
declared ``CLK`` where the source says ``clk`` is one such disagreement.

So this module has exactly two powers.  It can **decline** -- say it could not
establish anything -- and it can **refuse** a declaration the source
contradicts.  It cannot fill a field in.  ``no build-time path derives a Kernel
from RTL``: discovery would make a grammar version bump able to change which
designs exist.

Three things measured against real sources rather than assumed:

* **Resolved widths need a parameter binding.**  Neither ``replay_buffer`` nor
  ``dotp_axi`` gives its parameters defaults, so slang cannot elaborate either
  as a top level unbound.  The binding comes from the declaration, which is the
  right direction -- the checker is told what to check against.
* **Vendor primitives have no source in any closure we compile.**  ``DSP58``
  and its relatives are black-boxed, which cannot change the top's own ports.
* **FinnLib places ``\\`default_nettype`` inside module bodies.**  Illegal per
  LRM 22.8, tolerated by Vivado, rejected by slang, and present in many of the
  files we compile.  It is tolerated here by an explicit list rather than
  by relaxing the error filter, because the two are different: one names what
  is forgiven and why, the other forgives whatever turns up.  One more code
  Vivado accepts is forgiven only *inside* the constructs that keep it from a
  port or a parameter (``TOLERATED_WITHIN``), and declined anywhere else.

One thing slang will not do for us: an **undeclared parameter override is
silently ignored**.  So the override set is checked against the declared
parameters here, or a typo'd binding would pass as a match.

**A name is established before its value is.**  Every declared parameter is
reported by name, but its value only when it is an integer or a string (an
untyped parameter given a string literal is a string, not the literal's bits); an
unpacked array (``thresholding_axi``'s ``THRESHOLDS``) or a real
(``eltwise``'s ``B_SCALE``) is reported with the value ``None``, *not
established*.  The module is not declined for it: the ports and their widths
are what slang resolved under the binding, and nothing the checker compares
reads a parameter value.

**One read, beside the checker** (decision FS6): ``evaluate`` returns the
integer and integer-array constants a module derives at elaboration under a
binding, for an analysis that must agree with the RTL's own derivation (an
``input_gen``'s buffer size and pointer increments) rather than copy it. It
fills no declaration: a kernel's pins and parameters stay declared and checked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Union

import pyslang
from pyslang import ast, syntax

from finn.kernels.artifacts.abi import Direction, ObservedPort, Pin, check_against_rtl
from finn.kernels.artifacts.sources import include_directories, is_header

#: Diagnostics that cannot bear on ports or parameters, and are therefore not
#: grounds to decline.  Each entry is a decision with a reason, not a filter.
TOLERATED_DIAGNOSTICS = frozenset(
    {
        # FinnLib writes `default_nettype between the port list and the body.
        # Illegal per LRM 22.8 and unable to change a port width either way.
        "DiagCode(DirectiveInsideDesignElement)",
        # FinnLib's compressor sources declare `timescale and the rest of
        # FinnLib does not.  A time scale sets delay units, never a port or a parameter.
        "DiagCode(MissingTimeScale)",
    }
)

#: Diagnostics tolerated only *inside* the named constructs, and declined
#: anywhere else.  The same code elsewhere can reach a port or a parameter, so
#: an entry names the constructs that confine it, with the reason they do.
TOLERATED_WITHIN: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        # FinnLib's add_multi calls a function declared in its generate block to
        # build a localparam of that block (LRM 13.4.3 forbids it; Vivado
        # accepts it).  A constant of a generate block reaches module level only
        # by a hierarchical name, which slang refuses in a constant expression
        # (ConstEvalHierarchicalName, not tolerated), no port or module
        # parameter is declared inside one, and a generate condition only
        # chooses which blocks exist.  Called from module level, the same
        # function raises this code outside any generate construct: declined.
        # In an included header (add_multi's compressor schedule), the construct
        # confining it is the one around the `include.
        "DiagCode(ConstEvalFunctionInsideGenerate)": frozenset(
            {"IfGenerate", "LoopGenerate", "CaseGenerate"}
        ),
    }
)

_DIRECTIONS = {
    "ArgumentDirection.In": Direction.IN,
    "ArgumentDirection.Out": Direction.OUT,
    "ArgumentDirection.InOut": Direction.INOUT,
}


@dataclass(frozen=True)
class ExtractedModule:
    """What the source says, as far as the checker could establish it.

    Every port is established with its direction and resolved width, and every
    parameter and localparam by name, in declaration order.  A value is
    established only when it is an integer or a string; any other value (an
    unpacked array, a real, a type) is ``None``: the name is known, the value is
    not, and nothing here stands in for it.
    """

    name: str
    ports: tuple[ObservedPort, ...]
    parameters: tuple[tuple[str, int | str | None], ...]
    local_parameters: tuple[tuple[str, int | str | None], ...]

    @property
    def unestablished(self) -> tuple[str, ...]:
        """Parameters and localparams declared by name whose value is not established."""
        return tuple(
            name for name, value in (*self.parameters, *self.local_parameters) if value is None
        )


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


def _string_literal(member: ast.ParameterSymbol) -> str | None:
    """The string an untyped parameter holds (``parameter RAM_STYLE = "auto"``): its
    value, the default or the binding's, is a string literal and its declaration states
    no type, sign or range. SystemVerilog types such a parameter as the literal's bits,
    so its constant is an integer (``"auto"``: 1635087471), which is not what it says."""
    declared = member.declaredType.typeSyntax
    if declared is None or declared.kind != syntax.SyntaxKind.ImplicitType or str(declared).strip():
        return None
    expression = member.initializer
    while expression is not None and expression.kind == ast.ExpressionKind.Conversion:
        expression = expression.operand
    if expression is None or expression.kind != ast.ExpressionKind.StringLiteral:
        return None
    return str(expression.value)


def _constant(member: ast.ParameterSymbol) -> int | str | None:
    """An integer or string parameter value; anything else is not established."""
    inner = member.value.value
    if isinstance(inner, str):
        return inner
    literal = _string_literal(member)
    if literal is not None:
        return literal
    if not isinstance(inner, pyslang.SVInt) or inner.hasUnknown:
        return None
    return int(inner)


def _confining(
    trees: Sequence[syntax.SyntaxTree], kinds: frozenset[str]
) -> dict[str, list[pyslang.SourceRange]]:
    """The source range of every construct of ``kinds`` in ``trees``, by kind."""
    found: dict[str, list[pyslang.SourceRange]] = {kind: [] for kind in kinds}

    def record(node: syntax.SyntaxNode) -> None:
        found[node.kind.name].append(node.sourceRange)

    table = {getattr(syntax.SyntaxKind, kind): record for kind in kinds}
    for tree in trees:
        tree.root.visit(lookup_table=table)
    return found


def _within(
    sources: pyslang.SourceManager,
    location: pyslang.SourceLocation,
    ranges: Sequence[pyslang.SourceRange],
) -> bool:
    """Whether ``location``, or an `include that brought its text in, lies in ``ranges``.

    A location in a macro expansion is taken where the macro was used.
    """
    while True:
        location = sources.getFullyOriginalLoc(location)
        if any(
            location.buffer == area.start.buffer
            and area.start.offset <= location.offset < area.end.offset
            for area in ranges
        ):
            return True
        if not sources.isIncludedFileLoc(location):
            return False
        location = sources.getIncludedFrom(location.buffer)


def _errors(
    compilation: ast.Compilation,
    trees: Sequence[syntax.SyntaxTree],
    sources: pyslang.SourceManager,
) -> list[pyslang.Diagnostic]:
    """The errors that are grounds to decline: every one not tolerated where it arose."""
    errors = [
        diagnostic
        for diagnostic in compilation.getAllDiagnostics()
        if diagnostic.isError() and str(diagnostic.code) not in TOLERATED_DIAGNOSTICS
    ]
    if not any(str(diagnostic.code) in TOLERATED_WITHIN for diagnostic in errors):
        return errors
    ranges = _confining(trees, frozenset().union(*TOLERATED_WITHIN.values()))
    return [
        diagnostic
        for diagnostic in errors
        if not any(
            _within(sources, diagnostic.location, ranges[kind])
            for kind in TOLERATED_WITHIN.get(str(diagnostic.code), ())
        )
    ]


def _report(
    sources: pyslang.SourceManager, diagnostics: list[pyslang.Diagnostic]
) -> tuple[str, ...]:
    engine = pyslang.DiagnosticEngine(sources)
    client = pyslang.TextDiagnosticClient()
    engine.addClient(client)
    for diagnostic in diagnostics:
        engine.issue(diagnostic)
    return tuple(line for line in client.getString().splitlines() if line)


@dataclass
class _Elaborated:
    """An elaborated top's body, with what it reads kept alive: pyslang's symbols refer
    to the compilation, the source manager and the trees without owning them."""

    body: ast.InstanceBodySymbol
    compilation: ast.Compilation
    sources: pyslang.SourceManager
    trees: tuple[syntax.SyntaxTree, ...]


def _elaborate(
    files: Sequence[Path], top: str, parameters: Sequence[tuple[str, str]]
) -> _Elaborated | Declined:
    options = ast.CompilationOptions()
    options.topModules = {top}
    # A vendor primitive has no source in any closure we compile.  Treating it
    # as a black box is the only way to reach the top's own ports, and it
    # cannot change them.
    options.flags = ast.CompilationFlags.IgnoreUnknownModules
    options.paramOverrides = [f"{name}={value}" for name, value in parameters]
    compilation = ast.Compilation(pyslang.Bag([options]))

    sources = pyslang.SourceManager()
    for directory in include_directories(files):
        sources.addUserDirectories(str(directory))
    trees = [
        syntax.SyntaxTree.fromFile(str(path), sources) for path in files if not is_header(path)
    ]
    for tree in trees:
        compilation.addSyntaxTree(tree)

    errors = _errors(compilation, trees, sources)
    if errors:
        return Declined("elaboration failed", _report(sources, errors))

    instances = [
        instance for instance in compilation.getRoot().topInstances if instance.name == top
    ]
    if not instances:
        return Declined("no such top-level module", (top,))
    return _Elaborated(instances[0].body, compilation, sources, tuple(trees))


def _undeclared(
    body: ast.InstanceBodySymbol, parameters: Sequence[tuple[str, str]]
) -> Declined | None:
    """A binding naming a parameter the module does not declare: slang ignores such an
    override, so a typo would otherwise pass as agreement."""
    declared = {
        member.name
        for member in body
        if type(member).__name__ in ("ParameterSymbol", "TypeParameterSymbol")
        and not member.isLocalParam
    }
    unknown = sorted({name for name, _ in parameters} - declared)
    if unknown:
        return Declined(
            "parameter binding names something the module does not declare", tuple(unknown)
        )
    return None


def extract(
    files: Sequence[Path], top: str, parameters: Sequence[tuple[str, str]] = ()
) -> Extraction:
    """Elaborate ``top`` and report its ports, widths, parameter names and values.

    ``parameters`` is the binding the declaration supplies.  Without it a
    module whose parameters have no defaults cannot be elaborated at all, and
    the checker declines rather than inventing widths.  A parameter value that
    is not an integer or a string is reported as ``None`` rather than declining
    the module (``ExtractedModule``).  Headers among ``files`` are not parsed
    on their own; their directories are searched for every `include.
    """

    elaborated = _elaborate(files, top, parameters)
    if isinstance(elaborated, Declined):
        return elaborated
    body = elaborated.body

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

    declared: list[tuple[str, int | str | None]] = []
    local: list[tuple[str, int | str | None]] = []
    for member in body:
        kind = type(member).__name__
        if kind == "ParameterSymbol":
            # Not an integer or a string: the name is established, the value is not.
            value = _constant(member)
        elif kind == "TypeParameterSymbol":
            value = None
        else:
            continue
        (local if member.isLocalParam else declared).append((member.name, value))

    undeclared = _undeclared(body, parameters)
    if undeclared is not None:
        return undeclared

    return ExtractedModule(top, tuple(ports), tuple(declared), tuple(local))


Evaluated = Union[int, tuple[int, ...]]
"""An integer constant, or an unpacked array of them."""


def _evaluated(value: pyslang.ConstantValue) -> Evaluated | None:
    inner = value.value
    if isinstance(inner, list):
        elements = [_evaluated(element) for element in inner]
        if not all(isinstance(element, int) for element in elements):
            return None
        return tuple(element for element in elements if isinstance(element, int))
    if not isinstance(inner, pyslang.SVInt) or inner.hasUnknown:
        return None
    return int(inner)


def evaluate(
    files: Sequence[Path],
    top: str,
    parameters: Sequence[tuple[str, str]],
    names: Sequence[str],
) -> Mapping[str, Evaluated] | Declined:
    """The values slang evaluates for ``top``'s parameters and localparams ``names`` under
    the binding ``parameters``: each an integer or an unpacked array of integers.

    The one place a value is read from the RTL rather than checked against it: an
    analysis that must agree with what the RTL derives at elaboration (an
    ``input_gen``'s buffer, decision FS6) reads it here instead of copying the
    derivation. It declines when the module does not elaborate, the binding names
    an undeclared parameter, or a named value is missing or of another kind.
    """

    elaborated = _elaborate(files, top, parameters)
    if isinstance(elaborated, Declined):
        return elaborated
    body = elaborated.body
    undeclared = _undeclared(body, parameters)
    if undeclared is not None:
        return undeclared
    found: dict[str, Evaluated] = {}
    for member in body:
        if type(member).__name__ == "ParameterSymbol" and member.name in names:
            value = _evaluated(member.value)
            if value is None:
                return Declined("not an integer or an array of integers", (member.name,))
            found[member.name] = value
    missing = [name for name in names if name not in found]
    if missing:
        return Declined("no such parameter or localparam", tuple(missing))
    return MappingProxyType(found)


def check_abi(
    pins: Sequence[Pin],
    files: Sequence[Path],
    top: str,
    parameters: Sequence[tuple[str, str]] = (),
) -> tuple[str, ...] | Declined:
    """Refuse declared pins the source contradicts, or decline.

    Returns the empty tuple when the declaration and the source agree, a
    non-empty tuple of refusals when they do not, and ``Declined`` when the
    checker could not establish either.  Three outcomes, because collapsing
    "agrees" and "could not tell" is how a guarantee becomes a claim.

    The comparison reads the ports alone -- names, directions and the widths
    slang resolved under the binding -- so a parameter whose value is not
    established cannot enter it.
    """

    extracted = extract(files, top, parameters)
    if isinstance(extracted, Declined):
        return extracted
    return check_against_rtl(pins, extracted.ports)


__all__ = [
    "TOLERATED_DIAGNOSTICS",
    "TOLERATED_WITHIN",
    "Declined",
    "Extraction",
    "ExtractedModule",
    "Evaluated",
    "check_abi",
    "evaluate",
    "extract",
]
