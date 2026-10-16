# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A flat netlist of black-box instances, read as ports, instances and what drives what.

``finn.kernels.artifacts.build`` emits a module that instantiates library modules and
joins them with continuous assignments through one net per instance port. A test of it
means the structure: the top's ports (direction, width), each instance (its module, its
parameters' elaborated values, the ports left open), and the drives (which instance input
or top output takes which top input, instance output, bit range of one, or constant).
``connections`` elaborates the text with pyslang and reads exactly that, so a reformatting
of the text, a renamed internal net or a changed order does not fail a test, and a wrong
wire in a line the test does not quote cannot pass.

The instanced modules must be defined in the text (``black_box`` writes an empty one from
a port list), so pyslang elaborates their parameters and gives their ports' directions
and widths. Only what the emitter writes is read: another construct is refused
(``NetlistError``) and not skipped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pyslang
from pyslang import ast, syntax


class NetlistError(ValueError):
    """The text does not elaborate, or has a construct this reader does not know."""


@dataclass(frozen=True)
class Port:
    """A module port: its direction (``in``, ``out`` or ``inout``) and its width in bits."""

    direction: str
    width: int


@dataclass(frozen=True)
class Pin:
    """A port of the top (``instance`` is ``None``) or of an instance, whole or a bit range.

    ``bits`` is ``(msb, lsb)``, or ``None`` for the whole port: a range that spans the
    whole port is reported as ``None``, so a drive is one value however it was written.
    """

    instance: str | None
    port: str
    bits: tuple[int, int] | None = None


@dataclass(frozen=True)
class Const:
    """A constant of ``width`` bits."""

    width: int
    value: int


@dataclass(frozen=True)
class Inverted:
    """The logical or bitwise inverse of a pin (``!x``, ``~x``)."""

    operand: Pin


Source = Pin | Const | Inverted


@dataclass(frozen=True)
class Drive:
    """``sink`` (a top output or an instance input, a bit range of it) takes ``source``."""

    sink: Pin
    source: Source


@dataclass(frozen=True)
class Instance:
    """One instance: its module, its parameters as elaborated, its ports' directions (the
    module's own, from its black box) and the ports left open. A port's width is the
    width of what it is connected to: see ``Pin``."""

    module: str
    parameters: Mapping[str, Value]
    directions: Mapping[str, str]
    open: frozenset[str]


@dataclass(frozen=True)
class Netlist:
    """A top module: its ports, its instances by name, and its drives."""

    top: str
    ports: Mapping[str, Port]
    instances: Mapping[str, Instance]
    drives: tuple[Drive, ...]

    def driving(self, instance: str | None, port: str) -> tuple[Drive, ...]:
        """The drives into ``port`` of ``instance`` (the top itself for ``None``)."""
        return tuple(
            drive
            for drive in self.drives
            if drive.sink.instance == instance and drive.sink.port == port
        )

    def sources(self, instance: str | None, port: str) -> frozenset[Drive]:
        """``driving`` as a set, to compare against the drives a test states."""
        return frozenset(self.driving(instance, port))


def _parameter(name: str, value: str) -> str:
    """The declaration of a parameter the instance binds to ``value``: a string, an
    unpacked array of integers (``'{1, 2}``) or, by default, an integer."""
    if value.startswith('"'):
        return f'parameter string {name} = ""'
    if value.startswith("'{"):
        return f"parameter int {name} [{value.count(',') + 1}] = {value}"
    return f"parameter {name} = 0"


def black_box(
    name: str, ports: Mapping[str, Port], parameters: Mapping[str, str] | None = None
) -> str:
    """An empty module ``name`` with ``ports`` and ``parameters`` (names with their value
    text; see ``_parameter``), for ``connections`` to instance."""
    declared = [_parameter(key, value) for key, value in (parameters or {}).items()]
    pins = [
        f"{_KEYWORDS[port.direction]} logic "
        + (f"[{port.width - 1}:0] " if port.width > 1 else "")
        + key
        for key, port in ports.items()
    ]
    header = f" #({', '.join(declared)})" if declared else ""
    return f"module {name}{header} ({', '.join(pins)});\nendmodule\n"


_KEYWORDS = {"in": "input", "out": "output", "inout": "inout"}

_DIRECTIONS = {
    "ArgumentDirection.In": "in",
    "ArgumentDirection.Out": "out",
    "ArgumentDirection.InOut": "inout",
}


def _width(thing: ast.Symbol | ast.Expression) -> int:
    width = getattr(thing.type, "bitstreamWidth", None)
    if not isinstance(width, int) or width <= 0:
        raise NetlistError(f"no width for {thing}")
    return width


def _integer(value: pyslang.ConstantValue | None, what: str) -> int:
    inner = None if value is None else value.value
    if not isinstance(inner, pyslang.SVInt) or inner.hasUnknown:
        raise NetlistError(f"{what} is not a constant integer")
    return int(inner)


Value = int | str | tuple[int, ...]
"""A parameter's elaborated value: an integer, a string or an unpacked array of integers."""


def _value(value: pyslang.ConstantValue, what: str) -> Value:
    inner = value.value
    if isinstance(inner, str):
        return inner
    if isinstance(inner, list):
        return tuple(_integer(element, what) for element in inner)
    return _integer(value, what)


class _Reader:
    def __init__(self, body: ast.InstanceBodySymbol) -> None:
        self.ports: dict[str, Port] = {}
        for port in body.portList:
            if type(port).__name__ != "PortSymbol":
                raise NetlistError(f"unsupported port kind: {port.name}")
            direction = _DIRECTIONS.get(str(port.direction))
            if direction is None:
                raise NetlistError(f"unsupported port direction: {port.name}")
            self.ports[port.name] = Port(direction, _width(port))
        self.nets: dict[str, int] = {
            member.name: _width(member)
            for member in body
            if type(member).__name__ in ("VariableSymbol", "NetSymbol")
        }
        self.instances: dict[str, Instance] = {}
        self.pins: dict[str, Pin] = {}
        for member in body:
            if type(member).__name__ == "InstanceSymbol":
                self._instance(member)
        self.drives = tuple(
            self._drive(member.assignment)
            for member in body
            if type(member).__name__ == "ContinuousAssignSymbol"
        )
        for member in body:
            if type(member).__name__ not in _KNOWN:
                raise NetlistError(f"unsupported construct: {type(member).__name__}")

    def _instance(self, instance: ast.InstanceSymbol) -> None:
        parameters: dict[str, Value] = {}
        for member in instance.body:
            if type(member).__name__ != "ParameterSymbol" or member.isLocalParam:
                continue
            parameters[member.name] = _value(member.value, f"parameter {member.name}")
        directions: dict[str, str] = {}
        opened: set[str] = set()
        for connection in instance.portConnections:
            name = connection.port.name
            directions[name] = _DIRECTIONS.get(str(connection.port.direction), "inout")
            expression = connection.expression
            if isinstance(expression, ast.AssignmentExpression):
                expression = expression.left
            if expression is None or expression.kind == ast.ExpressionKind.EmptyArgument:
                opened.add(name)
                continue
            net = self._net(expression, name)
            if net in self.pins:
                raise NetlistError(f"net {net} joins two instance ports")
            self.pins[net] = Pin(instance.name, name)
        self.instances[instance.name] = Instance(
            instance.definition.name,
            MappingProxyType(parameters),
            MappingProxyType(directions),
            frozenset(opened),
        )

    def _net(self, expression: ast.Expression, port: str) -> str:
        if not isinstance(expression, ast.NamedValueExpression):
            raise NetlistError(f"port {port} is not connected to a plain net")
        name = str(expression.symbol.name)
        if name in self.ports:
            raise NetlistError(f"port {port} is connected straight to the top's port {name}")
        return name

    def _pin(self, expression: ast.Expression) -> Pin:
        """A net, or a bit or part select of one, as the pin it stands for."""
        bits: tuple[int, int] | None
        if isinstance(expression, ast.NamedValueExpression):
            name, bits = str(expression.symbol.name), None
        elif isinstance(expression, ast.RangeSelectExpression) and isinstance(
            expression.value, ast.NamedValueExpression
        ):
            if expression.selectionKind != ast.RangeSelectionKind.Simple:
                raise NetlistError("an indexed part select is not read")
            name = str(expression.value.symbol.name)
            bits = (
                _integer(expression.left.constant, "a range's msb"),
                _integer(expression.right.constant, "a range's lsb"),
            )
        elif isinstance(expression, ast.ElementSelectExpression) and isinstance(
            expression.value, ast.NamedValueExpression
        ):
            name = str(expression.value.symbol.name)
            index = _integer(expression.selector.constant, "a bit select's index")
            bits = (index, index)
        else:
            raise NetlistError(f"not a net or a select of one: {expression}")
        if name in self.ports:
            pin = Pin(None, name, bits)
            width = self.ports[name].width
        elif name in self.pins:
            pin = Pin(self.pins[name].instance, self.pins[name].port, bits)
            width = self.nets[name]
        else:
            raise NetlistError(f"net {name} is on no port of the top or of an instance")
        if bits is not None and bits[0] - bits[1] + 1 == width:
            return Pin(pin.instance, pin.port)
        return pin

    def _source(self, expression: ast.Expression) -> Source:
        if expression.constant is not None:
            return Const(_width(expression), _integer(expression.constant, "a constant"))
        if isinstance(expression, ast.ConversionExpression):
            if _width(expression.operand) != _width(expression):
                raise NetlistError(f"a conversion that changes the width: {expression}")
            return self._source(expression.operand)
        if isinstance(expression, ast.UnaryExpression) and str(expression.op) in (
            "UnaryOperator.LogicalNot",
            "UnaryOperator.BitwiseNot",
        ):
            return Inverted(self._pin(expression.operand))
        return self._pin(expression)

    def _drive(self, assignment: ast.Expression) -> Drive:
        if not isinstance(assignment, ast.AssignmentExpression):
            raise NetlistError(f"not an assignment: {assignment}")
        return Drive(self._pin(assignment.left), self._source(assignment.right))


_KNOWN = frozenset(
    {
        "PortSymbol",
        "VariableSymbol",
        "NetSymbol",
        "InstanceSymbol",
        "ContinuousAssignSymbol",
        "ParameterSymbol",
    }
)


def connections(source: str, top: str) -> Netlist:
    """Elaborate ``top`` of ``source`` and read its ports, instances and drives."""
    options = ast.CompilationOptions()
    options.topModules = {top}
    compilation = ast.Compilation(pyslang.Bag([options]))
    sources = pyslang.SourceManager()
    compilation.addSyntaxTree(syntax.SyntaxTree.fromText(source, sources))
    errors = [diagnostic for diagnostic in compilation.getAllDiagnostics() if diagnostic.isError()]
    if errors:
        engine = pyslang.DiagnosticEngine(sources)
        client = pyslang.TextDiagnosticClient()
        engine.addClient(client)
        for diagnostic in errors:
            engine.issue(diagnostic)
        raise NetlistError("elaboration failed:\n" + client.getString())
    tops = [instance for instance in compilation.getRoot().topInstances if instance.name == top]
    if not tops:
        raise NetlistError(f"no such top-level module: {top}")
    read = _Reader(tops[0].body)
    return Netlist(top, MappingProxyType(read.ports), MappingProxyType(read.instances), read.drives)
