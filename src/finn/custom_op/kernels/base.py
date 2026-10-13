# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""KernelOp: an ONNX node tied to a design space of kernels.

A KernelOp is a qonnx ``CustomOp`` whose node stands for a kernel of ``finn.kernels``
with its choices still open. It does not own the hardware of the graph: the channels
between nodes belong to the tensors they carry, and a partition's shell root (``shell``)
places the same kernels on channels the nodes share.

Three terms, used throughout the package:

- the node's *facts*: what it reads from the model it is attached to, which are its
  tensors' shapes and datatypes, the values of the initializers it holds (weights,
  thresholds) and the build target (``read_target``);
- its *node root*: its kernel with a channel on each input and output (``roots``),
  wired by its ``wiring``;
- its *point*: the node root bound to its facts (cached by facts, ``cache``), with the
  node's stored choices applied (``replay``).

Conversion asks two questions of a node. Does the op's pattern cover it? ``match`` reads
only the graph's structure (op types, connections, attributes), never a datatype or a
value, and never changes the model. Will a kernel build it? ``admission`` asks the
kernels. A missing shape or datatype raises ``FactUnstated``, and the node stays on the
host; anything else wrong with the facts is an error (``KernelOpError``).

``execute_node`` is the reference for what the op covers, and must equal ONNX on every
node the op converts (FINN executes ONNX with exact integers, which importing ``finn``
turns on).
The test harness checks this (``finn.harness.reference``); builds do not.

A node's attributes are its semantic attributes, which are part of the operation
(Thresholding's ``bias``), and its kernel's choices (``compute.packed.pe``), each set to
its baseline at conversion. A channel's choices are stored on its tensor instead, and
only in a partition's body (``CHANNEL``).

**Wiring.** Each ONNX input and output that carries a stream (a *port*: ``x``, ``w``,
``y``) meets the kernel as a ``Wiring``: an input's the kernel's Param its tensor is
given as, an output's the kernel's view of the tensor it writes (``tensor``,
``x_tensor``, ``y_tensor``: an output's tensor is the kernel's statement, never given
to it), and the kernel's interface its channel takes as an end (``end``, ``x_end``, a
view of a ``finn.kernels.base.PortEnd``). A parameter port,
whose initializer the node may hold (MatMul's weights), also names the kernel's Param
the initializer's values are given as (``contents``, ``w_contents``) and the kernel's
views of what its channel carries, downstream of the kernel: the tensor (``carried``,
``w_carried``) and the values its source stores (``stored``, ``w_stored``). The node
root and the shell root place the kernel and its channels by it: each tensor and
value to the kernel, each channel's tensor (a parameter channel's from the kernel's
views) and its end from the kernel (``finn.kernels.channels.end_of``). Facts flow one
way: tensor, kernel, end, channel.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeGuard

from onnx import helper
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import get_by_name

from finn.core.graph.annotations.channel import CHANNEL, channel_choices, value_type
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.facts import edge_tensor
from finn.core.space import (
    Available,
    Finding,
    FindingKind,
    Rejected,
    Space,
    inspection,
)
from finn.custom_op.kernels.cache import BIND_CACHE, Facts
from finn.custom_op.kernels.replay import committed, refusal, typed_choices
from finn.custom_op.kernels.roots import node_root, placed
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.tensor import Tensor
from finn.kernels.base import Kernel
from finn.kernels.configure import describe
from finn.platform.target import Target

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

ONNX_TYPES = {"int": "i", "bool": "i", "str": "s"}

Shapes = dict[str, tuple[tuple[int, ...], QONNXDataType]]


# -- patterns ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Wiring:
    """How one port of a KernelOp meets its kernel (the module docstring): the kernel's
    Param of an input's tensor, or its view of an output's, and the view of its
    interface's end; for a parameter port,
    the kernel's Param of the held values and its views of the tensor and the values the
    port's channel carries."""

    tensor: str
    end: str
    contents: str | None = None
    carried: str | None = None
    stored: str | None = None

    @property
    def parameter(self) -> bool:
        """Whether it wires a parameter port: its contents and the kernel's views named."""
        return None not in (self.contents, self.carried, self.stored)


@dataclass(frozen=True)
class Match:
    """What a KernelOp's pattern covers: the ONNX ``nodes``, its anchor first, and the
    values of the op's semantic ``attributes`` (``KernelOp.semantic``) read from them."""

    nodes: tuple[NodeProto, ...]
    attributes: Mapping[str, object]


def node_attributes(node: NodeProto) -> dict[str, Any]:
    """A node's attributes, by name, as values."""
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def unstated(owner: str, message: str, **details: object) -> Finding:
    """A finding for a shape or datatype the graph lacks (``fact-unstated``)."""
    return Finding(FindingKind.LIMITATION, "fact-unstated", owner, message, tuple(details.items()))


def refused(owner: str, code: str, message: str, **details: object) -> Finding:
    """A pattern's refusal, by its code."""
    return Finding(FindingKind.REJECTION, code, owner, message, tuple(details.items()))


# -- the op ------------------------------------------------------------------------------


def _is_choice_value(value: object, kind: str, cases: tuple[str, ...]) -> TypeGuard[int | str]:
    """Whether ``value`` is a value of a choice attribute of ONNX type ``kind`` (a
    string for ``"s"``, an integer otherwise), among its ``cases`` if it has any."""
    fits = isinstance(value, str) if kind == "s" else isinstance(value, int)
    return fits and (not cases or value in cases)


class KernelOp(CustomOp):
    """Base class of the KernelOps (see the module docstring). A subclass sets, in its
    own body:

    - ``op_type``, and ``op_version``, its kernel's version;
    - ``kernel``, the kernel class, and ``member``, its name in the node root;
    - ``formals``, the kernel arguments read from the graph (``facts``);
    - ``ports``, the channel name of each ONNX input (None for an input read as an
      argument, such as Thresholding's thresholds), and ``outputs``, of each output;
    - ``wiring``, how each port meets the kernel (``Wiring``, the module docstring);
    - ``parameters``, the ports whose value may be an initializer the node holds
      (MatMul's weights), each wired with its contents and the kernel's views;
    - ``semantic``, its semantic attributes;
    - ``anchor``, the ``(domain, op_type)`` its pattern starts from, and ``match``.

    The node root is generated from these (``root``), and a partition's shell root
    places the kernel by them (``place``).
    """

    wants_model = True
    op_type: ClassVar[str]
    op_version: ClassVar[int]
    kernel: ClassVar[type[Kernel]]
    member: ClassVar[str]
    formals: ClassVar[tuple[str, ...]]
    ports: ClassVar[tuple[str | None, ...]]
    outputs: ClassVar[tuple[str, ...]] = ("y",)
    wiring: ClassVar[Mapping[str, Wiring]]
    parameters: ClassVar[tuple[str, ...]] = ()
    semantic: ClassVar[dict[str, tuple[str, bool, object]]] = {}
    anchor: ClassVar[tuple[str, str]]
    _roots: ClassVar[dict[type[KernelOp], type[Kernel]]] = {}
    _schemas: ClassVar[dict[type[KernelOp], dict[str, tuple[str, tuple[str, ...]]]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "kernel" not in cls.__dict__:
            return  # an abstract base
        version = cls.__dict__.get("op_version")
        if type(version) is not int or version < 1:
            raise TypeError(f"{cls.__qualname__} must state a positive int op_version")
        anchor = getattr(cls, "anchor", None)
        if not (
            isinstance(anchor, tuple)
            and len(anchor) == 2
            and all(isinstance(part, str) for part in anchor)
        ):
            raise TypeError(f"{cls.__qualname__} must state its anchor, (domain, op_type)")
        ports = {port for port in (*cls.ports, *cls.outputs) if port is not None}
        wiring = getattr(cls, "wiring", {})
        if set(wiring) != ports:
            raise TypeError(
                f"{cls.__qualname__} must state the wiring of each port {sorted(ports)}"
            )
        unvalued = [port for port in cls.parameters if not wiring[port].parameter]
        if unvalued:
            raise TypeError(
                f"{cls.__qualname__}: the parameter ports {unvalued} name no contents and views"
            )

    @property
    def label(self) -> str:
        """The node's name, or its op type if it has none."""
        name: str = self.onnx_node.name or self.onnx_node.op_type
        return name

    # -- pattern --------------------------------------------------------------------------

    @classmethod
    def match(cls, model: ModelWrapper, node: NodeProto) -> Match | Rejected:
        """What this op covers starting from ``node``, a node at its ``anchor``, or the
        pattern's findings. Reads structure only; never changes ``model``."""
        raise NotImplementedError

    # -- schema ---------------------------------------------------------------------------

    @classmethod
    def attribute(cls, key: str) -> str | None:
        """The node attribute that stores a node-root key: the key without the kernel's
        member name. None for a channel's key, which is stored on its tensor
        (``CHANNEL``)."""
        head, _, rest = key.partition(".")
        return rest if head == cls.member else None

    @classmethod
    def node_key(cls, attribute: str) -> str:
        """An attribute's node-root key."""
        return f"{cls.member}.{attribute}"

    @classmethod
    def root(cls) -> type[Kernel]:
        """The op's node root class, generated from its placement once per process."""
        if cls not in KernelOp._roots:
            KernelOp._roots[cls] = node_root(cls)
        return KernelOp._roots[cls]

    @classmethod
    def schema(cls) -> dict[str, tuple[str, tuple[str, ...]]]:
        """The choice attributes: name -> (ONNX type, a selector's cases), from the
        decision keys of the kernel in the node root."""
        if cls not in KernelOp._schemas:
            found: dict[str, tuple[str, tuple[str, ...]]] = {}
            for item in inspection.decisions(cls.root()):
                name = cls.attribute(item.key)
                if name is not None:
                    kind = ONNX_TYPES[value_type(item)]
                    found[name] = (kind, item.cases if item.selector else ())
            KernelOp._schemas[cls] = dict(sorted(found.items()))
        return KernelOp._schemas[cls]

    def get_nodeattr_types(self) -> dict[str, tuple[Any, ...]]:
        types: dict[str, tuple[Any, ...]] = dict(self.semantic)
        for name, (kind, cases) in self.schema().items():
            default: object = "" if kind == "s" else 0
            types[name] = (kind, False, default, set(cases)) if cases else (kind, False, default)
        return types

    # -- facts and binding ----------------------------------------------------------------

    def model(self) -> ModelWrapper:
        """The model this op is attached to."""
        model = self._model
        if model is None:
            raise KernelOpError(f"{self.label}: no model attached (use get_customop_wrapper)")
        return model

    def target(self) -> Target:
        """The build target stored in the attached model."""
        try:
            return read_target(self.model())
        except KernelOpError as error:
            raise KernelOpError(f"{self.label}: {error}") from error

    def facts(self) -> Facts:
        """What binding this node reads, from the attached model."""
        raise NotImplementedError

    def input_edges(self) -> dict[str, Tensor]:
        """The tensor each input channel carries, by port (``edge_tensor``)."""
        model, node = self.model(), self.onnx_node
        return {
            port: edge_tensor(model, tensor, self.label)
            for port, tensor in zip(self.ports, node.input)
            if port is not None
        }

    def output_edges(self) -> dict[str, Tensor]:
        """The tensor each output channel carries, by port; known once inference has
        written it."""
        model, node = self.model(), self.onnx_node
        return {
            port: edge_tensor(model, tensor, self.label)
            for port, tensor in zip(self.outputs, node.output)
        }

    def base(self) -> Kernel:
        """The node root bound to this node's facts, with nothing chosen."""
        return BIND_CACHE.point(self.facts())

    def _channel_hint(self, names: Iterable[str]) -> str:
        """A hint for an error that names ``names``: if one is a channel's key (named
        under a port), channel choices are stored on the tensor."""
        ports = {port for port in (*self.ports, *self.outputs) if port}
        if any(name.partition(".")[0] in ports for name in names):
            return f"; a channel's choices are its tensor's ({CHANNEL.name}, save_channels)"
        return ""

    def choices(self) -> dict[str, object]:
        """The node's choices, by attribute: only those present. An attribute neither
        semantic nor a choice is refused."""
        schema = self.schema()
        found: dict[str, object] = {}
        for attribute in self.onnx_node.attribute:
            name = attribute.name
            if name in self.semantic:
                continue
            if name not in schema:
                raise KernelOpError(
                    f"{self.label}: {name} is not a choice of {self.op_type}"
                    + self._channel_hint([name]),
                    (name,),
                )
            found[name] = attribute.s.decode() if schema[name][0] == "s" else int(attribute.i)
        return found

    def point(self, extra: Mapping[str, object] | None = None) -> Kernel:
        """The node's point: its node root bound to its facts, with its stored choices
        (and ``extra``, by attribute) applied, and those stored on the tensors of the
        initializers it holds (``channel_choices``)."""
        return self._replayed({**self.choices(), **(extra or {})})

    def _replayed(self, choices: Mapping[str, object]) -> Kernel:
        """``point``, with ``choices`` (by attribute) in place of the node's."""
        facts = self.facts()
        wanted = {self.node_key(name): value for name, value in choices.items()}
        names = {key: key.partition(".")[2] for key in wanted}
        for port, tensor in self.owned(facts).items():
            for key, value in channel_choices(self.model(), tensor).items():
                wanted[f"{port}.{key}"] = value
                names[f"{port}.{key}"] = f"{tensor}.{key}"
        if not wanted:
            return BIND_CACHE.point(facts)
        typed = typed_choices([self.root()], wanted)

        def build(base: Kernel) -> Kernel:
            replayed = committed(base, typed)
            if isinstance(replayed, dict):
                raise refusal(
                    self.label, {names.get(key, key): why for key, why in replayed.items()}
                )
            return replayed

        return BIND_CACHE.configured(facts, typed, build)

    def save(self, choices: Mapping[str, object | None]) -> None:
        """Store ``choices`` as the node's attributes, over its stored ones, after
        checking them all together on a fresh point; ``None`` clears one. Channel
        choices go through ``shell.save_channels``."""
        schema = self.schema()
        unknown = sorted(set(choices) - set(schema))
        if unknown:
            raise KernelOpError(
                f"{self.label}: {unknown} are not choices of {self.op_type}"
                + self._channel_hint(unknown),
                tuple(unknown),
            )
        merged: dict[str, int | str] = {}
        for name, value in {**self.choices(), **choices}.items():
            if value is None:
                continue
            kind, cases = schema[name]
            if not _is_choice_value(value, kind, cases):
                raise KernelOpError(
                    f"{self.label}: {name} = {value!r} is not an {kind!r} value"
                    + (f" among {list(cases)}" if cases else ""),
                    (name,),
                )
            merged[name] = value
        self._replayed(merged)  # refuses before any write; a cleared choice is not replayed
        for name in schema:
            present = get_by_name(self.onnx_node.attribute, name)
            if present is not None:
                self.onnx_node.attribute.remove(present)
        for name, value in sorted(merged.items()):
            self.set_nodeattr(name, int(value) if isinstance(value, bool) else value)

    # -- in a shell root ------------------------------------------------------------------

    def inputs(self) -> dict[str, str]:
        """The tensor of each of this node's input channels, by port."""
        return {port: tensor for port, tensor in zip(self.ports, self.onnx_node.input) if port}

    def owned(self, facts: Facts) -> dict[str, str]:
        """The tensor of each parameter port whose initializer this node holds
        (``facts``, the node's), by port."""
        return {port: self.onnx_node.input[self.ports.index(port)] for port in facts.owned}

    def place(
        self,
        facts: Facts,
        tensors: Mapping[str, Tensor],
        values: Mapping[str, object],
    ) -> Kernel:
        """This node's kernel in a shell root: its arguments the values in ``facts``, its
        ports' tensors ``tensors`` and its held initializers' values ``values`` (each by
        graph name): the placement its node root is generated from."""
        names = self.inputs() | dict(zip(self.outputs, self.onnx_node.output))
        given = {port: tensors[name] for port, name in names.items()}
        held = {port: values[name] for port, name in self.owned(facts).items()}
        return placed(type(self), facts.formals(), given, held)

    # -- inference ------------------------------------------------------------------------

    def output_tensors(self) -> dict[str, Tensor]:
        """Each output's tensor, by graph name: what the kernel states it writes, bound on
        the node's inputs (each output's ``Wiring.tensor``), one rule for every op. It may
        read an input's value (MatMul's result range comes from its weights)."""
        kernel = getattr(BIND_CACHE.inputs(self.facts()), self.member)
        found: dict[str, Tensor] = {}
        for port, name in zip(self.outputs, self.onnx_node.output):
            answer = kernel.query(getattr(type(kernel), self.wiring[port].tensor))
            if not isinstance(answer, Available):
                raise KernelOpError(f"{self.label}: {describe([answer])}", findings=answer.findings)
            found[name] = answer.value
        return found

    def infer_output_tensors(self, model: ModelWrapper) -> Shapes:
        """Each output's ONNX shape and datatype (``output_tensors``)."""
        self.attach_model(model)
        return {
            name: (tensor.shape, tensor.element.dtype)
            for name, tensor in self.output_tensors().items()
        }

    def make_shape_compatible_op(self, model: ModelWrapper) -> Any:
        ((dims, _),) = self.infer_output_tensors(model).values()
        return self.make_const_shape_op(list(dims))

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        for name, (_, dtype) in self.infer_output_tensors(model).items():
            model.set_tensor_datatype(name, dtype)

    def refusals(self, point: Kernel) -> tuple[Finding, ...]:
        """What stops a kernel from building ``point``, whatever its open choices: the
        kernel's admission as far as it is decided, and each open decision with no
        viable case (``decision-no-viable-case``, why each case was refused as its
        causes). Empty when nothing does."""
        found: list[Finding] = []
        admitted = inspection.admission(getattr(point, self.member))
        if isinstance(admitted, Rejected):
            found.extend(admitted.findings)
        for item in inspection.viable(point):
            if not item.cases:
                detail = "; ".join(f"{case}: {why}" for case, why in item.refused.items())
                found.append(
                    Finding(
                        FindingKind.REJECTION,
                        "decision-no-viable-case",
                        item.key,
                        f"no case is viable: {detail}",
                        causes=candidate_refusals(point, item.key, tuple(item.refused)),
                    )
                )
        return tuple(found)

    def admission(self) -> tuple[Finding, ...]:
        """The kernels' refusals of this node, bound on its inputs only: what conversion
        asks before it converts a node."""
        return self.refusals(BIND_CACHE.inputs(self.facts()))

    def verify_node(self) -> list[str]:
        """Why the node's stored choices are refused, or each of ``refusals`` of its
        point, as lines with owner and code. Empty when accepted."""
        try:
            point = self.point()
        except KernelOpError as error:
            return [str(error)]
        return [
            f"{self.label}: {finding.owner}: {finding.code}: {finding.message}"
            for finding in self.refusals(point)
        ]


def candidate_refusals(point: Space, key: str, cases: tuple[str, ...]) -> tuple[Finding, ...]:
    """Why each of ``cases`` of the selector ``key`` is refused on ``point``: the
    admission of the case's candidate once the case is committed, each finding once. A
    decision over values gives none: its own finding says why."""
    decisions = {item.key: item for item in inspection.decisions(point)}
    if not decisions[key].selector:
        return ()
    selectors = {name for name, item in decisions.items() if item.selector}
    found: list[Finding] = []
    for case in cases:
        report = point.try_with_choices({decisions[key].reference: case})
        if not report.accepted:
            for outcome in report.outcomes:
                if isinstance(outcome.result, Rejected):
                    found.extend(outcome.result.findings)
            continue
        candidate: Any = report.instance
        walked: list[str] = []
        for part in key.split("."):
            # A key names a case after its selector (compute.packed.pe): the selector's
            # member is already that case's candidate.
            if ".".join(walked) not in selectors:
                candidate = getattr(candidate, part)
            walked.append(part)
        admitted = inspection.admission(candidate)
        if isinstance(admitted, Rejected):
            found.extend(admitted.findings)
    return tuple(dict.fromkeys(found))


def kernel_op(model: ModelWrapper, node: NodeProto) -> KernelOp:
    """The KernelOp of ``node``, attached to ``model``; a node of any other op is
    refused."""
    op = model.get_customop_wrapper(node)
    if not isinstance(op, KernelOp):
        raise KernelOpError(f"{node.name or node.op_type}: {node.op_type} is not a KernelOp")
    return op


__all__ = [
    "KernelOp",
    "Match",
    "Wiring",
    "Shapes",
    "kernel_op",
    "node_attributes",
    "refused",
    "unstated",
]
