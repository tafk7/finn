# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""KernelOp: an ONNX node tied to a design space of kernels.

A KernelOp is a qonnx ``CustomOp`` whose node stands for a kernel of ``finn.kernels``
with its choices still open. It does not own the hardware of the graph: the channels
between nodes belong to the tensors they carry, and a selection of nodes is bound as a
network (``network.network``), each node's kernel on channels the nodes share.

Three terms, used throughout the package:

- the node's *facts*: what it reads from the model it is attached to, which are its
  kernel's formals, its input tensors as values (``edge_tensor``: each at the graph's
  rank, a tensor stated once), the values of the initializers it holds (weights,
  thresholds) and the build target (``read_target``);
- its *kernel*: its kernel class bound to its facts, nothing chosen (``bound``);
- its *point*: its kernel with the node's stored choices committed (``point``).

Each is bound from the node's facts as it is asked for (``finn.kernels.composer.Facts``);
nothing is kept between asks.

Conversion asks two questions of a node. Does the op's pattern cover it? ``match`` reads
only the graph's structure (op types, connections, attributes), never a datatype or a
value, and never changes the model. Will a kernel build it? ``defaulted`` applies the
op's policies (``policies``, ``finn.kernels.policies``) once, on the kernel alone:
refused where its facts alone refuse it, or where a choice in play can be given no value,
its fixed one or a first valid one. Then each of its streams (``streamed``) must meet an
open AXIS boundary (``finn.transformation.kernels.convert``): a check, by a channel's own
rules, whose channels are dropped; a partition's channels are made at the cut. A missing
shape or datatype raises ``FactUnstated``, and the node stays on the host; anything else
wrong with the facts is an error (``KernelOpError``). Conversion stores the kernel's
choices only, and writes nothing on a tensor.

``execute_node`` is the reference for what the op covers, and must equal ONNX on every
node the op converts (FINN executes ONNX with exact integers, which importing ``finn``
turns on).
The test harness checks this (``finn.harness.reference``); builds do not.

A node's attributes are its semantic attributes, which are part of the operation
(Thresholding's ``bias``), and its kernel's choices, named by the kernel's own decision
keys (``pe``), each set by the op's policy at conversion (a required
one is left open, for a strategy to set). A channel's
choices are stored on its tensor instead, and only in a partition's body (``CHANNEL``).
Only the compiler writes a choice (``persist``): ``get_nodeattr`` of a choice answers what
the node stores or refuses it as absent, and ``set_nodeattr`` of one is refused.

**Who calls what.** One writer, the readers beside it, and a guard on the door:

- *Writes*, all through ``persist`` (``network.persist``, which writes past the guard) and,
  for conversion, ``network.persist_kernel``: conversion writes the node's kernel as its
  policies leave it (``defaulted``), only the node's choices. The cut persists the body's
  choices, the channels' by the channel's policies (``shell.make_cut_choices``). Every
  strategy persists the network it returns (``explore_kernel_choices``). A user sets a
  choice by ``Hold`` in a step after the cut, which the strategies apply: no one writes an
  attribute.
- *Reads*: ``choices`` (the attributes the node holds), ``stored`` (as the kernel's
  Decisions take them), ``point`` (the kernel with them committed), ``verify_node`` and
  every bind of a network. They never write: a stored choice the facts refuse is
  refused by name (``point``, ``verify_node``) or dropped as stale where a network is
  bound (``Partition.stale``).
- *The guard*: ``get_nodeattr`` and ``set_nodeattr`` of a KernelOp, the way qonnx's
  transformations reach an attribute. They guard the choice keys only: reading an absent
  choice is refused (``choice-absent``), setting one is refused (``choice-set-refused``),
  each naming the node and the key, and the second the route, ``Hold``. Any other
  attribute is qonnx's.

Its outputs are its kernel's statement, one rule for every op: the kernel bound on its
inputs states each output's tensor (``output_tensors``), its shape and its element, a
datatype over a range (MatMul's results over their exact range), a value of its facts
read whatever the kernel's validity (a refused kernel is ``verify_node``'s to report),
which inference writes on the graph, the range with the datatype
(``finn.core.graph.facts.state``). The graph's annotation of an output is the record
of that statement: a network that finds the two differ refuses the graph as stale
(``network.network``, ``tensor-stale``). The next node reads it as its input's tensor
(``edge_tensor``), its range included.

**Wiring.** Each ONNX input and output that carries a stream (a *port*: ``x``, ``w``,
``y``) meets the kernel as a ``Wiring``: an input's the kernel's Param its tensor is
given as, an output's the kernel's derived value of the tensor it writes (``tensor``,
``x_tensor``, ``result_tensor``: an output's tensor is the kernel's statement, never
given to it), and the view of the contract its channel takes of the kernel's port
(``contract``, ``x_contract``, a view of a ``finn.kernels.base.PortContract``). A parameter port,
whose initializer the node may hold (MatMul's weights), also names the kernel's Param
the held values are given as, in place of its tensor (``contents``; MatMul's ``weights``,
the same Param, known weights or a tensor) and the kernel's derived values of what its
channel carries, downstream of the kernel: the tensor (``carried``,
``carried_weights``) and the values its source stores (``stored``,
``stored_weights``). The graph layer declares the kernel and its channels by it
(``finn.custom_op.kernels.network.declaration``): each channel's tensor (a parameter
channel's from the kernel's values, ``Stated``) and its port as the channel's side
(``Port``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, TypeGuard, cast

import numpy.typing as npt
from onnx import helper
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import get_by_name

from finn.core.graph.annotations.channel import CHANNEL, value_type
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.facts import edge_tensor, initializer, state
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import (
    Available,
    ConfigurationError,
    Rejected,
    RequestError,
    Space,
    Unresolved,
    bind,
    inspection,
)
from finn.dataflow.tensor import Tensor
from finn.kernels.base import KernelSpace
from finn.kernels.composer import Facts
from finn.kernels.configure import describe
from finn.kernels.configure import refusal as refusal_of
from finn.kernels.policies import Alone, Defaulted, Policy, defaulted, stated
from finn.kernels.port import port_surface
from finn.kernels.sides import Side
from finn.platform.target import Target

if TYPE_CHECKING:
    from onnx import AttributeProto, NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

ONNX_TYPES = {"int": "i", "bool": "i", "str": "s"}

#: What qonnx's ``get_nodeattr`` returns and ``set_nodeattr`` takes.
_NodeAttr = int | float | str | bool | npt.NDArray[Any] | list[str | int | float] | None


def _decoded(attribute: AttributeProto, kind: str) -> int | str:
    """A choice attribute's value: its string, or its integer."""
    return attribute.s.decode() if kind == "s" else int(attribute.i)


# -- patterns ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Wiring:
    """How one port of a KernelOp meets its kernel (the module docstring): the kernel's
    Param of an input's tensor, or its derived value of an output's, and the view of its
    port's contract; for a parameter port, the kernel's Param of the held values and its
    derived values of the tensor and the values the port's channel carries."""

    tensor: str
    contract: str
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


def typed(space_type: type[Space], choices: Mapping[str, object]) -> dict[str, object]:
    """``choices``, by decision key of ``space_type``, as its Decisions take them: an ONNX
    ``i`` attribute of a ``bool`` decision becomes a bool."""
    if not choices:
        return {}
    bools = {item.key for item in inspection.decisions(space_type) if value_type(item) == "bool"}
    return {key: bool(value) if key in bools else value for key, value in choices.items()}


def why(findings: Iterable[Finding]) -> str:
    """Why a stored choice is refused, as text: ``inapplicable``, or each finding's
    owner, code and message."""
    found = tuple(findings)
    if found and all(finding.code == "inapplicable" for finding in found):
        return "inapplicable"
    return "; ".join(map(str, found))


def reasons(
    found: Mapping[str, Iterable[Finding]], standing: Mapping[str, Iterable[str]] | None = None
) -> str:
    """Why each key of ``found`` (a choice's or a rule's) is refused, as text, in key order:
    the key once, then each of its findings (its code and message where the key owns it),
    and the stored choices ``standing`` in its way where any do."""
    standing = standing or {}

    def text(key: str, finding: Finding) -> str:
        return f"{finding.code}: {finding.message}" if finding.owner == key else str(finding)

    def stood(key: str) -> str:
        keys = tuple(standing.get(key, ()))
        return f" (standing: {', '.join(keys)})" if keys else ""

    return "; ".join(
        f"{key}: " + "; ".join(text(key, finding) for finding in found[key]) + stood(key)
        for key in sorted(found)
    )


def replayed(
    point: Space, stored: Mapping[str, object]
) -> tuple[Space, dict[str, tuple[Finding, ...]]]:
    """``point``, a kernel bound with nothing stored, with its ``stored`` choices bound
    (the engine's ``bind``), and what refuses them, by key, each with its findings: a key
    that names no choice (``not-a-decision``), a value outside its domain over the kernel's
    facts, a request the engine refuses as malformed (``request-refused``, each key), and
    each rule the values set together break, by the rule's key (nothing stands in a rule's
    way at a bind, which sets every stored choice)."""
    if not stored:
        return point, {}
    every = inspection.decisions(inspection.model(point))
    decisions = {item.key: item.reference for item in every}
    refused: dict[str, tuple[Finding, ...]] = {
        key: (
            Finding(FindingKind.REJECTION, "not-a-decision", key, "not a Decision of this member"),
        )
        for key in stored
        if key not in decisions
    }
    if refused:
        return point, refused
    try:
        report = bind(point, {decisions[key]: value for key, value in stored.items()})
    except (RequestError, ConfigurationError) as error:
        return point, {
            key: (Finding(FindingKind.REJECTION, "request-refused", key, str(error)),)
            for key in stored
        }
    for outcome in report.outcomes:
        if outcome.status == "dropped":
            result = outcome.result
            refused[outcome.owner] = (
                result.findings if isinstance(result, (Rejected, Unresolved)) else ()
            )
    refused |= {rule.rule: rule.result.findings for rule in report.broken}
    return report.instance, refused


def refusal(
    owner: str,
    label: str,
    found: Mapping[str, tuple[Finding, ...]],
    standing: Mapping[str, tuple[str, ...]] | None = None,
    *,
    node: str | None = None,
    tensor: str | None = None,
) -> KernelOpError:
    """The error for choices replay refused (``choices-refused``, by ``owner``): each
    refused choice or broken rule by its key, once, with its findings and the stored
    choices standing in its way (``reasons``); every such finding, each once, about the
    ``node`` or ``tensor``, its causes."""
    subjects = {name: value for name, value in (("node", node), ("tensor", tensor)) if value}
    message = f"{label}: refused choices: {reasons(found, standing)}"
    keys = tuple(sorted(found))
    causes = tuple(
        dict.fromkeys(finding.about(**subjects) for key in keys for finding in found[key])
    )
    details = (("keys", keys), *subjects.items())
    return KernelOpError(
        message,
        keys,
        findings=(
            Finding(FindingKind.REJECTION, "choices-refused", owner, message, details, causes),
        ),
    )


# -- the op ------------------------------------------------------------------------------


def _is_choice_value(value: object, kind: str, cases: tuple[str, ...]) -> TypeGuard[int | str]:
    """Whether ``value`` is a value of a choice attribute of ONNX type ``kind`` (a
    string for ``"s"``, an integer otherwise), among its ``cases`` if it has any."""
    fits = isinstance(value, str) if kind == "s" else isinstance(value, int)
    return fits and (not cases or value in cases)


class KernelOp(CustomOp):
    """Base class of the KernelOps (see the module docstring). A subclass sets, in its
    own body:

    - ``op_type``, and ``op_version``, the version of its schema (its own, no kernel's);
    - ``kernel``, the kernel class;
    - ``facts``, the method that reads the kernel's arguments from the graph (each op
      names them where it builds them) and binds the node's kernel on them beside its
      tensors (``binding``);
    - ``ports``, the channel name of each ONNX input (None for an input read as an
      argument, such as Thresholding's thresholds), and ``outputs``, of each output;
    - ``wiring``, how each port meets the kernel (``Wiring``, the module docstring);
    - ``parameters``, the ports whose value may be an initializer the node holds
      (MatMul's weights), each wired with its contents and the kernel's views;
    - ``semantic``, its semantic attributes;
    - ``anchor``, the ``(domain, op_type)`` its pattern starts from, and ``match``;
    - ``policies``, its kernel's choices' policies (``finn.kernels.policies``): class-level
      Decision references (``MemStreamKernel.ram_style``) to ``Fixed``, ``FirstValid`` or
      ``Required``, each applying to every placement of that class below its kernel;
      checked when the class is made (``stated``). Empty by default: a choice it does not
      list is first valid, in listed order. Two ops of one kernel class (MatMul and Conv)
      that state one state it once, on a shared base.

    Its choice attributes are its kernel class's decision keys (``schema``); a network
    binds its kernel by its facts and wires its channels by its ``wiring``.
    """

    wants_model = True
    op_type: ClassVar[str]
    op_version: ClassVar[int]
    kernel: ClassVar[type[KernelSpace]]
    ports: ClassVar[tuple[str | None, ...]]
    outputs: ClassVar[tuple[str, ...]] = ("y",)
    wiring: ClassVar[Mapping[str, Wiring]]
    parameters: ClassVar[tuple[str, ...]] = ()
    semantic: ClassVar[dict[str, tuple[str, bool, object]]] = {}
    anchor: ClassVar[tuple[str, str]]
    policies: ClassVar[Mapping[object, Policy]] = MappingProxyType({})

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        kernel = getattr(cls, "kernel", None)
        if kernel is not None and ("kernel" in cls.__dict__ or "policies" in cls.__dict__):
            cls.policies = stated(cls.__qualname__, kernel, cls.policies)
        if "kernel" not in cls.__dict__:
            return  # an abstract base, or one changing only its policies
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
    def schema(cls) -> dict[str, tuple[str, tuple[str, ...]]]:
        """The choice attributes: name -> (ONNX type, a selector's cases), its kernel
        class's decision keys, in the flow order (``inspection.decisions``)."""
        return {
            item.key: (ONNX_TYPES[value_type(item)], item.cases if item.selector else ())
            for item in inspection.decisions(cls.kernel)
        }

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
            raise KernelOpError.refused(
                "model-unattached",
                self.op_type,
                f"{self.label}: no model attached (get the op with kernel_op(model, node))",
                node=self.label,
            )
        return model

    def target(self) -> Target:
        """The build target stored in the attached model."""
        try:
            return read_target(self.model())
        except KernelOpError as error:
            raise KernelOpError(
                f"{self.label}: {error}",
                findings=tuple(finding.about(node=self.label) for finding in error.findings),
            ) from error

    def facts(self) -> Facts:
        """What binding this node's kernel reads, from the attached model (``binding``)."""
        raise NotImplementedError

    def binding(
        self,
        formals: Mapping[str, object],
        values: Mapping[str, object] | None = None,
    ) -> Facts:
        """The node's facts: its kernel class and its arguments, the ``formals`` it reads
        from the graph beside its tensors, each input port's tensor (``edge_tensor``), and
        the values it makes of the initializers it holds, by port (``owned``; MatMul's
        ``KnownWeights``), given as the kernel's contents in place of a tensor."""
        held = values or {}
        tensors = {
            self.wiring[port].tensor: tensor for port, tensor in self.input_edges(skip=held).items()
        }
        contents = {cast(str, self.wiring[port].contents): each for port, each in held.items()}
        return Facts(type(self).kernel, {**formals, **tensors, **contents})

    def input_edges(self, skip: Iterable[str] = ()) -> dict[str, Tensor]:
        """The tensor of each input port, by port (``edge_tensor``), but those of ``skip``."""
        model, node, skipped = self.model(), self.onnx_node, set(skip)
        return {
            port: edge_tensor(model, tensor, self.label)
            for port, tensor in zip(self.ports, node.input)
            if port is not None and port not in skipped
        }

    def bound(self) -> KernelSpace:
        """The node's kernel bound to its facts, nothing chosen: its own design space."""
        point: KernelSpace = self.facts().bind()  # type: ignore[assignment]
        return point

    def _channel_hint(self, names: Iterable[str]) -> str:
        """A hint for an error that names ``names``: if one is a channel's key (named
        under a port), channel choices are stored on the tensor."""
        ports = {port for port in (*self.ports, *self.outputs) if port}
        if any(name.partition(".")[0] in ports for name in names):
            return (
                f"; a channel's choices are its tensor's ({CHANNEL.name}): set them in a step "
                "after the cut, with Hold"
            )
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
                raise KernelOpError.refused(
                    "choice-unknown",
                    self.op_type,
                    f"{self.label}: {name} is not a choice of {self.op_type}"
                    + self._channel_hint([name]),
                    (name,),
                    node=self.label,
                )
            found[name] = _decoded(attribute, schema[name][0])
        return found

    def get_nodeattr(self, name: str) -> _NodeAttr:
        """The attribute ``name``: a choice as the node stores it (as ``choices`` decodes
        it), refused (``choice-absent``) where the node holds none, never a made-up 0 or
        "". A choice is not stored when it is open (no policy or strategy has set it), or
        its scope is absent (under a case not selected) and nothing set it there; the
        configuration is read from ``point()``. Any other name is qonnx's."""
        schema = self.schema()
        if name not in schema:
            return super().get_nodeattr(name)
        held = get_by_name(self.onnx_node.attribute, name)
        if held is None:
            raise KernelOpError.refused(
                "choice-absent",
                self.op_type,
                f"{self.label}: {name} is not stored: it is open, or under a case not selected "
                "and never set; read the op's configuration from KernelOp.point()",
                (name,),
                node=self.label,
            )
        return _decoded(held, schema[name][0])

    def set_nodeattr(self, name: str, value: _NodeAttr) -> None:
        """Set the attribute ``name``: a choice is refused (``choice-set-refused``), the
        compiler sets it (``persist`` writes the node itself); any other name is qonnx's."""
        if name in self.schema():
            raise KernelOpError.refused(
                "choice-set-refused",
                self.op_type,
                f"{self.label}: {name} is a choice, set by the compiler; to fix one, set it "
                "in a step after the cut, with Hold",
                (name,),
                node=self.label,
            )
        super().set_nodeattr(name, value)

    def stored(self) -> dict[str, object]:
        """The node's choices (``choices``), as its kernel's Decisions take them."""
        return typed(type(self).kernel, self.choices())

    def point(self, extra: Mapping[str, object] | None = None) -> KernelSpace:
        """The node's point: its kernel bound to its facts, with its stored choices (and
        ``extra``, by attribute) committed together; refused, each choice by name."""
        return self._replayed({**self.choices(), **(extra or {})})

    def _replayed(self, choices: Mapping[str, object]) -> KernelSpace:
        """``point``, with ``choices`` (by attribute) in place of the node's: bound on its
        kernel (``replayed``), as a partition binds its stored choices; refused where a
        value is outside its domain, or a rule they break together, each by name with its
        findings (``refusal``)."""
        found, refused = replayed(self.bound(), typed(type(self).kernel, choices))
        if refused:
            raise refusal(self.op_type, self.label, refused, node=self.label)
        point: KernelSpace = found  # type: ignore[assignment]
        return point

    def _saved(self, choices: Mapping[str, object]) -> dict[str, int | str]:
        """What ``persist`` stores on this node, its whole: ``choices`` (by attribute),
        the committed Decisions of the node's kernel in the network it persists. ``persist``
        alone calls it (a route to a choice is ``Hold`` in a step after the cut, ``persist``
        for the compiler, ``support.configure.configured`` for a test). Each key is checked
        to be one of the schema's and each value for its attribute's type and encoding, so a
        write cannot fail halfway; no replay, the network has bound them. Writes nothing."""
        schema = self.schema()
        unknown = sorted(set(choices) - set(schema))
        if unknown:
            raise KernelOpError.refused(
                "choice-unknown",
                self.op_type,
                f"{self.label}: {unknown} are not choices of {self.op_type}"
                + self._channel_hint(unknown),
                tuple(unknown),
                node=self.label,
            )
        merged: dict[str, int | str] = {}
        for name, value in choices.items():
            kind, cases = schema[name]
            if not _is_choice_value(value, kind, cases):
                raise KernelOpError.refused(
                    "choice-value-invalid",
                    self.op_type,
                    f"{self.label}: {name} = {value!r} is not an {kind!r} value"
                    + (f" among {list(cases)}" if cases else ""),
                    (name,),
                    node=self.label,
                )
            merged[name] = value
        return merged

    def _write(self, merged: Mapping[str, int | str]) -> None:
        """Store ``merged`` (``_saved``'s) as the node's choice attributes, its whole;
        ``persist``'s alone. It writes the NodeProto itself, past the guard of
        ``set_nodeattr``."""
        for name in self.schema():
            present = get_by_name(self.onnx_node.attribute, name)
            if present is not None:
                self.onnx_node.attribute.remove(present)
        for name, value in sorted(merged.items()):
            self.onnx_node.attribute.append(
                helper.make_attribute(name, int(value) if isinstance(value, bool) else value)
            )

    # -- in a network ---------------------------------------------------------------------

    def inputs(self) -> dict[str, str]:
        """The tensor of each of this node's input channels, by port."""
        return {port: tensor for port, tensor in zip(self.ports, self.onnx_node.input) if port}

    def streamed(
        self, point: KernelSpace
    ) -> tuple[tuple[str, Side | tuple[Finding, ...], bool], ...]:
        """Each streamed port of this node at ``point``, its kernel's design space: the
        tensor it carries, by graph name, the side it gives a channel (its kernel's
        ``<p>_side``, labelled below the node's name) or, where ``point`` does not know that
        side (it waits on a choice, or its kernel refuses it), the side's findings, and
        whether it is an output; its inputs first, in port order. A streamed port is an AXIS
        one: not a parameter port whose initializer the node holds (its channel's source
        feeds it), nor an input read as an argument. A port ``point`` does not place is
        none."""
        model, node = self.model(), self.onnx_node
        held = set(self.owned())
        found = [
            (port, tensor, False)
            for port, tensor in self.inputs().items()
            if port not in held and initializer(model, tensor) is None
        ]
        found += [(port, tensor, True) for port, tensor in zip(self.outputs, node.output)]
        sides: list[tuple[str, Side | tuple[Finding, ...], bool]] = []
        for port, tensor, output in found:
            name = self.wiring[port].contract.removesuffix("_contract")
            reference = port_surface(type(point), name).side
            assert reference is not None
            answer = point.query(reference)
            if isinstance(answer, Available) and isinstance(answer.value, Side):
                sides.append((tensor, answer.value.under(self.label), output))
            elif isinstance(answer, (Rejected, Unresolved)):
                sides.append((tensor, answer.findings, output))
        return tuple(sides)

    def owned(self) -> dict[str, str]:
        """The tensor of each parameter port whose initializer this node holds, by port."""
        model, node = self.model(), self.onnx_node
        found = {port: node.input[self.ports.index(port)] for port in self.parameters}
        return {
            port: tensor for port, tensor in found.items() if initializer(model, tensor) is not None
        }

    # -- inference ------------------------------------------------------------------------

    def output_tensors(self) -> dict[str, Tensor]:
        """Each output's tensor, by graph name: what the kernel bound on the node's inputs
        states it writes (each output's ``Wiring.tensor``), one rule for every op: a value
        of its facts, whatever the kernel's validity (``verify_node`` reports a refused
        kernel). It may read an input's value (MatMul's result range comes from its
        weights)."""
        kernel = self.bound()
        found: dict[str, Tensor] = {}
        for port, name in zip(self.outputs, self.onnx_node.output):
            answer = kernel.query(getattr(type(kernel), self.wiring[port].tensor))
            if not isinstance(answer, Available):
                raise KernelOpError(
                    f"{self.label}: {describe([answer])}",
                    findings=tuple(finding.about(node=self.label) for finding in answer.findings),
                )
            found[name] = answer.value
        return found

    def infer_output_tensors(self, model: ModelWrapper) -> dict[str, Tensor]:
        """Each output's tensor, by graph name (``output_tensors``), on ``model``."""
        self.attach_model(model)
        return self.output_tensors()

    def make_shape_compatible_op(self, model: ModelWrapper) -> Any:
        """The stand-in qonnx's ``InferShapes`` asks a custom op for: a constant of the
        output's shape. It assumes one output: an op that lengthens ``outputs`` fails at
        the unpacking and must override it. FINN's own inference (``infer_node``) does
        not call it for a KernelOp."""
        (tensor,) = self.infer_output_tensors(model).values()
        return self.make_const_shape_op(list(tensor.shape))

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Each output's datatype and range, as its kernel states them (``state``)."""
        for name, tensor in self.infer_output_tensors(model).items():
            state(model, name, tensor.element)

    def refusals(self, point: KernelSpace) -> tuple[Finding, ...]:
        """What stops its kernel from building ``point``: its validity's refusal, the
        findings of its false pieces (``finn.kernels.configure.refusal``). Empty where it is
        valid or open."""
        return refusal_of(inspection.validity(point))

    def defaulted(self) -> Defaulted[KernelSpace] | tuple[Finding, ...]:
        """Conversion's answer (the module docstring): the node's kernel bound to its facts
        with the op's policies applied once (``finn.kernels.policies.defaulted``), every
        choice in play given its fixed or first valid value, a required one left open, and
        each ranking a policy could not read named (``Defaulted.fallbacks``); else why not:
        its kernel's refusal on its facts alone, or each choice no value can be set for
        (``choice-no-valid-value``, one per choice, owned by its key, each value's refusal
        among its causes). Conversion writes the point it gets here on the node
        (``persist_kernel``)."""
        done = defaulted(self.bound(), Alone(type(self)))
        if done.refused:
            return done.refused[""]
        return done.unset or done

    def verify_node(self) -> list[str]:
        """Why the node's stored choices are refused, or each of ``refusals`` of its
        point, as lines with owner and code. Empty when accepted."""
        try:
            point = self.point()
        except KernelOpError as error:
            return [str(error)]
        return [f"{self.label}: {finding}" for finding in self.refusals(point)]


def kernel_op(model: ModelWrapper, node: NodeProto) -> KernelOp:
    """The KernelOp of ``node``, attached to ``model``; a node of any other op is
    refused."""
    op = model.get_customop_wrapper(node)
    if not isinstance(op, KernelOp):
        raise KernelOpError.refused(
            "not-a-kernel-op",
            "kernel_op",
            f"{node.name or node.op_type}: {node.op_type} is not a KernelOp",
            node=node.name or node.op_type,
        )
    return op


__all__ = [
    "KernelOp",
    "Match",
    "Wiring",
    "kernel_op",
    "node_attributes",
    "reasons",
    "refusal",
    "refused",
    "replayed",
    "typed",
    "unstated",
    "why",
]
