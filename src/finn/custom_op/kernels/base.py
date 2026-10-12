# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""KernelOp: a qonnx ``CustomOp`` that binds one kernel point.

An op reads its facts from the attached model only: input shapes and datatype
annotations, initializers admitted by their value summary, and the build target
from the model's ``finn.platform`` metadata. An initializer the node owns is a
value its channel carries (``Facts.values``), and the channel's tensor states
its range. It states its placement once, as data (``kernel``, ``formals``,
``references``, ``parameters``; ``finn.custom_op.kernels.roots``): its node
root is generated from it, and a shell root places the same kernel. It
binds its node root through the bind cache, on its inputs for inference and
whole for its choices, replays the choices its node holds and those its owned
parameter channels' tensors state, and answers the compiler's queries from the
result.

Each node is asked two questions, each with one owner:

- **identity**, the op's pattern: the node it starts from (``anchor``: a domain and
  an op type) and ``match``, which reads the graph around that node by its structure
  only (the op types and connectivity, which inputs are facts, the semantic
  attributes, what fixes the axes; never a datatype, a value or the platform) and
  answers with the nodes the op covers and its semantic attributes (``Match``), or
  the Space's ``Rejected`` with the pattern's findings: a refusal by its code, or a
  structural fact it needs that the graph does not state (``fact-unstated``).
  ``match`` never changes the model; ``ToKernelOps`` applies a match;
- **realizability**, the kernels': ``admission`` asks them on the node's inputs
  (``refusals``: the kernel's admission and each Decision with no viable case, by
  why each of its cases is refused), which ``verify_node`` also reports on the
  node's point.

A fact binding reads that the graph does not state raises ``FactUnstated``, which
conversion reports; any other refusal of the facts is a contradiction
(``KernelOpError``).

An op's ``execute_node`` is the computational reference for what its pattern covers
(PRINCIPLES 8), defined for every input domain the ONNX it covers is, and equal to the
ONNX semantics of the covered nodes on every node the op converts. Integers are exact on
both sides: FINN executes ONNX with exact integers (``finn.core.onnx_exec``), as the
hardware computes them. The harness checks the
reference against ONNX, every value equal (``finn.harness.reference``); no build
re-checks it.

Two kinds of attribute:

- **semantic**: part of the operation, stated by the graph (Thresholding's
  ``bias``); required, never a choice;
- **choice**: one per decision key of the op's kernel in its node root,
  unprefixed (``compute.packed.pe``), each at its baseline from conversion on
  (``ToKernelOps``) until a strategy replaces it; absent only on a node no conversion
  created (made by hand) or for a required choice no strategy has set. The ONNX type
  is the key's value semantics' (``int`` and ``bool``: ``i``, ``str``: ``s``),
  and a selector lists its cases. The node's attributes are the persisted form.

A channel's choices are not its nodes': they are stated on the tensor it carries
(``CHANNEL``, the ``finn.channel`` namespace of the tensor's typed metadata), and
only in a partition's body, a model whose nodes are all KernelOps
(``channel_choices``; a shell root replays them, and only a shell root, which holds
both of a channel's ends, checks them: ``shell.save_channels``). A graph with nodes on
the host (the whole graph, before the cut) states none.

Replay commits the node's choices atomically on its cached base point, with the
choices of each parameter channel the node owns, as the initializer's tensor states
them. A choice nested under a selector nobody committed (``compute.packed.pe`` with
``compute`` open) applies when the selector is forced (its one viable case),
and forced cases are never committed, so nothing else happens at replay and
nothing forced reaches a node. A refusal names every refused key, an
inapplicable one too (it carries no finding of its own).
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
from finn.kernels.channels import Channel
from finn.kernels.configure import describe
from finn.kernels.target import Target

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

ONNX_TYPES = {"int": "i", "bool": "i", "str": "s"}

Shapes = dict[str, tuple[tuple[int, ...], QONNXDataType]]


# -- patterns ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    """What a KernelOp's pattern covers: the ONNX ``nodes``, its anchor first, and the
    op's semantic ``attributes`` (``KernelOp.semantic``) as the covered nodes state
    them."""

    nodes: tuple[NodeProto, ...]
    attributes: Mapping[str, object]


def node_attributes(node: NodeProto) -> dict[str, Any]:
    """A node's attributes, by name, as values."""
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def unstated(owner: str, message: str, **details: object) -> Finding:
    """A pattern's finding for a fact it needs that the graph does not state."""
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
    """One ONNX node's binding of one kernel point; see the module docstring.

    An op class states ``op_version`` (its kernel's ``version``) in its own body; its
    ``op_type`` is qonnx's (stated in its own body, or the name its domain exports
    it under). It states its placement, from which its node root is generated
    (``root()``) and by which a partition places it (``place``):

    - ``kernel``, the kernel class it binds, and ``member``, its name in the root;
    - ``formals``, the kernel's formals the op reads from the graph (``facts``);
    - ``ports``, each ONNX input's channel (``None``: an input that is a fact, never
      a channel), and ``outputs``, each ONNX output's;
    - ``references``, the kernel's reference input each port's channel binds;
    - ``parameters``, the ports whose channel may carry a value the node owns (an
      initializer): the channel's ``contents``, which its source stores.

    And its pattern (the module docstring): ``anchor``, the ``(domain, op_type)`` of the
    ONNX node it starts from, and ``match``.
    """

    wants_model = True
    op_type: ClassVar[str]
    op_version: ClassVar[int]
    kernel: ClassVar[type[Kernel]]
    member: ClassVar[str]
    formals: ClassVar[tuple[str, ...]]
    ports: ClassVar[tuple[str | None, ...]]
    outputs: ClassVar[tuple[str, ...]] = ("y",)
    references: ClassVar[Mapping[str, str]]
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

    @property
    def label(self) -> str:
        name: str = self.onnx_node.name or self.onnx_node.op_type
        return name

    # -- pattern --------------------------------------------------------------------------

    @classmethod
    def match(cls, model: ModelWrapper, node: NodeProto) -> Match | Rejected:
        """What this op covers from ``node``, a node at its ``anchor``, or the pattern's
        findings; by the graph's structure only, never a datatype or a value, ``model``
        unchanged (module docstring)."""
        raise NotImplementedError

    # -- schema ---------------------------------------------------------------------------

    @classmethod
    def attribute(cls, key: str) -> str | None:
        """A node-root key's attribute: the kernel's, unprefixed; none for a channel's,
        which its tensor states (``CHANNEL``)."""
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
        model = self._model
        if model is None:
            raise KernelOpError(f"{self.label}: no model attached (use get_customop_wrapper)")
        return model

    def target(self) -> Target:
        try:
            return read_target(self.model())
        except KernelOpError as error:
            raise KernelOpError(f"{self.label}: {error}") from error

    def facts(self) -> Facts:
        """What binding this node reads, from the attached model."""
        raise NotImplementedError

    def input_edges(self) -> dict[str, Tensor]:
        """The tensor of each input channel of the node root, by port, as the graph states
        it (an initializer's over its values' range)."""
        model, node = self.model(), self.onnx_node
        return {
            port: edge_tensor(model, tensor, self.label)
            for port, tensor in zip(self.ports, node.input)
            if port is not None
        }

    def output_edges(self) -> dict[str, Tensor]:
        """The tensor of each output channel of the node root, by port, as the graph
        states it: known once inference has written it."""
        model, node = self.model(), self.onnx_node
        return {
            port: edge_tensor(model, tensor, self.label)
            for port, tensor in zip(self.outputs, node.output)
        }

    def base(self) -> Kernel:
        """The node root bound from this node's facts, nothing chosen."""
        return BIND_CACHE.point(self.facts())

    def _channel_hint(self, names: Iterable[str]) -> str:
        """Where ``names`` are stated, if they are a channel's (one named under a port):
        on its tensor."""
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
        """The node root with the node's choices (and ``extra``, by attribute) replayed,
        and those of each parameter channel the node owns, which the initializer's
        tensor states (``channel_choices``)."""
        return self._replayed({**self.choices(), **(extra or {})})

    def _replayed(self, choices: Mapping[str, object]) -> Kernel:
        """The node root with ``choices`` (by attribute, in place of the node's) replayed,
        and those of each parameter channel the node owns."""
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
        """Commit ``choices`` (made on purpose, the kernel's) over the node's persisted
        ones: every one replayed on a fresh base and written only if accepted; ``None``
        clears a choice. A channel's are its tensor's (``shell.save_channels``)."""
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
        """The tensor of each parameter port whose value this node owns (``facts``, the
        node's), by port: its channel is this node's to declare, from the graph, with the
        value (``Facts.values``) as its contents."""
        return {port: self.onnx_node.input[self.ports.index(port)] for port in facts.owned}

    def place(self, facts: Facts, channels: Mapping[str, Channel]) -> Kernel:
        """This node's kernel, from its ``facts``, on a shell root's ``channels`` (by
        tensor), its formals literals: the placement its node root is generated from."""
        tensors = self.inputs() | dict(zip(self.outputs, self.onnx_node.output))
        on = {port: channels[tensor] for port, tensor in tensors.items()}
        return placed(type(self), facts.formals(), on)

    # -- inference ------------------------------------------------------------------------

    def output_tensors(self) -> Shapes:
        """Each output's ONNX shape and datatype, from the kernel's fact-level views
        (``view``)."""
        raise NotImplementedError

    def infer_output_tensors(self, model: ModelWrapper) -> Shapes:
        self.attach_model(model)
        return self.output_tensors()

    def make_shape_compatible_op(self, model: ModelWrapper) -> Any:
        ((dims, _),) = self.infer_output_tensors(model).values()
        return self.make_const_shape_op(list(dims))

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        for name, (_, dtype) in self.infer_output_tensors(model).items():
            model.set_tensor_datatype(name, dtype)

    def refusals(self, point: Kernel) -> tuple[Finding, ...]:
        """What refuses ``point``, a binding of this node's root, whatever its open choices
        are: the kernel's ``admission`` as far as it is decided (a constraint that refuses
        on the facts alone, while another waits on a choice), and each open Decision of the
        root with no viable case (``decision-no-viable-case``), the refusal of each of its
        candidates as the finding's causes. Empty when nothing refuses it."""
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
        """The kernels' refusals of this node on its inputs (``refusals`` of its root bound
        on them), before any output is known: what conversion asks of a candidate. A fact
        the graph does not state raises ``FactUnstated``, a contradiction
        ``KernelOpError``."""
        return self.refusals(BIND_CACHE.inputs(self.facts()))

    def verify_node(self) -> list[str]:
        """Replay's refusal, or each of ``refusals`` of the node's point, with its owner
        and code. None when accepted."""
        try:
            point = self.point()
        except KernelOpError as error:
            return [str(error)]
        return [
            f"{self.label}: {finding.owner}: {finding.code}: {finding.message}"
            for finding in self.refusals(point)
        ]

    def view(self, name: str) -> Any:
        """A fact-level view of the op's kernel in its node root bound on its inputs
        (``result_tensor``, ``result_dtype``): what inference reads, before any output of
        the node is known. It reads the kernel's facts and its input channels' tensors and
        values (MatMul's result range, its weight channel's value)."""
        kernel = getattr(BIND_CACHE.inputs(self.facts()), self.member)
        answer = kernel.query(getattr(type(kernel), name))
        if not isinstance(answer, Available):
            raise KernelOpError(f"{self.label}: {describe([answer])}", findings=answer.findings)
        return answer.value


def candidate_refusals(point: Space, key: str, cases: tuple[str, ...]) -> tuple[Finding, ...]:
    """Why each of ``cases`` of the Decision over nodes ``key`` is refused on ``point``:
    its candidate's admission once the case is committed, each finding once. A Decision
    over values (its cases are values, not candidates) gives none: its own finding says
    why."""
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
    "Shapes",
    "kernel_op",
    "node_attributes",
    "refused",
    "unstated",
]
