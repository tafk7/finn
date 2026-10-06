# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""KernelOp: a qonnx ``CustomOp`` that binds one kernel point.

An op reads its facts from the attached model only: input shapes and datatype
annotations, initializers admitted by their value summary, and the build target
from the model's ``finn.platform`` metadata. It states its placement once, as
data (``kernel``, ``formals``, ``references``, ``parameters``;
``finn.custom_op.kernels.roots``): its node root is generated from it, and a
partition root places the same kernel. It binds through the bind cache, its
kernel alone for inference and its node root for its choices, replays the
choices its node holds, and answers the compiler's queries from the result.

Two kinds of attribute:

- **semantic**: part of the operation, stated by the graph (Thresholding's
  ``bias``); required, never a choice;
- **choice**: one per decision key of the op's node root, sparse, absent
  meaning open. The kernel's keys are unprefixed (``compute.packed.pe``), a
  channel's sit under the op's port name (``x.adapter…``, ``w.transport``,
  ``y.transport``). A channel's choices are its consumer's; an output
  channel's are its producer's only where no KernelOp consumes it (a graph
  output), so ``y.*`` is set on a node whose output leaves the graph. The ONNX type
  is the key's value semantics' (``int`` and ``bool``: ``i``, ``str``: ``s``),
  and a selector lists its cases. The node's attributes are the persisted form.

Replay commits the node's choices atomically on its cached base point. A choice
nested under a selector nobody committed (``compute.packed.pe`` with
``compute`` open) applies when the selector is forced (its one viable case),
and forced cases are never committed, so nothing else happens at replay and
nothing forced reaches a node. A refusal names every refused key, an
inapplicable one too (it carries no finding of its own).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import fields
from math import prod
from typing import TYPE_CHECKING, Any, ClassVar, TypeGuard, TypeVar

from qonnx.analysis.tensor_value_summary import (
    UnsupportedTensorValueError,
    initializer_value_summary,
)
from qonnx.core.metadata import Key, MetadataError, Namespace
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import get_by_name

from finn.core.space import Available, Inapplicable, Rejected, Space, inspection
from finn.custom_op.kernels.cache import BIND_CACHE, Facts
from finn.custom_op.kernels.roots import node_root, placed
from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    canonical_qonnx_datatype,
    ordinary_integer_bounds,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.configure import describe
from finn.kernels.target import DspBlock, Platform, Target

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

S = TypeVar("S", bound=Space)

PLATFORM = Namespace("finn.platform", version=1, inherit=True)
"""The build target, typed graph metadata (qonnx's ``qonnx.core.metadata``): the part
and the platform (``finn.kernels.target.Platform``: the clock period and the
capabilities), every key stated. It inherits: a subgraph body reads its parent's."""

PLATFORM_KEYS: dict[str, Key[Any]] = dict(
    part=PLATFORM.key("part", str),
    period_ns=PLATFORM.key("period_ns", float, check=lambda v: v > 0, expect="a period > 0"),
    dsp=PLATFORM.key("dsp", DspBlock),
    uram=PLATFORM.key("uram", bool),
    uram_init=PLATFORM.key("uram_init", bool),
    clk2x=PLATFORM.key("clk2x", bool),
    control_ports=PLATFORM.key("control_ports", int, check=lambda v: v >= 0, expect="a count"),
    memory_ports=PLATFORM.key("memory_ports", int, check=lambda v: v >= 0, expect="a count"),
    aie=PLATFORM.key("aie", bool),
)

PLATFORM_FIELDS = tuple(field.name for field in fields(Platform))
"""The ``finn.platform`` keys that are ``Platform``'s fields: all but the part."""

ONNX_TYPES = {"int": "i", "bool": "i", "str": "s"}

Shapes = dict[str, tuple[tuple[int, ...], QONNXDataType]]


class KernelOpError(ValueError):
    """A node an op cannot bind or configure: a missing or refused fact, or a refused
    choice. ``keys`` names the refused choices, by attribute, when there are any."""

    def __init__(self, message: str, keys: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.keys = keys


# -- facts -------------------------------------------------------------------------------


def datatype(model: ModelWrapper, tensor: str, label: str) -> QONNXDataType:
    """The annotation of ``tensor``; an unannotated tensor is refused (qonnx reads it
    as its container type, FLOAT32, which is not a statement)."""
    if not model.has_tensor_datatype(tensor):
        produced = model.find_producer(tensor) is not None
        raise KernelOpError(
            f"{label}: {tensor} has no datatype annotation (absent is not FLOAT32"
            + ("; InferKernelTensors states each node's outputs in order)" if produced else ")")
        )
    return canonical_qonnx_datatype(model.get_tensor_datatype(tensor))


def shape(model: ModelWrapper, tensor: str, label: str) -> tuple[int, ...]:
    """The shape of ``tensor``; one not known yet is refused."""
    found = model.get_tensor_shape(tensor)
    if not found or any(type(dim) is not int or dim < 1 for dim in found):
        raise KernelOpError(f"{label}: {tensor} has no shape yet (run InferKernelTensors)")
    return tuple(found)


def rows(dims: tuple[int, ...]) -> tuple[int, int]:
    """Leading axes are rows: (1, M, K) is M rows of K."""
    return prod(dims[:-1]), dims[-1]


def edge_tensor(model: ModelWrapper, tensor: str, label: str) -> Tensor:
    """The tensor an edge's channel carries, as the graph states it: its shape as
    rows and its annotation."""
    return Tensor(rows(shape(model, tensor, label)), ScalarEncoding(datatype(model, tensor, label)))


def read_target(model: ModelWrapper) -> Target:
    """The build target, from the model's ``finn.platform`` metadata (a subgraph body
    opened through its parent reads the parent's).

    The one reader of the target; ``write_target`` is the one writer. A key missing
    or malformed is refused.
    """
    try:
        stated = model.namespace(PLATFORM)
    except MetadataError as error:
        raise KernelOpError(f"the model's target is not one: {error}") from error
    missing = [name for name in PLATFORM_KEYS if name not in stated]
    if missing:
        raise KernelOpError(
            f"the model states no target (finn.platform: {', '.join(missing)} missing; "
            "run ToKernelOps)"
        )
    platform = Platform(**{name: stated[name] for name in PLATFORM_FIELDS})
    return Target(stated["part"], platform)


def write_target(model: ModelWrapper, target: Target) -> None:
    """State the build target in the model's ``finn.platform`` metadata, every key,
    where ``read_target`` reads it."""
    platform = target.platform
    if platform.dsp is None:
        raise KernelOpError(
            "a target states its DSP block (finn.transformation.kernels.resolve_target)"
        )
    values: dict[str, object] = dict(
        part=target.part, **{name: getattr(platform, name) for name in PLATFORM_FIELDS}
    )
    try:
        for name, key in PLATFORM_KEYS.items():
            key.encode(values[name])  # every key checked before any is written
    except MetadataError as error:
        raise KernelOpError(f"the target is not one: {error}") from error
    for name, key in PLATFORM_KEYS.items():
        model.set(key, values[name])


def admitted(model: ModelWrapper, tensor: str, dtype: QONNXDataType, label: str) -> str:
    """An initializer holds integers its annotation admits, by its value summary; the
    summary's ``content_digest``, which keys the node's binding."""
    try:
        summary = initializer_value_summary(model, tensor)
    except UnsupportedTensorValueError as error:
        raise KernelOpError(f"{label}: {tensor}: {error}") from error
    if summary is None:
        raise KernelOpError(f"{label}: {tensor} is not an initializer")
    if not summary.is_integral:
        raise KernelOpError(f"{label}: {tensor} holds values that are not integers")
    try:
        low, high = ordinary_integer_bounds(dtype)
    except DatatypeError:
        return str(summary.content_digest)  # not an ordinary integer: the kernel refuses it
    observed = (int(summary.minimum), int(summary.maximum))
    if not low <= observed[0] <= observed[1] <= high:
        raise KernelOpError(
            f"{label}: {tensor} is annotated {dtype.name} and holds values over {list(observed)}"
        )
    return str(summary.content_digest)


# -- replay ------------------------------------------------------------------------------


def value_type(item: inspection.DecisionInfo[object]) -> str:
    """A decision's value type by its semantics' name (``int``, ``bool``, ``str``)."""
    semantics = item.reference.semantics
    if semantics is None:
        raise KernelOpError(f"{item.key} has no value semantics")
    return semantics.name


def typed_choices(
    subjects: Iterable[Space | type[Space]], choices: Mapping[str, object]
) -> dict[str, object]:
    """``choices``, by decision key of ``subjects`` (spaces or space classes), as replay
    takes them: an ONNX ``i`` attribute of a ``bool`` decision becomes a bool."""
    kinds = {item.key: value_type(item) for each in subjects for item in inspection.decisions(each)}
    return {
        key: bool(value) if kinds.get(key) == "bool" else value for key, value in choices.items()
    }


def refusal(label: str, found: Mapping[str, str]) -> KernelOpError:
    """The error for choices replay refused: each one, by name, with why."""
    return KernelOpError(
        f"{label}: refused choices: "
        + "; ".join(f"{name}: {why}" for name, why in sorted(found.items())),
        tuple(sorted(found)),
    )


def committed(point: S, choices: Mapping[str, object]) -> S | dict[str, str]:
    """``choices`` committed on ``point`` atomically, or each refused key with why. An
    inapplicable key carries no finding of its own, so it is named here."""
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    found = {key: "not a choice here" for key in choices if key not in handles}
    if found:
        return found
    report = point.try_with_choices({handles[key]: value for key, value in choices.items()})
    if report.accepted:
        return report.instance
    for outcome in report.outcomes:
        if outcome.status == "refused":
            result = outcome.result
            found[outcome.owner] = (
                "inapplicable" if isinstance(result, Inapplicable) else describe([result])
            )
    return found or dict.fromkeys(choices, "refused together")


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
    - ``parameters``, for a port that may carry a value the node owns, the kernel's
      views of the tensor and of the value its channel carries.
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
    parameters: ClassVar[Mapping[str, tuple[str, str]]] = {}
    semantic: ClassVar[dict[str, tuple[str, bool, object]]] = {}
    _roots: ClassVar[dict[type[KernelOp], type[Kernel]]] = {}
    _schemas: ClassVar[dict[type[KernelOp], dict[str, tuple[str, tuple[str, ...]]]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "kernel" not in cls.__dict__:
            return  # an abstract base
        version = cls.__dict__.get("op_version")
        if type(version) is not int or version < 1:
            raise TypeError(f"{cls.__qualname__} must state a positive int op_version")

    @property
    def label(self) -> str:
        name: str = self.onnx_node.name or self.onnx_node.op_type
        return name

    # -- schema ---------------------------------------------------------------------------

    @classmethod
    def attribute(cls, key: str) -> str | None:
        """A node-root key's attribute: the kernel's unprefixed, an input or owned
        channel's as it is, an output channel's transport as it is. An output presents
        what is produced, so it opens no adapter of its own."""
        head, _, rest = key.partition(".")
        if head == cls.member:
            return rest
        if head in cls.outputs:
            return key if rest.partition(".")[0] == "transport" else None
        return key if head in cls.ports else None

    @classmethod
    def node_key(cls, attribute: str) -> str:
        """An attribute's node-root key."""
        head = attribute.partition(".")[0]
        return (
            attribute if head in cls.ports or head in cls.outputs else f"{cls.member}.{attribute}"
        )

    @classmethod
    def root(cls) -> type[Kernel]:
        """The op's node root class, generated from its placement once per process."""
        if cls not in KernelOp._roots:
            KernelOp._roots[cls] = node_root(cls)
        return KernelOp._roots[cls]

    @classmethod
    def schema(cls) -> dict[str, tuple[str, tuple[str, ...]]]:
        """The choice attributes: name -> (ONNX type, a selector's cases), from the node
        root's decision keys."""
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

    def edges(self) -> dict[str, Tensor]:
        """The tensor of each edge of the node root, by port, as the graph states it: each
        input channel but a parameter port's, and each output."""
        model, node = self.model(), self.onnx_node
        found = {
            port: edge_tensor(model, tensor, self.label)
            for port, tensor in zip(self.ports, node.input)
            if port is not None and port not in self.parameters
        }
        return found | {
            port: edge_tensor(model, tensor, self.label)
            for port, tensor in zip(self.outputs, node.output)
        }

    def base(self) -> Kernel:
        """The node root bound from this node's facts, nothing chosen."""
        return BIND_CACHE.point(self.facts())

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
                    f"{self.label}: {name} is not a choice of {self.op_type}", (name,)
                )
            found[name] = attribute.s.decode() if schema[name][0] == "s" else int(attribute.i)
        return found

    def node_part(self, facts: Facts, choices: Mapping[str, object]) -> dict[str, object]:
        """The choices a node root replays: its kernel's and its owned channels'. An
        edge's, input or output, belong to the partition root that declares the edge."""
        edges = ({port for port in self.ports if port} - set(facts.owned)) | set(self.outputs)
        return {
            name: value for name, value in choices.items() if name.partition(".")[0] not in edges
        }

    def point(self, extra: Mapping[str, object] | None = None) -> Kernel:
        """The node root with the node's choices (and ``extra``) replayed."""
        facts = self.facts()
        wanted = self.node_part(facts, {**self.choices(), **(extra or {})})
        if not wanted:
            return BIND_CACHE.point(facts)
        names = {self.node_key(name): name for name in wanted}
        typed = typed_choices(
            [self.root()], {self.node_key(name): value for name, value in wanted.items()}
        )

        def build(base: Kernel) -> Kernel:
            replayed = committed(base, typed)
            if isinstance(replayed, dict):
                raise refusal(
                    self.label, {names.get(key, key): why for key, why in replayed.items()}
                )
            return replayed

        return BIND_CACHE.configured(facts, typed, build)

    def save(self, choices: Mapping[str, object | None]) -> None:
        """Commit ``choices`` (made on purpose) over the node's persisted ones: replayed
        on a fresh base and written only if accepted; ``None`` clears a choice."""
        schema = self.schema()
        unknown = sorted(set(choices) - set(schema))
        if unknown:
            raise KernelOpError(f"{self.label}: {unknown} are not choices of {self.op_type}")
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
        self.point(merged)  # refuses before any write
        for name in schema:
            present = get_by_name(self.onnx_node.attribute, name)
            if present is not None:
                self.onnx_node.attribute.remove(present)
        for name, value in sorted(merged.items()):
            self.set_nodeattr(name, int(value) if isinstance(value, bool) else value)

    # -- in a partition root ---------------------------------------------------------------

    def owned(self) -> dict[str, str]:
        """The tensor of each parameter port whose value this node owns, by port: its
        channel is this node's to declare, and carries the kernel's views."""
        return {port: self.onnx_node.input[self.ports.index(port)] for port in self.facts().owned}

    def place(self, channels: Mapping[str, Channel]) -> tuple[Kernel, dict[str, str]]:
        """This node's kernel on a partition's ``channels`` (by tensor), its formals
        literals: the placement its node root is generated from. And the tensor of each
        of its input channels, by port."""
        facts = self.facts()
        inputs = {port: tensor for port, tensor in zip(self.ports, self.onnx_node.input) if port}
        tensors = inputs | dict(zip(self.outputs, self.onnx_node.output))
        on = {port: channels[tensor] for port, tensor in tensors.items()}
        return placed(type(self), facts.formals(), on, facts.owned), inputs

    # -- inference ------------------------------------------------------------------------

    def normalize_inputs(self) -> None:
        """Rewrite this node's value inputs into the form its facts read, from inputs
        already inferred; ``InferKernelTensors`` calls it before the outputs. None by
        default."""

    def output_tensors(self) -> Shapes:
        """Each output's ONNX shape and datatype, from the kernel's fact-level views."""
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

    def verify_node(self) -> list[str]:
        """Replay's refusal, or the kernel's own (its ``admission``); none when accepted."""
        try:
            point = self.point()
        except KernelOpError as error:
            return [str(error)]
        kernel = getattr(point, self.member)
        result = kernel.inspect(type(kernel).admission).result
        return [f"{self.label}: {describe([result])}"] if isinstance(result, Rejected) else []

    def view(self, name: str) -> Any:
        """A fact-level view of the op's kernel, bound alone from the node's formals
        (``result_tensor``, ``result_dtype``): what inference reads, before any output of
        the node is known."""
        kernel = BIND_CACHE.kernel(self.facts())
        answer = kernel.query(getattr(type(kernel), name))
        if not isinstance(answer, Available):
            raise KernelOpError(f"{self.label}: {describe([answer])}")
        return answer.value


def kernel_op(model: ModelWrapper, node: NodeProto) -> KernelOp:
    """The KernelOp of ``node``, attached to ``model``; a node of any other op is
    refused."""
    op = model.get_customop_wrapper(node)
    if not isinstance(op, KernelOp):
        raise KernelOpError(f"{node.name or node.op_type}: {node.op_type} is not a KernelOp")
    return op


__all__ = [
    "PLATFORM",
    "PLATFORM_FIELDS",
    "PLATFORM_KEYS",
    "KernelOp",
    "KernelOpError",
    "admitted",
    "committed",
    "datatype",
    "edge_tensor",
    "kernel_op",
    "read_target",
    "refusal",
    "rows",
    "shape",
    "typed_choices",
    "write_target",
]
