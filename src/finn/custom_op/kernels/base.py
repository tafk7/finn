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
whole for its choices, replays the choices its node holds, and answers the
compiler's queries from the result.

Each node is asked three questions, each with one owner (KT18):

- **identity**, the op's pattern: the node it starts from (``anchor``: a domain and
  an op type) and ``match``, which reads the graph around that node by its structure
  only (the op types and connectivity, which inputs are facts, the semantic
  attributes, what fixes the axes; never a datatype, a value, a container or the
  platform) and answers with the nodes the op covers and its semantic attributes
  (``Match``), or the Space's ``Rejected`` with the pattern's findings: a refusal by
  its code, or a structural fact it needs that the graph does not state
  (``fact-unstated``). ``match`` never changes the model; ``ToKernelOps`` applies a
  match;
- **domain**, the op's reference: ``exact``, on the node once its inputs are
  normalized, refuses a node whose facts (its datatypes, values and containers) let
  the ONNX it covers compute other values than the reference, each refusal by its
  code;
- **realizability**, the kernels': ``admission`` asks them on the node's inputs
  (``refusals``: the kernel's admission and each Decision with no viable case, by
  why each of its cases is refused), which ``verify_node`` also reports on the
  node's point.

A fact binding reads that the graph does not state raises ``FactUnstated``, which
conversion reports; any other refusal of the facts is a contradiction
(``KernelOpError``).

An op's ``execute_node`` is the computational reference for what its pattern covers
(PRINCIPLES 8), with ``normalize_inputs`` before it, defined for every input domain
the ONNX it covers is, and equal to the ONNX semantics of the covered nodes on every
node the op converts: a rewrite ``normalize_inputs`` makes keeps those values, and a
node where they would differ is refused by ``exact``. The harness checks the
reference against ONNX, every value equal (``finn.harness.reference``); no build
re-checks it.

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

from array import array
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from math import prod
from typing import TYPE_CHECKING, Any, ClassVar, TypeGuard, TypeVar

import numpy as np
import numpy.typing as npt
from onnx import helper
from qonnx.analysis.tensor_value_summary import (
    UnsupportedTensorValueError,
    initializer_value_summary,
)
from qonnx.core.metadata import JSON, Key, MetadataError, Namespace
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import get_by_name

from finn.core.space import (
    Available,
    Finding,
    FindingKind,
    Inapplicable,
    Rejected,
    Space,
    inspection,
)
from finn.custom_op.kernels.cache import BIND_CACHE, Facts
from finn.custom_op.kernels.roots import node_root, placed
from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    canonical_qonnx_datatype,
    is_ordinary_integer,
    ordinary_integer_bounds,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.configure import describe
from finn.kernels.target import DspBlock, Fabric, Platform, Target
from finn.kernels.utilization import Resources
from finn.kernels.values.domains import stored_element
from finn.kernels.values.semantics import IntegerTensorValue, integer_bytes, integer_digest

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

S = TypeVar("S", bound=Space)

PLATFORM = Namespace("finn.platform", version=1, inherit=True)
"""The build target, typed graph metadata (qonnx's ``qonnx.core.metadata``): the part,
the shell, the board (``null`` when none is stated) and the platform
(``finn.kernels.target.Platform``: the clock period and the capabilities, its resources
``null`` when the part's are not known), every key stated. It inherits: a subgraph body
reads its parent's. A model stating another shape of it (a key it does not declare, or
one missing) is refused, each key named: it is converted again (``ToKernelOps``)."""

RESOURCE_COUNTS = tuple(field.name for field in fields(Resources))


def _resources_or_none(value: object) -> bool:
    return value is None or (
        isinstance(value, dict)
        and set(value) == set(RESOURCE_COUNTS)
        and all(type(count) is int and count >= 0 for count in value.values())
    )


PLATFORM_KEYS: dict[str, Key[Any]] = dict(
    part=PLATFORM.key("part", str, check=bool, expect="a part name"),
    shell=PLATFORM.key("shell", str, check=bool, expect="a shell name"),
    board=PLATFORM.key(
        "board",
        JSON,
        check=lambda v: v is None or (isinstance(v, str) and bool(v)),
        expect="a board name, or null when none is stated",
    ),
    period_ns=PLATFORM.key("period_ns", float, check=lambda v: v > 0, expect="a period > 0"),
    dsp=PLATFORM.key("dsp", DspBlock),
    fabric=PLATFORM.key("fabric", Fabric),
    uram=PLATFORM.key("uram", bool),
    uram_init=PLATFORM.key("uram_init", bool),
    resources=PLATFORM.key(
        "resources",
        JSON,
        check=_resources_or_none,
        expect=f"the counts {', '.join(RESOURCE_COUNTS)}, or null when unknown",
    ),
)

PLATFORM_FIELDS = tuple(field.name for field in fields(Platform))
"""The ``finn.platform`` keys that are ``Platform``'s fields: all but the part, the
shell and the board."""

ONNX_TYPES = {"int": "i", "bool": "i", "str": "s"}

Shapes = dict[str, tuple[tuple[int, ...], QONNXDataType]]


class KernelOpError(ValueError):
    """A node an op cannot bind or configure: a missing or refused fact, or a refused
    choice. ``keys`` names the refused choices, by attribute, when there are any."""

    def __init__(self, message: str, keys: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.keys = keys


class FactUnstated(KernelOpError):
    """A fact binding reads that the graph does not state (``tensor``'s shape or its
    datatype annotation): not a contradiction, so conversion leaves the node on the
    host with a ``fact-unstated`` finding instead of stopping the build."""

    def __init__(self, message: str, tensor: str) -> None:
        super().__init__(message)
        self.tensor = tensor


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


# -- facts -------------------------------------------------------------------------------


def datatype(model: ModelWrapper, tensor: str, label: str) -> QONNXDataType:
    """The annotation of ``tensor``; an unannotated tensor is refused (qonnx reads it
    as its container type, FLOAT32, which is not a statement)."""
    if not model.has_tensor_datatype(tensor):
        produced = model.find_producer(tensor) is not None
        raise FactUnstated(
            f"{label}: {tensor} has no datatype annotation (absent is not FLOAT32"
            + ("; InferKernelTensors states each node's outputs in order)" if produced else ")"),
            tensor,
        )
    return canonical_qonnx_datatype(model.get_tensor_datatype(tensor))


def known_shape(model: ModelWrapper, tensor: str) -> tuple[int, ...] | None:
    """The shape of ``tensor`` if the graph states it: None for none, or for one with an
    extent it does not state (qonnx reads both as a list, the first as empty)."""
    found = model.get_tensor_shape(tensor)
    if not found or any(type(dim) is not int or dim < 1 for dim in found):
        return None
    return tuple(found)


def shape(model: ModelWrapper, tensor: str, label: str) -> tuple[int, ...]:
    """The shape of ``tensor``; one not known yet is refused."""
    found = known_shape(model, tensor)
    if found is None:
        raise FactUnstated(f"{label}: {tensor} has no shape yet (run InferKernelTensors)", tensor)
    return found


def rows(dims: tuple[int, ...]) -> tuple[int, int]:
    """Leading axes are rows: (1, M, K) is M rows of K."""
    return prod(dims[:-1]), dims[-1]


def edge_tensor(model: ModelWrapper, tensor: str, label: str) -> Tensor:
    """The tensor a channel carries, as the graph states it: its shape as rows and its
    annotation; an initializer of integers (``admitted``) as its owner states it, over
    its values' range in the encoding they need (``stored_element``: INT8-typed ternary
    weights are ``INT2 over [-1, 1]``)."""
    dtype = datatype(model, tensor, label)
    stored = model.get_initializer(tensor)
    element = (
        stored_element(dtype, (int(stored.min()), int(stored.max())))
        if stored is not None and is_ordinary_integer(dtype)
        else ScalarEncoding(dtype)
    )
    return Tensor(rows(shape(model, tensor, label)), element)


def read_target(model: ModelWrapper) -> Target:
    """The build target, from the model's ``finn.platform`` metadata (a subgraph body
    opened through its parent reads the parent's).

    The one reader of the target; ``write_target`` is the one writer. A key missing,
    malformed or not declared (another shape's) is refused, by name.
    """
    try:
        stated = model.namespace(PLATFORM)
    except MetadataError as error:
        raise KernelOpError(
            f"the model's target is not one: {error}; run ToKernelOps to state it again"
        ) from error
    missing = [name for name in PLATFORM_KEYS if name not in stated]
    if missing:
        raise KernelOpError(
            f"the model states no target (finn.platform: {', '.join(missing)} missing; "
            "run ToKernelOps)"
        )
    values = {name: stated[name] for name in PLATFORM_FIELDS}
    if values["resources"] is not None:
        values["resources"] = Resources(**values["resources"])
    platform = Platform(**values)
    return Target(
        part=stated["part"], platform=platform, shell=stated["shell"], board=stated["board"]
    )


def write_target(model: ModelWrapper, target: Target) -> None:
    """State the build target in the model's ``finn.platform`` metadata, every key,
    where ``read_target`` reads it."""
    platform = target.platform
    if platform.dsp is None:
        raise KernelOpError("a target states its DSP block (finn.platform.resolve_target)")
    values: dict[str, object] = dict(
        part=target.part,
        shell=target.shell,
        board=target.board,
        **{name: getattr(platform, name) for name in PLATFORM_FIELDS},
    )
    if platform.resources is not None:
        values["resources"] = asdict(platform.resources)
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
    try:
        low, high = ordinary_integer_bounds(dtype)
    except DatatypeError:
        # Not an ordinary integer (FLOAT32 holds 0.5): no contradiction, the kernel refuses it.
        return str(summary.content_digest)
    if not summary.is_integral:
        raise KernelOpError(f"{label}: {tensor} holds values that are not integers")
    if summary.minimum is None or summary.maximum is None:  # integral with no range: empty
        raise KernelOpError(f"{label}: {tensor} is empty: it holds no values")
    observed = (int(summary.minimum), int(summary.maximum))
    if not low <= observed[0] <= observed[1] <= high:
        raise KernelOpError(
            f"{label}: {tensor} is annotated {dtype.name} and holds values over {list(observed)}"
        )
    return str(summary.content_digest)


def integer_tensor(values: npt.NDArray[Any]) -> IntegerTensorValue:
    """An initializer of integers (``admitted``) as an integer tensor value: its shape,
    range and digest read from the array, its integers loaded only when first read (a
    memory image packs them)."""
    found = np.asarray(values)
    shape = tuple(int(extent) for extent in found.shape)
    least, greatest = int(found.min()), int(found.max())
    if not -(2**63) <= least <= greatest < 2**63:
        return IntegerTensorValue.flat(shape, [int(value) for value in found.ravel()])
    stored = array("q")
    stored.frombytes(found.astype("=i8").tobytes())
    digest = integer_digest(shape, integer_bytes(stored))
    return IntegerTensorValue(shape, (least, greatest), digest, lambda: stored)


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
    - ``parameters``, the ports whose channel may carry a value the node owns (an
      initializer): the channel's ``contents``, which its source stores.

    And its pattern (the module docstring): ``anchor``, the ``(domain, op_type)`` of the
    ONNX node it starts from, and ``match``; and its domain step, ``exact``.
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

    def exact(self) -> tuple[Finding, ...]:
        """The domain step: the refusals of this node where, on its facts once its inputs
        are normalized, the ONNX it covers executed in its containers could compute other
        values than its reference, each by its code (``refused``). Empty where they
        agree; by default, an op whose reference agrees on every node it converts. A
        fact it reads that the graph does not state raises ``FactUnstated``."""
        return ()

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

    def edges(self) -> dict[str, Tensor]:
        """The tensor of each channel of the node root, by port, inputs and outputs."""
        return self.input_edges() | self.output_edges()

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
        edge's, input or output, belong to the root that declares the edge."""
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

    def normalize_inputs(self) -> None:
        """Rewrite this node's value inputs into the form its facts read, from inputs
        already inferred; ``InferKernelTensors`` calls it before the outputs. None by
        default."""

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
            raise KernelOpError(f"{self.label}: {describe([answer])}")
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
    "PLATFORM",
    "PLATFORM_FIELDS",
    "PLATFORM_KEYS",
    "KernelOp",
    "FactUnstated",
    "KernelOpError",
    "admitted",
    "candidate_refusals",
    "committed",
    "datatype",
    "edge_tensor",
    "integer_tensor",
    "kernel_op",
    "read_target",
    "refusal",
    "rows",
    "shape",
    "typed_choices",
    "write_target",
]
