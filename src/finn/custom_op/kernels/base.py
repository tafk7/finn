# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""KernelOp: a qonnx ``CustomOp`` that binds one kernel point (boundary note D1).

An op reads its facts from the attached model only (H-002, H-006's reading
rule): input shapes and datatype annotations, initializers admitted by their
value summary, and the build target from the model's metadata. It binds its
node root (``finn.custom_op.kernels.roots``) through the bind cache, replays the
choices its node holds, and answers the compiler's queries from the result.

Two kinds of attribute:

- **semantic**: part of the operation, stated by the graph (Thresholding's
  ``bias``); required, never a choice;
- **choice**: one per decision key of the op's node roots, sparse, absent
  meaning open. The kernel's keys are unprefixed (``compute.packed.pe``), an
  input or owned stream's sit under the op's port name (``x.adapter…``,
  ``w.transport``), and an output stream's belong to its consumer. The ONNX type
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

from collections.abc import Mapping
from math import prod
from typing import Any, ClassVar, TypeVar

from qonnx.analysis.tensor_value_summary import (
    UnsupportedTensorValueError,
    initializer_value_summary,
)
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import get_by_name

from finn.core.space import Available, Inapplicable, Rejected, Space, inspection
from finn.custom_op.kernels.cache import BIND_CACHE, Facts
from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    canonical_qonnx_datatype,
    ordinary_integer_bounds,
)
from finn.dataflow.tensor import Tensor
from finn.kernels.base import Kernel
from finn.kernels.configure import describe
from finn.kernels.streams import Stream
from finn.kernels.target import DspBlock

S = TypeVar("S", bound=Space)

TARGET_DSP, TARGET_PERIOD = "finn_target_dsp", "finn_target_period_ns"
"""The model metadata that states the build target (phase 1's home for it)."""

ONNX_TYPES = {"int": "i", "bool": "i", "str": "s"}

Shapes = dict[str, tuple[tuple[int, ...], QONNXDataType]]


class KernelOpError(ValueError):
    """A node an op cannot bind or configure: a missing or refused fact, or a refused
    choice. ``keys`` names the refused choices, by attribute, when there are any."""

    def __init__(self, message: str, keys: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.keys = keys


# -- facts -------------------------------------------------------------------------------


def datatype(model: Any, tensor: str, label: str) -> QONNXDataType:
    """The annotation of ``tensor``; an unannotated tensor is refused (qonnx reads it
    as its container type, FLOAT32, which is not a statement)."""
    if not model.has_tensor_datatype(tensor):
        produced = model.find_producer(tensor) is not None
        raise KernelOpError(
            f"{label}: {tensor} has no datatype annotation (absent is not FLOAT32"
            + ("; InferKernelTensors states each node's outputs in order)" if produced else ")")
        )
    return canonical_qonnx_datatype(model.get_tensor_datatype(tensor))


def shape(model: Any, tensor: str, label: str) -> tuple[int, ...]:
    """The shape of ``tensor``; one not known yet is refused."""
    found = model.get_tensor_shape(tensor)
    if not found or any(type(dim) is not int or dim < 1 for dim in found):
        raise KernelOpError(f"{label}: {tensor} has no shape yet (run InferKernelTensors)")
    return tuple(found)


def rows(dims: tuple[int, ...]) -> tuple[int, int]:
    """Leading axes are rows: (1, M, K) is M rows of K."""
    return prod(dims[:-1]), dims[-1]


def target(model: Any) -> tuple[DspBlock, float]:
    """The build target: the DSP block and the clock period, from the model's metadata.

    The one reader of the target; ``write_target`` is the one writer. Its long-term
    home is a typed platform field on the model (the qonnx track's Q6), which moves
    these two functions only.
    """
    dsp, period = model.get_metadata_prop(TARGET_DSP), model.get_metadata_prop(TARGET_PERIOD)
    if dsp is None or period is None:
        raise KernelOpError(f"the model states no target ({TARGET_DSP}, {TARGET_PERIOD})")
    try:
        return DspBlock[dsp], float(period)
    except (KeyError, ValueError) as error:
        raise KernelOpError(f"the model's target is not one: {dsp!r}, {period!r}") from error


def write_target(model: Any, dsp: DspBlock, period_ns: float) -> None:
    """State the build target in the model's metadata, where ``target`` reads it."""
    model.set_metadata_prop(TARGET_DSP, dsp.name)
    model.set_metadata_prop(TARGET_PERIOD, repr(float(period_ns)))


def admitted(model: Any, tensor: str, dtype: QONNXDataType, label: str) -> str:
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


def value_type(item: inspection.DecisionInfo[object]) -> str:
    """A decision's value type by its semantics' name (``int``, ``bool``, ``str``)."""
    semantics = item.reference.semantics
    if semantics is None:
        raise KernelOpError(f"{item.key} has no value semantics")
    return semantics.name


class KernelOp(CustomOp):  # type: ignore[misc]
    """One ONNX node's binding of one kernel point; see the module docstring.

    An op class states ``op_type`` (its class name) and ``op_version`` (its kernel's
    ``version``) in its own body, and names its node-root classes (``roots``), its
    kernel's member in them (``member``), and each ONNX input's stream (``ports``;
    ``None``: an input that is a fact, never a stream).
    """

    wants_model = True
    op_type: ClassVar[str]
    op_version: ClassVar[int]
    roots: ClassVar[tuple[type[Kernel], ...]]
    member: ClassVar[str]
    ports: ClassVar[tuple[str | None, ...]]
    semantic: ClassVar[dict[str, tuple[str, bool, object]]] = {}
    _schemas: ClassVar[dict[type[KernelOp], dict[str, tuple[str, tuple[str, ...]]]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "roots" not in cls.__dict__:
            return  # an abstract base
        stated = {name: cls.__dict__.get(name) for name in ("op_type", "op_version")}
        if stated["op_type"] != cls.__name__:
            raise TypeError(f"{cls.__qualname__} must state op_type = {cls.__name__!r}")
        version = stated["op_version"]
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
        stream's as it is; ``None`` for an output stream's (its consumer's)."""
        head, _, rest = key.partition(".")
        if head == cls.member:
            return rest
        return key if head in cls.ports else None

    @classmethod
    def node_key(cls, attribute: str) -> str:
        """An attribute's node-root key."""
        head = attribute.partition(".")[0]
        return attribute if head in cls.ports else f"{cls.member}.{attribute}"

    @classmethod
    def schema(cls) -> dict[str, tuple[str, tuple[str, ...]]]:
        """The choice attributes: name -> (ONNX type, a selector's cases), the union
        of the node-root classes' decision keys."""
        if cls not in KernelOp._schemas:
            found: dict[str, tuple[str, tuple[str, ...]]] = {}
            for root in cls.roots:
                for item in inspection.decisions(root):
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

    def model(self) -> Any:
        model = getattr(self, "_model", None)
        if model is None:
            raise KernelOpError(f"{self.label}: no model attached (use get_customop_wrapper)")
        return model

    def target(self) -> tuple[DspBlock, float]:
        try:
            return target(self.model())
        except KernelOpError as error:
            raise KernelOpError(f"{self.label}: {error}") from error

    def facts(self) -> Facts:
        """What binding this node reads, from the attached model."""
        raise NotImplementedError

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

    def _typed(self, choices: Mapping[str, object]) -> dict[str, object]:
        """Node-root keys and values: an ``i`` attribute of a ``bool`` decision is a bool."""
        kinds = {
            item.key: value_type(item) for root in self.roots for item in inspection.decisions(root)
        }
        typed: dict[str, object] = {}
        for name, value in choices.items():
            key = self.node_key(name)
            typed[key] = bool(value) if kinds.get(key) == "bool" else value
        return typed

    def node_part(self, facts: Facts, choices: Mapping[str, object]) -> dict[str, object]:
        """The choices a node root replays: its kernel's and its owned streams'. An input
        edge's belong to the partition root that declares the edge."""
        edges = {port for port in self.ports if port} - set(facts.owned)
        return {
            name: value for name, value in choices.items() if name.partition(".")[0] not in edges
        }

    def point(self, extra: Mapping[str, object] | None = None) -> Kernel:
        """The node root with the node's choices (and ``extra``) replayed."""
        facts = self.facts()
        wanted = self.node_part(facts, {**self.choices(), **(extra or {})})
        if not wanted:
            return BIND_CACHE.point(facts)
        typed = self._typed(wanted)

        def build(base: Kernel) -> Kernel:
            replayed = committed(base, typed)
            if isinstance(replayed, dict):
                names = {self.node_key(name): name for name in wanted}
                refused = {names.get(key, key): why for key, why in replayed.items()}
                raise KernelOpError(
                    f"{self.label}: refused choices: "
                    + "; ".join(f"{name}: {why}" for name, why in sorted(refused.items())),
                    tuple(sorted(refused)),
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
        merged = {
            name: value
            for name, value in {**self.choices(), **choices}.items()
            if value is not None
        }
        for name, value in merged.items():
            kind, cases = schema[name]
            fits = isinstance(value, str) if kind == "s" else isinstance(value, int)
            if not fits or (cases and value not in cases):
                raise KernelOpError(
                    f"{self.label}: {name} = {value!r} is not an {kind!r} value"
                    + (f" among {list(cases)}" if cases else ""),
                    (name,),
                )
        self.point(merged)  # refuses before any write
        for name in schema:
            present = get_by_name(self.onnx_node.attribute, name)
            if present is not None:
                self.onnx_node.attribute.remove(present)
        for name, value in sorted(merged.items()):
            self.set_nodeattr(name, int(value) if isinstance(value, bool) else value)

    def drop_inapplicable(self) -> tuple[str, ...]:
        """Remove the node's choices its node root cannot apply, and name them: for a
        transformation that changes the node's node-root class, as the upgrade rule
        does for one node. An input edge's choices are the partition's, which drops
        the stale ones (a lifted initializer's source)."""
        facts = self.facts()
        mine = self.node_part(facts, self.choices())
        found = committed(BIND_CACHE.point(facts), self._typed(mine)) if mine else {}
        if not isinstance(found, dict):
            return ()
        names = {self.node_key(name): name for name in mine}
        dropped = tuple(sorted(names[key] for key, why in found.items() if why == "inapplicable"))
        for name in dropped:
            self.onnx_node.attribute.remove(get_by_name(self.onnx_node.attribute, name))
        return dropped

    # -- in a partition root ---------------------------------------------------------------

    def owned_streams(self) -> dict[str, Stream]:
        """The streams this node declares beside its outputs, by tensor: a stored
        parameter's, its tensor the kernel's view (D5, D6)."""
        return {}

    def place(self, streams: Mapping[str, Stream]) -> tuple[Kernel, dict[str, str]]:
        """This node's kernel on a partition's ``streams`` (by tensor), the graph's pins as
        keywords; and the tensor of each of its input and owned streams, by port."""
        raise NotImplementedError

    # -- inference ------------------------------------------------------------------------

    def normalize_inputs(self) -> None:
        """Rewrite this node's value inputs into the form its facts read, from inputs
        already inferred; ``InferKernelTensors`` calls it before the outputs. None by
        default."""

    def output_tensors(self) -> Shapes:
        """Each output's ONNX shape and datatype, from the node root's views."""
        raise NotImplementedError

    def infer_output_tensors(self, model: Any) -> Shapes:
        self.attach_model(model)
        return self.output_tensors()

    def make_shape_compatible_op(self, model: Any) -> Any:
        ((dims, _),) = self.infer_output_tensors(model).values()
        return self.make_const_shape_op(list(dims))

    def infer_node_datatype(self, model: Any) -> None:
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

    def view(self, name: str) -> Tensor:
        """A fact-level view of the node root (``y_tensor``, ``w_tensor``)."""
        base = self.base()
        answer = base.query(getattr(type(base), name))
        if not isinstance(answer, Available):
            raise KernelOpError(f"{self.label}: {describe([answer])}")
        tensor: Tensor = answer.value
        return tensor


__all__ = [
    "KernelOp",
    "KernelOpError",
    "admitted",
    "committed",
    "datatype",
    "rows",
    "shape",
    "target",
    "write_target",
]
