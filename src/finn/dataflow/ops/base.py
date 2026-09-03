# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The DataflowOp: one source node, one frozen problem, one root occurrence.

A ``DataflowOp`` is a QONNX ``CustomOp`` that *wraps* a Space occurrence rather
than being one.  The node is QONNX's to construct and the occurrence is the
design space's to start, and pretending they are the same object would mean
either the graph reading the runtime or the runtime being reconstructed on every
attribute access.

```text
NodeProto + ModelWrapper + build config
    -> read once             SourceNode        (frozen; the graph is not reread)
    -> freeze                Problem snapshot  (with its fingerprint)
    -> start                 root occurrence   (the authored source Space)
    -> hydrate               node attributes replayed as assignments
    -> ask                   network(), association()
```

**Persistence is one authority.**  Everything that survives a save lives on the
node: the operation's stable scope id, its family and version, the fingerprint
of the problem the choices were made against, and one attribute per persistent
Decision.  A reload replays exactly those, in declaration order, through the
ordinary public assignment API -- there is no second path by which a point can
come into existence.

**A changed problem makes the occurrence stale, and nothing more.**  The
fingerprint is compared and mismatch is refused; assignments are never rebased
onto facts they were not made against.  Reconstruction is the caller's explicit
act, which is the U1 decision applied here rather than re-litigated.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

from qonnx.custom_op.base import CustomOp  # type: ignore[import-not-found]

from finn.dataflow._engine import Answer, Decided, RequestError, Unresolved
from finn.dataflow.model.declarations import AuthoringError, Decision, Problem, Space
from finn.dataflow.model.occurrence import VariantView
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.ops.association import SourceAssociation
from finn.dataflow.ops.source import SourceNode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from onnx import NodeProto  # type: ignore[import-not-found]

#: The ONNX domain every logical dataflow operation is registered under.
DATAFLOW_DOMAIN = "finn.custom_op.dataflow"

#: Node attributes the layer itself owns.  Named once, here, because they are
#: the persistence authority and a second spelling of any of them would be a
#: second authority.
SCOPE_ID_ATTRIBUTE = "dataflow_scope_id"
FAMILY_ATTRIBUTE = "dataflow_family"
FAMILY_VERSION_ATTRIBUTE = "dataflow_family_version"
FINGERPRINT_ATTRIBUTE = "dataflow_problem_fingerprint"

RESERVED_ATTRIBUTES = frozenset(
    {SCOPE_ID_ATTRIBUTE, FAMILY_ATTRIBUTE, FAMILY_VERSION_ATTRIBUTE, FINGERPRINT_ATTRIBUTE}
)


class DataflowOpError(ValueError):
    """One source node cannot be projected, hydrated, or committed.

    Carries engine ``Finding`` values when it has them, because a refusal that
    cannot say which declaration refused is a refusal nobody can act on.
    """

    def __init__(self, message: str, findings: Sequence[object] = ()) -> None:
        super().__init__(message)
        self.findings = tuple(findings)


@dataclass(frozen=True, slots=True)
class AttributeCodec:
    """How one persistent Decision's value crosses the ONNX attribute boundary.

    Explicit in both directions, and never ``repr``/``eval``.  An attribute is
    the only thing that survives a save, so a value that cannot be written and
    read back to something the Decision's domain accepts must fail where it is
    written, not where it is next read.

    ``kind`` is the ONNX attribute type the encoded form occupies.  It lives on
    the codec because the codec is what decides the encoded form; declaring it
    separately in ``get_nodeattr_types`` is how an enum written as its own
    stable name ends up in an integer field.
    """

    encode: Callable[[Any], int | str]
    decode: Callable[[Any], object]
    kind: str = "i"


INT_CODEC = AttributeCodec(lambda value: int(value), lambda value: int(value))
BOOL_CODEC = AttributeCodec(lambda value: int(bool(value)), lambda value: bool(int(value)))
STRING_CODEC = AttributeCodec(
    lambda value: str(value),
    lambda value: value.decode("utf-8") if isinstance(value, bytes) else str(value),
    kind="s",
)


@dataclass(frozen=True, slots=True)
class DecisionAttribute:
    """One nested Decision persisted under one ONNX attribute name.

    ``navigate`` reaches the occurrence that owns the Decision, from the root,
    using nothing but the public descriptor navigation.  Declaring the route
    rather than a compiled path is what keeps persistence independent of the
    namespace the root happened to be started under, and what makes a rename of
    an authored member a type error instead of a silently dead attribute.
    """

    name: str
    navigate: Callable[[Space], Space]
    declaration: Decision[Any]
    codec: AttributeCodec

    def apply(self, root: Space, value: object) -> Space:
        owner = self.navigate(root)
        return owner.assign(self.declaration, value).root

    def read(self, root: Space) -> object:
        owner = self.navigate(root)
        return owner.answer(self.declaration)


@dataclass(frozen=True, slots=True)
class SelectorAttribute:
    """One Variant's chosen alternative, persisted under one attribute name.

    A selector is not a public ``Decision`` -- the Variant generates it -- so it
    is committed the way a caller commits it, through the bound view.  Selectors
    are applied before ordinary Decisions, because which alternative is live
    decides which Decisions exist at all.
    """

    name: str
    navigate: Callable[[Space], VariantView]

    def apply(self, root: Space, value: object) -> Space:
        view = self.navigate(root)
        return view.select(str(value)).root

    def read(self, root: Space) -> Answer[str]:
        return self.navigate(root).selected()


PersistentAttribute = DecisionAttribute | SelectorAttribute


class DataflowOp(CustomOp):  # type: ignore[misc]
    """One logical dataflow operation over one ONNX node."""

    #: The stable family identity written onto the node.  Two operations with
    #: the same family and version must agree about their source schema, their
    #: attribute names and their association, because a reload trusts them to.
    family: ClassVar[str] = ""
    family_version: ClassVar[str] = "1"

    #: The authored Space whose Problem members this operation supplies.
    source_space: ClassVar[type[Space]]

    #: The Decisions and selectors that survive a save, in application order.
    attributes: ClassVar[tuple[PersistentAttribute, ...]] = ()

    #: The namespace the root occurrence is started under.  Fixed per family so
    #: that a compiled path means the same thing in every graph.
    root_namespace: ClassVar[str] = "op"

    def __init__(self, onnx_node: NodeProto, onnx_opset_version: int = 1) -> None:
        super().__init__(onnx_node, onnx_opset_version)
        self._model: Any | None = None
        if not type(self).family:
            raise AuthoringError(f"{type(self).__name__} must declare a non-empty family")
        duplicates = sorted(
            {
                item.name
                for item in type(self).attributes
                if sum(other.name == item.name for other in type(self).attributes) > 1
            }
        )
        if duplicates:
            raise AuthoringError(
                f"{type(self).__name__} persists attribute {duplicates[0]!r} twice"
            )
        clashes = sorted({item.name for item in type(self).attributes} & RESERVED_ATTRIBUTES)
        if clashes:
            raise AuthoringError(
                f"{type(self).__name__} persists {clashes[0]!r}, which the dataflow layer owns"
            )

    # -- QONNX integration ----------------------------------------------------

    def wants_model(self) -> bool:
        """Every projection needs the graph, so the wrapper is always attached."""

        return True

    def attach_model(self, model: Any) -> DataflowOp:
        self._model = model
        return self

    def get_nodeattr_types(self) -> Mapping[str, tuple[str, bool, object]]:
        declared: dict[str, tuple[str, bool, object]] = {
            SCOPE_ID_ATTRIBUTE: ("s", False, ""),
            FAMILY_ATTRIBUTE: ("s", False, ""),
            FAMILY_VERSION_ATTRIBUTE: ("s", False, ""),
            FINGERPRINT_ATTRIBUTE: ("s", False, ""),
        }
        declared.update(self.source_nodeattr_types())
        for item in type(self).attributes:
            kind = "s" if isinstance(item, SelectorAttribute) else item.codec.kind
            declared[item.name] = (kind, False, "" if kind == "s" else 0)
        return declared

    def source_nodeattr_types(self) -> Mapping[str, tuple[str, bool, object]]:
        """Node attributes this operation reads as source facts.  Override."""

        return {}

    def make_shape_compatible_op(self, model: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not declare a shape-compatible op")

    def infer_node_datatype(self, model: Any) -> None:
        """Datatypes are source facts this layer reads, never ones it invents."""

    def execute_node(self, context: Any, graph: Any) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not execute its source semantics")

    def verify_node(self) -> list[str]:
        return []

    # -- identity -------------------------------------------------------------

    def scope_id(self) -> str:
        """The operation's stable identity, allocated on first use."""

        existing = self._attribute(SCOPE_ID_ATTRIBUTE)
        if existing:
            return existing
        allocated = f"dataflow_{uuid4().hex}"
        self.set_nodeattr(SCOPE_ID_ATTRIBUTE, allocated)
        return allocated

    def _attribute(self, name: str) -> str:
        for attribute in self.onnx_node.attribute:
            if attribute.name == name:
                value = attribute.s
                return value.decode("utf-8") if isinstance(value, bytes) else str(value)
        return ""

    def _present(self, name: str) -> Any | None:
        for attribute in self.onnx_node.attribute:
            if attribute.name == name:
                return attribute
        return None

    # -- source projection ----------------------------------------------------

    def _attached(self) -> Any:
        if self._model is None:
            raise DataflowOpError(
                f"{type(self).__name__} needs its ModelWrapper; reach it through "
                "model.get_customop_wrapper(node)"
            )
        return self._model

    def read_source(self) -> SourceNode:
        """Read this node's operands and attributes once.  Override to declare."""

        raise NotImplementedError(f"{type(self).__name__} does not declare its source operands")

    def source_facts(self, source: SourceNode, build: Any) -> Mapping[Problem[Any], object]:
        """Map the frozen source reading onto this operation's Problem members."""

        raise NotImplementedError(f"{type(self).__name__} does not declare its source facts")

    def start(self, build: Any) -> Space:
        """One fresh root occurrence over the frozen problem, with no assignments."""

        source = self.read_source()
        facts = self.source_facts(source, build)
        return type(self).source_space.start(facts, namespace=type(self).root_namespace)

    def occurrence(self, build: Any) -> Space:
        """The root occurrence with every persisted choice replayed onto it.

        Selectors first, then ordinary Decisions, each through the same public
        call a caller would make.  A stored fingerprint that does not match the
        problem this node now presents is refused here rather than being
        rebased: the choices were made against different facts and the layer
        does not guess which of them still hold.
        """

        root = self.start(build)
        stored = self._attribute(FINGERPRINT_ATTRIBUTE)
        if stored and stored != root.problem_fingerprint:
            raise DataflowOpError(
                f"{self.onnx_node.name} stores choices made against a different problem "
                f"(recorded {stored}, current {root.problem_fingerprint}); "
                "reconstruct explicitly rather than reinterpreting them"
            )
        family = self._attribute(FAMILY_ATTRIBUTE)
        if family and family != type(self).family:
            raise DataflowOpError(
                f"{self.onnx_node.name} stores choices for family {family!r}, "
                f"not {type(self).family!r}"
            )
        version = self._attribute(FAMILY_VERSION_ATTRIBUTE)
        if version and version != type(self).family_version:
            raise DataflowOpError(
                f"{self.onnx_node.name} stores choices for {family}@{version}, "
                f"and this build offers {type(self).family_version}"
            )
        return self._hydrate(root)

    def _hydrate(self, root: Space) -> Space:
        ordered: tuple[PersistentAttribute, ...] = (
            *(item for item in type(self).attributes if isinstance(item, SelectorAttribute)),
            *(item for item in type(self).attributes if isinstance(item, DecisionAttribute)),
        )
        for item in ordered:
            attribute = self._present(item.name)
            if attribute is None:
                continue
            raw = attribute.s if attribute.type == attribute.STRING else attribute.i
            try:
                value = (
                    STRING_CODEC.decode(raw)
                    if isinstance(item, SelectorAttribute)
                    else item.codec.decode(raw)
                )
                root = item.apply(root, value)
            except (RequestError, AuthoringError, ValueError, TypeError) as error:
                raise DataflowOpError(
                    f"{self.onnx_node.name} attribute {item.name!r} does not replay onto "
                    f"this problem: {error}",
                    getattr(error, "findings", ()),
                ) from error
        return root

    def is_stale(self, build: Any) -> bool:
        """Whether the recorded choices belong to the problem this node presents."""

        stored = self._attribute(FINGERPRINT_ATTRIBUTE)
        if not stored:
            return False
        return stored != self.start(build).problem_fingerprint

    # -- the two projections --------------------------------------------------

    def network(self, build: Any) -> Answer[DataflowNetwork]:
        """The selected Network of this source node.  Override to route."""

        raise NotImplementedError(f"{type(self).__name__} does not project a Network")

    def association(self, build: Any) -> Answer[SourceAssociation]:
        """Where each source operand's data crosses the selected Network."""

        raise NotImplementedError(f"{type(self).__name__} does not declare its association")

    # -- persistence ----------------------------------------------------------

    def commit(self, build: Any, values: Mapping[str, object]) -> Space:
        """Validate every choice against the point, then write the node once.

        Transactional by construction: the whole assignment sequence runs
        against the occurrence first, and the node is touched only after the
        last one is accepted.  A refusal therefore leaves the graph byte for
        byte as it was, which is checked rather than asserted -- the node is
        snapshotted and restored if any write raises.
        """

        by_name = {item.name: item for item in type(self).attributes}
        unknown = sorted(set(values) - set(by_name))
        if unknown:
            raise DataflowOpError(
                f"{type(self).__name__} does not persist {unknown[0]!r}; "
                f"it persists {sorted(by_name)}"
            )

        root = self.occurrence(build)
        fingerprint = root.problem_fingerprint
        ordered: list[PersistentAttribute] = [
            item
            for item in type(self).attributes
            if item.name in values and isinstance(item, SelectorAttribute)
        ] + [
            item
            for item in type(self).attributes
            if item.name in values and isinstance(item, DecisionAttribute)
        ]
        encoded: dict[str, int | str] = {}
        for item in ordered:
            value = values[item.name]
            try:
                root = item.apply(root, value)
            except (RequestError, AuthoringError, ValueError, TypeError) as error:
                raise DataflowOpError(
                    f"{self.onnx_node.name} refuses {item.name}={value!r}: {error}",
                    getattr(error, "findings", ()),
                ) from error
            encoded[item.name] = (
                STRING_CODEC.encode(value)
                if isinstance(item, SelectorAttribute)
                else item.codec.encode(value)
            )

        scope = self.scope_id()
        snapshot = self.onnx_node.SerializeToString(deterministic=True)
        try:
            for name, raw in encoded.items():
                self.set_nodeattr(name, raw)
            self.set_nodeattr(SCOPE_ID_ATTRIBUTE, scope)
            self.set_nodeattr(FAMILY_ATTRIBUTE, type(self).family)
            self.set_nodeattr(FAMILY_VERSION_ATTRIBUTE, type(self).family_version)
            self.set_nodeattr(FINGERPRINT_ATTRIBUTE, fingerprint)
        except Exception:
            restored = type(self.onnx_node)()
            restored.ParseFromString(snapshot)
            self.onnx_node.CopyFrom(restored)
            raise
        return root

    def clear(self) -> None:
        """Drop every recorded choice, keeping the operation's identity."""

        dropped = {item.name for item in type(self).attributes} | {
            FAMILY_ATTRIBUTE,
            FAMILY_VERSION_ATTRIBUTE,
            FINGERPRINT_ATTRIBUTE,
        }
        kept = [item for item in self.onnx_node.attribute if item.name not in dropped]
        del self.onnx_node.attribute[:]
        self.onnx_node.attribute.extend(kept)

    def recorded(self) -> Mapping[str, object]:
        """Exactly the choices physically present on this node."""

        present: dict[str, object] = {}
        for item in type(self).attributes:
            attribute = self._present(item.name)
            if attribute is None:
                continue
            raw = attribute.s if attribute.type == attribute.STRING else attribute.i
            present[item.name] = (
                STRING_CODEC.decode(raw)
                if isinstance(item, SelectorAttribute)
                else item.codec.decode(raw)
            )
        return MappingProxyType(present)


def unresolved_reason(answer: Answer[Any]) -> str:
    """A short account of why a projection is not available, for a message."""

    if isinstance(answer, Decided):
        return "decided"
    findings = getattr(answer, "findings", ())
    kind = "unresolved" if isinstance(answer, Unresolved) else "absent"
    codes = ", ".join(sorted({finding.code for finding in findings})) or "no findings"
    return f"{kind}: {codes}"


__all__ = [
    "BOOL_CODEC",
    "DATAFLOW_DOMAIN",
    "FAMILY_ATTRIBUTE",
    "FAMILY_VERSION_ATTRIBUTE",
    "FINGERPRINT_ATTRIBUTE",
    "INT_CODEC",
    "RESERVED_ATTRIBUTES",
    "SCOPE_ID_ATTRIBUTE",
    "STRING_CODEC",
    "AttributeCodec",
    "DataflowOp",
    "DataflowOpError",
    "DecisionAttribute",
    "PersistentAttribute",
    "SelectorAttribute",
    "unresolved_reason",
]
