# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The DataflowOp: a QONNX CustomOp that *is* the root Space of its own design.

The previous arrangement wrapped an occurrence.  That looked like separation of
concerns and was not: an operation had to re-derive its root on every question,
which meant the graph was reread on every question, which meant nothing was
actually frozen and ``op.assign(...)`` could not exist at all.  Everything a
caller wanted was reachable only through a method that rebuilt the world.

So the operation is the root::

    unbound   = model.get_customop_wrapper(node)   # QONNX constructs this
    op        = unbound.bind(model, build)         # an attached occurrence
    successor = op.assign(SomeDesign.pe, 2)        # ordinary Space lifecycle
    fresh     = op.rebind(model, build)            # an explicit refresh

``bind`` reads the graph exactly once into a frozen :class:`SourceBinding`,
freezes it as this Space's Problem values, and starts a root occurrence through
a *root factory* -- so the successor of an assignment is another bound
operation of the same class, carrying the same frozen source byte for byte.  A
child Design or Kernel is never built by that factory and therefore cannot
reach the node, the model, or the build configuration.

**A bound occurrence borrows nothing.**  ``onnx_node`` on a bound instance is a
deep copy, and ``_model`` is ``None``.  A later edit to the live graph cannot
change a bound answer; it can only make the occurrence stale, which is a
different and detectable thing.  Mutation finds the live node again by scope
id, at the moment effects are applied, which is why the applier needs a stable
identity rather than an object reference.

**Persistence is one authority.**  Everything that survives a save lives on the
node: the operation's scope id, its family and version, the fingerprint of the
problem the choices were made against, and one attribute per persistent
Decision.  A reload replays exactly those through the ordinary public
assignment API.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, cast

from qonnx.custom_op.base import CustomOp  # type: ignore[import-not-found]

from finn.dataflow._engine import Answer, Decided, RequestError, Unresolved
from finn.dataflow.model.declarations import (
    AuthoringError,
    Decision,
    OccurrenceContext,
    Problem,
    Space,
)
from finn.dataflow.model.occurrence import ProjectionAssessment, VariantView
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.ops.association import SourceAssociation
from finn.dataflow.ops.schema import (
    SOURCE_DECLARATION_TYPES,
    Attribute,
    BuildFact,
    DatatypeAttribute,
    SourceDeclaration,
    TensorDeclaration,
    lower_source_schema,
)
from finn.dataflow.ops.source import SourceError, SourceNode, read_source_node

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
    """One source node cannot be bound, projected, or committed.

    Carries engine ``Finding`` values when it has them, because a refusal that
    cannot say which declaration refused is a refusal nobody can act on.
    """

    def __init__(self, message: str, findings: Sequence[object] = ()) -> None:
        super().__init__(message)
        self.findings = tuple(findings)


@dataclass(frozen=True, slots=True)
class BuildFacts:
    """The scalars one operation extracted from the build configuration.

    Extracted, not referenced.  A design space holding a live config object
    would answer differently after an unrelated build setting changed, and
    nothing would detect it -- the fingerprint would not move, because nothing
    the fingerprint reads did.
    """

    values: Mapping[str, object]

    def __getitem__(self, name: str) -> object:
        return self.values[name]


@dataclass(frozen=True, slots=True)
class SourceBinding:
    """Everything a bound operation carries about the node it was bound to.

    Frozen and complete.  There is deliberately no ``ModelWrapper`` here and no
    build-configuration object: a bound occurrence that could reach either
    would be a bound occurrence that is not actually bound to anything.

    ``node`` is a deep copy.  ``node_identity`` is how the *live* node is found
    again when effects are applied, which is the only operation that needs it.
    """

    node: Any
    node_identity: str
    opset_version: int
    source: SourceNode
    build: BuildFacts


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


def source_declarations(
    operation_type: type[DataflowOp],
) -> tuple[tuple[str, SourceDeclaration], ...]:
    """One operation's source-schema members, in deterministic inherited order."""

    ordered: dict[str, SourceDeclaration] = {}
    for base in reversed(operation_type.__mro__):
        if not isinstance(base, type) or not issubclass(base, Space):
            continue
        for name, value in vars(base).items():
            if isinstance(value, SOURCE_DECLARATION_TYPES):
                ordered[name] = cast(SourceDeclaration, value)
    return tuple(ordered.items())


class DataflowOp(Space, CustomOp):  # type: ignore[misc]
    """One logical dataflow operation, and the root Space of its design.

    Two inheritances, and both are load-bearing.  ``Space`` contributes no
    ``__init__``; ``CustomOp``'s runs on the unbound path and is skipped on the
    bound one, where :meth:`_allocate_bound` states every field it sets.  The
    MRO therefore decides nothing.
    """

    #: The stable family identity written onto the node.  Two operations with
    #: the same family and version must agree about their source schema, their
    #: attribute names and their association, because a reload trusts them to.
    family: ClassVar[str] = ""
    family_version: ClassVar[str] = "1"

    #: The Decisions and selectors that survive a save, in application order.
    attributes: ClassVar[tuple[PersistentAttribute, ...]] = ()

    #: The namespace the root occurrence is started under.  Fixed per family so
    #: that a compiled path means the same thing in every graph.
    root_namespace: ClassVar[str] = "op"

    #: A class attribute, which is what QONNX declares and reads.  Defining it
    #: as a method here would make it accidentally truthy for every operation
    #: including one that meant to say no.
    wants_model: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: object) -> None:
        declarations = dict(vars(cls))
        schema = {
            name: cast(SourceDeclaration, value)
            for name, value in declarations.items()
            if isinstance(value, SOURCE_DECLARATION_TYPES)
        }
        for name, member in lower_source_schema(cls, schema).items():
            if name in declarations:
                raise AuthoringError(
                    f"{cls.__name__}.{name} collides with a member generated by the source "
                    "schema; rename the declaration whose expansion produced it"
                )
            setattr(cls, name, member)
        super().__init_subclass__(**kwargs)

    def __init__(self, onnx_node: NodeProto, onnx_opset_version: int = 1) -> None:
        """The *unbound* constructor.  QONNX calls this; ``bind`` does not."""

        CustomOp.__init__(self, onnx_node, onnx_opset_version)
        self._model: Any | None = None
        self._binding: SourceBinding | None = None
        _check_authoring(type(self))

    # -- allocation -----------------------------------------------------------

    @classmethod
    def _allocate_bound(cls, context: OccurrenceContext, binding: SourceBinding) -> DataflowOp:
        """Allocate one *bound* instance, stating every field it gets.

        ``CustomOp.__init__`` is deliberately not run: it would take the live
        node, and a bound occurrence answers from its frozen copy.  Everything
        the QONNX side reads is set explicitly here, so nothing depends on
        which constructor the MRO would have chosen.
        """

        del context
        instance = object.__new__(cls)
        instance.onnx_node = binding.node
        instance.onnx_opset_version = binding.opset_version
        instance._model = None
        instance._binding = binding
        return instance

    # -- binding --------------------------------------------------------------

    def bind(self, model: Any, build: Any) -> Any:
        """Read the graph once and return a bound occurrence of this class.

        The returned object is the same concrete operation class, attached, and
        with every persisted choice replayed onto it.
        """

        if self.recorded_scope_id() is None:
            raise SourceError(
                f"{self.onnx_node.name!r} has no dataflow scope id, so nothing can be "
                "addressed to it and no recorded choice can be attributed to it; run "
                "AssignDataflowScopeIds over the graph first.  Identity belongs to "
                "whoever constructs the node, and a query that allocated one would "
                "mutate the graph on read"
            )
        binding = self.read_binding(model, build)
        return type(self)._start_bound(binding, self._recorded_on(self.onnx_node))

    def rebind(self, model: Any, build: Any) -> Any:
        """Read the graph again, deliberately, and return a fresh bound occurrence.

        The explicit refresh.  A bound occurrence never notices a graph edit on
        its own -- that is the point of freezing -- so noticing one is an act
        the caller performs and can be seen doing.
        """

        return self.bind(model, build)

    @classmethod
    def _start_bound(cls, binding: SourceBinding, recorded: Mapping[str, object]) -> Any:
        facts = cls._problem_values(binding)
        root = cls.start(
            facts,
            namespace=cls.root_namespace,
            root_factory=lambda context: cls._allocate_bound(context, binding),
        )
        _check_recorded_identity(cls, binding, root)
        return _hydrate(cls, binding, root, recorded)

    @classmethod
    def _problem_values(cls, binding: SourceBinding) -> Mapping[Problem[Any], object]:
        """Lower the frozen reading onto this class's own Problem members.

        Mechanical: each declaration *is* the Problem it supplies, and says
        which operand or fact fills it, so nothing here is operation-specific
        and no operation writes a ``source_facts`` mapping by hand.
        """

        values: dict[Problem[Any], object] = {}
        for member_name, declaration in source_declarations(cls):
            problem = cast("Problem[Any]", declaration)
            if isinstance(declaration, TensorDeclaration):
                if not binding.source.has(member_name):
                    if not declaration.optional:
                        raise SourceError(
                            f"{binding.source.node_name} requires operand {member_name!r}"
                        )
                    continue
                values[problem] = binding.source.operand(member_name)
                continue
            if isinstance(declaration, BuildFact):
                if member_name in binding.build.values:
                    values[problem] = binding.build[member_name]
                continue
            values[problem] = binding.source.attributes[member_name]
        return values

    def read_binding(self, model: Any, build: Any) -> SourceBinding:
        """Freeze this node, its operands, and the build scalars, once."""

        operation = type(self)
        declarations = source_declarations(operation)
        inputs: list[tuple[int, str]] = []
        outputs: list[tuple[int, str]] = []
        optional: list[str] = []
        digested: list[str] = []
        attributes: dict[str, object] = {}
        build_values: dict[str, object] = {}
        for member_name, declaration in declarations:
            if isinstance(declaration, TensorDeclaration):
                (outputs if declaration.output else inputs).append((declaration.index, member_name))
                if declaration.optional:
                    optional.append(member_name)
                if declaration.fingerprint:
                    digested.append(member_name)
            elif isinstance(declaration, DatatypeAttribute):
                attributes[member_name] = _datatype_attribute(self, member_name, declaration)
            elif isinstance(declaration, Attribute):
                attributes[member_name] = declaration.value_type(self.get_nodeattr(member_name))
            elif isinstance(declaration, BuildFact):
                build_values[member_name] = _build_value(operation, member_name, declaration, build)
        source = read_source_node(
            model,
            self.onnx_node,
            inputs=tuple(name for _index, name in sorted(inputs)),
            outputs=tuple(name for _index, name in sorted(outputs)),
            optional_inputs=tuple(optional),
            digest_inputs=tuple(digested),
            attributes=attributes,
        )
        return SourceBinding(
            deepcopy(self.onnx_node),
            self.recorded_scope_id() or "",
            int(self.onnx_opset_version),
            source,
            BuildFacts(MappingProxyType(build_values)),
        )

    # -- what a bound occurrence knows ----------------------------------------

    @property
    def binding(self) -> SourceBinding:
        """The frozen source this occurrence was bound to."""

        if self._binding is None:
            raise DataflowOpError(
                f"{type(self).__name__} is not bound; call bind(model, build) to read the "
                "graph once and get an attached occurrence"
            )
        return self._binding

    @property
    def source(self) -> SourceNode:
        """The frozen reading of the node.  The live graph is not consulted."""

        return self.binding.source

    @property
    def is_bound(self) -> bool:
        return self._binding is not None

    # -- QONNX integration ----------------------------------------------------

    def attach_model(self, model: Any) -> DataflowOp:
        """Borrow a read reference, exactly as QONNX's own contract states.

        On an unbound instance the reference is kept for ``bind`` to read.  On
        a bound one this is a no-op: a bound occurrence answers from its frozen
        snapshot and attaching a graph to it would suggest otherwise.  Neither
        path mutates the graph.
        """

        if self._binding is None:
            self._model = model
        return self

    def get_nodeattr_types(self) -> Mapping[str, tuple[str, bool, object]]:
        declared: dict[str, tuple[str, bool, object]] = {
            SCOPE_ID_ATTRIBUTE: ("s", False, ""),
            FAMILY_ATTRIBUTE: ("s", False, ""),
            FAMILY_VERSION_ATTRIBUTE: ("s", False, ""),
            FINGERPRINT_ATTRIBUTE: ("s", False, ""),
        }
        for member_name, declaration in source_declarations(type(self)):
            if isinstance(declaration, DatatypeAttribute):
                declared[member_name] = ("s", False, declaration.default)
            elif isinstance(declaration, Attribute):
                kind = "s" if declaration.value_type is str else "i"
                declared[member_name] = (kind, False, declaration.default)
        for item in type(self).attributes:
            kind = "s" if isinstance(item, SelectorAttribute) else item.codec.kind
            declared[item.name] = (kind, False, "" if kind == "s" else 0)
        return declared

    def make_shape_compatible_op(self, model: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not declare a shape-compatible op")

    def infer_node_datatype(self, model: Any) -> None:
        """Datatypes are source facts this layer reads, never ones it invents."""

    def execute_node(self, context: Any, graph: Any) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not execute its source semantics")

    def verify_node(self) -> list[str]:
        return []

    # -- identity -------------------------------------------------------------

    def recorded_scope_id(self) -> str | None:
        """The scope id physically present on the node, or ``None``.

        A pure read.  Allocation belongs to the transformation that constructs
        the node, and a query that allocated one would be a query that mutates
        the graph -- which QONNX's own ``attach_model`` contract forbids and
        which would make two reads of the same graph disagree.
        """

        value = _string_attribute(self.onnx_node, SCOPE_ID_ATTRIBUTE)
        return value or None

    # -- the two projections --------------------------------------------------

    @property
    def dataflow(self) -> ProjectionAssessment[DataflowNetwork]:
        """Readiness, constraint acceptance and the selected Network.

        A full assessment rather than a bare ``Answer``.  Collapsing readiness,
        validity and availability into one value is what lets a rejected point
        be handed on as though it were merely incomplete.
        """

        raise NotImplementedError(f"{type(self).__name__} does not project a Network")

    @property
    def network(self) -> Answer[DataflowNetwork]:
        """The accepted Network, for a caller that wants only the reduction."""

        return self.dataflow.accepted_answer

    @property
    def association(self) -> Answer[SourceAssociation]:
        """Where each source operand's data crosses the selected Network."""

        raise NotImplementedError(f"{type(self).__name__} does not declare its association")

    # -- persistence ----------------------------------------------------------

    def graph_effects(self, *, require: Any = None) -> Any:
        """Plan the one change that records this occurrence's choices.

        Reads a point and produces a value; it writes nothing.  What it
        serializes is the *reachable* point, not the patch that produced it: a
        choice whose Decision is finally inapplicable here -- a weight-supply
        mode under a Design alternative no longer selected -- is dropped, so a
        reload cannot fail replaying something that has nowhere to go.
        """

        # Imported here because ``persistence`` names this module's attribute
        # authority, and the authority must not depend on the writer.
        from finn.dataflow.ops.persistence import (  # noqa: PLC0415
            CommitmentStage,
            GraphEffects,
            check_commitment,
            node_digest,
        )

        binding = self.binding
        if not binding.node_identity:
            raise DataflowOpError(
                "this operation has no dataflow scope id, so a change cannot be addressed "
                "to it; run AssignDataflowScopeIds over the graph first"
            )
        stage = CommitmentStage.DATAFLOW if require is None else require
        check_commitment({CommitmentStage.DATAFLOW: self.dataflow}, stage)

        encoded: dict[str, int | str] = {}
        for item in type(self).attributes:
            value = _reachable_value(self, item)
            if value is _UNREACHABLE:
                continue
            encoded[item.name] = (
                STRING_CODEC.encode(value)
                if isinstance(item, SelectorAttribute)
                else item.codec.encode(value)
            )
        return GraphEffects(
            binding.node_identity,
            type(self).family,
            type(self).family_version,
            stage,
            self.problem_fingerprint,
            node_digest(binding.node),
            MappingProxyType(encoded),
            MappingProxyType(dict(self.graph_datatypes())),
            MappingProxyType(dict(self.graph_shapes())),
        )

    def graph_shapes(self) -> Mapping[str, tuple[int, ...]]:
        """The output shapes this operation is authoritative for.  Override."""

        return {}

    def graph_datatypes(self) -> Mapping[str, Any]:
        """The output datatypes this operation is authoritative for.  Override."""

        return {}

    def commit(self, model: Any, *, require: Any = None) -> Any:
        """Plan and apply in one step, through exactly the path above."""

        from finn.dataflow.ops.persistence import apply_graph_effects  # noqa: PLC0415

        return apply_graph_effects(model, self.graph_effects(require=require))

    def recorded(self) -> Mapping[str, object]:
        """The choices this object represents.

        Two states, two honest answers.  An *unbound* operation is a view on a
        node, so it reports what is physically written there.  A *bound* one is
        a point, so it reports the choices reachable at that point -- which is
        what a later ``graph_effects`` would write, and which is not the same
        thing as the frozen node it was bound from.  Conflating them would make
        ``recorded()`` on a successor report the state before the assignment
        that produced it.
        """

        if self._binding is None:
            return self._recorded_on(self.onnx_node)
        present: dict[str, object] = {}
        for item in type(self).attributes:
            value = _reachable_value(self, item)
            if value is not _UNREACHABLE:
                present[item.name] = value
        return MappingProxyType(present)

    @classmethod
    def _recorded_on(cls, node: Any) -> Mapping[str, object]:
        present: dict[str, object] = {}
        for item in cls.attributes:
            attribute = _attribute(node, item.name)
            if attribute is None:
                continue
            raw = attribute.s if attribute.type == attribute.STRING else attribute.i
            present[item.name] = (
                STRING_CODEC.decode(raw)
                if isinstance(item, SelectorAttribute)
                else item.codec.decode(raw)
            )
        return MappingProxyType(present)


#: "This choice is not reachable at this point", distinguished from every real
#: value a Decision could hold -- including ``None``.
_UNREACHABLE = object()


def _reachable_value(operation: DataflowOp, item: PersistentAttribute) -> object:
    """One recorded choice, or ``_UNREACHABLE`` if it does not arise here.

    Navigation itself can fail -- a Decision under an unselected alternative has
    no occurrence to be read from -- and that is a legitimate answer, not an
    error.  It is the difference between "this point does not have that dial"
    and "this point is broken".
    """

    try:
        answer = item.read(operation)
    except (DataflowOpError, AuthoringError, RequestError):
        return _UNREACHABLE
    if isinstance(answer, Decided):
        return answer.value
    return _UNREACHABLE


def _check_authoring(operation_type: type[DataflowOp]) -> None:
    if not operation_type.family:
        raise AuthoringError(f"{operation_type.__name__} must declare a non-empty family")
    names = [item.name for item in operation_type.attributes]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise AuthoringError(
            f"{operation_type.__name__} persists attribute {duplicates[0]!r} twice"
        )
    clashes = sorted(set(names) & RESERVED_ATTRIBUTES)
    if clashes:
        raise AuthoringError(
            f"{operation_type.__name__} persists {clashes[0]!r}, which the dataflow layer owns"
        )


def _datatype_attribute(
    operation: DataflowOp, member_name: str, declaration: DatatypeAttribute
) -> object:
    from finn.dataflow.datatypes import canonical_qonnx_datatype  # noqa: PLC0415 - cycle
    from qonnx.core.datatype import DataType  # type: ignore[import-not-found] # noqa: PLC0415

    raw = operation.get_nodeattr(member_name) or declaration.default
    name = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return canonical_qonnx_datatype(DataType[name])


def _build_value(
    operation_type: type[DataflowOp],
    member_name: str,
    declaration: BuildFact,
    build: Any,
) -> object:
    try:
        return declaration.accessor(build)
    except (AttributeError, KeyError, TypeError) as error:
        if declaration.required and declaration.default is None:
            raise DataflowOpError(
                f"{operation_type.__name__} needs build fact {member_name!r}, and this "
                f"build configuration does not supply it: {error}"
            ) from error
        return declaration.default


def _check_recorded_identity(
    operation_type: type[DataflowOp], binding: SourceBinding, root: Space
) -> None:
    node = binding.node
    name = getattr(node, "name", "<node>")
    stored = _string_attribute(node, FINGERPRINT_ATTRIBUTE)
    if stored and stored != root.problem_fingerprint:
        raise DataflowOpError(
            f"{name} stores choices made against a different problem "
            f"(recorded {stored}, current {root.problem_fingerprint}); "
            "reconstruct explicitly rather than reinterpreting them"
        )
    family = _string_attribute(node, FAMILY_ATTRIBUTE)
    if family and family != operation_type.family:
        raise DataflowOpError(
            f"{name} stores choices for family {family!r}, not {operation_type.family!r}"
        )
    version = _string_attribute(node, FAMILY_VERSION_ATTRIBUTE)
    if version and version != operation_type.family_version:
        raise DataflowOpError(
            f"{name} stores choices for {family}@{version}, "
            f"and this build offers {operation_type.family_version}"
        )


def _hydrate(
    operation_type: type[DataflowOp],
    binding: SourceBinding,
    root: Space,
    recorded: Mapping[str, object],
) -> Any:
    """Replay the recorded choices through the ordinary public assignment API.

    Selectors first, then Decisions: which alternative is live decides which
    Decisions exist at all.  There is no second path by which a point comes
    into existence.
    """

    ordered: tuple[PersistentAttribute, ...] = (
        *(item for item in operation_type.attributes if isinstance(item, SelectorAttribute)),
        *(item for item in operation_type.attributes if isinstance(item, DecisionAttribute)),
    )
    for item in ordered:
        if item.name not in recorded:
            continue
        try:
            root = item.apply(root, recorded[item.name])
        except (RequestError, AuthoringError, ValueError, TypeError) as error:
            raise DataflowOpError(
                f"{binding.source.node_name} attribute {item.name!r} does not replay onto "
                f"this problem: {error}",
                getattr(error, "findings", ()),
            ) from error
    return root


def _attribute(node: Any, name: str) -> Any | None:
    for attribute in node.attribute:
        if attribute.name == name:
            return attribute
    return None


def _string_attribute(node: Any, name: str) -> str:
    attribute = _attribute(node, name)
    if attribute is None:
        return ""
    value = attribute.s
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


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
    "BuildFacts",
    "DataflowOp",
    "DataflowOpError",
    "DecisionAttribute",
    "PersistentAttribute",
    "SelectorAttribute",
    "SourceBinding",
    "source_declarations",
    "unresolved_reason",
]
