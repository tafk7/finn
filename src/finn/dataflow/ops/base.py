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

**Two deviations, stated rather than implied.**

*The projection is composed in Python, not compiled.*  ``DataflowOp`` is the
root Space, but ``dataflow`` is a generic composition of this operation's own
source assessment with the selected Design's assessment -- not one flat
compiled ``Projection`` declaration.  It has to be: a child's projection
*obligations* do not travel with its exported value, and which child is
selected is a runtime fact.  ``combine_assessments`` centralizes the reduction
so there is still exactly one definition of the normative ordering, but this is
not the "synthesize a Projection declaration" shape the Kernel and Design
layers use, and it should not be described as though it were.

*No folding abstraction is introduced.*  ``PE``, ``SIMD``, tiling and the rest
remain ordinary Decisions owned by the Design whose Region or Network structure
they affect.  They are persisted because they are committed Decisions beneath
the root, discovered generically -- not because anything here knows what
folding is.  A common optimization interface is deferred until several
materially different folding paradigms can inform it.

**Persistence is one authority.**  Everything that survives a save lives on the
node: the operation's scope id and one canonical state document.  What goes
into that document is *discovered from the compiled model* -- every selector
and every committed Decision beneath this root, named by root-relative compiled
path -- so an operation never lists its own choices and a Decision added three
levels down is persisted without anyone editing the operation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, cast

from qonnx.custom_op.base import CustomOp  # type: ignore[import-not-found]

from finn.dataflow._engine import Answer, Decided, RequestError, Unresolved
from finn.dataflow.model.declarations import (
    AuthoringError,
    Constraint as DeclaredConstraint,
    ConstraintGroup,
    OccurrenceContext,
    PersistentCodec,
    Problem,
    Space,
    check_canonical,
)
from finn.dataflow.model.occurrence import (
    PersistableChoice,
    ProjectionAssessment,
    occurrence_answer_at,
    occurrence_commit_paths,
    occurrence_persistable,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.ops.association import SourceAssociation
from finn.dataflow.ops.schema import (
    SOURCE_DECLARATION_TYPES,
    Attribute,
    BuildFact,
    DatatypeAttribute,
    SourceDeclaration,
    InputTensor,
    OutputTensor,
    lower_source_schema,
)
from finn.dataflow.ops.source import SourceError, SourceNode, read_source_node
from finn.dataflow.ops.state import (
    SELECTOR_CODEC,
    STATE_ATTRIBUTE,
    Assignment,
    DataflowState,
    DecisionCodec,
    DecodeError,
    decode_dataflow_state,
    encode_state,
    enum_codec,
    structural_codec,
)

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
    {
        SCOPE_ID_ATTRIBUTE,
        FAMILY_ATTRIBUTE,
        FAMILY_VERSION_ATTRIBUTE,
        FINGERPRINT_ATTRIBUTE,
        STATE_ATTRIBUTE,
    }
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

    #: The node as *bytes*, not as a protobuf.  A frozen dataclass holding a
    #: mutable message is not frozen: every successor shares one binding, so a
    #: single ``set_nodeattr`` would edit the snapshot the whole lineage reads.
    node_bytes: bytes
    node_identity: str
    opset_version: int
    source: SourceNode
    build: BuildFacts

    def materialize(self) -> Any:
        """A fresh private ``NodeProto`` for one occurrence to hold."""

        from onnx import NodeProto  # noqa: PLC0415 - deferred; onnx is a heavy import

        node = NodeProto()
        node.ParseFromString(self.node_bytes)
        return node


#: Kept as the layer's name for a Decision codec.  It is
#: :class:`~finn.dataflow.ops.state.DecisionCodec`: one identity, one version,
#: and an explicit pair of functions.  No ``repr``/``eval``, and no ONNX
#: attribute *kind*, because a value no longer occupies an attribute of its own
#: -- it is one entry in the canonical state document.
AttributeCodec = DecisionCodec


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

    #: The namespace the root occurrence is started under.  Fixed per family so
    #: that a compiled path means the same thing in every graph.
    root_namespace: ClassVar[str] = "op"

    #: A class attribute, which is what QONNX declares and reads.  Defining it
    #: as a method here would make it accidentally truthy for every operation
    #: including one that meant to say no.
    wants_model: ClassVar[bool] = True

    #: The constraints that gate this operation's *own* semantics, as distinct
    #: from any Design's.  Empty by default and combined into ``dataflow`` by
    #: the root projection, so a declared group is never compiled-but-unread.
    source_accepts: ClassVar[Any] = ConstraintGroup()

    def __init_subclass__(cls, **kwargs: object) -> None:
        declarations = dict(vars(cls))
        schema = {
            name: cast(SourceDeclaration, value)
            for name, value in declarations.items()
            if isinstance(value, SOURCE_DECLARATION_TYPES)
        }
        _check_indices(cls, schema)
        _check_constraints_are_classified(cls, declarations)
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
        # A private copy per occurrence, so a successor cannot observe -- or be
        # edited through -- the node its predecessor holds.
        instance.onnx_node = binding.materialize()
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

        return self._bind_with(model, self.build_facts(build))

    def _bind_with(self, model: Any, facts: BuildFacts) -> Any:
        """Bind against build scalars that have already been extracted."""

        binding = self._binding_with(model, facts)
        try:
            state = decode_dataflow_state(binding.materialize())
        except DecodeError as error:
            raise DataflowOpError(f"{binding.source.node_name}: {error}") from error
        return type(self)._start_bound(binding, state)

    def rebind(self, model: Any, build: Any = None) -> Any:
        """Locate the live node by scope id and read it again, deliberately.

        The explicit refresh, and it must go to the *live* node.  Calling
        ``bind`` on ``self`` would re-read the frozen copy this occurrence
        already holds: the tensor names, the node attributes, the operand list
        and the node name would all be the ones captured at binding time, so a
        refresh would faithfully reproduce the state it was supposed to notice
        had changed.  Only the datatype of an unchanged tensor would move, which
        is exactly the narrow case that hides the defect.

        Identity does the locating, not the object: the node may have been
        renamed, and the wrapper may outlive the graph it came from.

        ``build`` is optional, and its absence is not a default -- it is the
        occurrence's *own* frozen :class:`BuildFacts`, reused exactly.  Reading
        the graph again and reading the build configuration again are two
        different intentions, and only one of them is what a refresh after a
        commit means.  Passing a build is the explicit "and read the build
        context again" form.
        """

        from finn.dataflow.ops.persistence import find_node  # noqa: PLC0415 - see graph_effects

        node = find_node(model, self.binding.node_identity)
        fresh = type(self)(node, self.binding.opset_version)
        if build is None:
            return fresh._bind_with(model, self.binding.build)
        return fresh.bind(model, build)

    @classmethod
    def _start_bound(cls, binding: SourceBinding, recorded: DataflowState | None) -> Any:
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
            if isinstance(declaration, OutputTensor):
                # An observation, never a Problem: it does not reach the point.
                continue
            if isinstance(declaration, InputTensor):
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
        """Freeze the live node, its operands, and the build scalars, once.

        The *live* node, resolved from the supplied model by scope identity --
        the same rule ``rebind`` uses, so there is one.  Reading structure off
        ``self.onnx_node`` while reading tensor facts from ``model`` would let
        a wrapper from one graph and a same-named node in another combine into
        an occurrence describing neither.
        """

        return self._binding_with(model, self.build_facts(build))

    def build_facts(self, build: Any) -> BuildFacts:
        """The scalars this operation extracts from one build configuration.

        Separated from the reading of the node because the two are compared
        independently: a commit checks that a caller-supplied build says the
        same thing this occurrence already froze, and it must be able to do
        that *before* it touches the graph.
        """

        operation = type(self)
        values: dict[str, object] = {}
        for member_name, declaration in source_declarations(operation):
            if isinstance(declaration, BuildFact):
                values[member_name] = _build_value(operation, member_name, declaration, build)
        return BuildFacts(MappingProxyType(values))

    def _binding_with(self, model: Any, facts: BuildFacts) -> SourceBinding:
        node = self._live_node(model)
        return SourceBinding(
            node.SerializeToString(deterministic=True),
            self.recorded_scope_id() or "",
            int(self.onnx_opset_version),
            self._read_source(model, node),
            facts,
        )

    def _read_source(self, model: Any, node: Any) -> SourceNode:
        """The operand and attribute reading both entry points share.

        Everything here comes from the node and the model.  Nothing here needs
        a build configuration, which is exactly why QONNX's shape and datatype
        passes can use it.
        """

        operation = type(self)
        inputs: list[tuple[int, str]] = []
        outputs: list[tuple[int, str]] = []
        optional: list[str] = []
        digested: list[str] = []
        attributes: dict[str, object] = {}
        for member_name, declaration in source_declarations(operation):
            if isinstance(declaration, (InputTensor, OutputTensor)):
                (outputs if declaration.output else inputs).append((declaration.index, member_name))
                if isinstance(declaration, InputTensor):
                    if declaration.optional:
                        optional.append(member_name)
                    if declaration.fingerprint_initializer:
                        digested.append(member_name)
            elif isinstance(declaration, DatatypeAttribute):
                attributes[member_name] = _datatype_attribute(node, member_name, declaration)
            elif isinstance(declaration, Attribute):
                attributes[member_name] = _plain_attribute(node, member_name, declaration)
        return read_source_node(
            model,
            node,
            inputs=_positional(inputs),
            outputs=_positional(outputs),
            optional_inputs=tuple(optional),
            digest_inputs=tuple(digested),
            attributes=attributes,
        )

    def is_stale(self, problem: Any = None) -> bool:
        """Whether this occurrence's frozen source still matches the graph.

        Overridden because the inherited form re-reads a *problem source*, and
        a bound operation's problem source is a frozen record that by
        construction never changes.  Staleness here is a question about the
        graph, so it takes the graph.
        """

        if problem is None:
            raise DataflowOpError(
                "a DataflowOp's staleness is relative to a model and a build; call "
                "is_stale((model, build)) or compare rebind(...).problem_fingerprint"
            )
        model, build = problem
        from finn.dataflow.ops.persistence import find_node  # noqa: PLC0415

        node = find_node(model, self.binding.node_identity)
        fresh = type(self)(node, self.binding.opset_version)
        # Deliberately not ``rebind``: hydration refuses a recorded fingerprint
        # that no longer matches, and refusing is the very thing this method
        # exists to let a caller ask about without triggering.
        binding = fresh.read_binding(model, build)
        started = type(self).start(
            type(self)._problem_values(binding),
            namespace=type(self).root_namespace,
            root_factory=lambda context: type(self)._allocate_bound(context, binding),
        )
        return bool(started.problem_fingerprint != self.problem_fingerprint)

    def reconstruct(self, problem: Any = None, **kwargs: Any) -> Any:
        """Start again from this occurrence's own frozen source, with no choices.

        The way a *structural* choice is changed.  An immutable point does not
        rebase a committed selector, so switching from one Design alternative to
        another means a point that never had the first one -- which is this, and
        which keeps the same frozen binding and therefore the same problem
        identity.

        The inherited arbitrary-problem form is refused.  Combining a
        caller-supplied problem mapping with this operation's root factory would
        produce an occurrence whose Problem values and whose frozen
        ``SourceBinding`` describe different nodes, and every answer downstream
        would be drawn from whichever of the two happened to be consulted.
        """

        if problem is not None:
            raise DataflowOpError(
                "a DataflowOp reconstructs from the node it is bound to, not from an "
                "arbitrary problem mapping; use rebind(model, build) to read a changed "
                "graph, or reconstruct() to drop this occurrence's choices"
            )
        binding = self.binding
        expected = kwargs.get("expected_problem_fingerprint")
        if expected is not None and expected != self.problem_fingerprint:
            raise DataflowOpError(
                f"this operation's problem is {self.problem_fingerprint}, not {expected}"
            )
        return type(self)._start_bound(binding, None)

    def _live_node(self, model: Any) -> Any:
        """The one node in this model this wrapper addresses."""

        from finn.dataflow.ops.persistence import find_node  # noqa: PLC0415 - see graph_effects

        scope = self.recorded_scope_id()
        if scope is None:
            raise SourceError(
                f"{self.onnx_node.name!r} has no dataflow scope id, so nothing can be "
                "addressed to it and no recorded choice can be attributed to it; run "
                "AssignDataflowScopeIds over the graph first.  Identity belongs to "
                "whoever constructs the node, and a query that allocated one would "
                "mutate the graph on read"
            )
        try:
            return find_node(model, scope)
        except DataflowOpError as error:
            raise SourceError(str(error)) from error

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

    def set_nodeattr(self, name: str, value: Any) -> None:
        """Refused on a bound occurrence, which is a snapshot and not the graph.

        Inherited from ``CustomOp`` and perfectly reasonable there.  Here it
        would edit a frozen copy: the write would appear to succeed, the live
        graph would be untouched, and every later question would be answered
        from a snapshot that no longer matches anything.
        """

        if self._binding is not None:
            raise DataflowOpError(
                f"{type(self).__name__} is a bound occurrence and its node is a frozen "
                f"snapshot; {name!r} cannot be set through it.  Plan the change with "
                "graph_effects() and apply it with apply_graph_effects(model, effects)"
            )
        CustomOp.set_nodeattr(self, name, value)

    def get_nodeattr_types(self) -> Mapping[str, tuple[str, bool, object]]:
        declared: dict[str, tuple[str, bool, object]] = {
            SCOPE_ID_ATTRIBUTE: ("s", False, ""),
            STATE_ATTRIBUTE: ("s", False, ""),
        }
        for member_name, declaration in source_declarations(type(self)):
            if isinstance(declaration, DatatypeAttribute):
                declared[member_name] = ("s", False, declaration.default)
            elif isinstance(declaration, Attribute):
                kind = "s" if declaration.value_type is str else "i"
                declared[member_name] = (kind, False, declaration.default)
        return declared

    def source_snapshot(self, model: Any) -> SourceNode:
        """Read this node's operands and attributes, needing no build context.

        The QONNX boundary.  ``InferShapes`` and ``InferDataTypes`` run on the
        ordinary wrapper from ``get_customop_wrapper`` and have no synthesis
        configuration to give -- nor should they need one, because an output
        shape is a fact about the *operation*, not about the target it will be
        built for.

        So the source reading splits from the design-space binding: this needs
        a NodeProto and a ModelWrapper, and ``bind`` additionally freezes the
        build facts a Design and its Kernels require.  A bound occurrence
        answers from the snapshot it already has, so both paths reach the same
        declaration-owned output formula and cannot drift.
        """

        if self._binding is not None:
            return self._binding.source
        return self.read_source(model)

    def make_shape_compatible_op(self, model: Any) -> Any:
        """A shape-compatible stand-in for QONNX shape inference.

        Applies no transaction: shape inference asks what this node produces,
        it does not commit a design.
        """

        expected = self.expected_for(self.source_snapshot(model))
        first = next(iter(self.onnx_node.output), None)
        entry = expected.get(self._output_operand())
        if first is None or entry is None or entry[0] is None:
            raise DataflowOpError(
                f"{self.onnx_node.name} cannot state a shape-compatible op from these operands"
            )
        from onnx import helper  # noqa: PLC0415 - deferred, as above

        return helper.make_node("RandomNormal", [], [first], shape=list(entry[0]))

    def infer_node_datatype(self, model: Any) -> None:
        """QONNX's in-place API, taking its value from the same derivation."""

        expected = self.expected_for(self.source_snapshot(model))
        entry = expected.get(self._output_operand())
        if entry is None or entry[1] is None:
            return
        model.set_tensor_datatype(self.onnx_node.output[0], entry[1])

    def _output_operand(self) -> str:
        for member_name, declaration in source_declarations(type(self)):
            if isinstance(declaration, OutputTensor) and declaration.index == 0:
                return member_name
        raise DataflowOpError(f"{type(self).__name__} declares no output operand")

    def read_source(self, model: Any) -> SourceNode:
        """Read the live node into a frozen record, without any build context."""

        return self._read_source(model, self._live_node(model))

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
        """The operation's own projection: source semantics *and* the Network.

        Exporting the selected Design's ``network`` is necessary and not
        sufficient.  A value crosses the boundary; a projection's *obligations*
        do not.  So this combines four things, and each is load-bearing:

        - the operation's own ``source_accepts`` constraints, which would
          otherwise be compiled and consulted by nothing;
        - the selected Design's readiness, so an unresolved child is an
          unresolved operation;
        - the selected Design's constraints, so a Design that refuses this
          point makes the operation refuse it;
        - the Design's Network as the output.

        An *unselected* Design is an ordinary ``Unresolved``, not an exception.
        "Which alternative" is a Decision like any other, and a caller
        inspecting a half-configured operation should get an answer describing
        that, not a traceback.
        """

        return _root_projection(self)

    @property
    def network(self) -> Answer[DataflowNetwork]:
        """The accepted Network, for a caller that wants only the reduction."""

        return self.dataflow.accepted_answer

    def selected_dataflow(self) -> ProjectionAssessment[DataflowNetwork] | None:
        """The selected child Design's dataflow assessment, or ``None``.

        ``None`` means "the structural choice is still open", which is a point
        state and not an error.  Override to route; the combination with this
        operation's own source semantics is done for you.
        """

        raise NotImplementedError(f"{type(self).__name__} does not route to a Design")

    @property
    def association(self) -> Answer[SourceAssociation]:
        """Where each source operand's data crosses the selected Network."""

        raise NotImplementedError(f"{type(self).__name__} does not declare its association")

    # -- reconciliation --------------------------------------------------------

    def expected_for(self, source: SourceNode) -> Mapping[str, tuple[tuple[int, ...] | None, Any]]:
        """What this operation produces, given one frozen source reading.

        Written once and read by everything: the graph effects that repair an
        annotation, QONNX's shape and datatype passes, and the reconciliation
        report.  It takes the reading rather than reading ``self``, so the
        unbound QONNX path and the bound design-space path cannot end up with
        two formulas that disagree.

        Keyed by operand id.  ``None`` in either position means "this operation
        does not claim that", not "empty".
        """

        del source
        return {}

    def expected_outputs(self) -> Mapping[str, tuple[tuple[int, ...] | None, Any]]:
        """:meth:`expected_for` over this bound occurrence's own source."""

        return self.expected_for(self.source)

    def reconciliation(self) -> tuple[str, ...]:
        """Where the graph's annotations differ from what this operation derives.

        A *difference*, not a rejection.  A stale output annotation is the thing
        ``graph_effects`` exists to repair, so it must not also refuse the
        projection whose acceptance is required to commit that repair --
        otherwise the repair can never be applied and the node stays wrong.
        """

        differences: list[str] = []
        for operand_id, (shape, datatype) in self.expected_outputs().items():
            if not self.source.has(operand_id):
                continue
            observed = self.source.operand(operand_id)
            if shape is not None and observed.annotated and tuple(observed.shape) != tuple(shape):
                differences.append(
                    f"{operand_id}: the graph annotates shape {tuple(observed.shape)}, "
                    f"this operation produces {tuple(shape)}"
                )
            if datatype is not None and observed.datatype != datatype:
                differences.append(
                    f"{operand_id}: the graph annotates datatype {observed.datatype.name}, "
                    f"this operation produces {datatype.name}"
                )
        return tuple(differences)

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
        )

        binding = self.binding
        if not binding.node_identity:
            raise DataflowOpError(
                "this operation has no dataflow scope id, so a change cannot be addressed "
                "to it; run AssignDataflowScopeIds over the graph first"
            )
        stage = CommitmentStage.DATAFLOW if require is None else require
        check_commitment({CommitmentStage.DATAFLOW: self.dataflow}, stage)

        encoded = _serialize_point(self)
        return GraphEffects(
            binding.node_identity,
            type(self).family,
            type(self).family_version,
            stage,
            self.problem_fingerprint,
            sha256(binding.node_bytes).hexdigest(),
            encode_state(
                family=type(self).family,
                family_version=type(self).family_version,
                problem_fingerprint=self.problem_fingerprint,
                commitment_stage=stage.value,
                assignments=encoded,
            ),
            MappingProxyType(
                {
                    self.source.operand(operand_id).tensor: datatype
                    for operand_id, (_shape, datatype) in self.expected_outputs().items()
                    if datatype is not None and self.source.has(operand_id)
                }
            ),
            MappingProxyType(
                {
                    self.source.operand(operand_id).tensor: tuple(shape)
                    for operand_id, (shape, _datatype) in self.expected_outputs().items()
                    if shape is not None and self.source.has(operand_id)
                }
            ),
        )

    def commit(self, model: Any, build: Any = None, *, require: Any = None) -> Any:
        """Plan, apply, and return a bound occurrence of the post-commit graph.

        The lifecycle continues across the mutation boundary.  Returning the
        live ``NodeProto`` would end it there and force every caller to rebind
        by hand -- and a caller who forgot would carry on asking questions of an
        occurrence bound to the graph as it was *before* their own commit.

        **The build is this occurrence's own.**  A bound occurrence already
        froze its build scalars, and the effects being applied were derived
        from *those*.  Rebinding under a different build would leave the graph
        holding choices made against one configuration and hand back an
        occurrence describing another -- and the failure would surface as a
        successful commit, which is worse than an error.

        So ``build`` is optional and, when given, is checked for equivalence
        **before** anything is applied.  A mismatch raises with the graph
        untouched.  Changing build context is ``rebind(model, new_build)``,
        which is a separate intention and reads the graph rather than writing
        it.
        """

        from finn.dataflow.ops.persistence import apply_graph_effects  # noqa: PLC0415

        frozen = self.binding.build
        if build is not None:
            offered = self.build_facts(build)
            if dict(offered.values) != dict(frozen.values):
                differences = sorted(
                    name
                    for name in set(offered.values) | set(frozen.values)
                    if offered.values.get(name) != frozen.values.get(name)
                )
                raise DataflowOpError(
                    f"{self.source.node_name} is bound to build facts that differ from the "
                    f"ones offered at commit ({', '.join(differences)}); the effects about "
                    "to be applied were derived from the frozen ones.  Nothing was written; "
                    "call rebind(model, build) first if the build context really changed"
                )
        effects = self.graph_effects(require=require)
        apply_graph_effects(model, effects)
        return self.rebind(model)

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
        return MappingProxyType(
            {
                choice.path: answer.value
                for choice in occurrence_persistable(self)
                if isinstance(answer := occurrence_answer_at(self, choice.reference), Decided)
            }
        )

    @classmethod
    def _recorded_on(cls, node: Any) -> Mapping[str, object]:
        """Every choice physically present on this node, by compiled path."""

        try:
            state = decode_dataflow_state(node)
        except DecodeError as error:
            raise DataflowOpError(f"{node.name}: {error}") from error
        if state is None:
            return MappingProxyType({})
        return MappingProxyType({path: item.value for path, item in state.assignments.items()})


def _root_projection(operation: DataflowOp) -> ProjectionAssessment[DataflowNetwork]:
    """Combine the operation's own source semantics with its selected Design's.

    The reduction is the engine's, not a second one: readiness and constraint
    assessments are concatenated and handed to the same
    ``_reduce_projection`` ordering every other projection uses, so
    "Unresolved dominates, then absence, then refusal" cannot come to mean two
    slightly different things in two places.
    """

    from finn.dataflow.model.occurrence import (  # noqa: PLC0415 - see graph_effects
        combine_assessments,
    )

    source = operation.assess(type(operation).source_accepts)
    selected = operation.selected_dataflow()
    if selected is None:
        return cast(
            "ProjectionAssessment[DataflowNetwork]",
            combine_assessments("dataflow", (source,), None),
        )
    return cast(
        "ProjectionAssessment[DataflowNetwork]",
        combine_assessments("dataflow", (source, *selected.constraints), selected),
    )


def _codec_for(operation: DataflowOp, choice: PersistableChoice) -> PersistentCodec[Any]:
    """The codec that writes one choice, or a refusal naming the declaration.

    Refused when the model is first walked rather than when a value happens to
    need writing, so an unpersistable Decision is found by any test that saves
    anything -- not by the one configuration that first commits it.
    """

    if choice.selector:
        return SELECTOR_CODEC
    if choice.codec is not None:
        return cast("PersistentCodec[Any]", choice.codec)
    token = choice.reference.semantics.type_token
    if isinstance(token, type) and issubclass(token, Enum):
        return enum_codec(token)
    structural = structural_codec(token)
    if structural is not None:
        return structural
    raise AuthoringError(
        f"{choice.owner}.{choice.member} is a committable Decision of "
        f"{choice.reference.semantics.name}, which no structural codec covers; "
        "give it canonical=PersistentCodec(...).  A persisted value is never a repr"
    )


def _serialize_point(operation: DataflowOp) -> dict[str, Assignment]:
    """Every committed, reachable choice beneath this root, discovered generically.

    Nothing here knows what operation this is.  The choices come from the
    compiled model, their names are root-relative compiled paths, and their
    codecs come from their own declarations -- so a Decision added to a Design
    three levels down is persisted without anybody editing the operation, and
    two Decisions called ``tile`` in different subspaces cannot collide.

    *Reachable* is the filter: a Decision that is finally inapplicable at this
    point -- one under an alternative no longer selected -- is dropped rather
    than written and then unreplayable.
    """

    encoded: dict[str, Assignment] = {}
    for choice in occurrence_persistable(operation):
        codec = _codec_for(operation, choice)
        answer = occurrence_answer_at(operation, choice.reference)
        if not isinstance(answer, Decided):
            continue
        encoded[choice.path] = Assignment(
            codec.tag, check_canonical(codec.encode(answer.value), f"the codec {codec.tag}")
        )
    return encoded


def _check_indices(operation_type: type, schema: Mapping[str, Any]) -> None:
    """Refuse a sparse or repeated operand schema where it is written.

    An index is an *index*, not a hint about ordering.  Sorting by it and
    handing the names to a positional reader silently reinterprets
    ``InputTensor(index=3)`` as "the next one", which is exactly wrong the
    moment an operation declares a sparse or out-of-order schema -- and that is
    what optional operands make ordinary.
    """

    for what, wanted in (("input", False), ("output", True)):
        indices = sorted(
            declaration.index
            for declaration in schema.values()
            if isinstance(declaration, (InputTensor, OutputTensor)) and declaration.output is wanted
        )
        if indices != list(range(len(indices))):
            raise AuthoringError(
                f"{operation_type.__name__} declares {what} indices {indices}; they must be "
                "unique and contiguous from zero, because an ONNX operand list is positional"
            )


def _check_constraints_are_classified(
    operation_type: type[DataflowOp], declarations: Mapping[str, object]
) -> None:
    """Every authored operation Constraint gates the source semantics, explicitly.

    The rule the Kernel and Design layers already state, for the reason that
    applies identically here: an operation has exactly one group, and a
    constraint outside it is compiled, evaluated, and consulted by nothing --
    so an operation would silently stop refusing what its author wrote a
    refusal for.  ``source_accepts`` is small enough that omitting a name from
    it is easy and invisible, which is why it is checked rather than trusted.

    An *output* observation deliberately has no place here: it is a
    reconciliation difference, not a verdict, and a constraint is not how it is
    reported.
    """

    group = declarations.get("source_accepts")
    if group is not None and not isinstance(group, ConstraintGroup):
        raise AuthoringError(
            f"{operation_type.__name__}.source_accepts is a {type(group).__name__}; it names "
            "one ConstraintGroup of the constraints that gate this operation's own semantics"
        )
    grouped = (
        {id(item) for item in group.constraints} if isinstance(group, ConstraintGroup) else set()
    )
    ungrouped = sorted(
        name
        for name, declaration in declarations.items()
        if isinstance(declaration, DeclaredConstraint) and id(declaration) not in grouped
    )
    if ungrouped:
        raise AuthoringError(
            f"{operation_type.__name__} declares Constraint {ungrouped[0]!r} outside "
            "source_accepts; an operation says which projection each of its constraints "
            "gates, because a constraint in no group refuses nothing"
        )


def _positional(declared: list[tuple[int, str]]) -> tuple[str, ...]:
    return tuple(name for _index, name in sorted(declared))


def _check_authoring(operation_type: type[DataflowOp]) -> None:
    if not operation_type.family:
        raise AuthoringError(f"{operation_type.__name__} must declare a non-empty family")
    clashes = sorted(
        {name for name, _ in source_declarations(operation_type)} & RESERVED_ATTRIBUTES
    )
    if clashes:
        raise AuthoringError(
            f"{operation_type.__name__} declares source attribute {clashes[0]!r}, which the "
            "dataflow layer owns"
        )


def _raw_attribute(node: Any, name: str) -> Any | None:
    attribute = _attribute(node, name)
    if attribute is None:
        return None
    return attribute.s if attribute.type == attribute.STRING else attribute.i


def _datatype_attribute(node: Any, member_name: str, declaration: DatatypeAttribute) -> object:
    from finn.dataflow.datatypes import canonical_qonnx_datatype  # noqa: PLC0415 - cycle
    from qonnx.core.datatype import DataType  # type: ignore[import-not-found] # noqa: PLC0415

    raw = _raw_attribute(node, member_name)
    name = declaration.default if raw is None else raw
    text = name.decode("utf-8") if isinstance(name, bytes) else str(name)
    return canonical_qonnx_datatype(DataType[text])


def _plain_attribute(node: Any, member_name: str, declaration: Attribute) -> object:
    raw = _raw_attribute(node, member_name)
    if raw is None:
        return declaration.default
    if declaration.value_type is str:
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return declaration.value_type(raw)


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
    node = binding.materialize()
    name = binding.source.node_name
    try:
        state = decode_dataflow_state(node)
    except DecodeError as error:
        raise DataflowOpError(f"{name}: {error}") from error
    if state is None:
        return
    if state.family and state.family != operation_type.family:
        raise DataflowOpError(
            f"{name} stores choices for family {state.family!r}, not {operation_type.family!r}"
        )
    if state.family_version and state.family_version != operation_type.family_version:
        raise DataflowOpError(
            f"{name} stores choices for {state.family}@{state.family_version}, "
            f"and this build offers {operation_type.family_version}"
        )
    if state.problem_fingerprint and state.problem_fingerprint != root.problem_fingerprint:
        raise DataflowOpError(
            f"{name} stores choices made against a different problem "
            f"(recorded {state.problem_fingerprint}, current {root.problem_fingerprint}); "
            "reconstruct explicitly rather than reinterpreting them"
        )


def _hydrate(
    operation_type: type[DataflowOp],
    binding: SourceBinding,
    root: Any,
    state: DataflowState | None,
) -> Any:
    """Replay a recorded document onto a fresh root, strictly.

    Every stored path must be one this model still has, and every stored codec
    tag must be the one its declaration currently declares.  A permissive
    reader would drop a renamed declaration's assignment and hand back a design
    that is not the one that was saved -- which is precisely the failure
    persistence exists to prevent, and the one hardest to notice.

    Selectors are committed before Decisions, because which alternative is live
    decides which Decisions exist at all.  Each group goes through one
    path-keyed commit, which is the same engine operation ``assign`` performs.
    """

    if state is None:
        return root
    known = {choice.path: choice for choice in occurrence_persistable(root)}
    unknown = sorted(set(state.assignments) - set(known))
    if unknown:
        raise DataflowOpError(
            f"{binding.source.node_name} records a choice at {unknown[0]!r}, which this "
            f"build's {operation_type.__name__} has no declaration for.  A renamed or "
            "removed declaration is a schema change, not a value to discard"
        )
    for selectors in (True, False):
        values: dict[Any, object] = {}
        for path, entry in state.assignments.items():
            choice = known[path]
            if choice.selector is not selectors:
                continue
            codec = _codec_for(root, choice)
            if entry.codec != codec.tag:
                raise DataflowOpError(
                    f"{binding.source.node_name} recorded {path!r} with codec "
                    f"{entry.codec!r} and this build declares {codec.tag!r}; a changed "
                    "encoding is not reinterpreted"
                )
            try:
                values[choice.reference.path] = codec.decode(entry.value)
            except (ValueError, TypeError, DecodeError) as error:
                raise DataflowOpError(
                    f"{binding.source.node_name} cannot decode {path!r}: {error}"
                ) from error
        if not values:
            continue
        try:
            root = occurrence_commit_paths(root, values)
        except (RequestError, AuthoringError, ValueError, TypeError) as error:
            raise DataflowOpError(
                f"{binding.source.node_name} recorded choices that do not replay onto this "
                f"problem: {error}",
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
    "DATAFLOW_DOMAIN",
    "FAMILY_ATTRIBUTE",
    "FAMILY_VERSION_ATTRIBUTE",
    "FINGERPRINT_ATTRIBUTE",
    "RESERVED_ATTRIBUTES",
    "SCOPE_ID_ATTRIBUTE",
    "AttributeCodec",
    "BuildFacts",
    "DataflowOp",
    "DataflowOpError",
    "SourceBinding",
    "source_declarations",
    "unresolved_reason",
]
