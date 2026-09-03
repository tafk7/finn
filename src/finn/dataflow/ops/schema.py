# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The source schema: what an operation declares about the node it reads.

An operation is the root Space of its own design space, so the graph facts it
depends on are its ``Problem`` members.  Each declaration here *is* one -- an
``InputTensor`` is a ``Problem[SourceOperand]`` that also knows its ONNX index
-- so nothing has to translate between a schema and a design space, and the
generic compiler needs no new concept to see them.

What a tensor additionally needs is *facets*.  An operand is never one fact: it
is a shape, a rank, a datatype, whether an initializer is present, and -- for
an optional operand -- whether it is there at all.  Declaring five members per
tensor and keeping them consistent by hand is exactly the bookkeeping that goes
wrong silently, so one declaration lowers to explicit members::

    activation = InputTensor(index=0)

        activation                       Problem[SourceOperand]  (the declaration)
        activation__shape                Derived[tuple[int, ...]]
        activation__rank                 Derived[int]
        activation__datatype             Derived[QONNXDataType]
        activation__initializer_present  Derived[bool]
        activation__present              Derived[bool]   (optional operands)
        activation__initializer_digest   Derived[str]    (fingerprint=True)

The author writes none of the generated names.  ``activation.shape`` returns
the generated declaration, so ``@derived(..., shape=activation.shape)``
resolves to an ordinary declared member and every existing mechanism --
dependency traversal, diagnostics, the member index -- works on it unchanged.

**The facets exist from the constructor, not from lowering.**  They have to:
the author names ``weight.shape`` in the same class body that declares
``weight``, and ``__init_subclass__`` does not run until that body has
finished.  Lowering therefore only *binds* -- it writes each facet onto the
class under its generated name.

**Lowering is explicit and lives here.**  Nothing is added to ``model/`` for
it; the expansion runs in ``DataflowOp.__init_subclass__``, the same mechanism
``Kernel.__init_subclass__`` already uses.  There is no generic "declaration
provider" protocol, because a protocol would let any Space grow members by side
effect.

**Presence is exposed; constancy is not imposed.**  An ``InputTensor`` says
whether an initializer is there and stops.  Whether a constant operand is
*required*, *forbidden*, or merely *possible* is a claim about the operation's
mathematics, so it is written as a constraint by the layer that can argue for
it -- see the ownership table in the operation's own module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from finn.dataflow._engine import ValueSemantics
from finn.dataflow.model.declarations import (
    AuthoringError,
    CanonicalValue,
    CanonicalValueCodec,
    Derived,
    Problem,
    ValueSource,
    allow_absent,
    reject,
    semantics_for,
)
from finn.dataflow.model.semantics import (
    QONNX_DATATYPE_CODEC,
    QONNX_DATATYPE_VALUE_SEMANTICS,
)
from finn.dataflow.ops.source import SourceOperand

T = TypeVar("T")

#: The facets a declared tensor lowers to, in generation order.  The last two
#: are conditional on the declaration.
TENSOR_FACETS: tuple[str, ...] = (
    "shape",
    "rank",
    "datatype",
    "initializer_present",
    "present",
    "initializer_digest",
)


def _encode_operand(value: object) -> CanonicalValue:
    """One operand's canonical form for the problem fingerprint.

    Shape, datatype, initializer presence and -- when the declaration asked for
    it -- the initializer digest.  Deliberately **not** the tensor name: an
    operation does not depend on what its input is called, and folding the name
    in would make a graph-level rename invalidate every recorded choice as
    though the numerics had changed.

    The operand id *is* included, because two operands of the same shape and
    type are not interchangeable and the fingerprint has to say which is which.
    """

    operand = cast(SourceOperand, value)
    return {
        "operand": operand.id,
        "shape": list(operand.shape),
        "datatype": operand.datatype.name,
        "initializer": operand.initializer,
        "digest": operand.initializer_digest,
    }


SOURCE_OPERAND_CODEC: CanonicalValueCodec[object] = CanonicalValueCodec(
    "finn.dataflow.source_operand", 1, _encode_operand
)

SOURCE_OPERAND_SEMANTICS = semantics_for(SourceOperand)


class TensorDeclaration:
    """Marker for the two kinds of operand declaration.

    They are deliberately *not* the same kind of thing.  An input is a fact the
    design space depends on, so it is a ``Problem`` and it is part of the
    problem's identity.  An output is an annotation the operation is
    authoritative for and merely observes, so it is not a Problem at all and
    never reaches the point.
    """

    __slots__ = ()

    index: int
    output: bool
    member_name: str


@dataclass(frozen=True, slots=True, eq=False, init=False)
class InputTensor(TensorDeclaration, Problem[Any]):
    """One input operand: a Problem, plus the facets over it.

    ``optional=True`` admits ONNX's empty-string spelling of "not supplied", so
    a thresholded variant of an operation can make a whole operand conditional
    without anyone inventing a placeholder tensor.  ``fingerprint=True`` folds a
    digest of the operand's initializer into the problem identity, for an
    operand whose *values* change what is built.
    """

    # Defaulted only because ``Problem`` defaults every field it declares
    # above these; ``init=False`` means the real values always arrive through
    # ``__init__``.
    index: int = 0
    optional: bool = False
    fingerprint_initializer: bool = False
    output: bool = False
    facets: dict[str, Derived[Any]] = field(default_factory=dict)
    member_name: str = ""

    def __init__(self, index: int, *, optional: bool = False, fingerprint: bool = False) -> None:
        if not isinstance(index, int) or index < 0:
            raise AuthoringError("a tensor declaration needs a non-negative operand index")
        Problem.__init__(
            self,
            SOURCE_OPERAND_SEMANTICS,
            required=not optional,
            canonical=SOURCE_OPERAND_CODEC,
        )
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "optional", optional)
        object.__setattr__(self, "fingerprint_initializer", fingerprint)
        object.__setattr__(self, "output", False)
        object.__setattr__(self, "member_name", "")
        object.__setattr__(self, "facets", _tensor_facets(self))

    def __getattr__(self, facet: str) -> Derived[Any]:
        if facet.startswith("_"):
            raise AttributeError(facet)
        facets = object.__getattribute__(self, "facets")
        try:
            return cast("Derived[Any]", facets[facet])
        except KeyError:
            available = ", ".join(sorted(facets))
            raise AttributeError(
                f"a tensor declares no facet {facet!r}; it has {available}"
            ) from None

    @property
    def described(self) -> str:
        """How this operand is named in a diagnostic, before or after lowering."""

        return self.member_name or f"operand #{self.index}"


@dataclass(frozen=True, slots=True, eq=False)
class OutputTensor(TensorDeclaration):
    """One output operand, as the graph currently annotates it.

    **Not a Problem, and not in the point.**  The operation is authoritative for
    what its output shape and datatype should be; the graph's annotation is an
    observation to be compared against that and repaired.  A value that could
    reach a Decision domain or a constraint while staying out of the problem
    identity is a value whose recorded choices can be silently wrong -- so
    rather than an escape hatch on ``Problem``, an observation simply is not
    one.  It is read into the frozen ``SourceNode`` and reconciled there.

    The consequence is deliberate: an operation cannot feed its own output
    annotation into a Design.  If a datatype genuinely defines the operation's
    mathematics -- an accumulator width, say -- it is declared as a
    ``DatatypeAttribute`` and *derived onto* the output, not read back off it.
    """

    index: int
    output: bool = True
    member_name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or self.index < 0:
            raise AuthoringError("a tensor declaration needs a non-negative operand index")


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Attribute(Problem[Any]):
    """One ONNX node attribute this operation reads as a source fact.

    ``onnx=`` names the attribute *in the graph* when it differs from the
    member name.  The two are genuinely separate concerns: ``noActivation`` is
    a spelling FINN's graphs already carry and cannot be renamed without
    breaking every model on disk, while ``no_activation`` is what the design
    space reads.  Folding them together would force one of the two to be
    wrong, and a silent default is what a mismatched name produces -- an
    attribute nobody wrote, read as its default, with no diagnostic at all.
    """

    value_type: type[Any] = object
    default: Any = None
    member_name: str = ""
    onnx: str = ""

    def __init__(self, value_type: type[T], *, default: T, onnx: str = "") -> None:
        Problem.__init__(self, value_type)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "default", default)
        object.__setattr__(self, "member_name", "")
        object.__setattr__(self, "onnx", onnx)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class DatatypeAttribute(Problem[Any]):
    """One node attribute naming a QONNX datatype."""

    default: str = ""
    member_name: str = ""
    onnx: str = ""

    def __init__(self, *, default: str, onnx: str = "") -> None:
        Problem.__init__(self, QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
        object.__setattr__(self, "default", default)
        object.__setattr__(self, "member_name", "")
        object.__setattr__(self, "onnx", onnx)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class BuildFact(Problem[Any]):
    """One scalar taken from the build configuration at binding time.

    A scalar, extracted and frozen, never the configuration object.  A design
    space holding a live config would answer differently after an unrelated
    build setting changed, with nothing to detect it: the fingerprint would not
    move because nothing the fingerprint reads did.
    """

    value_type: type[Any] = object
    accessor: Callable[[Any], Any] = bool
    default: Any = None
    member_name: str = ""

    def __init__(
        self,
        value_type: type[T],
        *,
        accessor: Callable[[Any], T],
        default: T | None = None,
        required: bool = True,
    ) -> None:
        if not callable(accessor):
            raise AuthoringError("a BuildFact needs a callable accessor")
        Problem.__init__(self, value_type, required=required)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "accessor", accessor)
        object.__setattr__(self, "default", default)
        object.__setattr__(self, "member_name", "")


@dataclass(frozen=True, slots=True, eq=False, init=False)
class InitializerAnalysis(Problem[Any]):
    """One scalar an operation computes *from* an operand's initializer values.

    The narrow, deliberate exception to "no array enters the design space".
    Some facts genuinely depend on the weights themselves -- whether they use
    the minimum representable value, whether they are all zero -- and the
    alternative to computing them is either carrying the array into the point
    (which makes every query depend on megabytes and every fingerprint
    expensive) or reading the graph again later (which is the unfrozen
    occurrence this layer exists to prevent).

    So the array is read once, at binding time, and only the author's scalar
    survives.  The function takes the values and the operand's datatype and
    returns a value or ``None``; ``None``, and an absent or uninitialized
    operand, all mean "this analysis has no answer here", which reaches the
    design space as an ordinary absent Problem rather than a fabricated
    default.
    """

    operand: Any = None
    evaluate: Callable[[Any, Any], object] = field(
        default=cast("Callable[[Any, Any], object]", bool), repr=False
    )
    value_type: type[Any] = object
    member_name: str = ""

    def __init__(
        self,
        operand: InputTensor,
        value_type: type[T],
        *,
        evaluate: Callable[[Any, Any], T | None],
    ) -> None:
        if not isinstance(operand, InputTensor):
            raise AuthoringError("an InitializerAnalysis analyses one declared InputTensor")
        if not callable(evaluate):
            raise AuthoringError("an InitializerAnalysis needs a callable evaluate")
        Problem.__init__(self, value_type, required=False)
        object.__setattr__(self, "operand", operand)
        object.__setattr__(self, "evaluate", evaluate)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "member_name", "")


#: Every class-body value the source-schema lowering recognizes.
SOURCE_DECLARATION_TYPES: tuple[type, ...] = (
    InputTensor,
    OutputTensor,
    Attribute,
    DatatypeAttribute,
    BuildFact,
    InitializerAnalysis,
)

#: The union, for a caller that wants to name it.
SourceDeclaration = (
    InputTensor | OutputTensor | Attribute | DatatypeAttribute | BuildFact | InitializerAnalysis
)


def attribute_name(member_name: str, declaration: Attribute | DatatypeAttribute) -> str:
    """The name this attribute has *in the graph*."""

    return declaration.onnx or member_name


def facet_name(member_name: str, facet: str) -> str:
    return f"{member_name}__{facet}"


def _tensor_facets(declaration: InputTensor) -> dict[str, Derived[Any]]:
    """Build one tensor's facet properties over its own operand record.

    An optional operand's facets tolerate its absence and say so: ``present``
    answers ``False`` and every other facet refuses.  A refusal rather than a
    fabricated zero, because a shape of ``()`` for an operand that is not there
    is a value a reader would happily compute with.
    """

    source: ValueSource[object] = cast("ValueSource[object]", declaration)
    # Only an optional operand's facets tolerate absence.  A required operand
    # that is missing is a refused problem, long before a facet is asked.
    tolerant = allow_absent(source) if declaration.optional else source

    def facet(
        name: str,
        value_type: type[Any] | ValueSemantics[Any],
        read: Callable[[SourceOperand], object],
    ) -> Derived[Any]:
        def evaluate(*, operand: object) -> object:
            if not isinstance(operand, SourceOperand):
                return reject(
                    "source-operand-absent",
                    f"{declaration.described} is not supplied, so it has no {name}",
                    values={"operand": declaration.described},
                )
            return read(operand)

        return Derived(semantics_for(value_type), None, (("operand", tolerant),), evaluate)

    facets: dict[str, Derived[Any]] = {
        "shape": facet("shape", tuple, lambda operand: operand.shape),
        "rank": facet("rank", int, lambda operand: len(operand.shape)),
        "datatype": facet(
            "datatype", QONNX_DATATYPE_VALUE_SEMANTICS, lambda operand: operand.datatype
        ),
        "initializer_present": facet(
            "initializer_present", bool, lambda operand: operand.initializer
        ),
    }
    if declaration.optional:

        def present(*, operand: object) -> object:
            return isinstance(operand, SourceOperand)

        facets["present"] = Derived(semantics_for(bool), None, (("operand", tolerant),), present)
    if declaration.fingerprint_initializer:
        facets["initializer_digest"] = facet(
            "initializer_digest", str, lambda operand: operand.initializer_digest or ""
        )
    return facets


def lower_source_schema(
    owner: type,
    declarations: Mapping[str, SourceDeclaration],
) -> dict[str, object]:
    """Bind each tensor's facets to explicit class members.

    The declarations are already ``Problem`` members, so the generic compiler
    sees them without help.  What needs a name is each facet, and it gets a
    stable generated one -- ``f"{member}__{facet}"`` -- derived from the
    author's own member name.

    A collision is refused naming both sides, because the alternative is one
    declaration's facet silently replacing another's.
    """

    generated: dict[str, object] = {}
    for member_name, declaration in declarations.items():
        object.__setattr__(declaration, "member_name", member_name)
        if not isinstance(declaration, InputTensor):
            continue
        for facet, generated_declaration in declaration.facets.items():
            name = facet_name(member_name, facet)
            if name in generated:
                raise AuthoringError(
                    f"{owner.__name__} generates {name!r} twice; the {facet} facet of "
                    f"{member_name!r} collides with an earlier declaration's expansion"
                )
            generated[name] = generated_declaration
    return generated


__all__ = [
    "SOURCE_DECLARATION_TYPES",
    "SOURCE_OPERAND_CODEC",
    "SOURCE_OPERAND_SEMANTICS",
    "TENSOR_FACETS",
    "Attribute",
    "BuildFact",
    "DatatypeAttribute",
    "InitializerAnalysis",
    "InputTensor",
    "OutputTensor",
    "SourceDeclaration",
    "TensorDeclaration",
    "attribute_name",
    "facet_name",
    "lower_source_schema",
]
