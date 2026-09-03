# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One datatype identity for the whole dataflow stack: QONNX's own.

The stack used to convert a QONNX datatype into a local ``(type_id, bit_width)``
pair on the way in and reconstruct a QONNX name on the way out.  Two authorities
for one fact, and the reduction is lossy: ``TERNARY`` and ``INT2`` reduce to the
same pair despite different value domains, and every sixteen-bit floating format
reduces to ``("float", 16)``.

So there is no intermediate value.  A tensor's datatype travels from the
``ModelWrapper`` through the problem instance, the Region operand, Kernel
coverage, and artifact metadata as the same QONNX datatype *value*, and its
canonical name is the only thing ever persisted.

"Value", not "object", is exact: every ingestion boundary re-resolves through
the canonical name, so what a Region holds is a fresh allocation equal to what
the caller supplied rather than the caller's own instance.  That is deliberate
-- see below.

This module is the boundary that makes that safe.  It has **no FINN imports at
all**, so ``region.py`` can depend on it without a cycle and without dragging
the engine into the model layer -- a separation ``test_api_and_boundaries``
enforces.  The engine-side ``ValueSemantics`` declaration therefore lives in
``finn.dataflow.model.semantics``, built from the helpers here.

Three hazards it exists to close, each verified against the pinned QONNX:

- **The values are mutable and uninterned.**  ``DataType["INT8"]`` allocates a
  fresh object each call, and its ``_bitwidth`` can be reassigned in place --
  which renames it, changes its hash, and loses it from any dict already holding
  it.  Every value crossing this boundary is therefore *re-resolved*, never
  merely checked, so nothing downstream retains a caller's instance.
- **Recognition is weaker than resolution.**  ``isinstance(value,
  BaseDataType)`` admits subclasses whose canonical name QONNX cannot resolve.
  Recognizing such a value and then failing inside the snapshot would break the
  contract that a recognized value can be stored, so both hooks share one
  helper.
- **A datatype compares and hashes equal to its own canonical name.**
  ``DataType["INT8"] == "INT8"`` is true in both directions and the hashes
  agree, so a string and a datatype collide as mapping keys.  ``str`` is
  therefore not a member of this value domain, and the recognizer says so.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeGuard, cast

from qonnx.core.datatype import (  # type: ignore[import-not-found]
    BaseDataType,
    DataType,
)

__all__ = [
    "DATATYPE_PAYLOAD_KEY",
    "DatatypeError",
    "QONNXDataType",
    "QONNX_DATATYPE_TOKEN",
    "canonical_qonnx_datatype",
    "decode_datatype",
    "encode_datatype",
    "is_qonnx_datatype",
    "qonnx_datatype_width",
    "resolve_qonnx_datatype_name",
]


class QONNXDataType(Protocol):
    """The datatype surface the dataflow stack is allowed to depend on.

    A typing aid only.  QONNX ships no ``py.typed``, and the gate deliberately
    hides it from mypy (see ``scripts/check-dataflow-design.sh``), so annotating
    against ``BaseDataType`` would annotate against ``Any``.  This names what is
    actually used instead.

    It is **not** a second value representation, a registry, or an engine value
    token: runtime values are real ``BaseDataType`` instances and the engine
    token is ``BaseDataType`` itself.  Adding a method here is a claim that the
    dataflow stack needs it from every datatype -- check the caveats in
    ``open/qonnx-datatype-adoption.md`` §13 before doing so, because several
    QONNX methods are not total.
    """

    @property
    def name(self) -> str: ...

    def bitwidth(self) -> int: ...

    def signed(self) -> bool: ...

    def is_integer(self) -> bool: ...

    def is_fixed_point(self) -> bool: ...

    # PARTIAL.  Unlike everything above it, this is not defined for every QONNX
    # datatype: ``ScaledIntType`` raises.  It is declared because the stack does
    # genuinely need a datatype's representable range -- the narrow-weight
    # promise asks for the minimum, and the datatype-continuity audit compares
    # both endpoints so a reduced representation cannot stand in for the live
    # value. Hiding either call behind a cast would hide the dependency too.
    # Every caller must guard partial uses. Do not add further partial methods
    # without the same treatment; ``get_hls_datatype_str`` in particular raises
    # a bare ``AssertionError`` for arbitrary float formats, which no ``except``
    # clause should be catching.
    def min(self) -> int | float: ...

    def max(self) -> int | float: ...


class DatatypeError(ValueError):
    """A value was offered as a datatype and is not one this stack can hold."""


def _guarded(describe: str, call: Callable[[], object]) -> object:
    """Run one third-party call, turning any ordinary failure into a refusal.

    ``except Exception``, deliberately, rather than an enumerated tuple.  This
    boundary's whole contract is that recognition is *total*: a predicate named
    ``is_qonnx_datatype`` must answer for any object it is handed, and it
    cannot do that if the set of exceptions QONNX might raise has to be
    predicted in advance.

    The attempt to enumerate them is instructive about why it fails.
    ``resolve_datatype`` raises ``KeyError`` for an unknown prefix,
    ``ValueError`` where a bad name reaches ``int()``, and ``AssertionError``
    from the constructors' own validation -- three types, all discovered one at
    a time, and a ``BaseDataType`` subclass may raise anything at all from
    ``get_canonical_name()``. Each addition to a tuple is a fix for the case
    that was found rather than for the class of case.

    ``BaseException`` is *not* caught, so ``KeyboardInterrupt``,
    ``SystemExit``, and ``GeneratorExit`` still propagate: an interrupt is not
    the datatype declining to be a datatype.
    """

    try:
        return call()
    except Exception as error:  # noqa: BLE001 - see docstring: totality is the contract
        raise DatatypeError(f"{describe}: {type(error).__name__}: {error}") from error


def resolve_qonnx_datatype_name(name: str) -> QONNXDataType:
    """Resolve a datatype name **permissively**, as QONNX itself would.

    Accepts anything QONNX accepts, including spellings it merges or normalizes
    into another name -- ``UINT1`` resolves to ``BINARY``, and ``FLOAT<5,10,0>``
    to ``FLOAT<5,10,15>``, because QONNX reads a zero exponent bias as "unset"
    and substitutes its default.

    This is right for *source-facing* names: a legacy node attribute like
    ``accDataType`` is whatever a human or an older FINN wrote, and adopting
    QONNX means adopting its reading of it.

    It is wrong for persistence. Use :func:`decode_datatype` there, which
    refuses exactly the reinterpretations this function performs.
    """

    if not isinstance(name, str):
        raise DatatypeError(f"a datatype name must be a string, not {type(name)}")
    resolved = _guarded(f"no QONNX datatype is named {name!r}", lambda: DataType[name])
    return cast(QONNXDataType, resolved)


def canonical_qonnx_datatype(value: object) -> QONNXDataType:
    """Return ``value`` re-resolved through its canonical name.

    The single gate every datatype passes on its way into the stack, used as
    both the recognizer and the snapshot so that recognition cannot promise
    something resolution then refuses.

    Raises ``DatatypeError`` rather than returning ``None``: a caller that
    wants a predicate should use ``is_qonnx_datatype``, and a caller that
    reached here has already decided the value ought to be a datatype.
    """

    if isinstance(value, str):
        # Not merely unsupported -- actively dangerous.  A datatype compares and
        # hashes equal to its own canonical name, so admitting strings here
        # would make the persisted form and the live value interchangeable and
        # let them collide as mapping keys without raising.
        raise DatatypeError(f"a canonical name is not a datatype value; resolve {value!r} first")
    if not isinstance(value, BaseDataType):
        raise DatatypeError(f"not a QONNX datatype: {value!r}")
    # Asking a datatype its own name can fail too: ``get_canonical_name`` is a
    # method, and a subclass -- or an instance mutated into an inconsistent
    # state -- may raise anything at all from it.  Totality has to start here,
    # not at the resolution below.
    name = _guarded(f"{type(value).__name__} cannot name itself", lambda: value.name)
    if not isinstance(name, str):
        raise DatatypeError(f"a QONNX datatype must name itself with a string, got {name!r}")
    resolved = _guarded(
        f"{type(value).__name__} names itself {name!r}, which QONNX cannot resolve",
        lambda: DataType[name],
    )
    return _checked(resolved, offered=name, original=value)


def _checked(resolved: object, *, offered: str, original: object | None = None) -> QONNXDataType:
    """Establish that resolution actually landed on the datatype it was given.

    Two conditions, and it is worth being exact about what each one can catch,
    because QONNX identity is *by canonical name* and that limits the question
    a check like this can even ask.

    ``resolved.name == offered`` refuses a non-canonical spelling, so a caller
    cannot smuggle in an alias and have it silently rename itself later.  Here
    ``offered`` is always a live value's *own* ``name``, so this is a check that
    the datatype agrees with QONNX about what it is called.  The corresponding
    question for a *persisted* name is asked in :func:`decode_datatype`, which
    has to be strict for a different reason.

    ``resolved == original`` refuses a subclass that overrides equality.  It
    cannot detect a subclass that presents a valid canonical name while
    behaving differently, and that is not a gap: under QONNX's own identity
    rule such a value simply *is* that datatype, and re-resolution replaces it
    with the registered one regardless.

    Both reads below are themselves guarded.  Neither is obviously third-party
    -- ``resolved`` came from ``DataType[...]`` and is a registered QONNX value
    -- but ``name`` is a *second* ``get_canonical_name()`` call rather than a
    reuse of the first, and ``resolved != original`` reaches ``original``'s
    reflected ``__eq__`` whenever QONNX's own returns ``NotImplemented``.  That
    second one is the real hazard: ``original`` is the caller's object, so the
    comparison performed to decide whether a value may be recognized runs
    caller-controlled code.  An unguarded raise there escapes
    ``is_qonnx_datatype`` as something other than ``DatatypeError``, which is
    precisely the totality this boundary promises not to break.
    """

    name = _guarded(
        f"{type(resolved).__name__} cannot name itself",
        lambda: resolved.name,  # type: ignore[attr-defined]
    )
    if name != offered and original is not None:
        raise DatatypeError(f"{offered!r} is not the canonical spelling of {name!r}")
    if original is not None:
        differs = _guarded(
            f"{offered!r} could not be compared with the value it was read from",
            lambda: resolved != original,
        )
        if differs:
            raise DatatypeError(f"{offered!r} does not resolve back to the value it was read from")
    return resolved  # type: ignore[return-value]


def qonnx_datatype_width(value: object) -> int:
    """The bit width of ``value``, re-resolved first and then read under guard.

    Two things a bare ``value.bitwidth()`` gets wrong, and this exists because
    both matter at a *recognition* boundary rather than at a use site.

    It measures the wrong object.  The caller's instance is not what the stack
    will hold -- ingestion re-resolves -- so asking the caller's instance its
    width answers a question about a value that is about to be discarded.  For
    a well-behaved datatype the two agree; for a subclass that reports one width
    and names itself another they do not, and the registered value's width is
    the one that is true downstream.

    And it can raise.  ``bitwidth()`` is a method like any other, so a predicate
    built on it inherits whatever the method does.  Guarded here so callers keep
    a total ``DatatypeError`` contract instead of re-deriving it each time.
    """

    canonical = canonical_qonnx_datatype(value)
    width = _guarded(
        f"{type(canonical).__name__} cannot state its width",
        lambda: canonical.bitwidth(),
    )
    if not isinstance(width, int) or isinstance(width, bool):
        raise DatatypeError(f"a QONNX datatype must state an integer width, got {width!r}")
    return width


def is_qonnx_datatype(value: object) -> TypeGuard[QONNXDataType]:
    """Whether ``canonical_qonnx_datatype`` would accept ``value``.

    Defined as "the helper succeeds" rather than as a parallel set of
    conditions, so the recognizer and the snapshot cannot drift apart.
    """

    try:
        canonical_qonnx_datatype(value)
    except DatatypeError:
        return False
    return True


#: The key a persisted datatype appears under.  Named rather than bare so a
#: reader of a stored fingerprint can tell a datatype from a string that happens
#: to look like one, and so the old ``numeric_element_type`` form is a visibly
#: different shape rather than a silently reinterpreted one.
DATATYPE_PAYLOAD_KEY = "qonnx_datatype"


def encode_datatype(value: object) -> dict[str, str]:
    """Encode a datatype for a fingerprint or a persisted selection.

    The canonical name and nothing else.  Never the ``repr``, the class name,
    or a family-and-width pair: the first two have no declared contract and the
    third is the lossy reduction this module replaced -- it would give
    ``TERNARY`` and ``INT2`` the same fingerprint.
    """

    return {DATATYPE_PAYLOAD_KEY: canonical_qonnx_datatype(value).name}


def decode_datatype(payload: object) -> QONNXDataType:
    """Hydrate what ``encode_datatype`` wrote -- **strictly**.

    The persisted name must be the canonical name of what it resolves to.
    Anything else is refused rather than reinterpreted, because a persisted
    value is not a human's spelling: `encode_datatype` only ever writes
    canonical names, so a payload that resolves to a *different* name did not
    come from here, and quietly accepting it is exactly the silent
    reinterpretation this module exists to remove.

    Two spellings QONNX would happily accept, and this refuses:

    - ``FLOAT<5,10,0>`` resolves to ``FLOAT<5,10,15>``. QONNX reads a zero
      exponent bias as "unset" and substitutes its default, so the stored
      datatype comes back as a numerically different one. This is the same
      class of failure as ``TERNARY`` arriving as ``INT2``, one layer out.
    - ``UINT1`` resolves to ``BINARY``. Harmless in itself -- they are one
      value (§6.1) -- but refused anyway, because the rule is "canonical or
      nothing" and carving an exception for the alias that happens to be benign
      is how the general case gets let back in.

    Source-facing names are a different question and use
    :func:`resolve_qonnx_datatype_name`, which is permissive on purpose.
    """

    if not isinstance(payload, dict) or set(payload) != {DATATYPE_PAYLOAD_KEY}:
        raise DatatypeError(f"not an encoded datatype: {payload!r}")
    stored = payload[DATATYPE_PAYLOAD_KEY]
    resolved = resolve_qonnx_datatype_name(stored)
    if resolved.name != stored:
        raise DatatypeError(
            f"persisted datatype {stored!r} is not canonical; "
            f"QONNX resolves it to {resolved.name!r}"
        )
    return resolved


#: The one engine value domain for datatypes.
#:
#: ``type_token`` is ``BaseDataType`` and must stay that single object:
#: ``ValueSemantics.is_compatible_with`` compares tokens by identity, not by
#: subtyping, so a second token -- the protocol above, say -- would silently
#: partition the domain and make two datatype fields report that they cannot be
#: compared.
#: The single object every datatype value-semantics declaration must use as its
#: ``type_token``.  Exported so that the declaration -- which lives in
#: ``finn.dataflow.model.semantics``, because the model layer must not import the
#: engine -- names the same object this module recognizes against.
QONNX_DATATYPE_TOKEN = BaseDataType
