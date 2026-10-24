# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile declarations into models, and open a declaration's design space.

Vocabulary: a *declaration* (``Room(area=12)``) is a template; a *model* is its
compiled structure, with no inputs; a *design space* is a model with its inputs
bound and every choice open; a *configuration* is a design space after some
choices, and a *design point* one complete for the question asked.

``design_space(node)`` is the one step from a declaration to a design space. A
Space class is compiled once, into the model every design space of it shares; a
root supplies plain values for its value formals, bound as runtime inputs. A
root that supplies structure (a reference, a fresh Decision) is refused:
structure belongs to a Space class. Preparing a model freezes every node
declaration it instantiates, and ``design_space`` freezes its root: from then
on, assignment is refused.
"""

# Lazy imports break the declaration/configuration/evaluation cycle.
# ruff: noqa: PLC0415
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from ._configuration import Space
from ._nodes import (
    NodeDecl,
    missing_formal,
    node_record,
    space_formals,
    structural_settings,
    unsupplied_formals,
)
from .errors import DefinitionError, RequestError
from .references import ValueHandle, resolve_decision, resolve_reference

if TYPE_CHECKING:
    from ._running.per_class import PerClass
    from ._scopes import Address, ScopeRecord, Scopes
    from ._template import Compiled

_OWNER = "space.compiler"

S = TypeVar("S", bound=Space)


@dataclass(frozen=True, slots=True)
class Model(Generic[S]):
    """A compiled declaration: its root class's compiled form (``compiled``) and its scope
    records (``scopes``: each placement of a class, made as it is reached, keyed by path,
    held for the model's life; ``_scopes.Scopes``), with no inputs or choices. What is read
    of its structure alone (inspection's Decisions below a scope) is kept with it
    (``_read``). ``lock`` is the model's one lock (its scope records'): making a scope and
    every public question about a design space of it take it. Its ``per_class`` record is
    what the running type reads once per model: the inferred reads, the flow order, the
    presence readers and the model's one session, in which every public question runs
    (``_running.per_class.PerClass``). Two models are equal only where they are one (their
    scope records are)."""

    space_type: type[S]
    compiled: Compiled = field(repr=False, compare=False)
    scopes: Scopes = field(repr=False)
    lock: RLock = field(init=False, repr=False, compare=False)
    _read: dict[tuple[str, object], object] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    per_class: PerClass = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from ._running.per_class import PerClass

        object.__setattr__(self, "lock", self.scopes.lock)
        object.__setattr__(self, "per_class", PerClass(cast("Model[Space]", self)))

    @property
    def root(self) -> ScopeRecord:
        """The root's scope record."""
        return self.scopes.root

    def resolve(self, scope: ScopeRecord, reference: object) -> Address:
        """Resolve a declaration in this exact scope without consulting its class: a
        member's own address (a handle holds one; a scope's members and choices are read as
        what each record stands for there), never one standing for another."""

        if isinstance(reference, ValueHandle):
            if not self.scopes.holds(scope):
                raise RequestError("reference scope does not belong to this model")
            return reference._resolve(self.scopes)
        return resolve_reference(self.scopes, scope, reference)

    def decision(self, scope: ScopeRecord, reference: object) -> Address:
        """The owning decision ``reference`` may edit in this scope."""

        if isinstance(reference, ValueHandle):
            index = self.resolve(scope, reference)
            if index.kind != "decision":
                raise RequestError("parameter aliases are not independently editable")
            return index
        return resolve_decision(self.scopes, scope, reference)


_PREPARATION_LOCK = RLock()


def _constructor_classes(space_type: type[Space]) -> tuple[type[object], ...]:
    """Return every class whose constructor protocol affects this Space class."""

    result: list[type[object]] = []
    for base in space_type.__mro__:
        if base is Space:
            break
        result.append(base)
    return tuple(result)


def _validate_constructors(space_types: tuple[type[Space], ...]) -> None:
    checked: set[type[object]] = set()
    for space_type in space_types:
        for base in _constructor_classes(space_type):
            if base in checked:
                continue
            checked.add(base)
            if "__init__" in base.__dict__:
                raise DefinitionError.broken(
                    "instance-constructor-custom",
                    _OWNER,
                    f"{base.__qualname__}: custom instance __init__ is unsupported; "
                    "use declarations and ordinary helper methods",
                    declaration=base.__qualname__,
                )
            if "__new__" in base.__dict__:
                raise DefinitionError.broken(
                    "instance-constructor-custom",
                    _OWNER,
                    f"{base.__qualname__}: custom instance __new__ is unsupported",
                    declaration=base.__qualname__,
                )


def _definition_classes(space_types: tuple[type[Space], ...]) -> tuple[type[Space], ...]:
    """Include concrete Space classes and Space bases contributing effective declarations."""

    result: list[type[Space]] = []
    seen: set[type[Space]] = set()
    for space_type in space_types:
        for base in space_type.__mro__:
            if base is Space:
                break
            if issubclass(base, Space) and base not in seen:
                seen.add(base)
                result.append(base)
    return tuple(result)


def compile_model(space_type: type[S]) -> Model[S]:
    """The canonical model of a Space class whose formals are all runtime inputs: its
    compiled form, placed as scopes on demand (``_scopes``). Its class is compiled first
    (checked, with the cycles through the Spaces it places, each compiled before it); then
    views read out and the domain rule, each composed per class on the compiled forms up
    to this root (``_compose``)."""

    if not isinstance(space_type, type) or not issubclass(space_type, Space):
        raise DefinitionError.broken(
            "not-a-space", _OWNER, "compile_model requires a Space subclass"
        )
    with _PREPARATION_LOCK:
        cached = space_type.__dict__.get("_space_prepared_model")
        if cached is not None:
            if not isinstance(cached, Model) or cached.space_type is not space_type:
                raise DefinitionError.broken(
                    "model-cache-invalid", _OWNER, "invalid prepared-model cache on Space class"
                )
            return cached
        from ._compose import check_domains, check_views

        root = compiled(space_type)
        # Views are read out: only a view reads a view. Then every domain's membership
        # reads facts only. (What each method may read is inferred from its source.)
        check_views(root)
        check_domains(root)
        model = model_of(root)
        type.__setattr__(space_type, "_space_prepared_model", model)
        return cast(Model[S], model)


def model_of(
    root: Compiled, *, permuted: bool | None = None, eager: bool | None = None
) -> Model[Space]:
    """A fresh model of the compiled form ``root``, in the test modes the environment asks
    for (``_scopes.model_modes``) unless given; ``compile_model`` keeps one per class."""
    from ._scopes import Scopes, model_modes

    asked = model_modes()
    scopes = Scopes(
        root,
        permuted=asked[0] if permuted is None else permuted,
        eager=asked[1] if eager is None else eager,
    )
    return Model(root.space_type, root, scopes)


def compiled(space_type: type[Space]) -> Compiled:
    """A Space class's compiled form (``_template.compiled``)."""
    from ._template import compiled as compiled_form

    return compiled_form(space_type)


def root_record(node: object) -> NodeDecl:
    record = node_record(node)
    if record is None:
        raise RequestError(
            "design_space() takes a node declaration, as in design_space(House(budget=100))"
        )
    if record.placement is not None:
        raise RequestError(
            f"design_space() takes a root: {record.describe()} is already placed "
            f"({record.placement}); "
            "open the design space of the node that contains it"
        )
    structural = structural_settings(record)
    if structural:
        raise DefinitionError.broken(
            "root-structural-formal",
            _OWNER,
            f"design_space() of {record.describe()}: a root supplies plain values for its "
            f"value formals, and {', '.join(structural)} supplies structure (a reference "
            "or a Decision); write a subclass, or build one with composite()",
            declaration=record.describe(),
        )
    return record


def root_parameters(model: Model[Space], record: NodeDecl) -> dict[Address, object]:
    """The root's plain values, by formal: what its call or assignments gave (a formal
    nothing gives takes its default when the design space is opened, ``facts.open``)."""

    values: dict[Address, object] = {}
    root = model.scopes.root
    own = record.bindings
    for name in space_formals(record.space_type):
        index = root.named_members.get(name)
        if index is None or index.kind != "param" or name not in own:
            continue
        value = own[name]
        values[index] = getattr(value, "value", value) if _is_const(value) else value
    return values


def _is_const(value: object) -> bool:
    from .declarations import Const

    return isinstance(value, Const)


def design_space(node: S) -> S:
    """Open the design space of a root declaration: its inputs bound, every choice open.

    ``design_space(House(budget=100))`` is typed as ``House``: the empty
    configuration of the space, narrowed with ``with_choices``.
    """

    from .occurrence import opened  # noqa: PLC0415 - keep compilation evaluator-independent

    record = root_record(node)
    formals = space_formals(record.space_type)
    missing = [
        missing_formal(name, formals[name], None, record.space_type.__qualname__)
        for name in unsupplied_formals(record)
    ]
    if missing:
        raise DefinitionError.broken(
            "formal-unsupplied",
            _OWNER,
            f"{'; '.join(missing)}; while opening the design space of {record.describe()}",
            declaration=record.describe(),
        )
    try:
        model = cast(Model[Space], compile_model(record.space_type))
    except DefinitionError as error:
        raise DefinitionError(
            f"{error}; while opening the design space of {record.describe()}",
            findings=error.findings,
        ) from error
    record.freeze("design_space() took it as a root")
    return cast(S, opened(model, root_parameters(model, record)))


__all__ = ["Model", "compile_model", "design_space", "model_of"]
