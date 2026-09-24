# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Experimental ordinary-self evaluation. Not a supported production entry point.

Select replay, recursive, or optional greenlet scheduling with ``using_scheduler``.
Only direct concrete placements and nominal value types are supported here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
import importlib
import inspect
from threading import RLock
from types import MappingProxyType
from typing import Generic, Literal, Protocol, TypeVar, cast, get_origin, get_type_hints, overload

from typing_extensions import Self

from .errors import DefinitionError, EvaluationError, RequestError, ValueUnavailableError
from .results import (
    Available,
    Finding,
    FindingKind,
    Inapplicable,
    QueryResult,
    Rejected,
    Unresolved,
    ViewAssessment,
    ReadinessAssessment,
    assess_view,
    constraint_result,
    reject,
    require_value,
)
from .semantics import ValueSemantics, semantics_for

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
S = TypeVar("S", bound="Space")
Scheduler = Literal["replay", "recursive", "greenlet"]
_MODE: ContextVar[Scheduler] = ContextVar("self_prototype_scheduler", default="replay")
_ACTIVE: ContextVar[_Context | None] = ContextVar("self_prototype_active", default=None)
_PREPARE_LOCK = RLock()


@contextmanager
def using_scheduler(mode: Scheduler) -> Iterator[None]:
    token = _MODE.set(mode)
    try:
        yield
    finally:
        _MODE.reset(token)


class Declaration:
    name = ""
    owner: type[object] | None = None

    def __setattr__(self, name: str, value: object) -> None:
        if self.__dict__.get("_finalized", False):
            raise DefinitionError("prepared declaration is finalized")
        object.__setattr__(self, name, value)

    def __set_name__(self, owner: type[object], name: str) -> None:
        if self.owner is not None:
            raise DefinitionError("declaration cannot be reused in another slot")
        self.owner, self.name = owner, name


class Value(Declaration, Generic[T_co]):
    semantics: ValueSemantics[T_co]

    def __init__(self) -> None:
        self.when: Value[bool] | None = None

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...
    @overload
    def __get__(self, instance: Space, owner: type[object] | None = None) -> T_co: ...
    def __get__(self, instance: Space | None, owner: type[object] | None = None) -> Self | T_co:
        return self if instance is None else cast(T_co, _read(instance, self))


class Param(Value[T]):
    def __init__(self, value_type: type[T] | ValueSemantics[T]) -> None:
        super().__init__()
        self.semantics = semantics_for(value_type)


@dataclass(frozen=True)
class Domain(Generic[T]):
    accepts: Callable[..., bool]
    references: tuple[Value[object], ...] = ()


def domain(accepts: Callable[..., bool]) -> Domain[T]:
    return Domain(accepts)


def divisors_of(source: Value[int]) -> Domain[int]:
    def accepts(point: Space, candidate: int) -> bool:
        extent = _read(point, source)
        return candidate > 0 and cast(int, extent) % candidate == 0

    return Domain(accepts, (cast(Value[object], source),))


class Decision(Value[T]):
    def __init__(
        self,
        value_type: type[T] | ValueSemantics[T],
        *,
        domain: Domain[T],
        when: Value[bool] | None = None,
    ) -> None:
        super().__init__()
        self.semantics, self.domain, self.when = semantics_for(value_type), domain, when


class Derived(Value[T]):
    def __init__(self, function: Callable[..., T], *, when: Value[bool] | None = None) -> None:
        super().__init__()
        self.function, self.when = function, when
        # Preparation supplies semantics without executing the function.


def derived(function: Callable[..., T]) -> Derived[T]:
    return Derived(function)


class Constraint(Derived[bool]):
    pass


def constraint(function: Callable[..., bool]) -> Constraint:
    return Constraint(function)


class BoundView(Generic[T]):
    def __init__(self, instance: Space, declaration: View[T]) -> None:
        self.instance, self.declaration = instance, declaration

    def __call__(self) -> T:
        return cast(T, _read(self.instance, self.declaration))

    def inspect(self) -> ViewAssessment[T]:
        _driver_only("view inspection")
        entry = _evaluate(self.instance._snapshot, _key(self.instance, self.declaration))
        assert entry.assessment is not None
        # No live configurations appear in the assessment.
        return cast(ViewAssessment[T], _copy_assessment(self.declaration, entry.assessment))


class View(Declaration, Generic[T]):
    def __init__(
        self,
        function: Callable[..., T],
        *,
        constraints: Sequence[Constraint] = (),
        when: Value[bool] | None = None,
    ) -> None:
        self.function, self.constraints, self.when = function, tuple(constraints), when
        self.semantics: ValueSemantics[T]

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...
    @overload
    def __get__(self, instance: Space, owner: type[object] | None = None) -> BoundView[T]: ...
    def __get__(
        self, instance: Space | None, owner: type[object] | None = None
    ) -> Self | BoundView[T]:
        return self if instance is None else BoundView(instance, self)


def view(
    *, constraints: Sequence[Constraint] = (), when: Value[bool] | None = None
) -> Callable[[Callable[..., T]], View[T]]:
    def decorate(function: Callable[..., T]) -> View[T]:
        return View(function, constraints=constraints, when=when)

    return decorate


class Subspace(Declaration, Generic[S]):
    def __init__(self, family: type[S], **bindings: object) -> None:
        self.family, self.bindings = family, MappingProxyType(dict(bindings))

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...
    @overload
    def __get__(self, instance: Space, owner: type[object] | None = None) -> S: ...
    def __get__(self, instance: Space | None, owner: type[object] | None = None) -> Self | S:
        if instance is None:
            return self
        _same_snapshot(instance)
        scope = instance._snapshot.model.children[(instance._scope, self.name)]
        return cast(S, _attach(instance._snapshot, scope))


@dataclass(frozen=True)
class Change(Generic[T_co]):
    base: _Snapshot
    key: str
    value: T_co | None = None
    remove: bool = False


class BoundDecision(Generic[T]):
    def __init__(self, instance: Space, declaration: Decision[T]) -> None:
        self.instance, self.declaration = instance, declaration

    def change(self, value: T) -> Change[T]:
        _driver_only("change construction")
        return Change(self.instance._snapshot, _key(self.instance, self.declaration), value)

    def clear(self) -> Change[T]:
        _driver_only("clear construction")
        return Change(self.instance._snapshot, _key(self.instance, self.declaration), remove=True)


@dataclass(frozen=True)
class Update(Generic[S]):
    instance: S
    accepted: bool
    outcomes: Mapping[str, QueryResult[object]]


class SpaceMeta(type):
    def __call__(cls, **parameters: object) -> Space:
        return prepare(cast(type[Space], cls)).bind(**parameters)

    def __setattr__(cls, name: str, value: object) -> None:
        if cls.__dict__.get("_definition_finalized", False):
            raise DefinitionError("prepared family is finalized")
        super().__setattr__(name, value)

    def __delattr__(cls, name: str) -> None:
        if cls.__dict__.get("_definition_finalized", False):
            raise DefinitionError("prepared family is finalized")
        super().__delattr__(name)


class Space(metaclass=SpaceMeta):
    _snapshot: _Snapshot
    _scope: str

    def __init__(self, **parameters: object) -> None:
        """Typing signature; the metaclass performs allocation."""

    def __setattr__(self, name: str, value: object) -> None:
        _driver_only("configuration mutation")
        raise AttributeError("immutable configuration; use with_choices")

    def __delattr__(self, name: str) -> None:
        _driver_only("configuration mutation")
        raise AttributeError("immutable configuration")

    @property
    def root(self) -> Space:
        _same_snapshot(self)
        return _attach(self._snapshot, "")

    def query(self, reference: Value[T]) -> QueryResult[T]:
        _driver_only("status inspection")
        entry = _evaluate(self._snapshot, _key(self, reference))
        return cast(QueryResult[T], _copy_result(reference, entry.result))

    def explain(self, reference: Value[T] | View[T]) -> Mapping[str, tuple[str, ...]]:
        _driver_only("explanation")
        key = _key(self, reference)
        _evaluate(self._snapshot, key)
        pending, seen, result = [key], set(), {}
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            dependencies = self._snapshot.cache[current].reads
            result[current] = dependencies
            pending.extend(dependencies)
        return MappingProxyType(result)

    def field(self, reference: Decision[T]) -> BoundDecision[T]:
        _driver_only("field inspection")
        _key(self, reference)
        return BoundDecision(self, reference)

    def try_with_choices(self, /, *changes: Change[object], **choices: object) -> Update[Self]:
        return _update(self, changes, choices, monotone=False)

    def with_choices(self, /, *changes: Change[object], **choices: object) -> Self:
        result = self.try_with_choices(*changes, **choices)
        if not result.accepted:
            raise ValueUnavailableError(Rejected(()), context=result)
        return result.instance

    def commit(self, /, *changes: Change[object], **choices: object) -> Update[Self]:
        return _update(self, changes, choices, monotone=True)


@dataclass(frozen=True)
class _Node:
    key: str
    scope: str
    declaration: Value[object] | View[object]
    alias: str | None = None
    literal: object = None
    bound_literal: bool = False


@dataclass(frozen=True)
class Model(Generic[S]):
    family: type[S]
    nodes: Mapping[str, _Node]
    families: Mapping[str, type[Space]]
    children: Mapping[tuple[str, str], str]
    members: Mapping[tuple[str, Declaration], str]

    def bind(self, **parameters: object) -> S:
        _driver_only("configuration binding")
        expected = {
            key
            for key, node in self.nodes.items()
            if node.scope == "" and isinstance(node.declaration, Param)
        }
        if set(parameters) != expected:
            raise RequestError(f"parameters must be exactly {sorted(expected)}")
        for key, value in parameters.items():
            if not self.nodes[key].declaration.semantics.accepts(value):
                raise RequestError(f"{key}: invalid nominal parameter type")
        frozen = {
            key: self.nodes[key].declaration.semantics.freeze(value)
            for key, value in parameters.items()
        }
        snapshot = _Snapshot(
            cast(Model[Space], self), MappingProxyType(frozen), MappingProxyType({})
        )
        return cast(S, _attach(snapshot, ""))


def _namespace(family: type[Space]) -> dict[str, object]:
    return {name: value for base in reversed(family.__mro__) for name, value in vars(base).items()}


def _check_signature(function: Callable[..., object], count: int, owner: str) -> None:
    parameters = tuple(inspect.signature(function).parameters.values())
    if (
        len(parameters) != count
        or any(
            parameter.kind
            not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            or parameter.default is not inspect.Parameter.empty
            for parameter in parameters
        )
        or inspect.iscoroutinefunction(function)
        or inspect.isgeneratorfunction(function)
    ):
        raise DefinitionError(f"{owner}: requires {count} ordinary positional parameters")


def prepare(family: type[S]) -> Model[S]:
    _driver_only("preparation")
    with _PREPARE_LOCK:
        cached = family.__dict__.get("_prepared")
        if isinstance(cached, Model):
            return cast(Model[S], cached)
        nodes: dict[str, _Node] = {}
        families: dict[str, type[Space]] = {}
        children: dict[tuple[str, str], str] = {}
        members: dict[tuple[str, Declaration], str] = {}
        pending: list[tuple[str, type[Space], dict[str, object], tuple[type[Space], ...], str]] = [
            ("", cast(type[Space], family), {}, (), "")
        ]
        while pending:
            scope, current, bindings, ancestors, parent = pending.pop()
            if current in ancestors:
                raise DefinitionError(f"{scope}: recursive placement structure")
            namespace = _namespace(current)
            families[scope] = current
            if any(
                "__init__" in vars(base) or "__new__" in vars(base)
                for base in current.__mro__
                if base not in (Space, object)
            ):
                raise DefinitionError("custom constructors unsupported")
            params = {name for name, item in namespace.items() if isinstance(item, Param)}
            if scope and set(bindings) != params:
                raise DefinitionError(f"{scope}: child parameter binding names do not match")
            scope_nodes: list[_Node] = []
            for name, item in namespace.items():
                if not isinstance(item, Declaration):
                    continue
                if name.startswith("_") or name in vars(Space):
                    raise DefinitionError(f"reserved declaration name {name}")
                key = f"{scope}.{name}" if scope else name
                members[(scope, item)] = key
                if isinstance(item, Subspace):
                    children[(scope, name)] = key
                    pending.append(
                        (key, item.family, dict(item.bindings), (*ancestors, current), scope)
                    )
                    continue
                if not isinstance(item, (Value, View)):
                    raise DefinitionError(f"{key}: unsupported declaration")
                if isinstance(item, (Derived, View)):
                    _check_signature(item.function, 1, key)
                    annotation = get_type_hints(item.function, localns=namespace).get("return")
                    nominal = get_origin(annotation) or annotation
                    if not isinstance(nominal, type):
                        raise DefinitionError(f"{key}: a nominal return annotation is required")
                    if not hasattr(item, "semantics"):
                        item.semantics = semantics_for(nominal)
                if isinstance(item, Decision):
                    _check_signature(item.domain.accepts, 2, key)
                node = _Node(key, scope, item)
                if scope and isinstance(item, Param):
                    binding = bindings[name]
                    if isinstance(binding, Value):
                        try:
                            alias = members[(parent, binding)]
                        except KeyError as cause:
                            raise DefinitionError(f"{key}: foreign parameter binding") from cause
                        if not item.semantics.is_compatible_with(
                            nodes[alias].declaration.semantics
                        ):
                            raise DefinitionError(f"{key}: binding type mismatch")
                        node = _Node(key, scope, item, alias=alias)
                    else:
                        try:
                            frozen = item.semantics.freeze(binding)
                        except TypeError as cause:
                            raise DefinitionError(f"{key}: invalid bound literal") from cause
                        node = _Node(key, scope, item, literal=frozen, bound_literal=True)
                nodes[key] = node
                scope_nodes.append(node)
            for node in scope_nodes:
                refs: list[Declaration] = []
                if node.declaration.when is not None:
                    refs.append(node.declaration.when)
                if isinstance(node.declaration, Decision):
                    refs.extend(node.declaration.domain.references)
                if isinstance(node.declaration, View):
                    refs.extend(node.declaration.constraints)
                for reference in refs:
                    if (scope, reference) not in members:
                        raise DefinitionError(f"{node.key}: foreign known reference")
                guard = node.declaration.when
                if guard is not None and guard.semantics.type_token is not bool:
                    raise DefinitionError(f"{node.key}: applicability requires bool")
                if isinstance(node.declaration, View) and any(
                    not isinstance(item, Constraint) for item in node.declaration.constraints
                ):
                    raise DefinitionError(f"{node.key}: view constraint has wrong kind")
        model = Model(
            family,
            MappingProxyType(nodes),
            MappingProxyType(families),
            MappingProxyType(children),
            MappingProxyType(members),
        )
        for participating in families.values():
            for base in participating.__mro__:
                if base in (Space, object):
                    continue
                type.__setattr__(base, "_definition_finalized", True)
            for declaration in _namespace(participating).values():
                if isinstance(declaration, Declaration):
                    object.__setattr__(declaration, "_finalized", True)
        type.__setattr__(family, "_prepared", model)
        return model


@dataclass
class Work:
    callback_starts: int = 0
    getter_attempts: int = 0
    nodes_started: int = 0
    suspensions: int = 0
    max_pending: int = 0


@dataclass(eq=False)
class _Snapshot:
    model: Model[Space]
    parameters: Mapping[str, object]
    candidates: Mapping[str, object]
    cache: dict[str, _Entry] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock)
    work: Work = field(default_factory=Work)


@dataclass(frozen=True)
class _Entry:
    result: QueryResult[object]
    reads: tuple[str, ...] = ()
    assessment: ViewAssessment[object] | None = None


class _Suspend(BaseException):
    def __init__(self, key: str) -> None:
        self.key = key


@dataclass
class _Context:
    snapshot: _Snapshot
    key: str
    request: Callable[[str], None]
    reads: dict[str, None] = field(default_factory=dict)
    blocked: QueryResult[object] | None = None
    fault: EvaluationError | None = None
    pending: str | None = None
    role: str = "value"


def _driver_only(role: str) -> None:
    active = _ACTIVE.get()
    if active is not None:
        active.fault = EvaluationError(active.key, role, "driver-only API during computation")
        raise active.fault


def _same_snapshot(point: Space) -> None:
    active = _ACTIVE.get()
    if active is not None and active.snapshot is not point._snapshot:
        active.fault = EvaluationError(active.key, "field read", "cross-snapshot access")
        raise active.fault


def _key(point: Space, declaration: Declaration) -> str:
    try:
        return point._snapshot.model.members[(point._scope, declaration)]
    except KeyError as cause:
        raise RequestError("reference does not belong to this scope") from cause


def _attach(snapshot: _Snapshot, scope: str) -> Space:
    point = object.__new__(snapshot.model.families[scope])
    object.__setattr__(point, "_snapshot", snapshot)
    object.__setattr__(point, "_scope", scope)
    return point


def _copy_result(
    declaration: Value[T] | View[T], answer: QueryResult[object]
) -> QueryResult[object]:
    return (
        Available(declaration.semantics.freeze(answer.value))
        if isinstance(answer, Available)
        else answer
    )


def _copy_assessment(
    declaration: View[T], answer: ViewAssessment[object]
) -> ViewAssessment[object]:
    readiness = ReadinessAssessment(
        {
            key: value if key in answer.constraints.results else _copy_result(declaration, value)
            for key, value in answer.readiness.results.items()
        },
        answer.readiness.result,
    )
    return ViewAssessment(
        _copy_result(declaration, answer.output_result),
        readiness,
        answer.constraints,
        _copy_result(declaration, answer.accepted_result),
    )


def _read(point: Space, declaration: Value[T] | View[T]) -> object:
    _same_snapshot(point)
    key, snapshot = _key(point, declaration), point._snapshot
    snapshot.work.getter_attempts += 1
    active = _ACTIVE.get()
    if active is None:
        answer = _evaluate(snapshot, key).result
    else:
        active.reads[key] = None
        if key not in snapshot.cache:
            active.pending = key
            active.request(key)
            active.pending = None
        answer = snapshot.cache[key].result
        if not isinstance(answer, Available):
            active.blocked = answer
    return require_value(_copy_result(declaration, answer), context=key)


def _raw_read(point: Space, declaration: Value[object]) -> QueryResult[object]:
    """Engine-only assessment read; not an author status-observation escape."""
    active = _ACTIVE.get()
    assert active is not None
    key = _key(point, declaration)
    active.reads[key] = None
    if key not in point._snapshot.cache:
        active.pending = key
        active.request(key)
        active.pending = None
    return point._snapshot.cache[key].result


def _compute(context: _Context) -> _Entry:
    snapshot = context.snapshot
    node = snapshot.model.nodes[context.key]
    declaration = node.declaration
    point = _attach(snapshot, node.scope)
    answer: QueryResult[object]
    assessment: ViewAssessment[object] | None
    if declaration.when is not None:
        context.role = "applicability"
        if _read(point, cast(Value[object], declaration.when)) is False:
            answer = Inapplicable()
            assessment = (
                assess_view(answer, owner=node.key, applicability=answer)
                if isinstance(declaration, View)
                else None
            )
            return _Entry(answer, assessment=assessment)
    if node.alias is not None:
        target = snapshot.model.nodes[node.alias]
        return _Entry(Available(_read(_attach(snapshot, target.scope), target.declaration)))
    if isinstance(declaration, Param):
        return _Entry(
            Available(node.literal if node.bound_literal else snapshot.parameters[node.key])
        )
    if isinstance(declaration, Decision):
        if node.key not in snapshot.candidates:
            return _Entry(
                Unresolved(
                    (
                        Finding(
                            FindingKind.BLOCKER,
                            "decision-unassigned",
                            node.key,
                            "decision requires a commitment",
                        ),
                    )
                )
            )
        context.role = "admission/domain"
        snapshot.work.callback_starts += 1
        candidate = declaration.semantics.freeze(snapshot.candidates[node.key])
        accepted = declaration.domain.accepts(point, candidate)
        if type(accepted) is not bool:
            raise TypeError("domain must return bool")
        if not accepted:
            return _Entry(reject("domain-refused", "candidate outside domain", owner=node.key))
        return _Entry(Available(snapshot.candidates[node.key]))
    assert isinstance(declaration, (Derived, View))
    context.role = "view" if isinstance(declaration, View) else "derived"
    snapshot.work.callback_starts += 1
    value = declaration.function(point)
    answer = Available(declaration.semantics.freeze(value))
    if isinstance(declaration, Constraint):
        return _Entry(cast(QueryResult[object], constraint_result(cast(bool, value), node.key)))
    if isinstance(declaration, View):
        constraints = {
            _key(point, item): cast(QueryResult[bool], _raw_read(point, cast(Value[object], item)))
            for item in declaration.constraints
        }
        assessment = assess_view(answer, owner=node.key, constraints=constraints)
        return _Entry(assessment.accepted_result, assessment=assessment)
    return _Entry(answer)


def _attempt(context: _Context) -> _Entry:
    token = _ACTIVE.set(context)
    try:
        try:
            entry = _compute(context)
        except ValueUnavailableError:
            if context.blocked is None:
                raise
            entry = _Entry(context.blocked)
        if context.fault is not None:
            raise context.fault
        if context.pending is not None:
            raise _Suspend(context.pending)
        if context.blocked is not None:
            entry = _Entry(context.blocked)
        # A view blocked before its body completes still has an inspectable assessment.
        declaration = context.snapshot.model.nodes[context.key].declaration
        if isinstance(declaration, View) and entry.assessment is None:
            entry = _Entry(entry.result, assessment=assess_view(entry.result, owner=context.key))
        return _Entry(entry.result, tuple(context.reads), entry.assessment)
    except EvaluationError:
        raise
    except Exception as cause:
        raise EvaluationError(context.key, context.role, str(cause)) from cause
    finally:
        _ACTIVE.reset(token)


class _Greenlet(Protocol):
    gr_context: object
    dead: bool

    def switch(self, *args: object) -> object: ...
    def throw(self, exception: type[BaseException]) -> object: ...


def _evaluate(snapshot: _Snapshot, key: str) -> _Entry:
    with snapshot.lock:
        if key in snapshot.cache:
            return snapshot.cache[key]
        mode = _MODE.get()
        active: dict[str, _Context] = {}
        stack: list[str] = []
        continuations: dict[str, _Greenlet] = {}
        if mode == "greenlet":
            module = importlib.import_module("greenlet")
            parent = cast(_Greenlet, module.getcurrent())

        def request(dependency: str) -> None:
            if dependency in active:
                roles = [f"{item} ({active[item].role})" for item in stack]
                error = EvaluationError(
                    dependency, "dependency cycle", " -> ".join((*roles, dependency))
                )
                active[stack[-1]].fault = error
                raise error
            snapshot.work.suspensions += 1
            if mode == "recursive":
                visit(dependency)
            elif mode == "greenlet":
                parent.switch(dependency)
            else:
                raise _Suspend(dependency)

        def begin(current: str) -> _Context:
            context = _Context(snapshot, current, request)
            active[current] = context
            stack.append(current)
            snapshot.work.nodes_started += 1
            snapshot.work.max_pending = max(snapshot.work.max_pending, len(stack))
            return context

        def finish(current: str, entry: _Entry) -> None:
            snapshot.cache[current] = entry
            stack.pop()
            active.pop(current)
            continuations.pop(current, None)

        def visit(current: str) -> None:
            context = begin(current)
            finish(current, _attempt(context))

        if mode == "recursive":
            visit(key)
            return snapshot.cache[key]
        begin(key)
        try:
            while stack:
                current = stack[-1]
                context = active[current]
                try:
                    if mode == "greenlet":
                        continuation = continuations.get(current)
                        if continuation is None:
                            continuation = cast(
                                _Greenlet,
                                module.greenlet(
                                    lambda context=context: _attempt(context), parent=parent
                                ),
                            )
                            continuation.gr_context = copy_context()
                            continuations[current] = continuation
                        output = continuation.switch()
                        if not continuation.dead:
                            begin(cast(str, output))
                            continue
                        entry = cast(_Entry, output)
                    else:
                        # Retain reads across attempts; clear only transient suspension state.
                        context.pending = None
                        entry = _attempt(context)
                    finish(current, entry)
                except _Suspend as signal:
                    begin(signal.key)
            return snapshot.cache[key]
        finally:
            # Explicitly unwind suspended ordinary frames on error; no live task cache.
            for continuation in tuple(continuations.values()):
                if not continuation.dead:
                    try:
                        continuation.throw(GeneratorExit)
                    except GeneratorExit:
                        pass


def _update(
    point: S, changes: tuple[Change[object], ...], choices: Mapping[str, object], *, monotone: bool
) -> Update[S]:
    _driver_only("configuration update")
    snapshot = point._snapshot
    pending: dict[str, Change[object]] = {}
    keywords = tuple(
        Change(snapshot, f"{point._scope}.{name}" if point._scope else name, value)
        for name, value in choices.items()
    )
    for change in (*changes, *keywords):
        if not isinstance(change, Change) or change.base is not snapshot:
            raise RequestError("change must target this exact snapshot")
        node = snapshot.model.nodes.get(change.key)
        if node is None or not isinstance(node.declaration, Decision):
            raise RequestError("change must name an owned decision")
        if change.key in pending:
            raise RequestError("duplicate change")
        if change.remove and monotone:
            raise RequestError("monotone operation cannot clear")
        if not change.remove and not node.declaration.semantics.accepts(change.value):
            raise RequestError(f"{change.key}: invalid nominal candidate type")
        pending[change.key] = change
    prepared = {
        key: snapshot.model.nodes[key].declaration.semantics.freeze(change.value)
        for key, change in pending.items()
        if not change.remove
    }
    merged = dict(snapshot.candidates)
    for key, change in pending.items():
        if change.remove:
            merged.pop(key, None)
        else:
            if (
                monotone
                and key in merged
                and not snapshot.model.nodes[key].declaration.semantics.values_equal(
                    merged[key], prepared[key]
                )
            ):
                return Update(
                    point,
                    False,
                    MappingProxyType(
                        {
                            key: reject(
                                "commitment-conflict", "cannot revise monotone choice", owner=key
                            )
                        }
                    ),
                )
            merged[key] = prepared[key]
    if merged.keys() == snapshot.candidates.keys() and all(
        snapshot.model.nodes[key].declaration.semantics.values_equal(
            snapshot.model.nodes[key].declaration.semantics.freeze(value),
            snapshot.model.nodes[key].declaration.semantics.freeze(snapshot.candidates[key]),
        )
        for key, value in merged.items()
    ):
        return Update(point, True, MappingProxyType({}))
    trial = _Snapshot(snapshot.model, snapshot.parameters, MappingProxyType(merged))
    outcomes = {
        key: _copy_result(snapshot.model.nodes[key].declaration, _evaluate(trial, key).result)
        for key in merged
    }
    if any(not isinstance(answer, Available) for answer in outcomes.values()):
        return Update(point, False, MappingProxyType(outcomes))
    published = _Snapshot(snapshot.model, snapshot.parameters, MappingProxyType(merged))
    return Update(cast(S, _attach(published, point._scope)), True, MappingProxyType(outcomes))


def capture(point: Space) -> tuple[tuple[str, object], ...]:
    """Bounded in-process same-family representation: sparse detached choices only."""
    _driver_only("choice capture")
    return tuple(
        (key, point._snapshot.model.nodes[key].declaration.semantics.freeze(value))
        for key, value in sorted(point._snapshot.candidates.items())
    )


def replay(point: S, saved: tuple[tuple[str, object], ...]) -> Update[S]:
    _driver_only("choice replay")
    changes = tuple(Change(point._snapshot, key, value) for key, value in saved)
    return point.try_with_choices(*changes)
