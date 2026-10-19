# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The engine side of the harness: a model description built as an engine Space, and the
engine asked the model's questions, each answer put in the comparable form (``forms``).

Everything that depends on how the engine is **authored** or **asked** at this commit is in
this one module, so the engine core's change of API is a port of this file alone:

- **Spelling.** A rule of the description is a constraint in the model, and in the engine
  every rule is a ``@constraint`` of the Space that owns it, which its Space's validity
  counts. The spellings (``Spelling.rules``, beside each case) name where a rule came from
  and how it reads: ``Admission`` (the default), ``Plain``, ``Requirement`` (a rule over a
  choice and a fact, for some of the choice's cases) and ``Obligation`` (a rule stated
  beside a view of its Space) are each a plain ``@constraint`` reading its inputs
  explicitly; ``ByName`` is a ``@constraint`` method reading its inputs by attribute name,
  ``getattr(self, name)``. A view carries no rule. A domain with a membership predicate is
  a ``domain(accepts=, candidates=)``: membership is the model's whole membership, the
  enumeration the listed values. A domain's ``order`` is not built: no compared answer
  reads it.
- **Questions, today's engine's** (what the engine core's port replaces):

  - a stored choice is set, waiting (``inspection.unchecked``, with what it waits on),
    dormant (its decision inapplicable), or the legacy ``refused``; an unset one is open,
    absent, the legacy ``forced`` (``inspection.forced``) or ``refused``;
  - a change's report names refusals per choice (``ChangeOutcome``, named or reached):
    one whose own finding is ``domain-membership`` is an ``assignment`` refusal, any other
    the legacy ``choice``; it reports nothing put dormant, in play or opened;
  - possible values are ``inspection.viable``'s: an open choice neither stored nor forced,
    its viable cases (possible, nothing moving) and its other member values (impossible,
    with no refusal form: their reasons are text);
  - validity's ``"undecided"`` is ``"open"``;
  - a definition is refused for a dependency cycle only (``definition``): no domain reading
    a choice is refused.

The description becomes one Space class per scope, built with ``composite``. Every member
keeps the description's path as its key: a choice ``compute.packed.pe`` is the engine's
Decision ``compute.packed.pe``. A scope that reads something outside it takes it as a formal
(``in-<path>``), bound by its parent; a selector's value is read through a derived member
``key-<name>`` (``selected``); a child's guard is a derived member ``when-<name>``. Facts
are the root's optional formals: a missing fact is an unsupplied one.
"""

from __future__ import annotations

import ast
import inspect as pyinspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from core.space import reference as model
from core.space.reference.model import same
from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    EvaluationError,
    Inapplicable,
    Param,
    QueryResult,
    Rejected,
    Space,
    Unresolved,
    composite,
    constraint,
    derived,
    design_space,
    domain,
    inspection,
    occurrence,
    reject,
    selected,
    view,
)
from finn.core.space.edits import ChangeRequest

from .forms import (
    AnswerForm,
    ChangeForm,
    Code,
    DefinitionForm,
    OutcomeForm,
    PossibleForm,
    RefusalForm,
    RuleForm,
    StateForm,
    ValidityForm,
)


class NotTranslatable(Exception):
    """A description, a spelling or a change this harness cannot write in the engine. Never
    a disagreement: a test that meets one fails."""


# --- spelling -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Admission:
    """A ``@constraint`` of its Space (the default)."""


@dataclass(frozen=True)
class Plain:
    """A ``@constraint`` of its Space: every constraint counts, so the same as
    ``Admission``."""


@dataclass(frozen=True)
class Requirement:
    """A rule that ``choice``'s ``cases`` need ``fact``: a ``@constraint`` of its Space, like
    any rule. The model's rule must read ``choice`` and ``fact`` and hold exactly when the
    value is not one of ``cases`` or ``fact`` is truthy."""

    choice: str
    fact: str
    cases: tuple[object, ...]


@dataclass(frozen=True)
class ByName:
    """A ``@constraint`` method of its Space, reading each input by name (``getattr(self,
    name)`` over a list of names, through child nodes for a child's member): the engine
    discovers the reads when the body runs."""


@dataclass(frozen=True)
class Obligation:
    """A rule stated beside ``view``, a view of the rule's own Space: a ``@constraint`` of
    that Space, like any rule (a view carries none; it refuses when its Space is
    refused)."""

    view: str


Rule = Admission | Plain | Requirement | Obligation | ByName


@dataclass(frozen=True)
class Spelling:
    """How a case's rules are written today (by rule path; ``Admission`` when absent) and
    the value types the description does not state (by path; a listed domain's are read
    from its values, everything else is ``int``)."""

    rules: Mapping[str, Rule] = field(default_factory=dict)
    types: Mapping[str, type] = field(default_factory=dict)

    def rule(self, path: str) -> Rule:
        return self.rules.get(path, Admission())


# --- the translator ---------------------------------------------------------------------------


def _explicit(
    reads: tuple[str, ...], body: Callable[[dict[str, Any]], Any], returns: object
) -> Any:
    """A function of keyword arguments ``a0, a1, ...`` (one per read, in order) calling
    ``body`` with the mapping from read path to value: an explicit-input function the engine
    binds by its signature."""

    names = [f"a{index}" for index in range(len(reads))]

    def call(**arguments: Any) -> Any:
        return body({read: arguments[name] for read, name in zip(reads, names, strict=True)})

    call.__signature__ = pyinspect.Signature(  # type: ignore[attr-defined]
        [pyinspect.Parameter(name, pyinspect.Parameter.KEYWORD_ONLY) for name in names]
    )
    call.__annotations__ = {**dict.fromkeys(names, object), "return": returns}
    return call


def _value_type(values: Iterable[object], path: str) -> type:
    kinds = {type(value) for value in values}
    if len(kinds) == 1:
        (kind,) = kinds
        if kind in (bool, int, str):
            return kind
    if not kinds:
        return int
    raise NotTranslatable(f"{path}: state the value type of {sorted(map(str, kinds))}")


def _name(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _input(path: str) -> str:
    """A scope's formal taking ``path`` from outside it."""
    return "in-" + path.replace(".", "-")


class _Unchecked(model.Index):
    """The description's index without the model's definition checks (cycles, a domain
    reading a choice): a description the model refuses is still built in the engine, to see
    what the engine answers."""

    def _check_domains(self) -> None:
        return None

    def _check_cycles(self) -> None:
        return None


class _Builder:
    """The engine Space classes of one description, one per scope, built on demand."""

    def __init__(self, space: model.Space, spelling: Spelling, unchecked: bool = False) -> None:
        self.index = _Unchecked(space) if unchecked else space.validate()
        self.spelling = spelling
        index = self.index
        for path, rule in spelling.rules.items():
            if path not in index.constraints:
                raise NotTranslatable(f"{path}: spelled, but not a rule of the description")
            if isinstance(rule, Requirement) and (
                rule.choice not in index.choices or index.kind(rule.fact) is None
            ):
                raise NotTranslatable(f"{path}: its requirement names unknown nodes")
            if isinstance(rule, Obligation) and (
                rule.view not in index.views
                or index.views[rule.view][1] != index.constraints[path][1]
            ):
                raise NotTranslatable(f"{path}: an obligation of a view of its own Space")
        self.externals = {scope: self._externals(scope) for scope in index.scopes}
        #: per scope: the declaration of each member it owns, by model path or generated name
        self.declared: dict[str, dict[str, object]] = {scope: {} for scope in index.scopes}
        #: per scope: the node declaration of each child scope and case placed in it
        self.nodes: dict[str, dict[str, Any]] = {scope: {} for scope in index.scopes}
        self.selectors: dict[str, Any] = {}
        self.classes: dict[str, type[Space]] = {}
        self.pending: set[tuple[str, str]] = set()
        #: an engine alias key (a scope's formal) -> the model path it takes
        self.aliases: dict[str, str] = {}

    # -- reads

    def owner(self, path: str) -> str:
        """The scope a readable node belongs to; a fact is the root's."""
        return "" if path in self.index.facts else self.index.scope_of(path)

    def _reads(self, scope: str) -> set[str]:
        """Every read of a node at or below ``scope`` (the guards of the scopes below it
        included, its own guard not: its parent evaluates that)."""
        index = self.index
        found: set[str] = set()
        for path, choice in index.choices.items():
            if index.within(choice.scope, scope):
                found.update(choice.domain.reads)
        for table in (index.derived, index.views, index.constraints):
            for path, (node, owner) in table.items():
                if index.within(owner, scope):
                    found.update(node.reads)
        for path, info in index.scopes.items():
            if path != scope and info.when is not None and index.within(path, scope):
                found.update(info.when.reads)
        return found

    def _externals(self, scope: str) -> tuple[str, ...]:
        if scope == "":
            return ()
        return tuple(
            sorted(
                read
                for read in self._reads(scope)
                if not self.index.within(self.owner(read), scope)
            )
        )

    def value_type(self, path: str) -> object:
        if path in self.spelling.types:
            return self.spelling.types[path]
        info = self.index.choices.get(path)
        if info is not None:
            if info.cases is not None:
                return str
            if info.domain.values is not None:
                return _value_type(info.domain.values, path)
        return int

    def reference(self, scope: str, path: str) -> Any:
        """``path`` as an engine reference written in ``scope``'s class body."""
        if path in self.externals[scope]:
            return self.member(scope, _input(path))
        owner = self.owner(path)
        if owner == scope:
            return self.member(scope, path)
        if not self.index.within(owner, scope):
            raise NotTranslatable(f"{path} is not visible from {scope or 'the root'}")
        chain: list[str] = []
        current: str | None = owner
        while current != scope:
            assert current is not None
            chain.append(current)
            current = self.index.scopes[current].parent
        chain.reverse()
        node = self.node(scope, chain[0])
        for below in chain[1:]:
            info = self.index.scopes[below]
            if info.selector is None:
                node = getattr(node, _name(below))
            else:
                node = getattr(node, _name(info.selector))[info.case]
        member = (
            "key-" + _name(path)
            if path in self.index.choices and self.index.choices[path].cases is not None
            else _name(path)
        )
        return getattr(node, member)

    # -- members

    def member(self, scope: str, name: str) -> Any:
        """The declaration of ``name`` (a model path the scope owns, or a generated name)
        in ``scope``, created on first use."""
        declared = self.declared[scope]
        if name in declared:
            return declared[name]
        if (scope, name) in self.pending:
            raise NotTranslatable(f"{name}: its declaration reads itself")
        self.pending.add((scope, name))
        declared[name] = self._declare(scope, name)
        self.pending.discard((scope, name))
        return declared[name]

    def _declare(self, scope: str, name: str) -> object:
        index = self.index
        if name.startswith("in-"):
            return Param()
        if name.startswith("when-"):
            child = name.removeprefix("when-")
            return self.guard(scope, f"{scope}.{child}" if scope else child)
        if name in index.facts:
            return Param(required=False)
        if name in index.choices and index.choices[name].cases is not None:
            # a selector's value, read as its selected key
            return self._selector_key(scope, name)
        if name in index.choices:
            return self._choice(scope, name)
        if name in index.derived:
            return self._derived(scope, name)
        if name in index.views:
            return self._view(scope, name)
        if name in index.constraints:
            return self._constraint(scope, name)
        raise NotTranslatable(f"{name}: nothing to declare in {scope or 'the root'}")

    def _aliases(self, scope: str, reads: tuple[str, ...]) -> dict[str, Any]:
        return {f"a{position}": self.reference(scope, read) for position, read in enumerate(reads)}

    def _selector_key(self, scope: str, path: str) -> object:
        decision = self.selector(scope, path)
        return derived(a0=selected(decision))(_explicit(("key",), lambda read: read["key"], str))

    def selector(self, scope: str, path: str) -> Any:
        if path in self.selectors:
            return self.selectors[path]
        info = self.index.choices[path]
        assert info.cases is not None
        entries: dict[str, type[Space] | Space] = {}
        for case in info.cases:
            entries[case] = self.node(scope, f"{path}.{case}")
        self.selectors[path] = Decision(entries, required=info.required)
        return self.selectors[path]

    def _choice(self, scope: str, path: str) -> object:
        info = self.index.choices[path]
        domain_ = info.domain
        simple = domain_.values is not None and domain_.member is None and not domain_.reads
        if simple and domain_.values:
            return Decision(values=domain_.values, required=info.required)
        reads = domain_.reads
        names = [f"a{position}" for position in range(len(reads))]

        def readings(arguments: Mapping[str, object]) -> dict[str, object]:
            return {read: arguments[name] for read, name in zip(reads, names, strict=True)}

        def listing(arguments: Mapping[str, object]) -> tuple[object, ...] | None:
            if domain_.values is not None:
                return domain_.values
            if domain_.enumerate is not None:
                return tuple(domain_.enumerate(readings(arguments)))
            return None

        def accepts(*, candidate: object, **arguments: object) -> bool:
            values = listing(arguments)
            if values is not None and not any(same(item, candidate) for item in values):
                return False
            return domain_.member is None or bool(domain_.member(candidate, readings(arguments)))

        def candidates(**arguments: object) -> tuple[object, ...]:
            values = listing(arguments)
            assert values is not None
            return values

        built = domain(
            accepts=accepts,
            candidates=candidates if domain_.listed else None,
            **dict(zip(names, (self.reference(scope, read) for read in reads), strict=True)),
        )
        return Decision(domain=built, required=info.required)

    def _derived(self, scope: str, path: str) -> object:
        """A derived value; one whose function refuses rejects with the refusal's code."""
        node = self.index.derived[path][0]

        def body(read: dict[str, Any]) -> object:
            output = node.compute(read)
            if isinstance(output, model.Refuse):
                return reject(output.code, "the derived value refuses")
            return output

        returns = self.value_type(path) | Rejected
        return derived(**self._aliases(scope, node.reads))(_explicit(node.reads, body, returns))

    def _view(self, scope: str, path: str) -> object:
        node = self.index.views[path][0]

        def body(read: dict[str, Any]) -> object:
            output = node.body(read)
            if isinstance(output, model.Refuse):
                return reject(output.code, "the view refuses")
            return output

        returns = self.value_type(path) | Rejected
        return view(**self._aliases(scope, node.reads))(_explicit(node.reads, body, returns))

    def _constraint(self, scope: str, path: str) -> object:
        rule = self.index.constraints[path][0]

        def body(read: dict[str, Any]) -> bool | Rejected:
            return True if rule.predicate(read) else reject(rule.code, "the rule refuses")

        if isinstance(self.spelling.rule(path), ByName):
            return constraint(self._by_name(scope, path, rule.reads, body))
        return constraint(**self._aliases(scope, rule.reads))(
            _explicit(rule.reads, body, bool | Rejected)
        )

    def _by_name(
        self,
        scope: str,
        path: str,
        reads: tuple[str, ...],
        body: Callable[[dict[str, Any]], bool | Rejected],
    ) -> Any:
        """A method reading each of ``reads`` by attribute name from ``self``."""
        chains: list[tuple[str, tuple[str, ...]]] = []
        for read in reads:
            owner = self.owner(read)
            if read in self.index.facts or not self.index.within(owner, scope):
                raise NotTranslatable(f"{path}: a rule read by name reads inside its Space")
            chain = [_name(read)]
            current = owner
            while current != scope:
                info = self.index.scopes[current]
                if info.selector is not None or info.when is not None:
                    raise NotTranslatable(f"{path}: read by name through a plain child only")
                chain.append(_name(current))
                assert info.parent is not None
                current = info.parent
            chains.append((read, tuple(reversed(chain))))

        def method(self: Any) -> bool | Rejected:
            read: dict[str, Any] = {}
            for target, chain in chains:
                node = self
                for name in chain:
                    node = getattr(node, name)
                read[target] = node
            return body(read)

        method.__annotations__ = {"return": bool | Rejected}
        return method

    # -- scopes

    def node(self, scope: str, below: str) -> Any:
        """The node declaration placing the child scope or case ``below`` in ``scope``."""
        nodes = self.nodes[scope]
        if below in nodes:
            return nodes[below]
        info = self.index.scopes[below]
        space_type = self.space_class(below)
        if info.selector is not None:
            if info.when is not None:
                raise NotTranslatable(f"{below}: a selector's case takes no guard here")
            nodes[below] = space_type()
        elif info.when is None:
            nodes[below] = space_type()
        else:
            guard = self.member(scope, "when-" + _name(below))
            nodes[below] = space_type(when=guard)
        return nodes[below]

    def guard(self, scope: str, below: str) -> object:
        when = self.index.scopes[below].when
        assert when is not None
        if any(self.index.within(self.owner(read), below) for read in when.reads):
            raise NotTranslatable(f"{below}: its guard reads inside it")
        return derived(**self._aliases(scope, when.reads))(
            _explicit(when.reads, lambda read: bool(when.test(read)), bool)
        )

    def space_class(self, scope: str) -> type[Space]:
        if scope in self.classes:
            return self.classes[scope]
        index = self.index
        members: dict[str, object] = {}
        annotations: dict[str, object] = {}
        mine = [
            path
            for table in (index.choices, index.derived, index.views, index.constraints)
            for path in table
            if self.owner(path) == scope
        ]
        if scope == "":
            for fact in sorted(index.facts):
                members[fact] = self.member(scope, fact)
                annotations[fact] = self.value_type(fact)
        for path in self.externals[scope]:
            members[_input(path)] = self.member(scope, _input(path))
            annotations[_input(path)] = self.value_type(path)
        for path in mine:
            if path in index.choices and index.choices[path].cases is not None:
                members[_name(path)] = self.selector(scope, path)
                members["key-" + _name(path)] = self.member(scope, path)
                annotations[_name(path)] = object
                continue
            members[_name(path)] = self.member(scope, path)
            if path in index.choices:
                annotations[_name(path)] = self.value_type(path)
        for path in sorted(index.scopes):
            child = index.scopes[path]
            if child.parent == scope and child.selector is None:
                members[_name(path)] = self.node(scope, path)
                if child.when is not None:
                    members["when-" + _name(path)] = self.member(scope, "when-" + _name(path))
        # bind every placed node's formals (assigned after the node, as a class body may)
        for below, placed in self.nodes[scope].items():
            for path in self.externals[below]:
                setattr(placed, _input(path), self.reference(scope, path))
                self.aliases[f"{below}.{_input(path)}"] = path
        owned = {_name(path) for path in mine} | {
            _name(path) for path, child in index.scopes.items() if child.parent == scope
        }
        clashes = sorted(name for name in owned if "-" in name)
        if clashes:
            raise NotTranslatable(f"{scope}: names the harness generates: {clashes}")
        name = "Ref_" + (scope.replace(".", "_") or "root")
        self.classes[scope] = composite(name, members, annotations=annotations)
        return self.classes[scope]


# --- the engine's answers --------------------------------------------------------------------


@dataclass(frozen=True)
class Engine:
    """A description built as an engine Space class, with what maps the engine's owners
    back to the description's paths."""

    description: model.Space
    spelling: Spelling
    root: type[Space]
    index: model.Index
    aliases: Mapping[str, str]

    def open(self, facts: Mapping[str, object]) -> Point:
        """The configuration with ``facts`` and nothing committed."""
        return Point(self, design_space(self.root(**cast(dict[str, Any], facts))))

    def path(self, owner: str) -> str:
        """The model path an engine finding's owner names."""
        return self.aliases.get(owner, owner)

    def codes(self, findings: Iterable[Any], only: str | None = None) -> frozenset[Code]:
        """Findings as codes; ``only``: the findings owned by that node alone."""
        return frozenset(
            (finding.code, self.path(finding.owner))
            for finding in findings
            if only is None or finding.owner == only
        )

    def blockers(self, findings: Iterable[Any], asked: str | None = None) -> frozenset[str]:
        """What an answer waits on: its findings' owners, but for ``asked``, the node whose
        answer it is (the engine names a choice whose own check waits as the owner of what
        it waits on; the model names the inputs, and blockers compare as a subset)."""
        return frozenset(self.path(finding.owner) for finding in findings) - {asked}


def build(
    description: model.Space, spelling: Spelling | None = None, unchecked: bool = False
) -> Engine:
    """The engine Space class of ``description``, spelled as ``spelling`` says; with
    ``unchecked``, built even where the model refuses the description as no definition."""
    builder = _Builder(description, spelling or Spelling(), unchecked)
    root = builder.space_class("")
    return Engine(
        description,
        builder.spelling,
        root,
        builder.index,
        dict(builder.aliases),
    )


def definition(
    description: model.Space,
    spelling: Spelling | None = None,
    facts: Mapping[str, object] | None = None,
) -> DefinitionForm:
    """Whether the engine takes ``description`` as a definition: a dependency cycle reported
    when building the Space, or when asked (its validity, every rule and every view of the
    configuration with ``facts``), is ``"cycle"``; today's engine reports no domain reading
    a choice, and answers otherwise (``None``)."""
    try:
        engine = build(description, spelling, unchecked=True)
        point = engine.open(dict(facts or {}))
        point.validity()
        for path in sorted(engine.index.constraints):
            point.rule(path)
        for path in sorted(engine.index.views):
            point.answer(path)
    except DefinitionError as error:
        if any(finding.code == "cyclic-dependency" for finding in error.findings):
            return DefinitionForm("cycle")
        raise
    except EvaluationError as error:
        if error.role == "dependency cycle":
            return DefinitionForm("cycle")
        raise
    return DefinitionForm(None)


@dataclass(frozen=True)
class Point:
    """One engine configuration, asked the model's questions."""

    engine: Engine
    point: Space

    # -- handles

    def _decisions(self) -> dict[str, Any]:
        return {item.key: item.reference for item in inspection.decisions(self.point)}

    def _members(self) -> dict[str, Any]:
        return {item.key: item.reference for item in inspection.members(self.point)}

    def unchecked(self) -> dict[str, frozenset[str]]:
        """The stored choices kept unchecked, by path, each with what it waits on (the
        engine's list: the choices still open or unchecked, and the inputs not supplied,
        that its own check and its guard read)."""
        return {
            item.key: frozenset(self.engine.path(key) for key in item.blockers) - {item.key}
            for item in inspection.unchecked(self.point)
        }

    def stored(self) -> dict[str, object]:
        """The committed choices, by path."""
        current = occurrence.state(self.point)
        return {
            current.linked.nodes[index].key: value for index, value in current.assignments.items()
        }

    def _query(self, path: str) -> QueryResult[Any]:
        index = self.engine.index
        if path in index.choices and index.choices[path].cases is not None:
            scope = index.choices[path].scope
            key = f"{scope}.key-{_name(path)}" if scope else f"key-{_name(path)}"
            return self.point.query(self._members()[key])
        if path in index.choices:
            return self.point.query(self._decisions()[path])
        return self.point.query(self._members()[path])

    # -- the questions

    def state(self, path: str) -> StateForm:
        """A choice's state; ``forced`` and ``refused`` are today's engine's only."""
        engine = self.engine
        handle = self._decisions()[path]
        state = self.point.field(handle).state
        stored = self.stored()
        if path in stored:
            if isinstance(state, Inapplicable):
                return StateForm("dormant", stored[path])
            unchecked = self.unchecked()
            if path in unchecked:
                return StateForm("waiting", stored[path], blockers=unchecked[path])
            if isinstance(state, Rejected):
                return StateForm("refused")
            answer = self._query(path)
            if isinstance(answer, Available):
                return StateForm("set", stored[path])
            if isinstance(answer, Rejected):
                return StateForm("refused")
            if isinstance(state, Unresolved) or isinstance(answer, Unresolved):
                # an engine fault, not a disagreement: stored, read unresolved, not listed
                raise RuntimeError(f"{path}: stored and unresolved, but not listed unchecked")
            return StateForm("dormant", stored[path])
        if isinstance(state, Inapplicable):
            return StateForm("absent")
        if isinstance(state, Rejected):
            return StateForm("refused")
        if isinstance(state, Unresolved):
            return StateForm("open", blockers=engine.blockers(state.findings, path))
        forced = {item.key: item.value for item in inspection.forced(self.point)}
        if path in forced:
            return StateForm("forced", forced[path])
        answer = self._query(path)
        if isinstance(answer, Rejected):
            return StateForm("refused")
        blockers = (
            engine.blockers(answer.findings, path)
            if isinstance(answer, Unresolved)
            else frozenset()
        )
        return StateForm("open", blockers=blockers)

    def answer(self, path: str) -> AnswerForm:
        """A fact's, a choice's, a derived value's or a view's answer."""
        result = self._query(path)
        engine = self.engine
        if isinstance(result, Available):
            stored = self.stored()
            value = stored[path] if path in stored else result.value
            if path in engine.index.choices and engine.index.choices[path].cases is not None:
                value = result.value
            return AnswerForm("value", value)
        if isinstance(result, Unresolved):
            return AnswerForm("unresolved", blockers=engine.blockers(result.findings, path))
        if isinstance(result, Rejected):
            return AnswerForm("refused", codes=engine.codes(result.findings))
        return AnswerForm("inapplicable")

    def rule(self, path: str) -> RuleForm:
        """A rule's answer."""
        result = self.point.query(self._members()[path])
        return _rule(self.engine, result)

    def validity(self) -> ValidityForm:
        """The engine's validity of the configuration (``inspection.validity``): refused
        with its false pieces' codes, open with what its unknown pieces wait on (each
        piece's own key aside; today's ``"undecided"``), or valid."""
        engine = self.engine
        answer = inspection.validity(self.point)
        if answer.status == "refused":
            false = frozenset().union(
                *(engine.codes(piece.result.findings) for piece in answer.pieces)
            )
            return ValidityForm("refused", false)
        if answer.status == "undecided":
            blockers = frozenset().union(
                *(
                    engine.blockers(piece.result.findings, engine.path(piece.owner))
                    for piece in answer.pieces
                )
            )
            return ValidityForm("open", blockers=blockers)
        return ValidityForm("valid")

    def possible(self, path: str) -> PossibleForm:
        """A choice's possible values, from today's ``inspection.viable``: listed only for an
        open choice neither stored nor forced, whose cases are enumerable and whose guard
        does not wait. Its viable cases are possible, moving nothing (today's engine reports
        no movement); its other member values (a value outside membership aside) are
        impossible with no refusal form, since their reasons are text. In the declared list's
        order where the description fixes one (a selector's cases, a fixed list), else the
        viable cases before the others."""
        found = {item.key: item for item in inspection.viable(self.point)}
        if path not in found:
            return PossibleForm(None)
        item = found[path]
        info = self.engine.index.choices[path]
        outcomes: list[tuple[object, OutcomeForm]] = [
            (case, OutcomeForm("possible")) for case in item.cases
        ]
        for key, why in item.refused.items():
            if _OUTSIDE in why:
                continue  # not a member: the list holds members only
            value = key if info.cases is not None else ast.literal_eval(key)
            outcomes.append((value, OutcomeForm("impossible")))
        declared = info.cases if info.cases is not None else info.domain.values
        if declared is not None:
            order = list(declared)

            def position(entry: tuple[object, OutcomeForm]) -> int:
                at = [i for i, item in enumerate(order) if same(item, entry[0])]
                return at[0] if at else len(order)

            outcomes.sort(key=position)
        return PossibleForm(tuple(outcomes))

    def change(self, edit: model.Change) -> tuple[ChangeForm, Point | None]:
        """``edit`` applied: its comparable result, and the configuration it publishes.
        Today's report moves nothing dormant, in play or opened."""
        if edit.unset:
            raise NotTranslatable("the engine cannot set a fact back to unknown")
        engine = self.engine
        decisions = self._decisions()
        commits = {decisions[path]: value for path, value in edit.commit.items()}
        clears: list[ChangeRequest] = [
            self.point.field(decisions[path]).clear() for path in edit.clear
        ]
        if edit.facts:
            members = self._members()
            facts = {members[name]: value for name, value in edit.facts.items()}
            report = self.point.try_with_facts(facts, commits, *clears)
        else:
            report = self.point.try_with_choices(commits, *clears)
        if not report.accepted:
            refusals = frozenset(
                _refusal(engine, outcome.owner, outcome.result)
                for outcome in report.outcomes
                if outcome.status == "refused" and outcome.owner in decisions
            )
            return ChangeForm(False, refusals=refusals), None
        return ChangeForm(True), Point(engine, report.instance)


#: an engine finding's text (``owner: code: message``) for a value outside membership,
#: which ``inspection.viable`` gives as its reason
_OUTSIDE = ": domain-membership: "


def _rule(engine: Engine, result: QueryResult[Any]) -> RuleForm:
    if isinstance(result, Available):
        return RuleForm("true" if result.value else "false")
    if isinstance(result, Rejected):
        return RuleForm("false", codes=engine.codes(result.findings))
    if isinstance(result, Unresolved):
        return RuleForm("unknown", blockers=engine.blockers(result.findings))
    return RuleForm("inapplicable")


def _refusal(engine: Engine, owner: str, result: QueryResult[bool]) -> RefusalForm:
    """A choice refusing a change, named or reached, with its own codes (a refused guard's
    findings, reported beside them, are compared through validity; ``inactive`` for a
    choice whose scope is not present): an ``assignment`` refusal where its one own code is
    ``domain-membership`` (the type check), else the legacy ``choice``."""
    codes: frozenset[Code] = frozenset()
    if isinstance(result, Inapplicable):
        codes = frozenset({("inactive", owner)})
    elif isinstance(result, (Rejected, Unresolved)):
        codes = engine.codes(result.findings, only=owner)
    if codes == {("domain-membership", owner)}:
        return RefusalForm("assignment", owner, codes)
    return RefusalForm("choice", owner, codes)


__all__ = [
    "Admission",
    "ByName",
    "Engine",
    "NotTranslatable",
    "Obligation",
    "Plain",
    "Point",
    "Requirement",
    "Rule",
    "Spelling",
    "build",
    "definition",
]
