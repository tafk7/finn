# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The engine side of the harness: a model description built as an engine Space, and the
engine asked the model's questions, each answer put in the comparable form (``forms``).

Everything that depends on how the engine is **authored** or **asked** at this commit is in
this one module, so a milestone that changes either changes it here:

- **Spelling.** A rule of the description is a constraint in the model. The engine at this
  commit spells it one of six ways (``Spelling.rules``, beside each case):
  ``Admission`` (the default: a ``@constraint`` in its Space's ``admission =
  ConstraintGroup(...)``, the only rules the engine counts for a selector's case),
  ``Plain`` (a ``@constraint`` in no group, which the engine does not count),
  ``Requirement`` (``requires(...)`` on a choice's domain, refused at commit),
  ``Membership`` (a choice's domain reading the rule's other inputs: a linking rule
  written as membership), ``Obligation`` (a view's ``requires=``) and ``ByName`` (a
  ``@constraint`` method reading its inputs by attribute name, ``getattr(self, name)``,
  in its Space's admission group). A domain with a membership predicate is a
  ``domain(accepts=, candidates=)``: membership is the model's whole membership, the
  enumeration the listed values, unchecked by membership as the engine enumerates today.
- **Questions.** The engine has no query for a configuration's validity at this commit, nor
  a list of the choices a change keeps unchecked. ``Point.validity`` assembles validity
  from what the engine counts (each present Space's admission group and each choice's
  refusal), and ``Point.change`` lists the unchecked as the stored choices that read as
  unresolved.

The description becomes one Space class per scope, built with ``composite``. Every member
keeps the description's path as its key: a choice ``compute.packed.pe`` is the engine's
Decision ``compute.packed.pe``. A scope that reads something outside it takes it as a formal
(``in-<path>``), bound by its parent; a selector's value is read through a derived member
``key-<name>`` (``selected``); a child's guard is a derived member ``when-<name>``. Facts
are the root's optional formals: a missing fact is an unsupplied one.
"""

from __future__ import annotations

import inspect as pyinspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from core.space import reference as model
from core.space.reference.model import same
from finn.core.space import (
    Available,
    ConstraintGroup,
    Decision,
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
    requires,
    requiring,
    selected,
    view,
)
from finn.core.space.edits import ChangeRequest

from .forms import (
    AnswerForm,
    ChangeForm,
    Code,
    Refusal,
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
    """A rule in its Space's ``admission = ConstraintGroup(...)`` (the default)."""


@dataclass(frozen=True)
class Plain:
    """A ``@constraint`` of its Space, in no group."""


@dataclass(frozen=True)
class Requirement:
    """``requires(fact, ...)`` on ``choice``'s domain for ``cases``: the model's rule must
    read ``choice`` and ``fact`` and hold exactly when the value is not one of ``cases`` or
    ``fact`` is truthy."""

    choice: str
    fact: str
    cases: tuple[object, ...]


@dataclass(frozen=True)
class Membership:
    """The rule as part of ``choice``'s domain: its membership reads the rule's other
    inputs, and a value the rule makes false is outside the domain. ``choice`` is one of
    the rule's reads, in the rule's own scope."""

    choice: str


@dataclass(frozen=True)
class ByName:
    """A ``@constraint`` method of its Space, in its admission group, reading each input by
    name (``getattr(self, name)`` over a list of names, through child nodes for a child's
    member): the engine discovers the reads when the body runs."""


@dataclass(frozen=True)
class Obligation:
    """A view's obligation: ``@view(requires=(rule,))`` on ``view``, of the rule's own
    Space."""

    view: str


Rule = Admission | Plain | Requirement | Membership | Obligation | ByName


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


class _Acyclic(model.Index):
    """The description's index without the model's cycle check: a description the model
    refuses as a cycle is still built in the engine, to see what the engine answers."""

    def _check_cycles(self) -> None:
        return None


#: spellings that write a rule into a choice's domain, not as a member of its Space
_IN_DOMAIN = (Requirement, Membership)


class _Builder:
    """The engine Space classes of one description, one per scope, built on demand."""

    def __init__(self, space: model.Space, spelling: Spelling, cycles: bool = False) -> None:
        self.index = _Acyclic(space) if cycles else space.validate()
        self.spelling = spelling
        index = self.index
        self.requirements: dict[str, list[tuple[str, Requirement]]] = {}
        self.memberships: dict[str, list[str]] = {}
        for path, rule in spelling.rules.items():
            if path not in index.constraints:
                raise NotTranslatable(f"{path}: spelled, but not a rule of the description")
            if isinstance(rule, Membership):
                owner = index.constraints[path][1]
                if (
                    rule.choice not in index.constraints[path][0].reads
                    or rule.choice not in index.choices
                    or index.choices[rule.choice].scope != owner
                    or index.choices[rule.choice].cases is not None
                ):
                    raise NotTranslatable(f"{path}: a membership of a value choice it reads")
                self.memberships.setdefault(rule.choice, []).append(path)
            if isinstance(rule, Requirement):
                if rule.choice not in index.choices or index.kind(rule.fact) is None:
                    raise NotTranslatable(f"{path}: its requirement names unknown nodes")
                self.requirements.setdefault(rule.choice, []).append((path, rule))
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
        #: (engine owner, code) -> the model rule a requirement's finding is about
        self.subjects: dict[tuple[str, str], str] = {}

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
                found.update(rule.fact for _, rule in self.requirements.get(path, ()))
        for table in (index.derived, index.views, index.constraints):
            for path, (node, owner) in table.items():
                rule = self.spelling.rules.get(path)
                if index.within(owner, scope) and not isinstance(rule, Requirement):
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
            node = index.derived[name][0]
            body = node.compute
            return derived(**self._aliases(scope, node.reads))(
                _explicit(node.reads, body, self.value_type(name))
            )
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
        requirements = []
        for rule_path, rule in self.requirements.get(path, ()):
            code = self.index.constraints[rule_path][0].code
            finding = f"{code}: the rule refuses"
            requirements.append(
                requires(self.reference(scope, rule.fact), finding, cases=rule.cases)
            )
            self.subjects[(path, code)] = rule_path
        rules = [self.index.constraints[rule][0] for rule in self.memberships.get(path, ())]
        simple = (
            domain_.values is not None
            and domain_.member is None
            and not domain_.reads
            and not rules
        )
        if simple and domain_.values:
            return Decision(
                values=domain_.values, requires=tuple(requirements), required=info.required
            )
        reads = tuple(
            dict.fromkeys(
                (*domain_.reads, *(read for rule in rules for read in rule.reads if read != path))
            )
        )
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
            read = readings(arguments)
            if not all(rule.predicate({**read, path: candidate}) for rule in rules):
                return False
            return domain_.member is None or bool(domain_.member(candidate, read))

        def candidates(**arguments: object) -> tuple[object, ...]:
            values = listing(arguments)
            assert values is not None
            return values

        built = domain(
            accepts=accepts,
            candidates=candidates if domain_.listed else None,
            **dict(zip(names, (self.reference(scope, read) for read in reads), strict=True)),
        )
        if requirements:
            built = requiring(built, *requirements)
        return Decision(domain=built, required=info.required)

    def _view(self, scope: str, path: str) -> object:
        node = self.index.views[path][0]
        obligations = tuple(
            self.member(scope, rule_path)
            for rule_path, rule in sorted(self.spelling.rules.items())
            if isinstance(rule, Obligation) and rule.view == path
        )

        def body(read: dict[str, Any]) -> object:
            output = node.body(read)
            if isinstance(output, model.Refuse):
                return reject(output.code, "the view refuses")
            return output

        returns = self.value_type(path) | Rejected
        return view(requires=obligations, **self._aliases(scope, node.reads))(
            _explicit(node.reads, body, returns)
        )

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
            rule = self.spelling.rules.get(path)
            if isinstance(rule, _IN_DOMAIN):
                continue
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
        admitted = [
            self.member(scope, path)
            for path in mine
            if path in index.constraints
            and isinstance(self.spelling.rule(path), (Admission, ByName))
        ]
        if admitted:
            members["admission"] = ConstraintGroup(*admitted)
        # bind every placed node's formals (assigned after the node, as a class body may)
        for below, placed in self.nodes[scope].items():
            for path in self.externals[below]:
                setattr(placed, _input(path), self.reference(scope, path))
                self.aliases[f"{below}.{_input(path)}"] = path
        owned = {_name(path) for path in mine} | {
            _name(path) for path, child in index.scopes.items() if child.parent == scope
        }
        clashes = sorted(name for name in owned if name == "admission" or "-" in name)
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
    subjects: Mapping[tuple[str, str], str]

    def open(self, facts: Mapping[str, object]) -> Point:
        """The configuration with ``facts`` and nothing committed."""
        return Point(self, design_space(self.root(**cast(dict[str, Any], facts))))

    def path(self, owner: str) -> str:
        """The model path an engine finding's owner names."""
        return self.aliases.get(owner, owner)

    def codes(self, findings: Iterable[Any], only: str | None = None) -> frozenset[Code]:
        """Findings as codes; ``only``: the findings owned by that node alone."""
        return frozenset(
            (
                finding.code,
                self.subjects.get((finding.owner, finding.code), self.path(finding.owner)),
            )
            for finding in findings
            if only is None or finding.owner == only
        )

    def blockers(self, findings: Iterable[Any]) -> frozenset[str]:
        return frozenset(self.path(finding.owner) for finding in findings)


def build(
    description: model.Space, spelling: Spelling | None = None, cycles: bool = False
) -> Engine:
    """The engine Space class of ``description``, spelled as ``spelling`` says; with
    ``cycles``, built even where the model refuses the description as a cycle."""
    builder = _Builder(description, spelling or Spelling(), cycles)
    root = builder.space_class("")
    return Engine(
        description,
        builder.spelling,
        root,
        builder.index,
        dict(builder.aliases),
        dict(builder.subjects),
    )


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
        engine = self.engine
        handle = self._decisions()[path]
        state = self.point.field(handle).state
        stored = self.stored()
        if path in stored:
            if isinstance(state, Inapplicable):
                return StateForm("inactive", stored[path])
            if isinstance(state, Unresolved):
                return StateForm(
                    "unchecked", stored[path], blockers=engine.blockers(state.findings)
                )
            if isinstance(state, Rejected):
                return StateForm("refused", codes=engine.codes(state.findings))
            answer = self._query(path)
            if isinstance(answer, Available):
                return StateForm("readable", stored[path])
            if isinstance(answer, Unresolved):
                return StateForm(
                    "unchecked", stored[path], blockers=engine.blockers(answer.findings)
                )
            if isinstance(answer, Rejected):
                return StateForm("refused", codes=engine.codes(answer.findings))
            return StateForm("inactive", stored[path])
        if isinstance(state, Inapplicable):
            return StateForm("inactive")
        if isinstance(state, Rejected):
            return StateForm("refused", codes=engine.codes(state.findings))
        if isinstance(state, Unresolved):
            return StateForm("open", blockers=engine.blockers(state.findings))
        forced = {item.key: item.value for item in inspection.forced(self.point)}
        if path in forced:
            return StateForm("forced", forced[path])
        viable = {item.key: item.cases for item in inspection.viable(self.point)}
        answer = self._query(path)
        if isinstance(answer, Rejected):
            return StateForm("refused", codes=engine.codes(answer.findings))
        if path in viable:
            return StateForm("open", possible=tuple(viable[path]))
        blockers = (
            engine.blockers(answer.findings) if isinstance(answer, Unresolved) else frozenset()
        )
        return StateForm("open", blockers=blockers - {path})

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
            return AnswerForm("unresolved", blockers=engine.blockers(result.findings))
        if isinstance(result, Rejected):
            return AnswerForm("refused", codes=engine.codes(result.findings))
        return AnswerForm("inapplicable")

    def rule(self, path: str) -> RuleForm:
        """A rule's answer, read from wherever its spelling puts it."""
        spelled = self.engine.spelling.rule(path)
        if isinstance(spelled, _IN_DOMAIN):
            raise NotTranslatable(f"{path}: a rule in a domain has no answer of its own")
        result = self.point.query(self._members()[path])
        return _rule(self.engine, result)

    def validity(self) -> ValidityForm:
        """Assembled from what the engine counts at this commit: every present Space's
        admission group, and every choice that reads refused (a stored choice that reads
        unresolved, kept waiting, is an unknown piece)."""
        engine = self.engine
        false: set[Code] = set()
        blockers: set[str] = set()
        unknown = False
        members = self._members()
        for key in sorted(members):
            if key == "admission" or key.endswith(".admission"):
                answer = _rule(engine, self.point.query(members[key]))
                false |= answer.codes
                if answer.kind == "unknown":
                    unknown = True
                    blockers |= answer.blockers
        stored = self.stored()
        for path in sorted(engine.index.choices):
            state = self.state(path)
            if state.kind == "refused":
                false |= state.codes
            elif state.kind == "unchecked" and path in stored:
                unknown = True
                blockers |= state.blockers
        if false:
            return ValidityForm("refused", frozenset(false), frozenset(blockers))
        if unknown:
            return ValidityForm("undecided", blockers=frozenset(blockers))
        return ValidityForm("valid")

    def change(self, edit: model.Change) -> tuple[ChangeForm, Point | None]:
        """``edit`` applied: its comparable result, and the configuration it publishes."""
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
                Refusal(
                    outcome.owner,
                    "named" if outcome.requested else "reached",
                    _own_codes(engine, outcome.owner, outcome.result),
                )
                for outcome in report.outcomes
                if outcome.status == "refused" and outcome.owner in decisions
            )
            return ChangeForm(False, refusals=refusals), None
        successor = Point(engine, report.instance)
        stored = successor.stored()
        unchecked = frozenset(path for path in stored if successor.state(path).kind == "unchecked")
        return ChangeForm(True, unchecked=unchecked), successor


def _rule(engine: Engine, result: QueryResult[Any]) -> RuleForm:
    if isinstance(result, Available):
        return RuleForm("true" if result.value else "false")
    if isinstance(result, Rejected):
        return RuleForm("false", codes=engine.codes(result.findings))
    if isinstance(result, Unresolved):
        return RuleForm("unknown", blockers=engine.blockers(result.findings))
    return RuleForm("inapplicable")


def _own_codes(engine: Engine, owner: str, result: QueryResult[bool]) -> frozenset[Code]:
    """A refused choice's own codes: the findings it owns (a refused guard's, reported
    beside them, are compared through validity), ``inactive`` for a choice whose scope is
    not present."""
    if isinstance(result, Inapplicable):
        return frozenset({("inactive", owner)})
    if isinstance(result, (Rejected, Unresolved)):
        return engine.codes(result.findings, only=owner)
    return frozenset()


__all__ = [
    "Admission",
    "ByName",
    "Engine",
    "NotTranslatable",
    "Obligation",
    "Membership",
    "Plain",
    "Point",
    "Requirement",
    "Rule",
    "Spelling",
    "build",
]
