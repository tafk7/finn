# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The reference model's answers in the comparable form, and the comparison.

A case (``Case``) is a description, built in both the model and the engine; a ``Pair`` is
one configuration of it on both sides, reached by the same changes. Each question a test
asks compares one answer:

- **kinds, values, codes, verdicts and refusals exactly**: a state, a verdict, each
  refusal's kind, subject, codes and standing choices, what a change moves, each possible
  value's outcome in its list's order;
- **a change's lists as sets**: its refusals, the values new facts dropped and the rules
  they broke. Drops are compared as sets: the model sorts them by path, the engine reports
  them in its flow order;
- **blockers as a subset**: the engine runs opaque method bodies that stop at their first
  unavailable read, so it may name fewer of what an answer waits on than the model, never
  one the model does not name.

The engine's side is an implementation of the engine's public questions
(``support.space_oracle``), the harness's default by default (``implementation.default``:
the public API unless a session names another): ``Case.start(..., implementation=)`` names
another, such as a planted fault (``support.space_oracle.public.Planted``), which a hand
case must then report by question.

A disagreement raises :class:`Disagreement`, an ``AssertionError`` naming the case, the
question and both answers. A test may state the answer a ruling on the semantics gives in
place of the model's (``ruled=``) where the model has not yet been brought to it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields

from support.space_oracle import Implementation

from core.space import reference
from core.space.reference.model import same

from .engine import Engine, Point, Spelling, build, definition
from .forms import (
    AnswerForm,
    ChangeForm,
    Code,
    DefinitionForm,
    DroppedForm,
    Form,
    OutcomeForm,
    PossibleForm,
    RefusalForm,
    RuleForm,
    StateForm,
    ValidityForm,
)

#: the model's reason kinds as the engine's finding codes. A rule's, a view body's and a
#: derived value's reasons carry their own code (the engine's rejection by that code); a
#: refused input and a refused Space are expanded to the reasons that refuse them (the
#: engine carries those findings, not one of its own).
CODES: Mapping[reference.ReasonKind, str] = {
    reference.ReasonKind.OUTSIDE_DOMAIN: "domain-membership",
    reference.ReasonKind.EMPTY_DOMAIN: "domain-empty",
    # § 16.17: a present choice's membership input that does not apply
    reference.ReasonKind.INPUT_ABSENT: "domain-input-absent",
}

#: the code of a guard's validity piece, a guard refused through an input (owned by the
#: guarded scope), beside the codes of that input
GUARD_CODE = "guard-refused"

#: the reason kinds that carry their own code, the code of the engine's rejection
_OWN_CODE = (
    reference.ReasonKind.CONSTRAINT,
    reference.ReasonKind.VIEW_BODY,
    reference.ReasonKind.DERIVED_BODY,
)


class Disagreement(AssertionError):
    """The model and the engine answer one question differently."""

    def __init__(self, case: str, question: str, expected: Form, engine: Form) -> None:
        self.case, self.question, self.expected, self.engine = case, question, expected, engine
        super().__init__(self.describe())

    def describe(self) -> str:
        lines = [f"{self.case}: {self.question}"]
        for item in fields(self.expected):
            left = getattr(self.expected, item.name)
            right = getattr(self.engine, item.name)
            mark = "  " if _same(left, right) else "! "
            lines.append(f"  {mark}{item.name}: model {_show(left)} | engine {_show(right)}")
        return "\n".join(lines)


class SetupError(Exception):
    """A step that brings a configuration to its question failed on one side: the test is
    written wrongly, never an expected disagreement."""


def _show(value: object) -> str:
    if isinstance(value, frozenset):
        return "{" + ", ".join(sorted(map(_show, value))) + "}"
    if isinstance(value, RefusalForm):
        standing = f", standing {_show(value.standing)}" if value.standing else ""
        return f"{value.subject} ({value.kind}: {_show(value.codes)}{standing})"
    if isinstance(value, DroppedForm):
        return f"{value.subject} = {value.value!r} ({_show(value.codes)})"
    if isinstance(value, OutcomeForm):
        if value.kind == "impossible":
            return f"impossible {_show(value.refusals)}"
        moved = [
            f"{name} {_show(getattr(value, name))}"
            for name in ("dormant", "in_play", "opened")
            if getattr(value, name)
        ]
        return " ".join(("possible", *moved))
    if isinstance(value, tuple):
        return "(" + ", ".join(map(_show, value)) + ")"
    return repr(value)


def _same(left: object, right: object) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(map(_same, left, right))
    if isinstance(left, (frozenset, RefusalForm, OutcomeForm)) or left is None:
        return bool(left == right)
    return same(left, right)


# --- the model's answers in the comparable form ------------------------------------------------

#: what a refused input reads as, to expand its reasons
_Read = Callable[[str], reference.Answer]


def _codes(
    reasons: Iterable[reference.Reason],
    read: _Read,
    space: Callable[[str], frozenset[Code]] | None = None,
) -> frozenset[Code]:
    """``reasons`` as codes: own codes and mapped kinds as they are, a refused input by the
    reasons that refuse it (``read``), a refused Space by its false pieces (``space``)."""
    found: set[Code] = set()
    for reason in reasons:
        if reason.kind in _OWN_CODE:
            found.add((reason.code, reason.subject))
        elif reason.kind is reference.ReasonKind.INPUT_REFUSED:
            for path in reason.related:
                answer = read(path)
                if isinstance(answer, reference.Refused):
                    found |= _codes(answer.reasons, read, space)
        elif reason.kind is reference.ReasonKind.SPACE_REFUSED:
            assert space is not None, reason
            found |= space(reason.owner)
        else:
            found.add((CODES[reason.kind], reason.subject))
    return frozenset(found)


def _solution_codes(
    solution: reference.Solution, reasons: Iterable[reference.Reason]
) -> frozenset[Code]:
    return _codes(reasons, solution.read, lambda owner: validity_form(solution, owner).false)


def state_form(solution: reference.Solution, path: str) -> StateForm:
    state = solution.state(path)
    if isinstance(state, reference.Set):
        return StateForm("set", state.value)
    if isinstance(state, reference.Waiting):
        return StateForm("waiting", state.value, frozenset(state.waits))
    if isinstance(state, reference.Dormant):
        return StateForm("dormant", state.value)
    if isinstance(state, reference.Open):
        return StateForm("open", blockers=frozenset(state.waits))
    return StateForm("absent")


def answer_form(solution: reference.Solution, path: str) -> AnswerForm:
    answer = solution.read(path)
    if isinstance(answer, reference.Known):
        return AnswerForm("value", answer.value)
    if isinstance(answer, reference.Unresolved):
        return AnswerForm("unresolved", blockers=frozenset(answer.waits))
    if isinstance(answer, reference.Refused):
        return AnswerForm("refused", codes=_solution_codes(solution, answer.reasons))
    return AnswerForm("inapplicable")


def rule_form(solution: reference.Solution, path: str) -> RuleForm:
    piece = solution.piece(path)
    if not piece.applies:
        return RuleForm("inapplicable")
    if piece.verdict is reference.Verdict.TRUE:
        return RuleForm("true")
    if piece.verdict is reference.Verdict.FALSE:
        return RuleForm("false", codes=_solution_codes(solution, piece.reasons))
    return RuleForm("unknown", blockers=frozenset(piece.waits))


def _piece_codes(solution: reference.Solution, piece: reference.Piece) -> frozenset[Code]:
    codes = _solution_codes(solution, piece.reasons)
    if piece.kind is reference.PieceKind.GUARD:
        codes |= {(GUARD_CODE, piece.subject)}
    return codes


def validity_form(solution: reference.Solution, scope: str = "") -> ValidityForm:
    validity = solution.validity(scope)
    false = frozenset().union(*(_piece_codes(solution, piece) for piece in validity.false))
    if validity.verdict is reference.Verdict.TRUE:
        return ValidityForm("valid", false)
    if validity.verdict is reference.Verdict.FALSE:
        return ValidityForm("refused", false)
    return ValidityForm("open", false, frozenset(validity.waits))


def refusal_form(
    index: reference.Index, facts: Mapping[str, object], refusal: reference.ChangeRefusal
) -> RefusalForm:
    """One refusal of a change whose configuration has ``facts``. An assignment refusal's
    refused inputs are read over the facts alone, as membership reads them; a rule
    refusal's reasons are its origin's own."""
    read = reference.FactsOnly(index, facts).value
    if isinstance(refusal, reference.AssignmentRefusal):
        return RefusalForm("assignment", refusal.choice, _codes(refusal.reasons, read))
    codes = _codes(refusal.reasons, read)
    return RefusalForm("rule", refusal.origin, codes, frozenset(refusal.standing))


def change_form(
    configuration: reference.Configuration,
    edit: reference.Change,
    result: reference.Accepted | reference.Rejected,
) -> ChangeForm:
    """``edit``'s ``result`` on ``configuration``. A dropped value's type check and a rule's
    reasons are read over the facts the change publishes, as membership reads them."""
    facts = dict(configuration.facts)
    facts.update(dict.fromkeys(edit.unset))  # an optional fact unset is None (SZ100 (5))
    facts.update(edit.facts)
    index = configuration.index
    if isinstance(result, reference.Accepted):
        read = reference.FactsOnly(index, facts).value
        return ChangeForm(
            True,
            frozenset(result.dormant),
            frozenset(result.in_play),
            frozenset(result.opened),
            dropped=frozenset(
                DroppedForm(item.choice, item.value, _codes(item.reasons, read))
                for item in result.dropped
            ),
            broken=frozenset(refusal_form(index, facts, item) for item in result.broken),
        )
    refusals = frozenset(refusal_form(index, facts, item) for item in result.refusals)
    return ChangeForm(False, refusals=refusals)


def bind_form(
    space: reference.Space, facts: Mapping[str, object], stored: Mapping[str, object]
) -> tuple[ChangeForm, reference.Configuration]:
    """The model's ``bind`` of ``stored`` under ``facts``: its report as an accepted change
    of the configuration with ``facts`` and nothing stored, and what it publishes."""
    result = reference.bind(space, facts, stored)
    return change_form(reference.Configuration(space, facts), reference.Change(), result), (
        result.configuration
    )


def possible_form(solution: reference.Solution, path: str) -> PossibleForm:
    found = solution.possible(path)
    if found is None:
        return PossibleForm(None)
    configuration = solution.configuration
    outcomes: list[tuple[object, OutcomeForm]] = []
    for value, outcome in found:
        if isinstance(outcome, reference.Possible):
            moved = (frozenset(outcome.dormant), frozenset(outcome.in_play))
            outcomes.append((value, OutcomeForm("possible", *moved, frozenset(outcome.opened))))
        else:
            refusals = frozenset(
                refusal_form(configuration.index, configuration.facts, item)
                for item in outcome.refusals
            )
            outcomes.append((value, OutcomeForm("impossible", refusals=refusals)))
    return PossibleForm(tuple(outcomes))


# --- the comparison ------------------------------------------------------------------------------


def agrees(expected: Form, engine: Form) -> bool:
    """Whether the engine's answer is the model's: exactly, but for blockers (a subset)."""
    if type(expected) is not type(engine):
        return False
    for item in fields(expected):
        left, right = getattr(expected, item.name), getattr(engine, item.name)
        if item.name == "blockers":
            if not right <= left:
                return False
        elif not _same(left, right):
            return False
    return True


def _check(case: str, question: str, expected: Form, engine: Form) -> None:
    __tracebackhide__ = True  # a disagreement is shown at the test's question
    if not agrees(expected, engine):
        raise Disagreement(case, question, expected, engine)


class Case:
    """A description built in the model and in the engine (spelled as ``spelling`` says)."""

    def __init__(self, name: str, description: reference.Space, spelling: Spelling | None = None):
        self.name = name
        self.description = description
        self.engine: Engine = build(description, spelling)

    def start(
        self,
        facts: Mapping[str, object] | None = None,
        stored: Mapping[str, object] | None = None,
        implementation: Implementation | None = None,
    ) -> Pair:
        """The configuration with ``facts``, then ``stored`` committed as one change (a step:
        it must be accepted on both sides), the engine's side as ``implementation`` answers
        it (the harness's default by default, ``Engine.open``'s)."""
        facts = dict(facts or {})
        model = reference.Configuration(self.description, facts)
        pair = Pair(self, model, self.engine.open(facts, implementation))
        return pair.step(reference.Change(commit=stored)) if stored else pair


@dataclass(frozen=True)
class Pair:
    """One configuration of a case, on both sides."""

    case: Case
    model: reference.Configuration
    engine: Point

    def solution(self) -> reference.Solution:
        return self.model.solve()

    def step(self, edit: reference.Change) -> Pair:
        """A change that brings the configuration to a question: accepted on both sides, or
        the test is written wrongly (``SetupError``); nothing else is compared."""
        result = reference.change(self.model, edit)
        form, successor = self.engine.change(edit)
        if not isinstance(result, reference.Accepted) or successor is None:
            raise SetupError(
                f"{self.case.name}: a step must be accepted on both sides: {edit}: model "
                f"{type(result).__name__}, engine {'accepted' if form.accepted else form}"
            )
        return Pair(self.case, result.configuration, successor)

    # -- the questions

    def check_state(self, path: str, *, ruled: StateForm | None = None) -> None:
        __tracebackhide__ = True
        expected = ruled or state_form(self.solution(), path)
        _check(self.case.name, f"the state of {path}", expected, self.engine.state(path))

    def check_answer(self, path: str, *, ruled: AnswerForm | None = None) -> None:
        __tracebackhide__ = True
        expected = ruled or answer_form(self.solution(), path)
        _check(self.case.name, f"the answer of {path}", expected, self.engine.answer(path))

    def check_rule(self, path: str, *, ruled: RuleForm | None = None) -> None:
        __tracebackhide__ = True
        expected = ruled or rule_form(self.solution(), path)
        _check(self.case.name, f"the rule {path}", expected, self.engine.rule(path))

    def check_validity(self, *, ruled: ValidityForm | None = None) -> None:
        __tracebackhide__ = True
        expected = ruled or validity_form(self.solution())
        _check(self.case.name, "validity", expected, self.engine.validity())

    def check_possible(self, path: str, *, ruled: PossibleForm | None = None) -> None:
        """A choice's possible values, each with what setting it alone answers."""
        __tracebackhide__ = True
        expected = ruled or possible_form(self.solution(), path)
        _check(
            self.case.name, f"the possible values of {path}", expected, self.engine.possible(path)
        )

    def check_change(
        self, edit: reference.Change, *, ruled: ChangeForm | None = None
    ) -> Pair | None:
        """``edit`` compared; the successor where both accept it."""
        __tracebackhide__ = True
        result = reference.change(self.model, edit)
        expected = ruled or change_form(self.model, edit, result)
        form, point = self.engine.change(edit)
        _check(self.case.name, f"the change {_edit(edit)}", expected, form)
        if isinstance(result, reference.Accepted) and point is not None:
            return Pair(self.case, result.configuration, point)
        return None

    def check_bind(self, stored: Mapping[str, object]) -> Pair:
        """``stored`` bound under this design space's facts (the model's ``bind``, the
        engine's ``finn.core.space.bind``), compared; its successor. Asked of any design
        space with these facts: both sides read the bind against the facts with nothing
        stored, whatever the receiver stores."""
        __tracebackhide__ = True
        expected, configuration = bind_form(self.model.space, self.model.facts, stored)
        form, point = self.engine.bind(stored)
        _check(self.case.name, f"the bind of {dict(stored)}", expected, form)
        return Pair(self.case, configuration, point)

    def reach(self, edit: reference.Change) -> Pair:
        """``edit`` compared, then its successor: how a question is asked beyond a change
        the engine may refuse, so its refusal shows as the disagreement it is. A change the
        model refuses is a test written wrongly (``SetupError``)."""
        __tracebackhide__ = True
        successor = self.check_change(edit)
        if successor is None:
            raise SetupError(f"{self.case.name}: the model refuses {_edit(edit)}")
        return successor


def check_definition(
    name: str,
    description: reference.Space,
    spelling: Spelling | None = None,
    facts: Mapping[str, object] | None = None,
) -> None:
    """Whether ``description`` is a definition: the model's definition error (a cycle, a
    domain reading a choice) or none, against the engine's. The engine reports a cycle when
    building the Space, or when asked (its validity, every rule and every view of the
    configuration with ``facts``), as a dependency cycle; it answers otherwise."""
    __tracebackhide__ = True
    try:
        description.validate()
        expected = DefinitionForm(None)
    except reference.DefinitionError as error:
        expected = DefinitionForm(error.kind)
    _check(name, "the definition", expected, definition(description, spelling, facts))


def _edit(edit: reference.Change) -> str:
    parts = [f"commit {dict(edit.commit)}"] if edit.commit else []
    parts += [f"clear {list(edit.clear)}"] if edit.clear else []
    parts += [f"facts {dict(edit.facts)}"] if edit.facts else []
    parts += [f"unset {list(edit.unset)}"] if edit.unset else []
    return ", ".join(parts) or "(nothing)"


__all__ = [
    "CODES",
    "GUARD_CODE",
    "Case",
    "Disagreement",
    "Pair",
    "SetupError",
    "agrees",
    "answer_form",
    "bind_form",
    "change_form",
    "check_definition",
    "possible_form",
    "refusal_form",
    "rule_form",
    "state_form",
    "validity_form",
]
