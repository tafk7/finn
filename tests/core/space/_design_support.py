# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Design spaces of the running type (``finn.core.space._running``) for the engine's own
tests: a root declaration's design space, every choice open (``start``), or with stored
choices by key, checked by nothing (``built``); a node by key; a change by key
(``questions.change``); an answer by key; a scope's validity; and a refusal's codes. Each
question is asked in a session of its own with no cache."""

from __future__ import annotations

from collections.abc import Mapping

from core.space._evaluation_support import open_evaluation
from finn.core.space import Inapplicable, Rejected, Space, Unresolved, design_space, inspection
from finn.core.space._running import questions
from finn.core.space._running.evaluate import Session, evaluation
from finn.core.space._running.value import DesignSpace
from finn.core.space.ir import NodeId
from finn.core.space.occurrence import held
from finn.core.space.results import QueryResult


def start(declaration: Space) -> DesignSpace:
    """The design space of a root declaration: its facts bound, every choice open."""
    return held(design_space(declaration)).ds


def built(declaration: Space, choices: Mapping[str, object]) -> DesignSpace:
    """The design space of a root declaration's facts with exactly ``choices`` (by key)
    stored, built directly: no rule or type check."""
    return open_evaluation(declaration, choices).ds


def key(ds: DesignSpace, name: str) -> NodeId:
    return ds.model.linked.keys[name]


def change(
    ds: DesignSpace,
    choices: Mapping[str, object] | None = None,
    facts: Mapping[str, object] | None = None,
    clear: tuple[str, ...] = (),
) -> questions.Report:
    """``ds`` with ``choices`` set and ``clear`` cleared (by key), and ``facts``, as one
    change."""
    requests: list[tuple[NodeId, object]] = [
        (key(ds, name), value) for name, value in (choices or {}).items()
    ]
    requests += [(key(ds, name), questions.REMOVE) for name in clear]
    given = {key(ds, name): value for name, value in (facts or {}).items()}
    with ds.model.lock:
        return questions.change(ds, given, requests, Session())


def published(report: questions.Report) -> DesignSpace:
    """The design space an accepted change publishes."""
    assert report.published is not None, report
    return report.published


def read(ds: DesignSpace, name: str) -> QueryResult[object]:
    """``name``'s answer in ``ds``."""
    return evaluation(ds, Session()).result(key(ds, name))


def validity(ds: DesignSpace, scope: int = 0) -> inspection.Validity:
    """The validity of ``scope``'s Space in ``ds``."""
    return questions.validity(evaluation(ds, Session()), scope)


def codes(result: object) -> list[tuple[str, str]]:
    assert isinstance(result, (Inapplicable, Rejected, Unresolved))
    return [(finding.code, finding.owner) for finding in result.findings]
