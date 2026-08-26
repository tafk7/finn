# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
from dataflow.engine.helpers import INT, any_int_domain, path, specification, started

from finn.dataflow._engine import Decision, Engine, RequestError, ValidationError
from finn.dataflow._engine.safe import SafeFailure, SafeSuccess, capture


def test_safe_capture_is_an_explicit_opt_in_boundary() -> None:
    engine = Engine()
    point = started(engine, specification(decisions=(Decision(path("d"), INT, any_int_domain()),)))
    result = capture(lambda: engine.commit_assignments(point, {"unknown": 1}))
    assert isinstance(result, SafeFailure)
    assert result.succeeded is False
    assert isinstance(result.error, RequestError)
    with pytest.raises(RequestError):
        result.unwrap()

    success = capture(lambda: None)
    assert isinstance(success, SafeSuccess)
    assert success.succeeded is True
    assert success.unwrap() is None


def test_common_try_operations_are_explicit_nonthrowing_conveniences() -> None:
    engine = Engine()
    duplicate = specification(
        decisions=(
            Decision(path("d"), INT, any_int_domain()),
            Decision(path("d"), INT, any_int_domain()),
        )
    )
    assert isinstance(engine.try_validate(duplicate), ValidationError)

    space = engine.validate(specification())
    assert isinstance(engine.try_start(space, {"unknown": 1}), RequestError)

    point = engine.start(space, {})
    assert isinstance(engine.try_commit_assignments(point, {"unknown": 1}), RequestError)
