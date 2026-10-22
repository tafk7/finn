# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The per-class record (``_running.per_class.PerClass``) on stage 7's chain at 4 kernels
(``support.one_space.chain_space(2)``) answers as today's evaluator: every node's inferred
reads, the flow order, and every choice's and fact's presence readers."""

from __future__ import annotations

from typing import cast

from core.space._per_class_support import same_as_todays
from support.one_space import chain_space

from finn.core.space import Space
from finn.core.space.compiler import Model, compile_model


def test_the_per_class_record_answers_as_todays_evaluator_on_the_chain() -> None:
    whole, _ = chain_space(2)
    model = cast(Model[Space], compile_model(whole))
    assert len(model.per_class.flow()) > 4
    assert same_as_todays(model) == []
