# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in graph inspection helpers; not part of the package-root API."""

from __future__ import annotations

from .declarations import DesignSpaceSpec
from .facts import GraphAnalysis, compile_facts


def analyze_graph(specification: DesignSpaceSpec) -> GraphAnalysis:
    """Compile graph detail after the same specification has passed validation.

    This optional diagnostic seam deliberately trusts typed, semantically valid
    declarations; it does not duplicate the validator.
    """

    analysis, _plan = compile_facts(specification)
    return analysis


__all__ = ["GraphAnalysis", "analyze_graph"]
