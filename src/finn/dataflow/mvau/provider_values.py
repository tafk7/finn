# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Read a provider's declared parameter table out of a resolved design point.

``ProviderParameter`` says, for every value an RTL core takes, *where that
value is entitled to come from*.  This turns that declaration into the values
themselves: projected problem data is read from the problem, a decision from
the assignments, a derived property through the engine, and a provider constant
from the declaration.

Nothing here computes a parameter.  If a value is not reachable through one of
the four declared routes, the answer is a finding, not a fallback -- that is the
whole point of having declared it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from finn.dataflow.design import (
    Decided,
    DesignPoint,
    Engine,
    Finding,
    FindingKind,
    QualifiedPath,
)
from finn.dataflow.mvau.decomposed import ParameterOwnership, ProviderParameter
from finn.dataflow.region import NumericElementType

#: Where a failure to read the table is reported.
PROVIDER_VALUES_PATH = QualifiedPath("provider.mvau.parameters")


class ProviderValueError(ValueError):
    """A declared parameter could not be read from the point."""

    def __init__(self, findings: tuple[Finding, ...]) -> None:
        super().__init__("; ".join(item.message for item in findings))
        self.findings = findings


def _fail(code: str, message: str, path: QualifiedPath | None = None) -> ProviderValueError:
    return ProviderValueError(
        (
            Finding(
                FindingKind.BLOCKER,
                code,
                PROVIDER_VALUES_PATH,
                message,
                trace=() if path is None else (path,),
            ),
        )
    )


def _problem_value(point: DesignPoint, path: QualifiedPath) -> object:
    if path not in point.problem:
        raise _fail(
            "mvau-provider-problem-field-absent",
            "a provider parameter names a problem field this point does not carry",
            path,
        )
    raw = point.problem[path]
    # An element type reaches the RTL as its width.  Doing the projection here
    # rather than in each declaration keeps the table naming the *fact*, which
    # is what makes the audit readable.
    return raw.bit_width if isinstance(raw, NumericElementType) else raw


def resolve_provider_parameters(
    engine: Engine,
    point: DesignPoint,
    parameters: Sequence[ProviderParameter],
) -> Mapping[str, object]:
    """Every declared parameter's value, read the way its owner says to."""

    values: dict[str, object] = {}
    for parameter in parameters:
        if parameter.ownership is ParameterOwnership.CONSTANT:
            values[parameter.name] = parameter.value
            continue
        source = parameter.source
        if source is None:  # pragma: no cover - ProviderParameter forbids it
            raise _fail(
                "mvau-provider-parameter-sourceless",
                f"{parameter.name} is {parameter.ownership.value} but names no source",
            )
        if parameter.ownership is ParameterOwnership.PROBLEM:
            values[parameter.name] = _problem_value(point, source)
        elif parameter.ownership is ParameterOwnership.DECISION:
            if source not in point.assignments:
                raise _fail(
                    "mvau-provider-decision-unassigned",
                    f"{parameter.name} needs a committed decision that is not assigned",
                    source,
                )
            values[parameter.name] = point.assignments[source]
        else:
            answer = engine.query_property(point, source)
            if not isinstance(answer, Decided):
                raise ProviderValueError(
                    answer.findings
                    or (
                        Finding(
                            FindingKind.BLOCKER,
                            "mvau-provider-property-unresolved",
                            PROVIDER_VALUES_PATH,
                            f"{parameter.name} names a property that did not resolve",
                            trace=(source,),
                        ),
                    )
                )
            values[parameter.name] = answer.value
    return values


def scalar_parameters(
    values: Mapping[str, object],
) -> tuple[tuple[str, bool | int | float | str], ...]:
    """The same table as sorted scalar pairs, for a physical component.

    A non-scalar would silently stringify into generated Verilog, so it is
    refused here rather than discovered in a synthesis log.
    """

    bad = tuple(
        name for name, value in values.items() if type(value) not in (bool, int, float, str)
    )
    if bad:
        raise _fail(
            "mvau-provider-parameter-not-scalar",
            f"provider parameters must be scalar: {', '.join(sorted(bad))}",
        )
    return tuple((name, cast("bool | int | float | str", values[name])) for name in sorted(values))


__all__ = [
    "PROVIDER_VALUES_PATH",
    "ProviderValueError",
    "resolve_provider_parameters",
    "scalar_parameters",
]
