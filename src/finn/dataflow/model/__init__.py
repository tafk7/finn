# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative frontend for constructing ordinary dataflow design-space specs."""

from importlib import import_module
from typing import TYPE_CHECKING

from finn.dataflow.model.compiler import compile_space
from finn.dataflow.model.declarations import (
    AuthoringError,
    Constraint,
    ConstraintGroup,
    Decision,
    Derived,
    Domain,
    Input,
    Problem,
    Readiness,
    Space,
    Use,
    constraint,
    derived,
    divisors_of,
    domain,
    finite,
    reject,
    unresolved,
)

if TYPE_CHECKING:
    from finn.dataflow.model.kernel import Kernel, Parameter, configure_kernel

_LAZY_EXPORTS = {
    name: ("finn.dataflow.model.kernel", name)
    for name in ("Kernel", "Parameter", "configure_kernel")
}
_LAZY_EXPORTS.update(
    {
        name: ("finn.dataflow.model.kernel_artifacts", name)
        for name in (
            "kernel_source_derivation",
            "portable_kernel_component",
            "resolve_kernel_contributions",
        )
    }
)
_LAZY_EXPORTS.update(
    {name: ("finn.dataflow.model.dotp_axi", name) for name in ("DspBlock", "DotpAxiKernel")}
)


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "AuthoringError",
    "Constraint",
    "ConstraintGroup",
    "Decision",
    "DspBlock",
    "Derived",
    "Domain",
    "DotpAxiKernel",
    "Input",
    "Kernel",
    "Parameter",
    "Problem",
    "Readiness",
    "Space",
    "Use",
    "constraint",
    "compile_space",
    "configure_kernel",
    "derived",
    "divisors_of",
    "domain",
    "finite",
    "kernel_source_derivation",
    "portable_kernel_component",
    "reject",
    "resolve_kernel_contributions",
    "unresolved",
]
