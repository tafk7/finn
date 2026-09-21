# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical integer operand support for the admitted dot-product families."""

from finn.dataflow.model.logical.region import NumericElementType, element_width
from finn.dataflow.space.declarations import reject

_MULTIPLIABLE_FAMILIES = ("INT", "UINT")
_SIGNED_ROLES = frozenset({"weight", "accumulator", "output"})


def _is_twos_complement_integer(datatype: NumericElementType) -> bool:
    name = datatype.name
    return any(
        name.startswith(prefix) and name[len(prefix) :].isdigit()
        for prefix in _MULTIPLIABLE_FAMILIES
    )


def operand_types_supported(
    activation: NumericElementType,
    weight: NumericElementType,
    accumulator: NumericElementType,
    output: NumericElementType,
) -> object:
    rejected: dict[str, str] = {}
    for role, datatype in (
        ("activation", activation),
        ("weight", weight),
        ("accumulator", accumulator),
        ("output", output),
    ):
        if not _is_twos_complement_integer(datatype):
            rejected[role] = f"{datatype.name} is not a two's-complement integer"
        elif role in _SIGNED_ROLES and not datatype.signed():
            rejected[role] = f"{datatype.name} is unsigned; the core declares this role signed"
    if "accumulator" not in rejected and "output" not in rejected and output != accumulator:
        rejected["output"] = (
            f"{output.name} is not the accumulator {accumulator.name}; "
            "the core drives the accumulator straight out"
        )
    if rejected:
        return reject(
            "dotp-axi-numeric-types-unsupported",
            "this dot-product core multiplies two's-complement integers",
            values=rejected,
        )
    return True


def operand_widths_supported(activation: NumericElementType, weight: NumericElementType) -> object:
    if element_width(activation) < 2 or element_width(weight) < 2:
        return reject(
            "dotp-axi-operands-too-narrow",
            "the dot-product core needs at least two bits of each operand",
            values={
                "activation": element_width(activation),
                "weight": element_width(weight),
            },
        )
    return True


__all__ = ["operand_types_supported", "operand_widths_supported"]
