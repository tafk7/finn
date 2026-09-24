# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Expected failures: each BAD line must receive exactly one strict mypy error."""

from finn.kernels.space._self_prototype import (
    Decision,
    Domain,
    Param,
    Space,
    Subspace,
    derived,
    view,
)


class Child(Space):
    fact = Param(int)
    choice = Decision(int, domain=Domain(lambda self, candidate: candidate > 0))

    @derived
    def output(self) -> int:
        return self.fact + self.choice

    @view()
    def product(self) -> tuple[int, int]:
        return self.fact, self.output


class Family(Space):
    child = Subspace(Child, fact=1)


point = Family()
reference_as_value: int = Child.fact  # BAD
scalar_as_string: str = point.child.fact  # BAD
child_as_string: str = point.child  # BAD
derived_as_string: str = point.child.output  # BAD
product_as_int: int = point.child.product()  # BAD
point.with_choices(point.child.field(Child.choice).change("bad"))  # BAD
point.try_with_choices(point.child.field(Child.choice).change("bad"))  # BAD
point.commit(point.child.field(Child.choice).change("bad"))  # BAD
