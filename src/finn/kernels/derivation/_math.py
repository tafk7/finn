############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Vendored into FINN from brainsmith._internal.math for the dataflow-kernel
# derivation layer (self-contained, no cross-repo dependency).
############################################################################
"""Mathematical utilities for the vendored dataflow derivation layer."""


def divisors(n: int) -> set[int]:
    """Return all divisors of n.

    Uses sqrt optimization: only check up to sqrt(n), add both i and n/i.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    result = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            result.add(i)
            result.add(n // i)
    return result


__all__ = ["divisors"]
