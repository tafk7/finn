# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""ActivationReplay: the second operation, and the reason there is one.

One input, one output, no matrix, no initializer, one Design with no
alternatives and no selector.  It exists so that the DataflowOp lifecycle has
to work for something that is not MVAU: every piece of MVAU-shaped thinking
that leaked into the generic layer -- a mandatory weight operand, an assumed
Design choice, a two-node Network -- fails here and nowhere else.

It is not a toy.  The Region it selects is the canonical activation replay, the
Kernel is the reusable ``ReplayBufferKernel`` that carries its own numerical and
synthesis evidence, and the folding it exposes is the real one.
"""

__all__: list[str] = []
