# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The class-based oracle's tool (``tool``) with ``targets.TARGETS``: the toys and stage 7's
chain. TFC's target reads FINN's builder, so its tool is ``support.space_oracle_tfc``'s.

    python -m support.space_oracle dump TARGET OUT.json [--implementation M:A] [--permuted]
    python -m support.space_oracle diff A.json B.json
    python -m support.space_oracle walk TARGET --seeds A-B --steps N --against M:A
"""

import sys

from support.space_oracle.tool import main

sys.exit(main())
