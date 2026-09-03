# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""DataflowOps and their operation-owned Design inventories.

``finn.dataflow.ops.base`` holds the generic contract -- one source node, one
frozen problem, one root occurrence, one persistence authority -- and
``source`` and ``association`` hold the two value families it reads and
produces.  Concrete operations live beside their own Designs, for example
``finn.dataflow.ops.mvau``.

This namespace performs no eager import.  Importing the operation layer would
pull in QONNX and every Design and Kernel behind it, and the generic Space
substrate must stay usable without any of that.
"""

__all__: list[str] = []
