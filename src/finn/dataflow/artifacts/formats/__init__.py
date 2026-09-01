# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Distribution formats.

Each is a projection of one ``PortableComponent`` and reads nothing else.
Adding one costs an implementation, a registration, a round-trip conformance
test, and integration evidence -- and it must not require a Kernel or
Operation edit when the ``ComponentABI`` already carries what it needs.
"""

from finn.dataflow.artifacts.formats.rtl_module import RtlModuleDirectory
from finn.dataflow.artifacts.formats.tar import DeterministicTar

__all__ = ["DeterministicTar", "RtlModuleDirectory"]
