# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Operation-owned conditional input-supply authoring boundary.

The implementation remains byte-for-byte compatible with the original
``authoring.design`` definitions during the split; contributors import this
focused module and do not depend on inventory or realization internals.
"""

from finn.dataflow.authoring.design import (
    InputSupplyAlternative,
    InputSupplyContext,
    InputSupplyDeclaration,
    SupplierAttachment,
    attach_supplier_network,
    declare_input_supply,
)

__all__ = [
    "InputSupplyAlternative",
    "InputSupplyContext",
    "InputSupplyDeclaration",
    "SupplierAttachment",
    "attach_supplier_network",
    "declare_input_supply",
]
